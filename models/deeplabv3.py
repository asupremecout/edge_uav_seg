import torch
import torch.nn as nn
import torch.nn.functional as F



class BasicBlock(nn.Module):
    def __init__(self,in_channels,out_channels,dilation=1,expansion=4,stride=1):
        super().__init__()
        self.expansion=expansion
        mid_channels=in_channels//expansion

        self.conv1=nn.Conv2d(in_channels,mid_channels,kernel_size=1,stride=1,bias=False)
        self.bn1=nn.BatchNorm2d(mid_channels)
        self.relu1=nn.ReLU(inplace=True)

        self.conv2=nn.Conv2d(mid_channels,mid_channels,kernel_size=3,stride=stride,
                              padding=dilation,dilation=dilation,bias=False)
        self.bn2=nn.BatchNorm2d(mid_channels)
        self.relu2=nn.ReLU(inplace=True)

        self.conv3=nn.Conv2d(mid_channels,out_channels,kernel_size=1,stride=1,bias=False)
        self.bn3=nn.BatchNorm2d(out_channels)
        self.relu3=nn.ReLU(inplace=True)

        # 当通道数或空间尺寸变化时，对 identity 做 1×1 conv 投影
        if stride!=1 or in_channels!=out_channels:
            self.identity_proj=nn.Sequential(
                nn.Conv2d(in_channels,out_channels,kernel_size=1,stride=stride,bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.identity_proj=nn.Identity()

    def forward(self,x):
        identity=self.identity_proj(x)

        x=self.conv1(x)
        x=self.bn1(x)
        x=self.relu1(x)

        x=self.conv2(x)
        x=self.bn2(x)
        x=self.relu2(x)

        x=self.conv3(x)
        x=self.bn3(x)

        x+=identity
        x=self.relu3(x)

        return x



class ResNet101_for_DeepLabV3(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1=nn.Conv2d(3,64,kernel_size=7,stride=2,padding=3,bias=False)
        self.bn1=nn.BatchNorm2d(64)
        self.relu=nn.ReLU(inplace=True)
        self.maxpool=nn.MaxPool2d(kernel_size=3,stride=2,padding=1)

        self.layer1=nn.Sequential(
            BasicBlock(64,256,expansion=1),
            BasicBlock(256,256,expansion=4),
            BasicBlock(256,256,expansion=4),
        )

        self.layer2=nn.ModuleList([
            BasicBlock(256,512,expansion=2,stride=2),
            BasicBlock(512,512,expansion=4,stride=1),
            BasicBlock(512,512,expansion=4,stride=1),
            BasicBlock(512,512,expansion=4,stride=1),
        ])

        self.layer3=nn.ModuleList([
            BasicBlock(512,1024,expansion=2,stride=2),
        ]+[BasicBlock(1024,1024,expansion=4,stride=1) for _ in range(22)])

        self.layer4=nn.ModuleList([
            BasicBlock(1024,2048,expansion=2,stride=1,dilation=2),
            BasicBlock(2048,2048,expansion=4,stride=1,dilation=2),
            BasicBlock(2048,2048,expansion=4,stride=1,dilation=2),
        ])

    def forward(self,x):
        x=self.conv1(x)
        x=self.bn1(x)
        x=self.relu(x)
        x=self.maxpool(x)

        x=self.layer1(x)
        low_feat=x
        for block in self.layer2:
            x=block(x)
        for block in self.layer3:
            x=block(x)
        for block in self.layer4:
            x=block(x)

        return x,low_feat



class ASPP(nn.Module):
    def __init__(self,in_channels,out_channels,dilations=[6,12,18]):
        super().__init__()

        self.con1x1=nn.Conv2d(in_channels=in_channels,out_channels=out_channels,
                              kernel_size=1)
        self.bn1=nn.BatchNorm2d(out_channels)
        self.relu=nn.ReLU(inplace=True)

        self.mutil_scale=nn.ModuleList()
        for dilation in dilations:
            temp_layer=nn.Sequential(
                nn.Conv2d(in_channels=in_channels,out_channels=out_channels,kernel_size=3,dilation=dilation,padding=dilation),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
                )
            self.mutil_scale.append(temp_layer)

        self.global_avg_pooling=nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=in_channels,out_channels=out_channels,kernel_size=1,bias=True),
            nn.ReLU(inplace=True)
        )

        self.conv_fuse=nn.Conv2d(in_channels=out_channels*(2+len(dilations)),out_channels=out_channels,kernel_size=1,bias=False)
        self.bn_fuse=nn.BatchNorm2d(out_channels)
        self.relu_fuse=nn.ReLU(inplace=True)
        self.dropout=nn.Dropout2d(0.1)



    def forward(self,x): #x comes from resnet101

        feature1=self.relu(self.bn1(self.con1x1(x)))

        feature2=[]
        for conv in self.mutil_scale:
            feature2.append(conv(x))

        feature3=self.global_avg_pooling(x)
        feature3=F.interpolate(feature3, size=x.shape[2:], mode='bilinear', align_corners=True)

        feature=torch.cat([feature1]+feature2+[feature3],dim=1)

        feature=self.relu_fuse(self.bn_fuse(self.conv_fuse(feature)))
        feature=self.dropout(feature)

        return feature



class Decoder(nn.Module):
    def __init__(self,low_level_in_channels,low_level_out_channels,num_classes):
        super().__init__()
        self.conv_low_feat=nn.Sequential(
            nn.Conv2d(in_channels=low_level_in_channels,out_channels=low_level_out_channels,kernel_size=1,bias=False),
            nn.BatchNorm2d(low_level_out_channels),
            nn.ReLU(inplace=True)
        )#处理来自resnet101的低级特征，降低通道数，便于后续与ASPP输出的特征融合

        self.conv_fuse=nn.Sequential(
            nn.Conv2d(in_channels=low_level_out_channels+256,out_channels=256,kernel_size=3,stride=1,padding=1,bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.5),
            nn.Conv2d(in_channels=256,out_channels=256,kernel_size=3,stride=1,padding=1,bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            nn.Conv2d(in_channels=256,out_channels=num_classes,kernel_size=1,stride=1)
        )

    def forward(self,x,low_feat):
        low_feat=self.conv_low_feat(low_feat)

        x=F.interpolate(x,size=(low_feat.shape[2],low_feat.shape[3]),mode='bilinear',align_corners=True) #上采样x

        x=torch.cat([x,low_feat],dim=1)

        x=self.conv_fuse(x)

        return x



class DeepLabV3(nn.Module):
    def __init__(self,num_classes=8):
        super().__init__()
        self.aspp=ASPP(in_channels=2048,out_channels=256)
        self.resnet101=ResNet101_for_DeepLabV3()
        self.decoder=Decoder(low_level_in_channels=256,low_level_out_channels=48,num_classes=num_classes)

    def forward(self,x):
        input_size=x.shape[-2:]           # 保存原始 H,W 用于最终上采样
        low_resolution,low_feat=self.resnet101(x)

        aspp_out=self.aspp(low_resolution)

        result=self.decoder(aspp_out,low_feat)

        result=F.interpolate(result,size=input_size,mode='bilinear',align_corners=True)  # 4×上采样

        return result


def get_DeepLabV3(num_classes=8):
    return DeepLabV3(num_classes=num_classes)


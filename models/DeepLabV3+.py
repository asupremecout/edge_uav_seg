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


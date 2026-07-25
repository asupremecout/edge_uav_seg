import torch
import torch.nn as nn
import torch.nn.functional as F



class OverlapPatchEmbed(nn.Module):
    def __init__(self,in_channels,out_channels,stride=2,kernel_size=7):
        super().__init__()
        self.proj=nn.Conv2d(in_channels=in_channels,out_channels=out_channels,
                            stride=stride,kernel_size=kernel_size,padding=stride-1)
        self.norm=nn.LayerNorm(out_channels)

    def forward(self,x):
        x=self.proj(x) #b*c*h*w
        _,_,H,W=x.shape

        x=x.flatten(2) #b*c*(h*w)

        x=x.transpose(1,2) #b*(h*w)*c

        x=self.norm(x)

        return x,H,W

class EfficientAttention(nn.Module):
    def __init__(self,embed_dim,num_heads=8,sr_ratio=1):
        super().__init__()
        self.num_head=num_heads
        self.sr=sr_ratio
        self.q=nn.Linear(embed_dim,embed_dim)
        self.conv_kv=nn.Conv2d(embed_dim,embed_dim,kernel_size=sr_ratio,stride=sr_ratio)
        self.kv=nn.Linear(embed_dim,embed_dim*2)
        self.softmax=nn.Softmax(dim=-1)

    def forward(self,x):
        batch_size,N,C=x.shape
        head_dim=C//self.num_head

        Q=self.q(x)
        Q=Q.reshape(batch_size,N,self.num_head,head_dim).permute(0,2,1,3)

        if self.sr>1:
            H,W=self.H,self.W
            original=x.reshape(batch_size,H,W,C).permute(0,3,1,2)
            conv_kv=self.conv_kv(original)
            _,_,H_sr,W_sr=conv_kv.shape
            conv_kv=conv_kv.flatten(2).transpose(1,2)
            kv=self.kv(conv_kv)
            kv=kv.reshape(batch_size,-1,2,self.num_head,head_dim).permute(0,3,1,2,4)
            k=kv[:,:,:,0,:]
            v=kv[:,:,:,1,:]
        else:
            kv=self.kv(x)
            kv=kv.reshape(batch_size,-1,2,self.num_head,head_dim).permute(0,3,1,2,4)
            k=kv[:,:,:,0,:]
            v=kv[:,:,:,1,:]

        atten=(Q @ k.transpose(-2,-1)) * (head_dim**-0.5)
        atten=self.softmax(atten)
        out=(atten @ v).transpose(1,2).reshape(batch_size,N,C)
        return out


class MixedFFN(nn.Module):
    def __init__(self,in_feature,hidden_feature):
        super().__init__()
        self.fc1=nn.Linear(in_feature,hidden_feature)
        self.gelu=nn.GELU()
        self.dwconv=nn.Conv2d(stride=1,padding=1,kernel_size=3,in_channels=hidden_feature,out_channels=hidden_feature)
        self.fc2=nn.Linear(hidden_feature,in_feature)

    def forward(self,x):
        H,W=self.H,self.W
        x=self.fc1(x)
        x=self.gelu(x)
        x=x.transpose(1,2)
        x=x.reshape(x.shape[0],-1,H,W)
        x=self.dwconv(x)
        x=x.flatten(2)
        x=x.transpose(1,2)
        x=self.fc2(x)
        return x



class SegFormer_Encoder(nn.Module):

    def __init__(self,N1,N2,N3,N4):
        super().__init__()

        self.layer1=nn.ModuleList()
        self.layer1.append(OverlapPatchEmbed(in_channels=3,out_channels=32,stride=4,kernel_size=7))
        for _ in range(N1):
            self.layer1.append(nn.Sequential(
                nn.LayerNorm(32),
                EfficientAttention(embed_dim=32,num_heads=1,sr_ratio=8),
                nn.LayerNorm(32),
                MixedFFN(in_feature=32,hidden_feature=32*4)
            ))

        self.layer2=nn.ModuleList()
        self.layer2.append(OverlapPatchEmbed(in_channels=32,out_channels=64,stride=2,kernel_size=3))
        for _ in range(N2):
            self.layer2.append(nn.Sequential(
                nn.LayerNorm(64),
                EfficientAttention(embed_dim=64,num_heads=2,sr_ratio=4),
                nn.LayerNorm(64),
                MixedFFN(in_feature=64,hidden_feature=64*4)
            ))

        self.layer3=nn.ModuleList()
        self.layer3.append(OverlapPatchEmbed(in_channels=64,out_channels=160,stride=2,kernel_size=3))
        for _ in range(N3):
            self.layer3.append(nn.Sequential(
                nn.LayerNorm(160),
                EfficientAttention(embed_dim=160,num_heads=5,sr_ratio=2),
                nn.LayerNorm(160),
                MixedFFN(in_feature=160,hidden_feature=160*4)
            ))

        self.layer4=nn.ModuleList()
        self.layer4.append(OverlapPatchEmbed(in_channels=160,out_channels=256,stride=2,kernel_size=3))
        for _ in range(N4):
            self.layer4.append(nn.Sequential(
                nn.LayerNorm(256),
                EfficientAttention(embed_dim=256,num_heads=8,sr_ratio=1),
                nn.LayerNorm(256),
                MixedFFN(in_feature=256,hidden_feature=256*4)
            ))

    def forward(self,x):
        out1,H1,W1=self.layer1[0](x)
        for i in range(1,len(self.layer1)):
            self.layer1[i][1].H,self.layer1[i][1].W=H1,W1
            self.layer1[i][3].H,self.layer1[i][3].W=H1,W1
            out1=out1+self.layer1[i](out1)

        out2,H2,W2=self.layer2[0](out1.transpose(1,2).reshape(out1.shape[0],-1,H1,W1))
        for i in range(1,len(self.layer2)):
            self.layer2[i][1].H,self.layer2[i][1].W=H2,W2
            self.layer2[i][3].H,self.layer2[i][3].W=H2,W2
            out2=out2+self.layer2[i](out2)

        out3,H3,W3=self.layer3[0](out2.transpose(1,2).reshape(out2.shape[0],-1,H2,W2))
        for i in range(1,len(self.layer3)):
            self.layer3[i][1].H,self.layer3[i][1].W=H3,W3
            self.layer3[i][3].H,self.layer3[i][3].W=H3,W3
            out3=out3+self.layer3[i](out3)

        out4,H4,W4=self.layer4[0](out3.transpose(1,2).reshape(out3.shape[0],-1,H3,W3))  
        for i in range(1,len(self.layer4)):
            self.layer4[i][1].H,self.layer4[i][1].W=H4,W4
            self.layer4[i][3].H,self.layer4[i][3].W=H4,W4
            out4=out4+self.layer4[i](out4)

        return out1,out2,out3,out4,H1,W1,H2,W2,H3,W3,H4,W4

class SegFormer_Decoder(nn.Module):
    def __init__(self,num_classes=8):
        super().__init__()
        self.conv1=nn.Conv2d(in_channels=32,out_channels=32,kernel_size=1)
        self.conv2=nn.Conv2d(in_channels=64,out_channels=32,kernel_size=1)
        self.conv3=nn.Conv2d(in_channels=160,out_channels=32,kernel_size=1)
        self.conv4=nn.Conv2d(in_channels=256,out_channels=32,kernel_size=1)

        # 融合后的分割头
        self.seg_head=nn.Sequential(
            nn.Conv2d(32*4,128,kernel_size=3,padding=1,bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128,num_classes,kernel_size=1)
        )

    def forward(self,out1,out2,out3,out4,H1,W1,H2,W2,H3,W3,H4,W4):
        def reshape(x,H,W):
            return x.transpose(1,2).reshape(x.shape[0],-1,H,W)

        f1=reshape(out1,H1,W1)
        f2=reshape(out2,H2,W2)
        f3=reshape(out3,H3,W3)
        f4=reshape(out4,H4,W4)

        # 全部上采样到第一层输出的大小（H1*4, W1*4 = 输入 1/4 分辨率）
        target_size=(H1*4,W1*4)
        f1=F.interpolate(f1,size=target_size,mode='bilinear',align_corners=True)
        f2=F.interpolate(f2,size=target_size,mode='bilinear',align_corners=True)
        f3=F.interpolate(f3,size=target_size,mode='bilinear',align_corners=True)
        f4=F.interpolate(f4,size=target_size,mode='bilinear',align_corners=True)

        f1=self.conv1(f1)
        f2=self.conv2(f2)
        f3=self.conv3(f3)
        f4=self.conv4(f4)

        feat=torch.cat([f1,f2,f3,f4],dim=1)

        out=self.seg_head(feat)

        return out


class SegFormer(nn.Module):
    def __init__(self,N1,N2,N3,N4,num_classes=8):
        super().__init__()
        self.encoder=SegFormer_Encoder(N1,N2,N3,N4)
        self.decoder=SegFormer_Decoder(num_classes)

    def forward(self,x):
        input_size=x.shape[-2:]
        out1,out2,out3,out4,H1,W1,H2,W2,H3,W3,H4,W4=self.encoder(x)

        result=self.decoder(out1,out2,out3,out4,H1,W1,H2,W2,H3,W3,H4,W4)

        result=F.interpolate(result,size=input_size,mode='bilinear',align_corners=True)

        return result


def get_segformer(num_layers):
    if num_layers==3:
        model=SegFormer(N1=2,N2=2,N3=2,N4=2)
    elif num_layers==4:
        model=SegFormer(N1=3,N2=4,N3=6,N4=3)
    else:
        raise ValueError("num_layers must be 3 or 4")

    return model











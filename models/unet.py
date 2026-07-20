import torch
import torch.nn as nn

import torch.nn.functional as F

class DoubleConv(nn.Module): #蓝色箭头
    def __init__(self,in_channels,out_channels):
        super().__init__()
        self.layer=nn.Sequential(

            nn.Conv2d(in_channels=in_channels,out_channels=out_channels,kernel_size=3),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels=out_channels,out_channels=out_channels,kernel_size=3),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self,x):
        return self.layer(x)


class  DownSample(nn.Module):
    def __init__(self,in_channels,out_channels):
        self.layer=nn.Sequential(
            nn.MaxPool2d(stride=2,kernel_size=3),
            DoubleConv(in_channels,out_channels)
        )
    def forward(self,x):
        return self.layer(x)

class UpSample(nn.Module): #处理上采样：转置卷积（恢复分辨率） + 跳跃连接（拼接特征）
    def __init__(self,in_channels,out_channels):
        super().__init__()
        self.layer=nn.Sequential(

            nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2) 

        )
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self,x,skip_connect): # x: 来自下层（深层语义），skip_connect: 来自同层编码器（浅层细节）
        x=self.layer(x)
# x2（来自编码器）：形状为 (B, C2, H, W)
# x1（上采样后的解码器特征）：形状为 (B, C1, H, W)
# （注意：二者空间尺寸 H、W 相同，这是通过前面 F.pad 对齐的结果）
# 执行 torch.cat([x2, x1], dim=1) 后：
# 新张量 x 的形状变为 (B, C2 + C1, H, W)
        diffY=skip_connect.shape[-2]-x.shape[-2]

        diffX=skip_connect.shape[-1]-x.shape[-1]
        x = F.pad(x, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])

        x=torch.cat([x,skip_connect],dim=1)

        return x

class OutConv(nn.Module):
    def __init__(self,in_channels,num_classes):
        super().__init__()
        self.layer=nn.Conv2d(in_channels=in_channels,out_channels=num_classes,kernel_size=1)

    def forward(self,x):
        return self.layer(x)

class UNet(nn.Module):
    def __init__(self,in_channels=3,num_classes=6,out_features=64):
        super().__init__()
        self.in_channels=in_channels
        self.num_classes=num_classes

        self.inc=DoubleConv(in_channels,64)
        self.down1=DownSample(64,128)
        self.down2=DownSample(128,256)
        self.down3=DownSample(256,512)
        self.down4=DownSample(512,1024)

        self.up1=UpSample(1024,512)
        self.up2=UpSample(512,256)
        self.up3=UpSample(256,128)
        self.up4=UpSample(128,64)

        self.outc=OutConv(64,num_classes)

    def forward(self,x):
        x1=self.inc(x)
        x2=self.down1(x1)
        x3=self.down2(x2)
        x4=self.down3(x3)
        x5=self.down4(x4)

        x=self.up1(x5,x4)
        x=self.up2(x,x3)
        x=self.up3(x,x2)
        x=self.up4(x,x1)

        logits=self.outc(x)

        return logits
    
def get_unet(in_channels=3,num_classes=6,out_features=64):
    return UNet(in_channels=in_channels,num_classes=num_classes,out_features=out_features)

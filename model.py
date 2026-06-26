import torch.nn as nn
import torch
import torch.nn.functional as F
from torchvision.models import vgg16
from config import *

#普通残差网络
class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1),
            # nn.InstanceNorm2d(channels, eps=1e-5), 
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, 1, 1),
            # nn.InstanceNorm2d(channels, eps=1e-5), 
        )
    def forward(self, x):
        return x + self.conv(x)

#空洞残差网络
class DilatedResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels, 3, stride=1, padding=2, dilation=2),
            # nn.InstanceNorm2d(channels, eps=1e-5),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(channels, channels, 3, stride=1, padding=1, dilation=1),
            # nn.InstanceNorm2d(channels, eps=1e-5),
        )
        
    def forward(self, x):
        return x + self.conv(x)

#分解网络
class Decomposition(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(4, 64, 3, padding=1),
            nn.ReLU(True),
        )
        
        self.res_blocks = nn.Sequential(
            DilatedResBlock(64),
            DilatedResBlock(64),
            DilatedResBlock(64), 
        )
        
        self.conv2 = nn.Conv2d(64, 4, 3, padding=1)
        self.illumination_scale = 0.2
        self.reflectance_scale = 0.1

    def forward(self, img):
        x_max = torch.max(img, dim=1, keepdim=True)[0]
        
        x_input = torch.cat([img, x_max], dim=1)
        features = self.conv1(x_input)
        features = self.res_blocks(features)
        features = self.conv2(features)
        
        # R，L调节因子
        R_residual = torch.tanh(features[:, 0:3, :, :]) * self.reflectance_scale
        I_residual = torch.tanh(features[:, 3:4, :, :]) * self.illumination_scale

        L = torch.clamp(x_max + I_residual, min=1e-4, max=1.0)

        R_base = torch.clamp(img / (L + 1e-4), min=0.0, max=1.0)
        R = torch.clamp(R_base * (1.0 + R_residual), min=0.0, max=1.0)
        return R, L

#判别器
class Discriminator(nn.Module):
    def __init__(self, input_channels=3):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(input_channels, 64, 4, stride=2, padding=1),
            nn.InstanceNorm2d(64),
            nn.LeakyReLU(0.2),
            
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2),
            
            nn.Conv2d(128, 256, 4, stride=2, padding=1),
            nn.InstanceNorm2d(256),
            nn.LeakyReLU(0.2),
            
            nn.Conv2d(256, 512, 4, stride=1, padding=1), 
            nn.InstanceNorm2d(512),
            nn.LeakyReLU(0.2),
            
            nn.Conv2d(512, 1, 4, stride=1, padding=1),
        )

    def forward(self, x):
        return self.model(x)

#光照转化网络
class LCNet(nn.Module):
    def __init__(self,mode = "brighten"):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1, 64, 3, 1, 1),    
            nn.ReLU(inplace=True),
            ResBlock(64),     
            ResBlock(64),
            ResBlock(64),     
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, 3, 1, 1),
        )
        self.mode = mode
        self.sigmoid = nn.Sigmoid()
        self.iterations = 3
    
    def forward(self, x):
        delta_L = self.model(x)
        strength = 0.8 * self.sigmoid(delta_L)
        out = x
        #防止过曝，利用二次曲线，限制图像增亮或变暗的力度
        for _ in range(self.iterations):
            curve = out * (1.0 - out)
            if self.mode == "brighten":
                out = out + strength * curve
            else:
                out = out - strength * curve
        return torch.clamp(out, 0.0, 1.0)

#unet结构清洗R
class UNetDenoise(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        super(UNetDenoise, self).__init__()
        
        # --- 下采样 ---
        self.enc1 = self.conv_block(in_channels, 64)
        self.pool1 = nn.MaxPool2d(2)
        
        self.enc2 = self.conv_block(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        
        self.bottleneck = self.conv_block(128, 256)
        
        self.upconv2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.dec2 = self.conv_block(256, 128) 
        
        self.upconv1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.dec1 = self.conv_block(128, 64)
        
        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool1(enc1))

        bottleneck = self.bottleneck(self.pool2(enc2))

        dec2 = self.upconv2(bottleneck)
        dec2 = torch.cat((dec2, enc2), dim=1) 
        dec2 = self.dec2(dec2)
        
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.dec1(dec1)
        
        return x + 0.1 * self.final_conv(dec1)
    

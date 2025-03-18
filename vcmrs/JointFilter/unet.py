import torch
import torch.nn as nn
import torch.nn.functional as F

from e2evc.Utils import utils_nn

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super(DoubleConv, self).__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            #nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            utils_nn.IntConv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            #nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            utils_nn.IntConv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)
    
    
class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)
    
    
class Up(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        #self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.up = utils_nn.IntTransposedConv2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)
    

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        #self.conv = nn.Conv2d(in_channels, out_channels, kernel_si
        self.conv = utils_nn.IntConv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)
    
    

# Split the model into two parts: encoder and decoder
class Encoder(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        super(Encoder, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.inc = (DoubleConv(in_channels, 32))
        self.down1 = (Down(32, 64))
        self.down2 = (Down(64, 128))
        self.down3 = (Down(128, 256))
        self.down4 = (Down(256, 512))

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1) # 64
        x3 = self.down2(x2) # 128
        x4 = self.down3(x3) # 256
        x5 = self.down4(x4) # 512
        return x5, x4, x3, x2, x1

class Decoder(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        super(Decoder, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.up1 = (Up(512, 256))
        self.up2 = (Up(256, 128))
        self.up3 = (Up(128, 64))
        self.up4 = (Up(64, 32))
        self.outc = (OutConv(32, out_channels))

    def forward(self, x5, x4, x3, x2, x1):
        x = self.up1(x5, x4) # 256
        x = self.up2(x, x3) # 128
        x = self.up3(x, x2) # 64
        x = self.up4(x, x1) # 32
        logits = self.outc(x)
        return logits
    
class Unet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        super(Unet, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.inc = (DoubleConv(in_channels, 32))
        self.down1 = (Down(32, 64))
        self.down2 = (Down(64, 128))
        self.down3 = (Down(128, 256))
        self.down4 = (Down(256, 512))
        self.up1 = (Up(512, 256))
        self.up2 = (Up(256, 128))
        self.up3 = (Up(128, 64))
        self.up4 = (Up(64, 32))
        self.outc = (OutConv(32, out_channels))

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits
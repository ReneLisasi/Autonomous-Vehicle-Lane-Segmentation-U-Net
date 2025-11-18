import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder_Block(nn.Module):
    def __init__(self, in_channels, out_channels, First_Block = False):
        super(Encoder_Block, self).__init__()
        if First_Block == True:
            self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        else:
            self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=False)

        self.bn1 = nn.BatchNorm2d(out_channels)
        self.ReLU = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        if in_channels != out_channels:
            if First_Block == True:
                self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False)
            else:
                self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=2, bias=False)

    def forward(self, x):
        residual = x
        if hasattr(self, 'conv3'):
            residual = self.conv3(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.ReLU(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        out = self.ReLU(out)
        return out



class Decoder_Block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Decoder_Block, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.ReLU1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.ReLU2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        if in_channels != out_channels:
            self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)

    def forward(self, x):
        residual = x
        if hasattr(self, 'conv3'):
            residual = self.conv3(x)
        out = self.bn1(x)
        out = self.ReLU1(out)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.ReLU2(out)
        out = self.conv2(out)
        out += residual
        return out






class Res_UNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Res_UNet, self).__init__()
        #Encoder 1 64
        self.encoder_1 = Encoder_Block(in_channels, 64, First_Block = True)
        #Encoder 2 128
        self.encoder_2 = Encoder_Block(64, 128)
        #Encoder 3 256
        self.encoder_3 = Encoder_Block(128, 256)
        #Encoder 4 512
        self.encoder_4 = Encoder_Block(256, 512)
        
        #Decoder 1 256
        self.decoder_1 = Decoder_Block(512, 256)
        #Decoder 2 128
        self.decoder_2 = Decoder_Block(256, 128)
        #Decoder 3 64
        self.decoder_3 = Decoder_Block(128, 64)

        #to fit the shape
        self.conv1 = nn.Conv2d(512, 256, kernel_size=1, bias=False)
        self.conv2 = nn.Conv2d(256, 128, kernel_size=1, bias=False)
        self.conv3 = nn.Conv2d(128, 64, kernel_size=1, bias=False)

        #Final output 1
        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1, bias=False)

    def forward(self, x):
        #Encoder 1
        e_64 = self.encoder_1(x)
        #Encoder 2
        e_128 = self.encoder_2(e_64)
        #Encoder 3
        e_256 = self.encoder_3(e_128)
        #Encoder 4
        e_512 = self.encoder_4(e_256)

        #Decoder 1
        u1 = F.interpolate(e_512, scale_factor=2, mode='nearest')
        u1 = self.conv1(u1)
        c1 = torch.cat([u1, e_256], dim=1)
        d_256 = self.decoder_1(c1)

        #Decoder 2
        u2 = F.interpolate(d_256, scale_factor=2, mode='nearest')
        u2 = self.conv2(u2)
        c2 = torch.cat([u2, e_128], dim=1)
        d_128 = self.decoder_2(c2)

        #Decoder 3
        u3 = F.interpolate(d_128, scale_factor=2, mode='nearest')
        u3 = self.conv3(u3)
        c3 = torch.cat([u3, e_64], dim=1)
        d_64 = self.decoder_3(c3)

        #Final output
        logits  = self.final_conv(d_64)
        output = torch.sigmoid(logits)

        return output



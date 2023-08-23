""" Full assembly of the parts to form the complete network """

from pdb import set_trace
import torch.nn.functional as F

from .unet_parts import *


class UNet(nn.Module):
    def __init__(self, n_channels=3, n_classes=4, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        factor = 2 if bilinear else 1
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)
        self.out1 = OutConv(64, 2)
        self.out2 = OutConv(64, 3)
        self.out3 = OutConv(64, 2)
        self.out4 = OutConv(64, 3)
    def forward(self, x):
        #print(x.size())
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        #import pdb;pdb.set_trace()
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        #print('net1',x.size())
        x = self.up2(x, x3)
        #print(x.size())
        x = self.up3(x, x2)
        #print(x.size())
        x = self.up4(x, x1)
        #print(x.size())
        logits = self.outc(x)
        
        return logits,x5#,x5#,logits1,logits2,logits3,logits4




class UNet_ssl(nn.Module):
    def __init__(self, n_channels=3, n_classes=4, bilinear=True):
        super(UNet_ssl, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        factor = 2 if bilinear else 1
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024 // factor)

        self.rotate_classifier = nn.Linear(1024, n_classes)
        self.pooling=nn.AdaptiveAvgPool2d(1)
    def forward(self, x1,x2):
        f1=self.forward_one(x1)
        f1=f1.view(f1.shape[0],-1)
        f2=self.forward_one(x2)
        f2=f2.view(f2.shape[0],-1)
        r_outputs = self.rotate_classifier(torch.cat((f1, f2), 1))
        return r_outputs
    def forward_one(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x5=self.pooling(x5)
        return x5#,x5#,logits1,logits2,logits3,logits4

class UNet2(nn.Module):
    def __init__(self, n_channels=3, n_classes=3, bilinear=True):
        super(UNet2, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        factor = 2 if bilinear else 1
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up41 = Up(128, 64, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outifta = OutConv(64, n_classes)
        self.outta = OutConv(64,2)
        
    def forward(self, x):
        #print(x.size())
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        #import pdb;pdb.set_trace()
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        #print('net1',x.size())
        x = self.up2(x, x3)
        #print(x.size())
        x = self.up3(x, x2)
        #print(x.size())
        f_ifta = self.up4(x, x1)
        f_ta=self.up41(x, x1)
        #print(x.size())
        logits = self.outifta(f_ifta)
        f_ta=torch.mul(f_ta,f_ifta)

        logits1 = self.outta(f_ta)
        
        return logits1,x5



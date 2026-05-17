import torch
import torch.nn as nn
import torch.nn.functional as F

class SegNet(nn.Module):
    def __init__(self, n_channels=1, n_classes=2):
        super(SegNet, self).__init__()

        # Encoder
        self.enc_conv1 = nn.Sequential(
            nn.Conv2d(n_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.enc_conv2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.enc_conv3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        
        self.pool = nn.MaxPool2d(2, stride=2, return_indices=True)

        # Decoder
        self.unpool = nn.MaxUnpool2d(2, stride=2)
        
        self.dec_conv3 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.dec_conv2 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.dec_conv1 = nn.Sequential(
            nn.Conv2d(64, n_classes, kernel_size=3, padding=1)
        )

    def forward(self, x):
        # Encoder
        x = self.enc_conv1(x)
        x, id1 = self.pool(x)
        
        x = self.enc_conv2(x)
        x, id2 = self.pool(x)
        
        x = self.enc_conv3(x)
        x, id3 = self.pool(x)
        
        # Decoder
        x = self.unpool(x, id3)
        x = self.dec_conv3(x)
        
        x = self.unpool(x, id2)
        x = self.dec_conv2(x)
        
        x = self.unpool(x, id1)
        x = self.dec_conv1(x)
        
        return x

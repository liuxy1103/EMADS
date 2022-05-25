from operator import mod
import torch
import torch.nn as nn
import torch.nn.functional as F

class UNet3D(nn.Module):
    def __init__(self, in_channel=1, n_classes=2, bn=False):
        self.in_channel = in_channel
        self.n_classes = n_classes
        super(UNet3D, self).__init__()
        self.ec0 = self.encoder(self.in_channel, 8, bias=False, batchnorm=bn)
        self.ec1 = self.encoder(8, 16, bias=False, batchnorm=bn)
        self.ec2 = self.encoder(16, 16, bias=False, batchnorm=bn)
        self.ec3 = self.encoder(16, 32, bias=False, batchnorm=bn)
        self.ec4 = self.encoder(32, 32, bias=False, batchnorm=bn)
        self.ec5 = self.encoder(32, 64, bias=False, batchnorm=bn)
        self.ec6 = self.encoder(64, 64, bias=False, batchnorm=bn)
        self.ec7 = self.encoder(64, 128, bias=False, batchnorm=bn)

        self.pool0 = nn.MaxPool3d(2)
        self.pool1 = nn.MaxPool3d(2)
        self.pool2 = nn.MaxPool3d(2)

        self.dc9 = self.decoder(128, 128, kernel_size=3, stride=1, bias=False, padding=1)  # kernel_size=2, stride=2
        self.dc8 = self.decoder(64 + 128, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.dc7 = self.decoder(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.dc6 = self.decoder(64, 64, kernel_size=3, stride=1, bias=False, padding=1)  # kernel_size=2, stride=2
        self.dc5 = self.decoder(32 + 64, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.dc4 = self.decoder(32, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.dc3 = self.decoder(32, 32, kernel_size=3, stride=1, bias=False, padding=1)  # kernel_size=2, stride=2
        self.dc2 = self.decoder(16 + 32, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.dc1 = self.decoder(16, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.dc0 = self.decoder_sigmoid(16, n_classes, kernel_size=1, stride=1, bias=False)


    def encoder(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1,
                bias=True, batchnorm=False):
        if batchnorm:
            layer = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias),
                nn.BatchNorm3d(out_channels),
                nn.ReLU())
        else:
            layer = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias),
                nn.ReLU())
        return layer

    def decoder_sigmoid(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
                bias=True):
        layer = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride,
                               padding=padding, bias=bias),
            # nn.Sigmoid()
        )
        return layer

    def decoder(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
                bias=True):
        layer = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride,
                               padding=padding, bias=bias),
            nn.ReLU()
        )
        return layer

    def forward(self, x):
        e0 = self.ec0(x)
        syn0 = self.ec1(e0)
        e1 = self.pool0(syn0)
        e2 = self.ec2(e1)
        syn1 = self.ec3(e2)
        del e0, e1, e2

        e3 = self.pool1(syn1)
        e4 = self.ec4(e3)
        syn2 = self.ec5(e4)
        del e3, e4

        e5 = self.pool2(syn2)
        e6 = self.ec6(e5)
        e7 = self.ec7(e6)     
        del e5, e6

        d9 = torch.cat((self.dc9(F.interpolate(e7, size=syn2.shape[-3:], mode="trilinear")), syn2), dim=1)
        del e7, syn2

        d8 = self.dc8(d9)
        d7 = self.dc7(d8)
        del d9, d8

        d6 = torch.cat((self.dc6(F.interpolate(d7, size=syn1.shape[-3:], mode="trilinear")), syn1), dim=1)
        del d7, syn1

        d5 = self.dc5(d6)
        d4 = self.dc4(d5)
        del d6, d5

        d3 = torch.cat((self.dc3(F.interpolate(d4, size=syn0.shape[-3:], mode="trilinear")), syn0), dim=1)
        del d4, syn0

        d2 = self.dc2(d3)
        d1 = self.dc1(d2)
        del d3, d2

        d0 = self.dc0(d1)
        return d0

if __name__ == '__main__':
    input = torch.ones((1,1,62, 459, 459))
    model = UNet3D()
    out = model(input)
    print(out.shape)
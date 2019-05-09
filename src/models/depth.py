import torch.nn as nn
from src.models.utils import *

class depth(nn.Module):

    def __init__(self, feature_scale=1, n_classes=21, is_deconv=True, in_channels=3, is_batchnorm=True,n_bits = 8):
        super(depth, self).__init__()
        self.is_deconv      = is_deconv
        self.in_channels    = in_channels
        self.is_batchnorm   = is_batchnorm
        self.feature_scale  = feature_scale
        self.n_bits = n_bits
        filters = [32,64,128, 256, 512,512]
        filters = [int(x / self.feature_scale) for x in filters]
        # upsampling
        self.up_concat4 = Up_recon(filters[5], filters[4], self.is_deconv)
        self.up_concat3 = Up_recon(filters[4], filters[3], self.is_deconv)
        self.up_concat2 = Up_recon(filters[3], filters[2], self.is_deconv)
        self.up_concat1 = Up_recon(filters[2], filters[1], self.is_deconv)
        self.up_concat0 = Up_recon(filters[1], filters[0], self.is_deconv)
        # final conv (without any concat)
        self.final = nn.Conv2d(filters[0], n_classes, 1)

    def forward(self, inputs):
        up4 = self.up_concat4(inputs)
        up3 = self.up_concat3(up4)
        up2 = self.up_concat2(up3)
        up1 = self.up_concat1(up2)
        up0 = self.up_concat0(up1)
        final = self.final(up0)
        return final

class depth_L(nn.Module):
    def __init__(self, feature_scale=1, n_classes=21, is_deconv=True, in_channels=3, is_batchnorm=True, n_bits=8):
        super(depth_L, self).__init__()
        self.is_deconv = is_deconv
        self.in_channels = in_channels
        self.is_batchnorm = is_batchnorm
        self.feature_scale = feature_scale
        self.n_bits = n_bits
        filters = [32, 64, 128, 256, 256]
        filters = [int(x / self.feature_scale) for x in filters]
        # upsampling
        self.up_concat4 = Up_recon(filters[4], filters[3], self.is_deconv)
        self.up_concat3 = Up_recon(filters[3], filters[2], self.is_deconv)
        self.up_concat2 = Up_recon(filters[2], filters[1], self.is_deconv)
        self.up_concat1 = Up_recon(filters[1], filters[0], self.is_deconv)
        self.final = nn.Conv2d(filters[0], n_classes, 1)

    def forward(self, inputs):
        up4 = self.up_concat4(inputs)
        up3 = self.up_concat3(up4)
        up2 = self.up_concat2(up3)
        up1 = self.up_concat1(up2)
        final = self.final(up1)
        return final
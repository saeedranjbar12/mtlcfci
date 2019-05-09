import torch
import torch.nn as nn
from torch.autograd import Variable

#=========================================================
#            Reconstruction
#=========================================================
class Conv2_recon(nn.Module):
    def __init__(self, in_size, out_size, is_batchnorm):
        super(Conv2_recon, self).__init__()
        # saeed added padding  ======================> ADD LEAKY RELU  LeakyReLU
        self.conv1 = nn.Sequential(nn.Conv2d(in_size, out_size, 3, 1, 1), 
                                   ) #nn.BatchNorm2d(out_size) #,
        self.conv2 = nn.Sequential(nn.Conv2d(out_size, out_size, 3, 1, 1),
                                   ) #nn.BatchNorm2d(out_size), #,nn.ReLU()

    def forward(self, inputs):
        outputs = self.conv1(inputs)
        #outputs = self.conv2(outputs)
        return outputs

class Up_recon(nn.Module):
    def __init__(self, in_size, out_size, is_deconv):
        super(Up_recon, self).__init__()
        self.conv = Conv2_recon(out_size, out_size, True)
        if is_deconv:
            self.up = nn.ConvTranspose2d(in_size, out_size, kernel_size=2, stride=2)
        else:
            self.up = nn.UpsamplingBilinear2d(scale_factor=2)

    def forward(self, inputs2):
        outputs2 = self.up(inputs2)
        return self.conv((outputs2))

#=========================================================
#            SEGMENTATION
#=========================================================
class Conv2_discon(nn.Module):
    def __init__(self, in_size, out_size, is_batchnorm):
        super(Conv2_discon, self).__init__()
        #saeed added padding  ======================> ADD LEAKY RELU  LeakyReLU
        if is_batchnorm:
            self.conv1 = nn.Sequential(nn.Conv2d(in_size, out_size, 3, 1, 1),
                                       )#nn.ReLU() #nn.BatchNorm2d(out_size),
            self.conv2 = nn.Sequential(nn.Conv2d(out_size, out_size, 3, 1, 1),
                                      ) # nn.ReLU(), #nn.BatchNorm2d(out_size),
        else:
            self.conv1 = nn.Sequential(nn.Conv2d(in_size, out_size, 3, 1, 1),)
            self.conv2 = nn.Sequential(nn.Conv2d(out_size, out_size, 3, 1, 1),)
    def forward(self, inputs):
        outputs = self.conv1(inputs) 
        #outputs = self.conv1(outputs)
        return outputs

#======================================================================
class Up_disconnected(nn.Module):
    def __init__(self, in_size, out_size, is_deconv):
        super(Up_disconnected, self).__init__()
        self.conv = Conv2_discon(out_size, out_size, True)
        if is_deconv:
            self.up = nn.ConvTranspose2d(in_size, out_size, kernel_size=2, stride=2)
        else:
            self.up = nn.UpsamplingBilinear2d(scale_factor=2)

    def forward(self, inputs2):
        outputs2 = self.up(inputs2)
        return self.conv((outputs2))



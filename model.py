import torch.nn as nn
import torch.nn.functional as F
import torch
import torchvision.transforms as transforms

#----------------------
#      Conv unit
#----------------------
class ConvUnit(nn.Module):
    def __init__(self, in_layers, out_layers, kernel_size=4, stride=1, padding=0, normalize=False):
        super(ConvUnit, self).__init__()
        layers = [nn.Conv2d(in_layers, out_layers, kernel_size, stride, padding)]
        if(normalize):
            layers.append(nn.InstanceNorm2d(out_layers))
        layers.append(nn.ReLU())

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

#----------------------
#      Conv unit
#----------------------
class DeconvUnit(nn.Module):
    def __init__(self, in_layers, out_layers, kernel_size=4, stride=1, padding=0, normalize=False):
        super(DeconvUnit, self).__init__()
        layers = [nn.ConvTranspose2d(in_layers, out_layers, kernel_size, stride, padding)]
        if(normalize):
            layers.append(nn.InstanceNorm2d(out_layers))
        layers.append(nn.ReLU())

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

#----------------------
#     SR network
#----------------------
class SRNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        super(SRNet, self).__init__()
        conv1 = ConvUnit(in_channels, 64, 3, 1, 1)
        conv_layers = [ConvUnit(64, 64, 3, 1, 1) for i in range(2)]
        deconv_layers = [DeconvUnit(64, 64, 3, 1, 1) for i in range(2)]
        fin = DeconvUnit(64, out_channels, 3, 1, 1)
        self.model = nn.Sequential(conv1, *conv_layers, *deconv_layers, fin)

    def forward(self, x):
        return (x + self.model(x))

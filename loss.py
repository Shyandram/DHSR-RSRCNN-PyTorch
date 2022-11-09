import torch
import torchvision as tv
from torch import nn
from torchvision.models.vgg import vgg19_bn
class ContentLoss(nn.Module):
    def __init__(self, critirion='mse', device='cpu'):
        super().__init__()
        # VGGLoss
        vgg = vgg19_bn(pretrained=True
                       )
        loss_network = nn.Sequential(*list(vgg.features)).eval()
        for param in loss_network.parameters():
            param.requires_grad = False
        self.loss_network = loss_network.to(device)
        if critirion=='mse':
            self.critirion_loss = nn.MSELoss()
        else:
            self.critirion_loss = critirion
    def __call__(self, img, label):
        img_f = self.loss_network(img)
        label_f = self.loss_network(label)
        perception_loss = self.critirion_loss(img_f, label_f)
        return perception_loss
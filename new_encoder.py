import torch.nn as nn
from torchvision.models import densenet161, resnet152, vgg19
# import warnings
# warnings.filterwarnings("ignore") 

class Encoder(nn.Module):
    def __init__(self, network='resnet152'):
        super(Encoder, self).__init__()
        self.network = network
        if network == 'resnet152':
            self.net = resnet152(pretrained=True)
            self.net = nn.Sequential(*list(self.net.children())[:-2])
            self.dim = 2048
        elif network == 'densenet161':
            self.net = densenet161(pretrained=True)
            self.net = nn.Sequential(*list(list(self.net.children())[0])[:-1])
            self.dim = 1920
        else:
            self.net = vgg19(pretrained=True)
            self.net = nn.Sequential(*list(self.net.features.children())[:-1])
            self.dim = 512

    def forward(self, x):
        x = self.net(x)
        # print(x.size())
        
        # x = x.permute(0, 2, 3, 1)
        x = x.view(x.size(0), x.size(1), -1)
        # print(x.size())
        # exit()
        return x
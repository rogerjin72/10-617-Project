import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class Projector_(nn.Module):
    def __init__(self, expansion=0):
        super(Projector_, self).__init__()
        self.linear_1 = nn.Linear(512*expansion, 2048)
        self.linear_2 = nn.Linear(2048, 128)
    
    def forward(self, x):   
        output = self.linear_1(x)
        output = F.relu(output)
        output = self.linear_2(output)
        return output


def ResNet18(num_classes, contrastive_learning):
    model = models.resnet18(pretrained=True)
    if contrastive_learning:
        model.fc = Projector_(expansion=1)
    else:
        model.fc = nn.Linear(512, num_classes)
    
    return model


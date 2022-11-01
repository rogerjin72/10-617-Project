
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchsummary import summary

class Projector(nn.Module):
    def __init__(self, expansion=0):
        super(Projector, self).__init__()
        self.linear_1 = nn.Linear(512*expansion, 2048)
        self.linear_2 = nn.Linear(2048, 128)
    
    def forward(self, x):   
        output = self.linear_1(x)
        output = F.relu(output)
        output = self.linear_2(output)
        return output

def ResNet18(num_classes, contrastive_learning):
    model = models.resnet18(weights='ResNet18_Weights.IMAGENET1K_V1')
    if contrastive_learning:
        model.fc = Projector(expansion=1)
    else:
        model.fc = nn.Linear(512, num_classes)
    return model


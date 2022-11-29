
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchsummary import summary

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


from models.resnet import ResNet18_
from models.projector import Projector

class AuthorResnet(nn.Module):
    def __init__(self, num_classes, contrastive_learning):
        super(AuthorResnet, self).__init__()
        self.model = ResNet18_(num_classes,contrastive_learning)
        self.fc = Projector(expansion=1)
    
    def forward(self, x):
        out = self.model(x)
        out = self.fc(out)
        return out
         
def ResNet18(num_classes, contrastive_learning):
    # weights = models.ResNet18_Weights
    # kw = {"replace_stride_with_dilation":[True, True, True]}
    model = models.resnet18(pretrained=True)
    if contrastive_learning:
        model.fc = Projector_(expansion=1)
    else:
        model.fc = nn.Linear(512, num_classes)
    
    # model = AuthorResnet(num_classes, contrastive_learning)
    return model


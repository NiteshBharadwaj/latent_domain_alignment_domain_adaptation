import torch.nn as nn
import torch.nn.functional as F
from grad_reverse import grad_reverse
import torch.utils.model_zoo as model_zoo
from torchvision.models import AlexNet
import torchvision.models as models
import torch

def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1 or classname.find('ConvTranspose2d') != -1:
        nn.init.kaiming_uniform_(m.weight)
        nn.init.zeros_(m.bias)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.zeros_(m.bias)
    elif classname.find('Linear') != -1:
        nn.init.xavier_normal_(m.weight)
        nn.init.zeros_(m.bias)

model_urls = {
    'alexnet': 'https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth',
}
        
class Feature(nn.Module):
    def __init__(self, fineTune=False):
        super(Feature, self).__init__()
        self.model = models.alexnet(pretrained=True)
        self.features = self.model.features
        self.avgpool = self.model.avgpool
        self.classifier = nn.Sequential()
        self.avgpool.requires_grad = True
        for i in range(6):
            self.classifier.add_module("classifier"+str(i), self.model.classifier[i])
        for param in self.classifier:
            param.requires_grad = True
        for i,param in enumerate(self.features):
            if(i < 13):
                param.requires_grad = True
    
    
    def forward(self, x, reverse=False):
        
        x = self.features(x)
        x = self.avgpool(x)
        x_feat = x.view(x.size(0), -1)
        x = self.classifier(x_feat)
        return x, x_feat
    
    def get_parameters(self):
        parameters_list = [{"params":self.features.parameters(), "lr_mult":1, 'decay_mult':2}, \
                            {"params":self.classifier.parameters(), "lr_mult":1, 'decay_mult':2}]
        return parameters_list

class Predictor(nn.Module):
    def __init__(self, num_classes):
        super(Predictor, self).__init__()
        self.fc = nn.Linear(4096, num_classes)
        self.fc.apply(init_weights)
        self.fc.requires_grad = True
        
    def forward(self,x,reverse=True):
        x = self.fc(x)
        return x
    
    def get_parameters(self):
        parameters_list = [{"params":self.fc.parameters(), "lr_mult":10, 'decay_mult':2}]
        return parameters_list
    
class DomainPredictor(nn.Module):
    def __init__(self, num_domains, prob=0.5):
        super(DomainPredictor, self).__init__()
        self.feature = Feature(fineTune=True)
        for param in self.feature.classifier.parameters():
            param.requires_grad = True
        self.predictor = Predictor(num_domains)
    
    def forward(self, x, reverse = False):
        print('here3')
        x,_ = self.feature(x)
        output = self.predictor(x)
        return output,x
    
    def get_parameters(self):
        return self.feature.get_parameters()+self.predictor.get_parameters()


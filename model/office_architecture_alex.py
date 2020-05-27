import torch.nn as nn
import torch.nn.functional as F
from grad_reverse import grad_reverse
import torch.utils.model_zoo as model_zoo
from torchvision.models import AlexNet
import torchvision.models as models
import torch

model_urls = {
    'alexnet': 'https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth',
}
        
class Feature(nn.Module):
    def __init__(self):
        super(Feature, self).__init__()
        self.model = models.alexnet(pretrained=True)
        for param in self.model.parameters():
            param.requires_grad = False
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(0.5)
        self.fc1 = nn.Linear(256*6*6, 2048)
        self.bn1 = nn.BatchNorm1d(2048)
        self.fc2 = nn.Linear(2048, 1024)
        self.bn2 = nn.BatchNorm1d(1024)
        self.fc3 = nn.Linear(1024, 512)
        self.bn3 = nn.BatchNorm1d(512)
        self.fc4 = nn.Linear(512, 256)
        self.bn4 = nn.BatchNorm1d(256, affine=False)
    
    
    def forward(self, x, reverse=False):
#         x = self.model.conv1(x)
#         x = self.model.bn1(x)
#         x = self.model.relu(x)
#         x = self.model.maxpool(x)
        
#         x = self.model.layer1(x)
#         x = self.model.layer2(x)
        
#         x = self.model.layer3(x)
#         x = self.model.layer4(x)
        
#         x = self.model.avgpool(x)
        x = self.model.features(x)
        x = self.model.avgpool(x)
        
        x_feat = x.view(x.size(0), -1)
        
        x = self.drop(self.relu(self.bn1(self.fc1(x_feat))))
        x = self.drop(self.relu(self.bn2(self.fc2(x))))
        x = self.drop(self.relu(self.bn3(self.fc3(x))))
        x = self.bn4(self.fc4(x))
        
        return x, x_feat

class Predictor(nn.Module):
    def __init__(self, num_classes):
        super(Predictor, self).__init__()
        self.fc1 = nn.Linear(256, num_classes)
        self.bn1 = nn.BatchNorm1d(num_classes)
        self.relu = nn.ReLU()
        
    def forward(self,x,reverse=True):
        x = self.fc1(self.relu(x))
        return x
    
class DomainPredictor(nn.Module):
    def __init__(self, num_domains, prob=0.5):
        super(DomainPredictor, self).__init__()
        self.feature = Feature()
        self.fc1 = nn.Linear(256*6*6, 2048)
        self.bn1 = nn.BatchNorm1d(2048)
        self.fc2 = nn.Linear(2048, 1024)
        self.bn2 = nn.BatchNorm1d(1024)
        self.fc3 = nn.Linear(1024, 512)
        self.bn3 = nn.BatchNorm1d(512)
        self.fc4 = nn.Linear(512, num_domains)
        self.bn4 = nn.BatchNorm1d(num_domains)
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(0.5)
    
    def forward(self, x_feat, reverse = False):
        _,x_feat = self.feature(x_feat)
        x_feat = self.relu(x_feat)
        x = self.drop(self.relu(self.bn1(self.fc1(x_feat))))
        x = self.drop(self.relu(self.bn2(self.fc2(x))))
        x = self.drop(self.relu(self.bn3(self.fc3(x))))
        output = self.fc4(x)
        
        return output,x

# inputTorch = torch.ones([32, 3, 224, 224])
# i1,i2 = Feature(inputTorch)
# cl = Predictor(i1)
# d1,d2 = DomainPredictor(i2)



# from torchvision import models
# alex = models.alexnet()
# features_extracted = alex.features(inputTorch)
# print(features_extracted.size())
# features_extracted_1 = alex.avgpool(features_extracted)
# print(features_extracted_1.size())


import torch.nn as nn
import torch.nn.functional as F
from grad_reverse import grad_reverse
import torch.utils.model_zoo as model_zoo
from torchvision.models import AlexNet
import torchvision.models as models

model_urls = {
    'alexnet': 'https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth',
}
        
class Feature(nn.Module):
    def __init__(self):
        super(Feature, self).__init__()
        self.model = models.resnet50(pretrained=True)
        for param in self.model.parameters():
            param.requires_grad = False
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(0.5)
        self.fc1 = nn.Linear(2048, 1024)
        self.bn1 = nn.BatchNorm1d(1024)
        self.fc1.weight.data.normal_(0, 0.005)
        self.fc1.bias.data.fill_(0.1)
        self.fc2 = nn.Linear(1024, 512)
        self.bn2 = nn.BatchNorm1d(512)
        self.fc2.weight.data.normal_(0, 0.005)
        self.fc2.bias.data.fill_(0.1)
        self.fc3 = nn.Linear(512, 256)
        self.bn3 = nn.BatchNorm1d(256, affine=False)
        self.fc3.weight.data.normal_(0, 0.005)
        self.fc3.bias.data.fill_(0.1)
        
    
    def forward(self, x, reverse=False):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        
        #x_feat = x
        #print(x_feat.size())
        
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        
        x = self.model.avgpool(x)
        
        x_feat = x.view(x.size(0), -1)
        
        x = self.drop(self.relu(self.bn1(self.fc1(x_feat))))
        x = self.drop(self.relu(self.bn2(self.fc2(x))))
        x = self.bn3(self.fc3(x))
        
        return x, x_feat

class Predictor(nn.Module):
    def __init__(self, num_classes):
        super(Predictor, self).__init__()
        self.fc1 = nn.Linear(256, num_classes)
        self.bn1 = nn.BatchNorm1d(num_classes)
        self.relu = nn.ReLU()
        self.fc1.weight.data.normal_(0, 0.01)
        self.fc1.bias.data.fill_(0.0)
        
    def forward(self,x,reverse=True):
        x = self.fc1(self.relu(x))
        return x
    
# class DomainPredictor(nn.Module):
# 	def __init__(self, num_domains, prob=0.5):
# 		super(DomainPredictor, self).__init__()
# 		self.fc1 = nn.Linear(128*8*8, 4096)
# 		self.bn1_fc = nn.BatchNorm1d(4096)
# 		self.fc2 = nn.Linear(4096, 2048)
# 		self.bn2_fc = nn.BatchNorm1d(2048)
# 		self.fc3 = nn.Linear(2048, num_domains)

# 		self.prob = prob
# 		self.num_domains = num_domains

# 		self.relu = nn.ReLU(inplace=True)

# 	def set_lambda(self, lambd):
# 		self.lambd = lambd

# 	def forward(self, x_feat, reverse=False):
# 		x = self.relu(self.bn1_fc(self.fc1(x_feat)))
# 		x = F.dropout(x, training=self.training)
# 		if reverse:
# 			x = grad_reverse(x, self.lambd)
# 		x = self.relu(self.bn2_fc(self.fc2(x)))
# 		output = self.fc3(x)
# 		return output, x
    
# class DomainPredictor(nn.Module):
# 	def __init__(self, num_domains, prob=0.5):
# 		super(DomainPredictor, self).__init__()
# 		self.feature = Feature()
# 		self.conv1 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
# 		self.bn1 = nn.BatchNorm2d(512)
# 		self.relu = nn.ReLU()
# 		self.conv2 = nn.Conv2d(512, 1024, kernel_size=3, stride=1, padding=1)
# 		self.bn2 = nn.BatchNorm2d(1024)
# 		self.conv3 = nn.Conv2d(1024, 1024, kernel_size=3, stride=1, padding=1)
# 		self.bn3 = nn.BatchNorm2d(1024)
# 		self.avgpool = nn.AvgPool2d(28)
# 		self.fc4 = nn.Linear(1024, 256)
# 		self.bn4 = nn.BatchNorm1d(256)
# 		self.fc4.weight.data.normal_(0, 0.005)
# 		self.fc4.bias.data.fill_(0.1)
# 		self.fc5 = nn.Linear(256, num_domains)
# 		self.fc5.weight.data.normal_(0, 0.01)
# 		self.fc5.bias.data.fill_(0.0)
# 		self.drop = nn.Dropout(0.5)
# 		# self.idm = nn.Linear(128*32*32, num_domains)

# 		self.prob = prob
# 		self.num_domains = num_domains

# 		self.relu = nn.ReLU(inplace=True)

# 	def set_lambda(self, lambd):
# 		self.lambd = lambd

# 	def forward(self, x, reverse=False):
# 		_,x = self.feature(x)
# 		x = self.relu(self.bn1(self.conv1(x)))
# 		x = self.relu(self.bn2(self.conv2(x)))
# 		x = self.relu(self.bn3(self.conv3(x)))
# 		x = self.avgpool(x)
# 		x = x.view(x.shape[0],-1)
# 		x = self.drop(self.relu(self.bn4(self.fc4(x))))        
# 		output = self.fc5(x)

# 		return output,x   
    
class DomainPredictor(nn.Module):
    def __init__(self, num_domains, prob=0.5):
        super(DomainPredictor, self).__init__()
        self.feature = Feature()
        self.fc1 = nn.Linear(2048, 1024)
        self.bn1 = nn.BatchNorm1d(1024)
        self.fc2 = nn.Linear(1024, 512)
        self.bn2 = nn.BatchNorm1d(512)
        self.fc3 = nn.Linear(512, num_domains)
        self.bn3 = nn.BatchNorm1d(num_domains)
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(0.5)
    
    def forward(self, x_feat, reverse = False):
        _,x_feat = self.feature(x_feat)
        x_feat = self.relu(x_feat)
        x = self.drop(self.relu(self.bn1(self.fc1(x_feat))))
        x = self.drop(self.relu(self.bn2(self.fc2(x))))
        output = self.fc3(x)
        
        return output,x

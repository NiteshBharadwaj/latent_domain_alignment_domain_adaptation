import torch.nn as nn
import torch.nn.functional as F
from grad_reverse import grad_reverse
import torch.utils.model_zoo as model_zoo
from torchvision.models import AlexNet
import torchvision.models as models

model_urls = {
    'alexnet': 'https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth',
}

# class Feature(nn.Module):
# 	def __init__(self):
# 		super(Feature, self).__init__()
# 		self.model = models.resnet18(pretrained=True)
# 		for param in self.model.parameters():
# 			param.requires_grad = False

# 		self.fc2 = nn.Linear(512, 512)
# 		nn.init.xavier_uniform_(self.fc2.weight, .1)
# 		nn.init.constant_(self.fc2.bias, 0.)
# 		self.bn_fc2 = nn.BatchNorm1d(512)
# 		self.fc3 = nn.Linear(512, 256)
# 		nn.init.xavier_uniform_(self.fc3.weight, .1)
# 		nn.init.constant_(self.fc3.bias, 0.)
# 		self.bn_fc3 = nn.BatchNorm1d(256)

# 		# num_ftrs = self.model.fc.in_features
# 		# self.model.fc = nn.Linear(num_ftrs, 1716)
# 		# nn.init.xavier_uniform_(self.model.fc.weight, .1)
# 		# nn.init.constant_(self.model.fc.bias, 0.)

# 		# self.fc1 = nn.Linear(512*8*8, 8192)
# 		# self.bn1_fc = nn.BatchNorm1d(8192)
# 		# self.fc2 = nn.Linear(8192, 4096)
# 		# self.bn2_fc = nn.BatchNorm1d(4096)
# 		# self.fc3 = nn.Linear(4096, 2048)
# 		# self.bn3_fc = nn.BatchNorm1d(2048)

# 		self.relu = nn.ReLU(inplace=True)
# 		self.drop = nn.Dropout(0.5)

# 	def forward(self, x, reverse=False):
# 		x = self.model.conv1(x)
# 		x = self.model.bn1(x)
# 		x = self.model.relu(x)
# 		x = self.model.maxpool(x)

# 		x = self.model.layer1(x)
# 		x = self.model.layer2(x)

# 		x_feat = x.detach()

# 		x = self.model.layer3(x)
# 		x = self.model.layer4(x)

# 		x = self.model.avgpool(x)
# 		x = x.view(x.size(0), -1)

# 		x = self.drop(self.relu(self.bn_fc2(self.fc2(x.detach()))))
# 		x = self.relu(self.bn_fc3(self.fc3(x)))



# 		# x = self.relu(self.bn1_fc(self.fc1(x_feat)))
# 		# x = F.dropout(x, training=self.training)

# 		# x = self.relu(self.bn2_fc(self.fc2(x)))
# 		# x = F.dropout(x, training=self.training)
# 		# if reverse:
# 		# 	x = grad_reverse(x, self.lambd)

# 		# x = self.relu(self.bn3_fc(self.fc3(x)))

# 		return x, x_feat

# class Predictor(nn.Module):
# 	def __init__(self, num_classes, prob=0.5):
# 		super(Predictor, self).__init__()
# 		self.num_classes = num_classes
# 		# self.fc2 = nn.Linear(512, 2048)
# 		self.relu = nn.ReLU()
# 		self.drop = nn.Dropout(0.5)
# 		self.fc3 = nn.Linear(256, num_classes)
# 		nn.init.xavier_uniform_(self.fc3.weight, .1)
# 		nn.init.constant_(self.fc3.bias, 0.)
# # 		nn.init.xavier_uniform_(self.fc2.weight, .1)
# # 		nn.init.constant_(self.fc2.bias, 0.)
# 		# self.bn_fc2 = nn.BatchNorm1d(2048)
# 		self.prob = prob

# 	def set_lambda(self, lambd):
# 		self.lambd = lambd

# 	def forward(self, x, reverse=False):
# 		# x = self.relu(self.bn_fc2(self.fc2(x)))
# 		x = self.fc3(x)
# 		return x

# class DomainPredictor(nn.Module):
# 	def __init__(self, num_domains, prob=0.5):
# 		super(DomainPredictor, self).__init__()

# 		self.conv1 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)
# 		self.bn1 = nn.BatchNorm2d(256)
# 		self.relu = nn.ReLU()
# 		self.conv2 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1)
# 		self.bn2 = nn.BatchNorm2d(512)
# 		self.conv3 = nn.Conv2d(512, 1024, kernel_size=3, stride=2, padding=1)
# 		self.bn3 = nn.BatchNorm2d(1024)
# 		self.avgpool = nn.AvgPool2d(28)
# 		self.fc4 = nn.Linear(192, num_domains)
# 		# self.idm = nn.Linear(128*32*32, num_domains)

# 		self.prob = prob
# 		self.num_domains = num_domains

# 		self.relu = nn.ReLU(inplace=True)

# 	def set_lambda(self, lambd):
# 		self.lambd = lambd

# 	def forward(self, x, reverse=False):

# # 		x = self.relu(self.bn1(self.conv1(x)))
# # 		if reverse:
# # 			x = grad_reverse(x, self.lambd)
# # 		x = self.relu(self.bn2(self.conv2(x)))
# # 		x = self.relu(self.bn3(self.conv3(x)))
# 		x = self.avgpool(x)
# 		x = x.view(x.shape[0],-1)       
# 		output = self.fc4(x)

# 		# x = x_feat.view(x_feat.size(0), -1)
# 		# x = self.idm(x)

# 		return output, x
class Feature(nn.Module):
    def __init__(self):
        super(Feature, self).__init__()
        self.model = models.resnet18(pretrained=True)
        for param in self.model.parameters():
            param.requires_grad = False
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(0.5)
        self.fc1 = nn.Linear(512, 1024)
        self.bn1 = nn.BatchNorm1d(1024)
        self.fc2 = nn.Linear(1024, 512)
        self.bn2 = nn.BatchNorm1d(512)
        self.fc3 = nn.Linear(512, 256)
        self.bn3 = nn.BatchNorm1d(256)
    
    
    def forward(self, x, reverse=False):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        
        #print('conv1 : ', x.size())
        
        x = self.model.layer1(x)
        
        #print('layer1 : ', x.size())
        x = self.model.layer2(x)
        
        #print('layer2 : ', x.size())
        
        x = self.model.layer3(x)
        
        #print('layer3 : ', x.size())
        
        x = self.model.layer4(x)
        
        #print('layer4 : ', x.size())
        
        x = self.model.avgpool(x)
        
        #print('average pool : ', x.size())
        
        x_feat = x.view(x.size(0), -1)
        
        #print('x_feat : ', x.size())
        
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
        
    def forward(self,x,reverse=True):
        x = self.fc1(self.relu(x))
        return x
    
class DomainPredictor(nn.Module):
    def __init__(self, num_domains, prob=0.5):
        super(DomainPredictor, self).__init__()
        self.feature = Feature()
        self.fc1 = nn.Linear(512, 1024)
        self.bn1 = nn.BatchNorm1d(1024)
        self.fc2 = nn.Linear(1024, 512)
        self.bn2 = nn.BatchNorm1d(512)
        self.fc3 = nn.Linear(512, num_domains)
        self.bn3 = nn.BatchNorm1d(num_domains)
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(0.5)
    
    def forward(self, x_feat, reverse = False):
        _,x_feat = self.feature(x_feat)
        print('x feat dp : ', x_feat.size())
        x = self.drop(self.relu(self.bn1(self.fc1(x_feat))))
        x = self.drop(self.relu(self.bn2(self.fc2(x))))
        output = self.fc3(x)
        
        return output,x
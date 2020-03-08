import torch.nn as nn
import torch.nn.functional as F
from grad_reverse import grad_reverse
import torch.utils.model_zoo as model_zoo
from torchvision.models import AlexNet

model_urls = {
    'alexnet': 'https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth',
}

class Feature_AlexNet(nn.Module):
	def __init__(self):
		super(Feature_AlexNet, self).__init__()
		self.model = AlexNet()
		self.model.load_state_dict(model_zoo.load_url(model_urls['alexnet']))

		self.fc1 = nn.Linear(256*6*6, 4096)
		self.bn1_fc = nn.BatchNorm1d(4096)
		self.fc2 = nn.Linear(4096, 2048)
		self.bn2_fc = nn.BatchNorm1d(2048)

		self.relu = nn.ReLU(inplace=True)

	def forward(self, x, reverse=False):
		x = self.model.avgpool(self.model.features(x))
		x_feat = x.view(x.size(0), 256*6*6)

		x = self.relu(self.bn1_fc(self.fc1(x_feat)))
		x = F.dropout(x, training=self.training)
		if reverse:
			x = grad_reverse(x, self.lambd)
		x = self.relu(self.bn2_fc(self.fc2(x)))
		return x, x_feat

class Predictor_AlexNet(nn.Module):
	def __init__(self, num_classes, prob=0.5):
		super(Predictor_AlexNet, self).__init__()
		self.num_classes = num_classes
		self.fc3 = nn.Linear(2048, num_classes)
		self.bn_fc3 = nn.BatchNorm1d(num_classes)
		self.prob = prob

	def set_lambda(self, lambd):
		self.lambd = lambd

	def forward(self, x, reverse=False):
		x = self.fc3(x)
		return x

class DomainPredictor_AlexNet(nn.Module):
	def __init__(self, num_domains, prob=0.5):
		super(DomainPredictor_AlexNet, self).__init__()
		self.fc1 = nn.Linear(256*6*6, 4096)
		self.bn1_fc = nn.BatchNorm1d(4096)
		self.fc2 = nn.Linear(4096, 2048)
		self.bn2_fc = nn.BatchNorm1d(2048)
		self.fc3 = nn.Linear(2048, num_domains)

		self.prob = prob
		self.num_domains = num_domains

		self.relu = nn.ReLU(inplace=True)

	def set_lambda(self, lambd):
		self.lambd = lambd

	def forward(self, x_feat, reverse=False):
		x = self.relu(self.bn1_fc(self.fc1(x_feat)))
		x = F.dropout(x, training=self.training)
		if reverse:
			x = grad_reverse(x, self.lambd)
		x = self.relu(self.bn2_fc(self.fc2(x)))
		x = self.fc3(x)
		return x

# ----------------------------------- Noob Architecture ----------------------------------------
class Feature(nn.Module): # Input -> SxS
	def __init__(self):
		super(Feature, self).__init__()
		self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
		self.bn1 = nn.BatchNorm2d(32) 
		self.mp1 = nn.MaxPool2d(kernel_size=2, stride=2) # -> S/2

		self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
		self.bn2 = nn.BatchNorm2d(64) 
		self.mp2 = nn.MaxPool2d(kernel_size=2, stride=2) # -> S/4

		self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
		self.bn3 = nn.BatchNorm2d(64) 
		self.mp3 = nn.MaxPool2d(kernel_size=2, stride=2) # -> S/8

		self.conv4 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
		self.bn4 = nn.BatchNorm2d(128) 
		self.mp4 = nn.MaxPool2d(kernel_size=2, stride=2) # -> S/16

		self.fc1 = nn.Linear(128*8*8, 4096)
		self.bn1_fc = nn.BatchNorm1d(4096)
		self.fc2 = nn.Linear(4096, 2048)
		self.bn2_fc = nn.BatchNorm1d(2048)

		self.relu = nn.ReLU(inplace=True)

	def forward(self, x,reverse=False): # GOOD x_feat PLACE ??? IMPORTANT!!!
		x = self.mp1(self.relu(self.bn1(self.conv1(x))))
		x = self.mp2(self.relu(self.bn2(self.conv2(x))))
		x = self.mp3(self.relu(self.bn3(self.conv3(x))))
		x = self.mp4(self.relu(self.bn4(self.conv4(x))))


		x_feat = x.view(x.size(0), 128*8*8)
		x = self.relu(self.bn1_fc(self.fc1(x_feat)))
		x = F.dropout(x, training=self.training)
		if reverse:
			x = grad_reverse(x, self.lambd)
		x = self.relu(self.bn2_fc(self.fc2(x)))
		return x, x_feat

class Predictor(nn.Module):
	def __init__(self, num_classes, prob=0.5):
		super(Predictor, self).__init__()
		self.num_classes = num_classes
		self.fc3 = nn.Linear(2048, num_classes)
		self.bn_fc3 = nn.BatchNorm1d(num_classes)
		self.prob = prob

	def set_lambda(self, lambd):
		self.lambd = lambd

	def forward(self, x, reverse=False):
		x = self.fc3(x)
		return x

class DomainPredictor(nn.Module):
	def __init__(self, num_domains, prob=0.5):
		super(DomainPredictor, self).__init__()
		self.fc1 = nn.Linear(128*8*8, 4096)
		self.bn1_fc = nn.BatchNorm1d(4096)
		self.fc2 = nn.Linear(4096, 2048)
		self.bn2_fc = nn.BatchNorm1d(2048)
		self.fc3 = nn.Linear(2048, num_domains)

		self.prob = prob
		self.num_domains = num_domains

		self.relu = nn.ReLU(inplace=True)

	def set_lambda(self, lambd):
		self.lambd = lambd

	def forward(self, x_feat, reverse=False):
		x = self.relu(self.bn1_fc(self.fc1(x_feat)))
		x = F.dropout(x, training=self.training)
		if reverse:
			x = grad_reverse(x, self.lambd)
		x = self.relu(self.bn2_fc(self.fc2(x)))
		x = self.fc3(x)
		return x


# ----------------------------------- ------------- ----------------------------------------


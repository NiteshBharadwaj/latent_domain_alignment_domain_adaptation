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
        self.model = models.resnet18(pretrained=True)
        #for param in self.model.parameters():
        #    param.requires_grad = False

        # self.relu = nn.ReLU()
        # self.drop = nn.Dropout(0.5)
        # self.fc1 = nn.Linear(512, 1024)
        
        self.bn1 = nn.BatchNorm1d(512, affine=False)
        # self.fc2 = nn.Linear(1024, 512)
        # self.bn2 = nn.BatchNorm1d(512)
        #self.fc3 = nn.Linear(512, 256)
        #self.bn3 = nn.BatchNorm1d(256, affine=False)
        self.relu = nn.ReLU()

    def forward(self, x, reverse=False):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)

        x = self.model.layer1(x)
        x = self.model.layer2(x)

        x = self.model.layer3(x)
        x = self.model.layer4(x)

        x = self.model.avgpool(x)
        x_feat = x.view(x.size(0), -1)
        x = self.bn1(x.view(x.size(0), -1))
        #x2 = self.bn3(self.fc3(self.relu(x_feat)))
        # x = self.drop(self.relu(self.bn1(self.fc1(x_feat))))
        # x = self.drop(self.relu(self.bn2(self.fc2(x))))
        # x = self.bn3(self.fc3(x))

        return x, x_feat, x


class Predictor(nn.Module):
    def __init__(self, num_classes, inp_channels=512):
        super(Predictor, self).__init__()
        self.fc1 = nn.Linear(inp_channels, num_classes)
        self.bn1 = nn.BatchNorm1d(num_classes)
        self.relu = nn.ReLU()

    def forward(self, x, reverse=True):
        x = self.fc1(self.relu(x))
        return x


class DomainPredictor(nn.Module):
    def __init__(self, num_domains, prob=0.5, classaware_dp=False):
        super(DomainPredictor, self).__init__()
        self.dp_model = models.resnet18(pretrained=True)
        self.classaware_dp=classaware_dp
        for param in self.dp_model.conv1.parameters():
            param.requires_grad = False
        for param in self.dp_model.bn1.parameters():
            param.requires_grad = False
        for param in self.dp_model.layer1.parameters():
            param.requires_grad = False
        for param in self.dp_model.layer2.parameters():
            param.requires_grad = False

        self.fc5 = nn.Linear(512, 128)
        self.bn_fc5 = nn.BatchNorm1d(128)
        self.dp_layer = nn.Linear(128, num_domains)

        self.prob = prob
        self.num_domains = num_domains

        self.relu = nn.ReLU(inplace=True)

    def set_lambda(self, lambd):
        self.lambd = lambd

    def forward(self, x, reverse=False):
        if not self.classaware_dp:
            x = self.dp_model.conv1(x)
            x = self.dp_model.bn1(x)
            x = self.dp_model.relu(x)
            x = self.dp_model.maxpool(x)

            x = self.dp_model.layer1(x)
            x = self.dp_model.layer2(x)

            x = self.dp_model.layer3(x)
            x = self.dp_model.layer4(x)
            x = self.dp_model.avgpool(x)
            #import pdb
            #pdb.set_trace()
            x = x.view(x.size(0), -1)

        x = self.relu(self.bn_fc5(self.fc5(x)))

        #x = self.avgpool(x)
        #x = x.view(x.shape[0],-1)       
        #x = self.relu(self.bn_fc4(self.fc4(x)))

        dp_pred = self.dp_layer(x)

        return dp_pred, None


class DomainPredictorOLD(nn.Module):
    def __init__(self, num_domains, prob=0.5):
        super(DomainPredictorOLD, self).__init__()

        self.feature = Feature()
        self.fc1 = nn.Linear(512, 1024)
        self.bn1 = nn.BatchNorm1d(1024)
        self.fc2 = nn.Linear(1024, 512)
        self.bn2 = nn.BatchNorm1d(512)
        self.fc3 = nn.Linear(512, num_domains)
        self.bn3 = nn.BatchNorm1d(num_domains)
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(0.5)

    def forward(self, x_feat, reverse=False):
        _, x_feat = self.feature(x_feat)
        x_feat = self.relu(x_feat)
        x = self.drop(self.relu(self.bn1(self.fc1(x_feat))))
        x = self.drop(self.relu(self.bn2(self.fc2(x))))
        output = self.fc3(x)

        return output, x

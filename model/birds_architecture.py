import torch.nn as nn
import torch.nn.functional as F
from grad_reverse import grad_reverse
import torch.utils.model_zoo as model_zoo
from torchvision.models import AlexNet
import torchvision.models as models

model_urls = {
    'alexnet': 'https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth',
}

class Resnet50Fc(nn.Module):
  def __init__(self):
    super(Resnet50Fc, self).__init__()
    model_resnet50 = models.resnet50(pretrained=True)
    self.conv1 = model_resnet50.conv1
    self.bn1 = model_resnet50.bn1
    self.relu = model_resnet50.relu
    self.maxpool = model_resnet50.maxpool
    self.layer1 = model_resnet50.layer1
    self.layer2 = model_resnet50.layer2
    self.layer3 = model_resnet50.layer3
    self.layer4 = model_resnet50.layer4
    self.avgpool = model_resnet50.avgpool
    self.__in_features = model_resnet50.fc.in_features

  def forward(self, x):
    x = self.conv1(x)
    x = self.bn1(x)
    x = self.relu(x)
    x = self.maxpool(x)
    x = self.layer1(x)
    x = self.layer2(x)
    x = self.layer3(x)
    x = self.layer4(x)
    x = self.avgpool(x)
    x = x.view(x.size(0), -1)
    return x

  def output_num(self):
    return self.__in_features
class Feature(nn.Module):
    def __init__(self):
        super(Feature, self).__init__()
        self.model = Resnet50Fc()
        self.bottleneck_0 = nn.Linear(2048, 256)
        self.bottleneck_0.weight.data.normal_(0, 0.005)
        self.bottleneck_0.bias.data.fill_(0.1)
        self.bottleneck_layer = nn.Sequential(self.bottleneck_0, nn.ReLU(), nn.Dropout(0.5))
        self.bn1 = nn.BatchNorm1d(256,affine=False)

    def forward(self, x, reverse=False):
        features = self.model(x)
        out_bottleneck = self.bottleneck_layer(features)
        out_bottleneck = self.bn1(out_bottleneck)
        return out_bottleneck, features, out_bottleneck


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
        self.dp_model = Resnet50Fc()
        self.classaware_dp=classaware_dp
        for param in self.dp_model.conv1.parameters():
            param.requires_grad = False
        for param in self.dp_model.bn1.parameters():
            param.requires_grad = False
        for param in self.dp_model.layer1.parameters():
            param.requires_grad = True
        for param in self.dp_model.layer2.parameters():
            param.requires_grad = True
        self.inp_layer = 2048 if classaware_dp else 2048
        self.fc5 = nn.Linear(self.inp_layer, 128)
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

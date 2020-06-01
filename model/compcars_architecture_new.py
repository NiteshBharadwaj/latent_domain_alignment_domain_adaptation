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
    def __init__(self, is_fine_tune, is_extra_fc):
        super(Feature, self).__init__()
        self.is_extra_fc = is_extra_fc
        self.model = models.resnet18(pretrained=True)
        for param in self.model.parameters():
            param.requires_grad = True
        if not is_fine_tune:
            #for param in self.model.parameters():
            #    param.requires_grad = False
            for param in self.model.conv1.parameters():
                param.requires_grad = False
            for param in self.model.bn1.parameters():
                param.requires_grad = False
            for param in self.model.layer1.parameters():
                param.requires_grad = False
            for param in self.model.layer2.parameters():
                param.requires_grad = False

        self.fc2 = nn.Linear(512, 2048)
        nn.init.xavier_uniform_(self.fc2.weight, .1)
        nn.init.constant_(self.fc2.bias, 0.)
        self.bn_fc2 = nn.BatchNorm1d(2048)
        self.fc3 = nn.Linear(2048, 512)
        nn.init.xavier_uniform_(self.fc3.weight, .1)
        nn.init.constant_(self.fc3.bias, 0.)
        self.bn_fc3 = nn.BatchNorm1d(512, affine=False)

            # num_ftrs = self.model.fc.in_features
            # self.model.fc = nn.Linear(num_ftrs, 1716)
            # nn.init.xavier_uniform_(self.model.fc.weight, .1)
            # nn.init.constant_(self.model.fc.bias, 0.)

            # self.fc1 = nn.Linear(512*8*8, 8192)
            # self.bn1_fc = nn.BatchNorm1d(8192)
            # self.fc2 = nn.Linear(8192, 4096)
            # self.bn2_fc = nn.BatchNorm1d(4096)
            # self.fc3 = nn.Linear(4096, 2048)
            # self.bn3_fc = nn.BatchNorm1d(2048)

        self.relu = nn.ReLU(inplace=True)
        self.drop = nn.Dropout(0.5)

    def forward(self, x, reverse=False):
            
            x = self.model.conv1(x)
            x = self.model.bn1(x)
            x = self.model.relu(x)
            x = self.model.maxpool(x)

            x = self.model.layer1(x)
            x = self.model.layer2(x)

            #x_feat = x.detach()
            x_feat = x

            x = self.model.layer3(x_feat)
            x = self.model.layer4(x)

            x = self.model.avgpool(x)
            x = x.view(x.size(0), -1)

            if self.is_extra_fc:
                x = self.drop(self.relu(self.bn_fc2(self.fc2(x))))
                x = self.bn_fc3(self.fc3(x))
            else:
                x = self.bn_fc3(x)

            return x, x_feat

class Predictor(nn.Module):
    def __init__(self, num_classes):
        super(Predictor, self).__init__()
        self.fc1 = nn.Linear(512, num_classes)
        self.bn1 = nn.BatchNorm1d(num_classes)
        self.relu = nn.ReLU()

    def forward(self, x, reverse=True):
        x = self.fc1(self.relu(x))
        return x
class DomainPredictor_ResNet18(nn.Module):
    def __init__(self, num_domains, aux_classes, is_fine_tune, extra_fc, weight_reinit, prob=0.5):
        super(DomainPredictor_ResNet18, self).__init__()

        self.feature = Feature(is_fine_tune, extra_fc)
        if weight_reinit:
            for m in self.feature.model.layer3.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
            for m in self.feature.model.layer4.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)


        self.conv1 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(256)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(512)
        self.conv3 = nn.Conv2d(512, 1024, kernel_size=3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(1024)
        self.avgpool = nn.AvgPool2d(32)

        #self.fc4 = nn.Linear(128, 32)
        #self.bn_fc4 = nn.BatchNorm1d(32)
        #self.dp_layer = nn.Linear(32, num_domains)
        #self.aux_layer = nn.Linear(32, aux_classes)

        self.fc5 = nn.Linear(512, 128)
        self.bn_fc5 = nn.BatchNorm1d(128)
        self.dp_layer = nn.Linear(128, num_domains)
        self.aux_layer = nn.Linear(128, aux_classes)


        self.prob = prob
        self.num_domains = num_domains

        self.relu = nn.ReLU(inplace=True)

    def set_lambda(self, lambd):
        self.lambd = lambd

    def forward(self, x, reverse=False):
        _, x = self.feature(x)
        x = self.feature.model.layer3(x)
        x = self.feature.model.layer4(x)
        x = self.feature.model.avgpool(x)
        x = x.view(x.size(0), -1)
        
        x = self.relu(self.bn_fc5(self.fc5(x)))

        #x = self.avgpool(x)
        #x = x.view(x.shape[0],-1)       
        #x = self.relu(self.bn_fc4(self.fc4(x)))

        dp_pred = self.dp_layer(x)
        aux_pred = self.aux_layer(x)


        return dp_pred, aux_pred


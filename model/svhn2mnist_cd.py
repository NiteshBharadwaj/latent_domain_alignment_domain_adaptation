import torch.nn as nn
import torch.nn.functional as F
from grad_reverse import grad_reverse
import torchvision.models as models

# -------------------------------------------------------- ResNet18 -----------------------------------------------------

class Feature_ResNet18(nn.Module):
    def __init__(self):
        super(Feature_ResNet18, self).__init__()
        self.model = models.resnet18(pretrained=True)
        for param in self.model.parameters():
            param.requires_grad = False

        self.fc1 = nn.Linear(128, 128)
        self.bn_fc1 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 64)
        self.bn_fc2 = nn.BatchNorm1d(64, affine=False)


    def forward(self, x, reverse=False):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)

        x = self.model.layer1(x)
        x = self.model.layer2(x)

        x_feat = x

#        x = self.model.layer3(x)
#        x = self.model.layer4(x)

        x = self.model.avgpool(x)
        x = x.view(x.size(0), -1)

        x = F.dropout(F.relu(self.bn_fc1(self.fc1(x))), training=self.training)
        x = self.bn_fc2(self.fc2(x))

        return x, x_feat

class Predictor_ResNet18(nn.Module):
    def __init__(self, prob=0.5):
        super(Predictor_ResNet18, self).__init__()
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x, reverse=False):
        x = self.fc3(F.relu(x))
        return x

class DomainPredictor_ResNet18(nn.Module):
    def __init__(self, num_domains, prob=0.5):
        super(DomainPredictor_ResNet18, self).__init__()
        self.avgpool = nn.AvgPool2d(4)
        self.fc4 = nn.Linear(128, num_domains)

    def forward(self, x, reverse=False):
        x = self.avgpool(x)
        x = x.view(x.shape[0],-1)
        output = self.fc4(x)

        return output, x


# -------------------------------------------------NoobNet-------------------------------------------------------
class Feature(nn.Module):
    def __init__(self):
        super(Feature, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=5, stride=1, padding=2)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=5, stride=1, padding=2)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=5, stride=1, padding=2)
        self.bn3 = nn.BatchNorm2d(128)
        self.fc1 = nn.Linear(8192, 3072)
        self.bn1_fc = nn.BatchNorm1d(3072)
        self.bn1_fc1 = nn.BatchNorm1d(3072, affine=False)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.bn1(self.conv1(x))), stride=2, kernel_size=3, padding=1)
        x = F.max_pool2d(F.relu(self.bn2(self.conv2(x))), stride=2, kernel_size=3, padding=1)
        x = F.relu(self.bn3(self.conv3(x)))
        x = x.view(x.size(0), 8192)
        x_feat = x
        x_da = self.fc1(x)
        x = F.relu(self.bn1_fc(x_da))
        x = F.dropout(x, training=self.training)
        x_da = self.bn1_fc1(x_da)
        return x, x_feat, x_da


class Predictor(nn.Module):
    def __init__(self, prob=0.5):
        super(Predictor, self).__init__()
        self.fc1 = nn.Linear(8192, 3072)
        self.bn1_fc = nn.BatchNorm1d(3072)
        self.fc2 = nn.Linear(3072, 2048)
        self.bn2_fc = nn.BatchNorm1d(2048)
        self.fc3 = nn.Linear(2048, 10)
        self.bn_fc3 = nn.BatchNorm1d(10)
        self.prob = prob

    def set_lambda(self, lambd):
        self.lambd = lambd

    def forward(self, x, reverse=False):
        if reverse:
            x = grad_reverse(x, self.lambd)
        x = F.relu(self.bn2_fc(self.fc2(x)))
        x = self.fc3(x)
        return x



class DomainPredictor(nn.Module):
    def __init__(self, num_domains, prob=0.5):
        super(DomainPredictor, self).__init__()
        self.feat = Feature()
        self.fc1 = nn.Linear(8192, 3072)
        self.bn1_fc = nn.BatchNorm1d(3072)
        self.fc2 = nn.Linear(3072, 2048)
        self.bn2_fc = nn.BatchNorm1d(2048)
        self.fc3 = nn.Linear(2048, num_domains)
        self.bn_fc3 = nn.BatchNorm1d(10)
        self.prob = prob
        self.num_domains = num_domains
        self.lrelu = nn.ReLU()

    def set_lambda(self, lambd):
        self.lambd = lambd

    def forward(self, x_feat, reverse=False):
        _,x_feat,_ = self.feat(x_feat)
        x = self.lrelu(self.bn1_fc(self.fc1(x_feat)))
        x = F.dropout(x, training=self.training)
        if reverse:
            x = grad_reverse(x, self.lambd)
        x = self.lrelu(self.bn2_fc(self.fc2(x)))
        output = self.fc3(x)
        return output,x

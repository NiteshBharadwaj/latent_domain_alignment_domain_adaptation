import torch.nn as nn
import torch.nn.functional as F
from grad_reverse import grad_reverse
import torchvision.models as models


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
        self.fc2 = nn.Linear(3072, 2048)
        self.bn2_fc = nn.BatchNorm1d(2048, affine=False)
        self.lrelu = nn.ReLU()

    def forward(self, x,reverse=False):
        x = F.max_pool2d(self.lrelu(self.bn1(self.conv1(x))), stride=2, kernel_size=3, padding=1)
        x = F.max_pool2d(self.lrelu(self.bn2(self.conv2(x))), stride=2, kernel_size=3, padding=1)
        x = self.lrelu(self.bn3(self.conv3(x)))
        x_feat = x.view(x.size(0), 8192)
        x_early_align = 
        x = self.lrelu(self.bn1_fc(self.fc1(x_feat)))
        x = F.dropout(x, training=self.training)
        if reverse:
            x = grad_reverse(x, self.lambd)
#        x = F.relu(self.bn2_fc(self.fc2(x)))
        x = self.bn2_fc(self.fc2(x))
        return x, x_feat, x

class Predictor(nn.Module):
    def __init__(self, prob=0.5):
        super(Predictor, self).__init__()
        # self.fc1 = nn.Linear(8192, 3072)
        # self.bn1_fc = nn.BatchNorm1d(3072)
        # self.fc2 = nn.Linear(3072, 2048)
        # self.bn2_fc = nn.BatchNorm1d(2048)
        self.fc3 = nn.Linear(2048, 10)
        self.bn_fc3 = nn.BatchNorm1d(10)
        self.prob = prob

    def set_lambda(self, lambd):
        self.lambd = lambd

    def forward(self, x, reverse=False):
        # if reverse:
        #     x = grad_reverse(x, self.lambd)
        # x = F.relu(self.bn2_fc(self.fc2(x)))
#        x = self.fc3(x)
        x = self.fc3(F.relu(x))
        return x


class DomainPredictor(nn.Module):
    def __init__(self, num_domains, prob=0.5, classaware_dp=False):
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
        self.classaware_dp = classaware_dp

    def set_lambda(self, lambd):
        self.lambd = lambd

    def forward(self, x_feat, reverse=False):
        if not self.classaware_dp:
            _,x_feat,_ = self.feat(x_feat)
        x = self.lrelu(self.bn1_fc(self.fc1(x_feat)))
        x = F.dropout(x, training=self.training)
        if reverse:
            x = grad_reverse(x, self.lambd)
        x = self.lrelu(self.bn2_fc(self.fc2(x)))
        output = self.fc3(x)
        return output,x

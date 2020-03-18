
# import numpy as np
# import torch
# import torch.nn as nn
# import torchvision
# from torchvision import models
# from torch.autograd import Variable
# import math

# def calc_coeff(iter_num, high=1.0, low=0.0, alpha=10.0, max_iter=10000.0):
#     return np.float(2.0 * (high - low) / (1.0 + np.exp(-alpha*iter_num / max_iter)) - (high - low) + low)

# def init_weights(m):
#     classname = m.__class__.__name__
#     if classname.find('Conv2d') != -1 or classname.find('ConvTranspose2d') != -1:
#         nn.init.kaiming_uniform_(m.weight)
#         nn.init.zeros_(m.bias)
#     elif classname.find('BatchNorm') != -1:
#         nn.init.normal_(m.weight, 1.0, 0.02)
#         nn.init.zeros_(m.bias)
#     elif classname.find('Linear') != -1:
#         nn.init.xavier_normal_(m.weight)
#         nn.init.zeros_(m.bias)



# resnet_dict = {"ResNet18":models.resnet18, "ResNet34":models.resnet34, "ResNet50":models.resnet50, "ResNet101":models.resnet101, "ResNet152":models.resnet152}
# class ResNetFc(nn.Module):
#   def __init__(self, resnet_name, use_bottleneck=True, bottleneck_dim=256, new_cls=False, class_num=1000):
#     super(ResNetFc, self).__init__()
#     model_resnet = resnet_dict[resnet_name](pretrained=True)
#     self.conv1 = model_resnet.conv1
#     self.bn1 = model_resnet.bn1
#     self.relu = model_resnet.relu
#     self.maxpool = model_resnet.maxpool
#     self.layer1 = model_resnet.layer1
#     self.layer2 = model_resnet.layer2
#     self.layer3 = model_resnet.layer3
#     self.layer4 = model_resnet.layer4
#     self.avgpool = model_resnet.avgpool
#     self.feature_layers = nn.Sequential(self.conv1, self.bn1, self.relu, self.maxpool, \
#                          self.layer1, self.layer2, self.layer3, self.layer4, self.avgpool)

#     self.use_bottleneck = use_bottleneck
#     self.new_cls = new_cls
#     if new_cls:
#         if self.use_bottleneck:
#             self.bottleneck = nn.Linear(model_resnet.fc.in_features, bottleneck_dim)
#             self.fc = nn.Linear(bottleneck_dim, class_num)
#             self.bottleneck.apply(init_weights)
#             self.fc.apply(init_weights)
#             self.__in_features = bottleneck_dim
#         else:
#             self.fc = nn.Linear(model_resnet.fc.in_features, class_num)
#             self.fc.apply(init_weights)
#             self.__in_features = model_resnet.fc.in_features
#     else:
#         self.fc = model_resnet.fc
#         self.__in_features = model_resnet.fc.in_features

#   def forward(self, x):
#     x = self.feature_layers(x)
#     x = x.view(x.size(0), -1)
#     if self.use_bottleneck and self.new_cls:
#         x = self.bottleneck(x)
#     y = self.fc(x)
#     return x, y

#   def output_num(self):
#     return self.__in_features

#   def get_parameters(self):
#     if self.new_cls:
#         if self.use_bottleneck:
#             parameter_list = [{"params":self.feature_layers.parameters(), "lr_mult":1, 'decay_mult':2}, \
#                             {"params":self.bottleneck.parameters(), "lr_mult":10, 'decay_mult':2}, \
#                             {"params":self.fc.parameters(), "lr_mult":10, 'decay_mult':2}]
#         else:
#             parameter_list = [{"params":self.feature_layers.parameters(), "lr_mult":1, 'decay_mult':2}, \
#                             {"params":self.fc.parameters(), "lr_mult":10, 'decay_mult':2}]
#     else:
#         parameter_list = [{"params":self.parameters(), "lr_mult":1, 'decay_mult':2}]
#     return parameter_list



# print(models.resnet18(pretrained=True))


import torch.nn as nn
import torch.nn.functional as F
from grad_reverse import grad_reverse
import torch.utils.model_zoo as model_zoo
from torchvision.models import AlexNet
import torchvision.models as models

model_urls = {
    'alexnet': 'https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth',
}

# ---------------------------------------------- ResNet50 -----------------------------------------


class Feature_ResNet50(nn.Module):
    def __init__(self):
        super(Feature_ResNet50, self).__init__()
        self.model = models.resnet50(pretrained=True)

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

        # self.relu = nn.ReLU(inplace=True)

    def forward(self, x, reverse=False):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)

        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)

#         print(x.size())
        x_feat = x.view(x.size(0), 2048*7*7)

        x = self.model.avgpool(x)
        x = x.view(x.size(0), -1)
        # x = self.model.fc(x)


        # x = self.relu(self.bn1_fc(self.fc1(x_feat)))
        # x = F.dropout(x, training=self.training)

        # x = self.relu(self.bn2_fc(self.fc2(x)))
        # x = F.dropout(x, training=self.training)
        # if reverse:
        #   x = grad_reverse(x, self.lambd)

        # x = self.relu(self.bn3_fc(self.fc3(x)))

        return x, x_feat

class Predictor_ResNet50(nn.Module):
    def __init__(self, num_classes, prob=0.5):
        super(Predictor_ResNet50, self).__init__()
        self.num_classes = num_classes
        # self.fc3 = nn.Linear(2048, num_classes)

        self.fc3 = nn.Linear(2048, num_classes)
        nn.init.xavier_uniform_(self.fc3.weight, .1)
        nn.init.constant_(self.fc3.bias, 0.)

        self.bn_fc3 = nn.BatchNorm1d(num_classes)
        self.prob = prob

    def set_lambda(self, lambd):
        self.lambd = lambd

    def forward(self, x, reverse=False):
        x = self.fc3(x)
        return x

class DomainPredictor_ResNet50(nn.Module):
    def __init__(self, num_domains, prob=0.5):
        super(DomainPredictor_ResNet50, self).__init__()

        self.fc1 = nn.Linear(2048*7*7, 8192)
        self.bn1_fc = nn.BatchNorm1d(8192)
        self.fc2 = nn.Linear(8192, 4096)
        self.bn2_fc = nn.BatchNorm1d(4096)
        self.fc3 = nn.Linear(4096, 2048)
        self.bn3_fc = nn.BatchNorm1d(2048)
        self.fc4 = nn.Linear(2048, num_domains)

        self.prob = prob
        self.num_domains = num_domains

        self.relu = nn.ReLU(inplace=True)

    def set_lambda(self, lambd):
        self.lambd = lambd

    def forward(self, x_feat, reverse=False):

        x = self.relu(self.bn1_fc(self.fc1(x_feat)))
        x = F.dropout(x, training=self.training)

        x = self.relu(self.bn2_fc(self.fc2(x)))
        x = F.dropout(x, training=self.training)
        if reverse:
            x = grad_reverse(x, self.lambd)

        x = self.relu(self.bn3_fc(self.fc3(x)))
        x = self.fc4(x)
        return x

# ---------------------------------------------- ResNet18 -----------------------------------------

class Feature_ResNet18(nn.Module):
    def __init__(self):
        super(Feature_ResNet18, self).__init__()
        self.model = models.resnet18(pretrained=True)

        self.fc2 = nn.Linear(512, 2048)
        nn.init.xavier_uniform_(self.fc2.weight, .1)
        nn.init.constant_(self.fc2.bias, 0.)
        self.bn_fc2 = nn.BatchNorm1d(2048)

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

    def forward(self, x, reverse=False):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)

        x = self.model.layer1(x)
        x = self.model.layer2(x)

        x_feat = x

        x = self.model.layer3(x)
        x = self.model.layer4(x)

        x = self.model.avgpool(x)
        x = x.view(x.size(0), -1)

        x = self.relu(self.bn_fc2(self.fc2(x)))


        # x = self.relu(self.bn1_fc(self.fc1(x_feat)))
        # x = F.dropout(x, training=self.training)

        # x = self.relu(self.bn2_fc(self.fc2(x)))
        # x = F.dropout(x, training=self.training)
        # if reverse:
        #   x = grad_reverse(x, self.lambd)

        # x = self.relu(self.bn3_fc(self.fc3(x)))

        return x, x_feat

class Predictor_ResNet18(nn.Module):
    def __init__(self, num_classes, prob=0.5):
        super(Predictor_ResNet18, self).__init__()
        self.num_classes = num_classes
        # self.fc2 = nn.Linear(512, 2048)
        self.relu = nn.ReLU()
        self.fc3 = nn.Linear(2048, num_classes)
        nn.init.xavier_uniform_(self.fc3.weight, .1)
        nn.init.constant_(self.fc3.bias, 0.)
        # nn.init.xavier_uniform_(self.fc2.weight, .1)
        # nn.init.constant_(self.fc2.bias, 0.)
        # self.bn_fc2 = nn.BatchNorm1d(2048)
        self.prob = prob

    def set_lambda(self, lambd):
        self.lambd = lambd

    def forward(self, x, reverse=False):
        # x = self.relu(self.bn_fc2(self.fc2(x)))
        x = self.fc3(x)
        return x

class DomainPredictor_ResNet18(nn.Module):
    def __init__(self, num_domains, prob=0.5):
        super(DomainPredictor_ResNet18, self).__init__()

        self.conv1 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(256)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(512)
        self.conv3 = nn.Conv2d(512, 1024, kernel_size=3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(1024)
        self.avgpool = nn.AvgPool2d(4)
        self.fc4 = nn.Linear(1024, num_domains)

        # self.idm = nn.Linear(128*32*32, num_domains)

        self.prob = prob
        self.num_domains = num_domains

        self.relu = nn.ReLU(inplace=True)

    def set_lambda(self, lambd):
        self.lambd = lambd

    def forward(self, x_feat, reverse=False):

        x = self.relu(self.bn1(self.conv1(x_feat)))
        if reverse:
            x = grad_reverse(x, self.lambd)
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.avgpool(x)
        x = x.view(x.shape[0],-1)       
        x = self.fc4(x)

        # x = x_feat.view(x_feat.size(0), -1)
        # x = self.idm(x)

        return x

# ---------------------------------------------- ResNet18 -----------------------------------------


# class Feature_ResNet18(nn.Module):
#     def __init__(self):
#         super(Feature_ResNet18, self).__init__()
#         self.model = models.resnet18(pretrained=True)

#         # num_ftrs = self.model.fc.in_features
#         # self.model.fc = nn.Linear(num_ftrs, 1716)
#         # nn.init.xavier_uniform_(self.model.fc.weight, .1)
#         # nn.init.constant_(self.model.fc.bias, 0.)

#         # self.fc1 = nn.Linear(512*8*8, 8192)
#         # self.bn1_fc = nn.BatchNorm1d(8192)
#         # self.fc2 = nn.Linear(8192, 4096)
#         # self.bn2_fc = nn.BatchNorm1d(4096)
#         # self.fc3 = nn.Linear(4096, 2048)
#         # self.bn3_fc = nn.BatchNorm1d(2048)

#         # self.relu = nn.ReLU(inplace=True)

#     def forward(self, x, reverse=False):
#         x = self.model.conv1(x)
#         x = self.model.bn1(x)
#         x = self.model.relu(x)
#         x = self.model.maxpool(x)

#         x = self.model.layer1(x)
#         x = self.model.layer2(x)
#         x = self.model.layer3(x)
#         x = self.model.layer4(x)

        
#         x_feat = x.view(x.size(0), 512*7*7)

#         x = self.model.avgpool(x)
#         x = x.view(x.size(0), -1)
#         # x = self.model.fc(x)


#         # x = self.relu(self.bn1_fc(self.fc1(x_feat)))
#         # x = F.dropout(x, training=self.training)

#         # x = self.relu(self.bn2_fc(self.fc2(x)))
#         # x = F.dropout(x, training=self.training)
#         # if reverse:
#         #   x = grad_reverse(x, self.lambd)

#         # x = self.relu(self.bn3_fc(self.fc3(x)))

#         return x, x_feat

# class Predictor_ResNet18(nn.Module):
#     def __init__(self, num_classes, prob=0.5):
#         super(Predictor_ResNet18, self).__init__()
#         self.num_classes = num_classes
#         # self.fc3 = nn.Linear(2048, num_classes)

#         self.fc1 = nn.Linear(512, 256)
#         self.bn1_fc = nn.BatchNorm1d(256)
#         self.fc2 = nn.Linear(256, num_classes)
#         #self.bn2_fc = nn.BatchNorm1d(128)
#         # self.fc3 = nn.Linear(128, num_classes)

#         nn.init.xavier_uniform_(self.fc1.weight, .1)
#         nn.init.constant_(self.fc1.bias, 0.)
#         nn.init.xavier_uniform_(self.fc2.weight, .1)
#         nn.init.constant_(self.fc2.bias, 0.)
#         # nn.init.xavier_uniform_(self.fc3.weight, .1)
#         # nn.init.constant_(self.fc3.bias, 0.)

#         # self.bn_fc3 = nn.BatchNorm1d(num_classes)
#         # self.bn_fc4 = nn.BatchNorm1d(num_classes)
#         self.relu = nn.ReLU(inplace=True)
#         self.prob = prob

#     def set_lambda(self, lambd):
#         self.lambd = lambd

#     def forward(self, x, reverse=False):
#         x = self.relu(self.bn1_fc(self.fc1(x)))
#         #x = F.dropout(x, training=self.training)
#         return self.fc2(x)
        
#         x = self.relu(self.bn2_fc(self.fc2(x)))
#         x = F.dropout(x, training=self.training)
#         if reverse:
#             x = grad_reverse(x, self.lambd)

#         #x = self.relu(self.bn3_fc(self.fc3(x)))
#         x = self.fc3(x)
#         return x

# class DomainPredictor_ResNet18(nn.Module):
#     def __init__(self, num_domains, prob=0.5):
#         super(DomainPredictor_ResNet18, self).__init__()

#         self.fc1 = nn.Linear(512*7*7, 8192)
#         self.bn1_fc = nn.BatchNorm1d(8192)
#         self.fc2 = nn.Linear(8192, 4096)
#         self.bn2_fc = nn.BatchNorm1d(4096)
#         self.fc3 = nn.Linear(4096, 2048)
#         self.bn3_fc = nn.BatchNorm1d(2048)
#         self.fc4 = nn.Linear(2048, num_domains)
#         nn.init.xavier_uniform_(self.fc1.weight, .1)
#         nn.init.constant_(self.fc1.bias, 0.)
#         nn.init.xavier_uniform_(self.fc2.weight, .1)
#         nn.init.constant_(self.fc2.bias, 0.)
#         nn.init.xavier_uniform_(self.fc3.weight, .1)
#         nn.init.constant_(self.fc3.bias, 0.)
#         nn.init.xavier_uniform_(self.fc4.weight, .1)
#         nn.init.constant_(self.fc4.bias, 0.)

#         self.prob = prob
#         self.num_domains = num_domains

#         self.relu = nn.ReLU(inplace=True)

#     def set_lambda(self, lambd):
#         self.lambd = lambd

#     def forward(self, x_feat, reverse=False):

#         x = self.relu(self.bn1_fc(self.fc1(x_feat)))
#         x = F.dropout(x, training=self.training)

#         x = self.relu(self.bn2_fc(self.fc2(x)))
#         x = F.dropout(x, training=self.training)
#         if reverse:
#             x = grad_reverse(x, self.lambd)

#         x = self.relu(self.bn3_fc(self.fc3(x)))
#         x = self.fc4(x)
#         return x

# ---------------------------------------------- AlexNet -----------------------------------------

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


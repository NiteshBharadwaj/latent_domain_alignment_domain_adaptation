from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import mmd
import msda
from torch.autograd import Variable
from model.build_gen_digits import Generator as Generator_digit, Classifier as Classifier_digit, \
    DomainPredictor as DP_Digit
from model.build_gen import Generator as Generator_cars, Classifier as Classifier_cars, DomainPredictor as DP_cars
from datasets.dataset_read import dataset_read, dataset_hard_cluster, dataset_combined
from datasets.cars import cars_combined
import numpy as np
import math
from scipy.stats import entropy

# Training settings
class Solver(object):
    def __init__(self, args, batch_size=64,
                 target='mnist', learning_rate=0.0002, interval=100, optimizer='adam'
                 , checkpoint_dir=None, save_epoch=10):
        self.batch_size = batch_size
        self.target = target
        self.checkpoint_dir = checkpoint_dir
        self.save_epoch = save_epoch
        self.use_abs_diff = args.use_abs_diff

        self.args = args

        self.best_loss = 9999999

        print('dataset loading')
        if args.data == 'digits':
            if args.dl_type == 'original':
                self.datasets, self.dataset_test, self.dataset_valid = dataset_read(target, self.batch_size)
            elif args.dl_type == 'hard_cluster':
                self.datasets, self.dataset_test, self.dataset_valid = dataset_hard_cluster(target, self.batch_size,args.num_domain)
            elif args.dl_type == 'soft_cluster':
                self.datasets, self.dataset_test, self.dataset_valid = dataset_combined(target, self.batch_size,args.num_domain)
            elif args.dl_type == 'source_only':
                self.datasets, self.dataset_test, self.dataset_valid = dataset_combined(target, self.batch_size,args.num_domain)
            else:
                raise Exception('Type of experiment undefined')

            print('load finished!')
            num_classes = 10
            num_domains = args.num_domain

            self.G = Generator_digit()
            self.C1 = Classifier_digit()
            self.C2 = Classifier_digit()
            self.DP = DP_Digit(num_domains)
        elif args.data == 'cars':
            if args.dl_type == 'soft_cluster':
                self.datasets, self.dataset_test, self.dataset_valid = cars_combined(target, self.batch_size)
            print('load finished!')
            num_classes = 163
            num_domains = args.num_domain
            self.G = Generator_cars()
            self.C1 = Classifier_cars(num_classes)
            self.C2 = Classifier_cars(num_classes)
            self.DP = DP_cars(num_domains)
        # print(self.dataset['S1'].shape)
        print('model_loaded')

        if args.eval_only:
            checkpoint = torch.load('%s/%s_model_best.pth' % (self.checkpoint_dir, self.target))
            self.G.load_state_dict(checkpoint['G_state_dict'])
            self.C1.load_state_dict(checkpoint['C1_state_dict'])
            self.C2.load_state_dict(checkpoint['C2_state_dict'])
            self.DP.load_state_dict(checkpoint['DP_state_dict'])

            self.opt_g.load_state_dict(checkpoint['G_state_dict_opt'])
            self.opt_c1.load_state_dict(checkpoint['C1_state_dict_opt'])
            self.opt_c2.load_state_dict(checkpoint['C2_state_dict_opt'])
            self.opt_dp.load_state_dict(checkpoint['DP_state_dict_opt'])

        self.G.cuda()
        self.C1.cuda()
        self.C2.cuda()
        self.DP.cuda()
        self.interval = interval

        self.set_optimizer(which_opt=optimizer, lr=learning_rate)
        self.lr = learning_rate
        print('initialize complete')

    def set_optimizer(self, which_opt='momentum', lr=0.001, momentum=0.9):
        if which_opt == 'momentum':
            self.opt_g = optim.SGD(self.G.parameters(),lr=lr, weight_decay=0.0005, momentum=momentum)

            self.opt_c1 = optim.SGD(self.C1.parameters(), lr=lr, weight_decay=0.0005, momentum=momentum)
            self.opt_c2 = optim.SGD(self.C2.parameters(), lr=lr, weight_decay=0.0005, momentum=momentum)
            self.opt_dp = optim.SGD(self.DP.parameters(), lr=lr, weight_decay=0.0005, momentum=momentum)

        if which_opt == 'adam':
            self.opt_g = optim.Adam(self.G.parameters(), lr=lr, weight_decay=0.0005)

            self.opt_c1 = optim.Adam(self.C1.parameters(), lr=lr, weight_decay=0.0005)
            self.opt_c2 = optim.Adam(self.C2.parameters(), lr=lr, weight_decay=0.0005)
            self.opt_dp = optim.Adam(self.DP.parameters(), lr=lr, weight_decay=0.0005)

    def reset_grad(self):
        self.opt_g.zero_grad()
        self.opt_c1.zero_grad()
        self.opt_c2.zero_grad()
        self.opt_dp.zero_grad()

    def ent(self, output):
        return - torch.mean(output * torch.log(output + 1e-6))

    def discrepancy(self, out1, out2):
        return torch.mean(torch.abs(F.softmax(out1) - F.softmax(out2)))

    def feat_all_domain(self, img_s, img_t):
        feat_source = []
        for i in range(len(img_s)):
            feat_source.append(self.G(img_s[i])[0])

        return feat_source, self.G(img_t)[0]

    def C1_all_domain(self, feat_s, feat_t):
        C1_feat_source = []
        for i in range(len(feat_s)):
            C1_feat_source.append(self.C1(feat_s[i]))
        return C1_feat_source, self.C1(feat_t)

    def C2_all_domain(self, feat_s, feat_t):
        C2_feat_source = []
        for i in range(len(feat_s)):
            C2_feat_source.append(self.C2(feat_s[i]))
        return C2_feat_source, self.C2(feat_t)

    def softmax_loss_all_domain(self, output_s, label_s):
        criterion = nn.CrossEntropyLoss().cuda()
        softmax_loss = []
        for i in range(len(output_s)):
            softmax_loss.append(criterion(output_s[i], label_s[i]))
        return softmax_loss

    def loss_all_domain(self, img_s, img_t, label_s):

        feat_s, feat_t = self.feat_all_domain(img_s, img_t)

        C1_feat_s, C1_feat_t = self.C1_all_domain(feat_s, feat_t)
        C2_feat_s, C2_feat_t = self.C2_all_domain(feat_s, feat_t)

        loss_msda = 1e-4 * msda.msda_regulizer(feat_s, feat_t, 5)
        loss_source_C1 = self.softmax_loss_all_domain(C1_feat_s, label_s)
        loss_source_C2 = self.softmax_loss_all_domain(C2_feat_s, label_s)
        return loss_source_C1, loss_source_C2, loss_msda

    def feat_soft_all_domain(self, img_s, img_t):
        return self.G(img_s), self.G(img_t)

    def C1_all_domain_soft(self, feat1, feat_t):
        return self.C1(feat1), self.C1(feat_t)

    def C2_all_domain_soft(self, feat1, feat_t):
        return self.C2(feat1), self.C2(feat_t)

    def softmax_loss_all_domain_soft(self, output, label_s):
        criterion = nn.CrossEntropyLoss().cuda()
        return criterion(output, label_s)

    def entropy_loss(self, output):
        criterion = HLoss().cuda()
        return criterion(output)

    def get_kl_loss(self, domain_probs):
        bs, num_domains = domain_probs.size()
        domain_prob_sum = domain_probs.sum(0)
        uniform_prob = (torch.ones(num_domains)*(1/num_domains)).cuda()
        return (domain_prob_sum*(domain_prob_sum.log()-uniform_prob.log())).sum()


    def loss_soft_all_domain(self, img_s, img_t, label_s):
        feat_s_comb, feat_t_comb = self.feat_soft_all_domain(img_s, img_t)
        feat_s, conv_feat_s = feat_s_comb
        feat_t, conv_feat_t = feat_t_comb
        domain_logits = self.DP(conv_feat_s.detach())
        entropy_loss, domain_prob = self.entropy_loss(domain_logits)
        # print(domain_prob)

        # kl_loss = self.get_kl_loss(domain_prob)
        # kl_loss = kl_loss * 0.1
        kl_loss = 0

        if (math.isnan(entropy_loss.data.item())):
            raise Exception('entropy loss is nan')
        entropy_loss = entropy_loss * 0.1

        output_s_c1, output_t_c1 = self.C1_all_domain_soft(feat_s, feat_t)
        output_s_c2, output_t_c2 = self.C2_all_domain_soft(feat_s, feat_t)

        loss_msda = msda.msda_regulizer_soft(feat_s, feat_t, 5, domain_prob.detach()) * 1e-4
        if (math.isnan(loss_msda.data.item())):
            raise Exception('msda loss is nan')
        loss_s_c1 = \
            self.softmax_loss_all_domain_soft(output_s_c1, label_s)
        if (math.isnan(loss_s_c1.data.item())):
            raise Exception(' c1 loss is nan')
        loss_s_c2 = \
            self.softmax_loss_all_domain_soft(output_s_c2, label_s)
        if (math.isnan(loss_s_c2.data.item())):
            raise Exception(' c2 loss is nan')
        return loss_s_c1, loss_s_c2, loss_msda, entropy_loss, kl_loss


class HLoss(nn.Module):
    def __init__(self):
        super(HLoss, self).__init__()

    def forward(self, x):
        input_ = F.softmax(x, dim=1)
        mask = input_.ge(0.000001)
        mask_out = torch.masked_select(input_, mask)
        entropy = -(torch.sum(mask_out * torch.log(mask_out)))
        return entropy / float(input_.size(0)), input_ + 1e-5

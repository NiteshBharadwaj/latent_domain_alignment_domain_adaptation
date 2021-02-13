import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import mmd
import msda
import classwise_da
from torch.autograd import Variable
from model.build_gen_digits import Generator as Generator_digit, Classifier as Classifier_digit, \
    DomainPredictor as DP_Digit
from model.build_gen import Generator as Generator_cars, Classifier as Classifier_cars, DomainPredictor as DP_cars
from model.build_gen_office import Generator as Generator_office, Classifier as Classifier_office, DomainPredictor as DP_office
from datasets.dataset_read import dataset_read, dataset_hard_cluster, dataset_combined
from datasets.cars import cars_combined
from datasets.office import office_combined
import numpy as np
import math
from scipy.stats import entropy
from matplotlib import pyplot as plt
from PIL import Image

import random

# Training settings
class Solver(object):
    def __init__(self, args, batch_size=64,
                 target='mnist', learning_rate=0.0002, interval=100, optimizer='adam'
                 , checkpoint_dir=None, save_epoch=10, class_disc= False):
        self.batch_size = batch_size
        self.target = target
        self.checkpoint_dir = checkpoint_dir
        self.save_epoch = save_epoch
        self.use_abs_diff = args.use_abs_diff
        self.dl_type = args.dl_type

        self.args = args

        self.best_loss = 9999999
        self.best_acc = 0
        print('dataset loading')
        if args.data == 'digits':
            if args.dl_type == 'original':
                self.datasets, self.dataset_test, self.dataset_valid = dataset_read(target, self.batch_size)
            elif args.dl_type == 'hard_cluster':
                self.datasets, self.dataset_test, self.dataset_valid = dataset_hard_cluster(target, self.batch_size,args.num_domain)
            elif args.dl_type == 'soft_cluster':
                self.datasets, self.dataset_test, self.dataset_valid, self.classwise_dataset, self.is_multi, self.usps_only = dataset_combined(target, self.batch_size,args.num_domain, args.office_directory, args.seed, 
                        usps_less_data_protocol = args.usps_less_data_protocol)
            elif args.dl_type == 'source_only':
                self.datasets, self.dataset_test, self.dataset_valid, self.classwise_dataset, self.is_multi, self.usps_only = dataset_combined(target, self.batch_size,args.num_domain, args.office_directory, args.seed,
                        usps_less_data_protocol = args.usps_less_data_protocol)
            elif args.dl_type == 'source_target_only':
                self.datasets, self.dataset_test, self.dataset_valid, self.classwise_dataset, self.is_multi, self.usps_only = dataset_combined(target, self.batch_size,args.num_domain, args.office_directory, args.seed,
                        usps_less_data_protocol = args.usps_less_data_protocol)
            elif args.dl_type=='classwise_msda':
                self.datasets, self.dataset_test, self.dataset_valid, self.classwise_dataset, self.is_multi, self.usps_only = dataset_combined(target, self.batch_size,args.num_domain, args.office_directory, args.seed,
                        usps_less_data_protocol = args.usps_less_data_protocol)
            else:
                raise Exception('Type of experiment undefined')

            print('load finished!')
            self.num_classes = 10
            #if args.dl_type=='source_target_only':
            #    args.num_domain = 4 # To maintain reproducibility for a seed. Num domains is not used but can introduce a bit of randomness
            num_domains = args.num_domain
            self.num_domains = num_domains
            self.entropy_wt = args.entropy_wt
            self.msda_wt = args.msda_wt
            self.kl_wt = args.kl_wt
            self.to_detach = args.to_detach
            self.G = Generator_digit(cd=class_disc, usps_only=self.usps_only)
            self.C1 = Classifier_digit(cd=class_disc, usps_only=self.usps_only)
            self.C2 = Classifier_digit(cd=class_disc, usps_only=self.usps_only)
            self.DP = DP_Digit(num_domains,cd=class_disc, usps_only=self.usps_only)
        elif args.data == 'cars':
            if args.dl_type == 'soft_cluster':
                self.datasets, self.dataset_test, self.dataset_valid = cars_combined(target, self.batch_size)
            elif args.dl_type == 'source_target_only':
                self.datasets, self.dataset_test, self.dataset_valid = cars_combined(target, self.batch_size)
            elif args.dl_type == 'source_only':
                self.datasets, self.dataset_test, self.dataset_valid = cars_combined(target, self.batch_size)
            print('load finished!')
            self.entropy_wt = args.entropy_wt
            self.msda_wt = args.msda_wt
            self.to_detach = args.to_detach
            self.num_classes = 163
            num_domains = args.num_domain
            self.num_domains = num_domains
            self.G = Generator_cars()
            self.C1 = Classifier_cars(self.num_classes)
            self.C2 = Classifier_cars(self.num_classes)
            self.DP = DP_cars(num_domains)
        elif args.data == 'office':
            if args.dl_type == 'soft_cluster':
                self.datasets, self.dataset_test, self.dataset_valid, self.classwise_dataset = office_combined(target, self.batch_size, args.office_directory, args.seed)
            elif args.dl_type == 'source_target_only':
                self.datasets, self.dataset_test, self.dataset_valid, self.classwise_dataset = office_combined(target, self.batch_size, args.office_directory, args.seed)
            elif args.dl_type == 'source_only':
                self.datasets, self.dataset_test, self.dataset_valid, self.classwise_dataset = office_combined(target, self.batch_size, args.office_directory, args.seed)

            print('load finished!')
            self.entropy_wt = args.entropy_wt
            self.msda_wt = args.msda_wt
            self.kl_wt = args.kl_wt
            self.to_detach = args.to_detach
            self.num_classes = 31
            num_domains = args.num_domain
            self.num_domains = num_domains
            self.G = Generator_office()
            self.C1 = Classifier_office(self.num_classes)
            self.C2 = Classifier_office(self.num_classes)
            self.DP = DP_office(num_domains)
        # print(self.dataset['S1'].shape)
        print('model_loaded')

        self.set_optimizer(which_opt=optimizer, lr=learning_rate)
        print('ARGS EVAL ONLY : ', args.eval_only)
        if args.saved_model_dir != 'na':
            print('Loading model from: ','%s' % (args.saved_model_dir))
            checkpoint = torch.load('%s' % (args.saved_model_dir))
            self.G.load_state_dict(checkpoint['G_state_dict'])
            self.C1.load_state_dict(checkpoint['C1_state_dict'])
            self.C2.load_state_dict(checkpoint['C2_state_dict'])
            self.DP.load_state_dict(checkpoint['DP_state_dict'])

            self.opt_g.load_state_dict(checkpoint['G_state_dict_opt'])
            self.opt_c1.load_state_dict(checkpoint['C1_state_dict_opt'])
            self.opt_c2.load_state_dict(checkpoint['C2_state_dict_opt'])
            self.opt_dp.load_state_dict(checkpoint['DP_state_dict_opt'])

        if args.eval_only:
            print('Loading state from: ','%s/%s_model_best.pth' % (self.checkpoint_dir, self.target))
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
        if args.data=='cars':
            milestones = [100]
        else:
            milestones = [100]
        self.sche_g = torch.optim.lr_scheduler.MultiStepLR(self.opt_g, milestones, gamma=0.1)
        self.sche_c1 = torch.optim.lr_scheduler.MultiStepLR(self.opt_c1, milestones, gamma=0.1)
        self.sche_c2 = torch.optim.lr_scheduler.MultiStepLR(self.opt_c2, milestones, gamma=0.1)
        self.sche_dp = torch.optim.lr_scheduler.MultiStepLR(self.opt_dp, milestones, gamma=0.1)

        self.lr = learning_rate
        print('initialize complete')

    def set_optimizer(self, which_opt='momentum', lr=0.001, momentum=0.9):
        if which_opt == 'momentum':
            self.opt_g = optim.SGD(self.G.parameters(),lr=lr, weight_decay=0.0005, momentum=momentum)

            self.opt_c1 = optim.SGD(self.C1.parameters(), lr=lr, weight_decay=0.0005, momentum=momentum)
            self.opt_c2 = optim.SGD(self.C2.parameters(), lr=lr, weight_decay=0.0005, momentum=momentum)
            self.opt_dp = optim.SGD(self.DP.parameters(), lr=lr/100.0, weight_decay=0.0005, momentum=momentum)

        if which_opt == 'adam':
            self.opt_g = optim.Adam(self.G.parameters(), lr=lr, weight_decay=0.0005)

            self.opt_c1 = optim.Adam(self.C1.parameters(), lr=lr, weight_decay=0.0005)
            self.opt_c2 = optim.Adam(self.C2.parameters(), lr=lr, weight_decay=0.0005)
            self.opt_dp = optim.Adam(self.DP.parameters(), lr=lr/100.0, weight_decay=0.0005)

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
        # Takes input source and target images returns the feature from feature extractor
        return self.G(img_s), self.G(img_t)

    def C1_all_domain_soft(self, feat1, feat_t):
        #Takes source and target features from feature extractor and returns classifier output features
        return self.C1(feat1), self.C1(feat_t)

    def C2_all_domain_soft(self, feat1, feat_t):
        return self.C2(feat1), self.C2(feat_t)

    def softmax_loss_all_domain_soft(self, output, label_s):
        # takes logits and labels and  returns the cross entropy loss 
        criterion = nn.CrossEntropyLoss().cuda()
        return criterion(output, label_s)

    def entropy_loss(self, output):
        criterion = HLoss().cuda()
        return criterion(output)

    def get_kl_loss(self, domain_probs):
        #IGNORE
        bs, num_domains = domain_probs.size()
        domain_prob_sum = domain_probs.sum(0)/bs
        uniform_prob = (torch.ones(num_domains)*(1/num_domains)).cuda()
        return (domain_prob_sum*(domain_prob_sum.log()-uniform_prob.log())).sum()

    def get_domain_entropy(self, domain_probs):
        bs, num_domains = domain_probs.size()
        domain_prob_sum = domain_probs.sum(0)/bs
        mask = domain_prob_sum.ge(0.000001)
        domain_prob_sum = domain_prob_sum*mask + (1-mask.int())*1e-5
        return -(domain_prob_sum*(domain_prob_sum.log())).mean()

    def loss_soft_all_domain(self, img_s, img_t, label_s, epoch, img_s_cl, force_attach = False, single_domain_mode=False):
        # Takes source images, target images, source labels and returns classifier loss, domain adaptation loss and entropy loss
        feat_s_comb, feat_t_comb = self.feat_soft_all_domain(img_s, img_t)
        feat_s, conv_feat_s, feat_da_s = feat_s_comb
        feat_t, conv_feat_t, feat_da_t = feat_t_comb

        if self.to_detach:
            domain_logits, _ = self.DP(img_s)
            cl_s_logits,_ = self.DP(img_s_cl)
        else:
            domain_logits, _ = self.DP(img_s)
            cl_s_logits,_ = self.DP(img_s_cl)
        entropy_loss, domain_prob = self.entropy_loss(domain_logits)

        _,cl_s_prob = self.entropy_loss(cl_s_logits)

        kl_loss = -self.get_domain_entropy(cl_s_prob)
        kl_loss = kl_loss * self.kl_wt
        
        if (math.isnan(entropy_loss.data.item())):
            raise Exception('entropy loss is nan')
        entropy_loss = entropy_loss * self.entropy_wt

        output_s_c1, output_t_c1 = self.C1_all_domain_soft(feat_s, feat_t)
        output_s_c2, output_t_c2 = self.C2_all_domain_soft(feat_s, feat_t)
        if self.to_detach and not force_attach:
            loss_msda_nc2, loss_msda_nc1 = msda.msda_regulizer_soft(feat_da_s, feat_da_t, 5, domain_prob.detach(), single_domain_mode=single_domain_mode)
        else:
            loss_msda_nc2, loss_msda_nc1 = msda.msda_regulizer_soft(feat_da_s, feat_da_t, 5, domain_prob, single_domain_mode=single_domain_mode)
        loss_msda_nc2 = loss_msda_nc2*self.msda_wt
        loss_msda_nc1 = loss_msda_nc1*self.msda_wt
        if (math.isnan(loss_msda_nc2.data.item())):
            raise Exception('msda loss nc2 is nan')
        loss_s_c1 = \
            self.softmax_loss_all_domain_soft(output_s_c1, label_s)
        if (math.isnan(loss_s_c1.data.item())):
            raise Exception(' c1 loss is nan')
        loss_s_c2 = \
            self.softmax_loss_all_domain_soft(output_s_c2, label_s)
        if (math.isnan(loss_s_c2.data.item())):
            raise Exception(' c2 loss is nan')
        if (math.isnan(kl_loss.data.item())):
            raise Exception(' kl loss is nan')
        #print(loss_s_c1, loss_s_c2, loss_msda, entropy_loss, kl_loss, domain_prob)
        #print(self.DP.fc3.weight)
#        print("loss_s_c1", loss_s_c1, "loss_s_c2", loss_s_c2, "loss_msda", loss_msda, "entropy_loss", entropy_loss, "kl_loss", kl_loss)
        return loss_s_c1, loss_s_c2, loss_msda_nc2, loss_msda_nc1, entropy_loss, kl_loss, domain_prob

    def loss_domain_class_mmd(self, img_s, img_t, label_s, epoch, img_s_cl, force_attach = False, single_domain_mode=False):
        feat_s_comb, feat_t_comb = self.feat_soft_all_domain(img_s, img_t)
        feat_s, _, feat_da_s = feat_s_comb
        feat_t, _, feat_da_t = feat_t_comb

        domain_logits, _ = self.DP(img_s)
        cl_s_logits,_ = self.DP(img_s_cl)

        entropy_loss, domain_prob_s = self.entropy_loss(domain_logits)

        _,cl_s_prob = self.entropy_loss(cl_s_logits)

        kl_loss = -self.get_domain_entropy(cl_s_prob)
        kl_loss = kl_loss * self.kl_wt

        if (math.isnan(entropy_loss.data.item())):
            raise Exception('entropy loss is nan')
        entropy_loss = entropy_loss * self.entropy_wt

        output_s_c1, output_t_c1 = self.C1_all_domain_soft(feat_s, feat_t)
        output_s_c2, output_t_c2 = self.C2_all_domain_soft(feat_s, feat_t)

        # _, class_prob_s = self.entropy_loss(output_s_c1)
        _, class_prob_t = self.entropy_loss(output_t_c1)

        if self.to_detach and not force_attach:
            intra_domain_mmd_loss, inter_domain_mmd_loss = classwise_da.class_da_regulizer_soft(feat_da_s, feat_da_t, 5, self.get_one_hot_encoding(label_s, self.num_classes).cuda(), class_prob_t.detach(), domain_prob_s.detach())
        else:
            intra_domain_mmd_loss, inter_domain_mmd_loss = classwise_da.class_da_regulizer_soft(feat_da_s, feat_da_t, 5, self.get_one_hot_encoding(label_s, self.num_classes).cuda(), class_prob_t, domain_prob_s)
        intra_domain_mmd_loss = intra_domain_mmd_loss*self.msda_wt
        inter_domain_mmd_loss = inter_domain_mmd_loss*self.msda_wt

        if (math.isnan(intra_domain_mmd_loss.data.item())):
            raise Exception('intra_domain_mmd_loss is nan')
        loss_s_c1 = \
            self.softmax_loss_all_domain_soft(output_s_c1, label_s)
        if (math.isnan(loss_s_c1.data.item())):
            raise Exception(' c1 loss is nan')
        loss_s_c2 = \
            self.softmax_loss_all_domain_soft(output_s_c2, label_s)
        if (math.isnan(loss_s_c2.data.item())):
            raise Exception(' c2 loss is nan')
        if (math.isnan(kl_loss.data.item())):
            raise Exception(' kl loss is nan')

        return loss_s_c1, loss_s_c2, intra_domain_mmd_loss, inter_domain_mmd_loss, entropy_loss, kl_loss, domain_prob
    
    def get_one_hot_encoding(self, labels, num_classes):
        y = torch.eye(num_classes)
        return y[labels]


class HLoss(nn.Module):
    def __init__(self):
        super(HLoss, self).__init__()

    def forward(self, x):
        domain_prob = F.softmax(x, dim=1)
        b = F.softmax(x, dim=1) * F.log_softmax(x, dim=1)
        b = -1.0 * b.mean()
        return b, domain_prob + 1e-5

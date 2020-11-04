from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import mmd
import msda
from torch.autograd import Variable
from model.build_gen_digits import Generator as Generator_digit, Classifier as Classifier_digit, DomainPredictor as DP_Digit
from model.build_gen import Generator as Generator_cars, Classifier as Classifier_cars, DomainPredictor as DP_cars
from model.build_gen_office_caltech import Generator as Generator_office_caltech, Classifier as Classifier_office_caltech, DomainPredictor as DP_office_caltech
from model.build_gen_pacs import Generator as Generator_pacs, Classifier as Classifier_pacs, DomainPredictor as DP_pacs
from model.build_gen_domainnet import Generator as Generator_domainnet, Classifier as Classifier_domainnet, DomainPredictor as DP_domainnet


from datasets.dataset_read_cluster_testing import dataset_read, dataset_hard_cluster, dataset_combined
from datasets.cars import cars_combined
from datasets.office import office_combined
from datasets.office_caltech import office_caltech_combined
from datasets.pacs import pacs_combined

from datasets.domainnet import domainnet_combined

import numpy as np
import math
from scipy.stats import entropy
from matplotlib import pyplot as plt
from PIL import Image

# Training settings
class Solver(object):
    def __init__(self, args, batch_size=64,
                 target='mnist', learning_rate=0.002, interval=8, optimizer='adam'
                 , checkpoint_dir=None, save_epoch=10):
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
                self.datasets, self.dataset_test, self.dataset_valid, self.classwise_dataset = dataset_combined(target, self.batch_size,args.num_domain, args.office_directory, args.seed)
                if self.args.clustering_only:
                    _, _, self.dataset_svhn, _ = dataset_combined('mnist_svhn', self.batch_size,args.num_domain, args.office_directory, args.seed)
                    _, _, self.dataset_usps, _ = dataset_combined('mnist_usps', self.batch_size,args.num_domain, args.office_directory, args.seed)
                    _, _, self.dataset_mnist, _ = dataset_combined('mnist_mnist', self.batch_size,args.num_domain, args.office_directory, args.seed)
                    _, _, self.dataset_syn, _ = dataset_combined('mnist_syn', self.batch_size,args.num_domain, args.office_directory, args.seed)
            elif args.dl_type == 'source_only':
                self.datasets, self.dataset_test, self.dataset_valid, self.classwise_dataset = dataset_combined(target, self.batch_size,args.num_domain, args.office_directory, args.seed)
            elif args.dl_type == 'source_target_only':
                self.datasets, self.dataset_test, self.dataset_valid, self.classwise_dataset = dataset_combined(target, self.batch_size,args.num_domain, args.office_directory, args.seed)
            else:
                raise Exception('Type of experiment undefined')

            print('load finished!')
            num_classes = 10
            num_domains = args.num_domain
            self.num_domains = num_domains
            self.entropy_wt = args.entropy_wt
            self.msda_wt = args.msda_wt
            self.kl_wt = args.kl_wt
            self.to_detach = args.to_detach
            self.G = Generator_digit()
            self.C1 = Classifier_digit()
            self.C2 = Classifier_digit()
            self.DP = DP_Digit(num_domains)
        elif args.data == 'cars':
            if args.dl_type == 'soft_cluster':
                self.datasets, self.dataset_test, self.dataset_valid = cars_combined(target, self.batch_size)
            elif args.dl_type == 'source_target_only':
                self.datasets, self.dataset_test, self.dataset_valid = cars_combined(target, self.batch_size)
            elif args.dl_type == 'source_only':
                self.datasets, self.dataset_test, self.dataset_valid = cars_combined(target, self.batch_size)
            print('load finished!')
            self.entropy_wt = args.entropy_wt
            self.kl_wt = args.kl_wt
            self.msda_wt = args.msda_wt
            self.to_detach = args.to_detach
            num_classes = 163
            num_domains = args.num_domain
            self.num_domains = num_domains
            self.G = Generator_cars()
            self.C1 = Classifier_cars(num_classes)
            self.C2 = Classifier_cars(num_classes)
            self.DP = DP_cars(num_domains)
        elif args.data == 'office':
            if args.dl_type == 'soft_cluster':
                self.datasets, self.dataset_test, self.dataset_valid, self.classwise_dataset = office_combined(target, self.batch_size, args.office_directory, args.seed, args.num_workers)
                #if self.args.clustering_only:
                    #_, _, self.dataset_amazon, _ = office_combined('dwa', self.batch_size, args.office_directory, args.seed, args.num_workers)
                #    _, _, self.dataset_dslr, _ = office_combined('awd', self.batch_size, args.office_directory, args.seed, args.num_workers)
                #    _, _, self.dataset_webcam, _ = office_combined('adw', self.batch_size, args.office_directory, args.seed, args.num_workers)
            elif args.dl_type == 'source_target_only':
                self.datasets, self.dataset_test, self.dataset_valid, self.classwise_dataset = office_combined(target, self.batch_size, args.office_directory, args.seed, args.num_workers)
            elif args.dl_type == 'source_only':
                self.datasets, self.dataset_test, self.dataset_valid, self.classwise_dataset = office_combined(target, self.batch_size, args.office_directory, args.seed, args.num_workers)

            print('load finished!')
            self.entropy_wt = args.entropy_wt
            self.msda_wt = args.msda_wt
            self.kl_wt = args.kl_wt
            self.to_detach = args.to_detach
            num_classes = 31
            num_domains = args.num_domain
            self.num_domains = num_domains
            from model.build_gen_office import Generator as Generator_office, Classifier as Classifier_office, DomainPredictor as DP_office
            if(args.network == 'alex'):
                from model.build_gen_office import GeneratorAlex as Generator_office, ClassifierAlex as Classifier_office, DomainPredictorAlex as DP_office
                
            
            
            
            
            self.G = Generator_office()
            self.C1 = Classifier_office(num_classes)
            self.C2 = Classifier_office(num_classes)
            self.DP = DP_office(num_domains)

        elif args.data == 'office_caltech':
            if args.dl_type == 'soft_cluster':

                if args.dl_type == 'soft_cluster':
                    self.datasets, self.dataset_test, self.classwise_dataset = office_caltech_combined(target,self.batch_size,
                                                                                                        args.office_caltech_directory,
                                                                                                        args.seed,
                                                                                                        args.num_workers)

            elif args.dl_type == 'source_target_only':

                self.datasets, self.dataset_test, self.classwise_dataset = office_caltech_combined(target,self.batch_size,
                                                                                                    args.office_caltech_directory,
                                                                                                    args.seed,args.num_workers)


            elif args.dl_type == 'source_only':

                self.datasets, self.dataset_test, self.classwise_dataset = office_caltech_combined(target,self.batch_size,
                                                                                                    args.office_caltech_directory,
                                                                                                    args.seed,
                                                                                                    args.num_workers)

            print('load finished!')
            self.entropy_wt = args.entropy_wt
            self.msda_wt = args.msda_wt
            self.kl_wt = args.kl_wt
            self.to_detach = args.to_detach
            num_classes = 10
            num_domains = args.num_domain
            self.num_domains = num_domains
            self.G = Generator_office_caltech()
            self.C1 = Classifier_office_caltech(num_classes)
            self.C2 = Classifier_office_caltech(num_classes)
            self.DP = DP_office_caltech(num_domains)

        elif args.data == 'pacs':
            if args.dl_type == 'soft_cluster':

                    self.datasets, self.dataset_test, self.dataset_valid, self.classwise_dataset = pacs_combined(
                                                                                                    target,
                                                                                                    self.batch_size,
                                                                                                    args.pacs_directory,
                                                                                                    args.seed,
                                                                                                    args.num_workers)

            elif args.dl_type == 'source_target_only':

                self.datasets, self.dataset_test, self.dataset_valid, self.classwise_dataset = pacs_combined(
                                                                                                target,
                                                                                                self.batch_size,
                                                                                                args.pacs_directory,
                                                                                                args.seed,
                                                                                                args.num_workers)


            elif args.dl_type == 'source_only':

                self.datasets, self.dataset_test, self.dataset_valid, self.classwise_dataset = pacs_combined(
                                                                                                target,
                                                                                                self.batch_size,
                                                                                                args.pacs_directory,
                                                                                                args.seed,
                                                                                                args.num_workers)

            print('load finished!')
            self.entropy_wt = args.entropy_wt
            self.msda_wt = args.msda_wt
            self.kl_wt = args.kl_wt
            self.to_detach = args.to_detach
            num_classes = 7
            num_domains = args.num_domain
            self.num_domains = num_domains
            self.G = Generator_pacs()
            self.C1 = Classifier_pacs(num_classes)
            self.C2 = Classifier_pacs(num_classes)
            self.DP = DP_pacs(num_domains)

        elif args.data == 'domainnet':
            if args.dl_type == 'soft_cluster':
                self.datasets, self.dataset_test, self.dataset_valid, self.classwise_dataset = domainnet_combined(
                                                                                                    target,
                                                                                                    self.batch_size,
                                                                                                    args.domainnet_directory,
                                                                                                    args.seed,
                                                                                                    args.num_workers)

            elif args.dl_type == 'source_target_only':
                self.datasets, self.dataset_test, self.dataset_valid, self.classwise_dataset = domainnet_combined(
                                                                                                target,
                                                                                                self.batch_size,
                                                                                                args.domainnet_directory,
                                                                                                args.seed,
                                                                                                args.num_workers)


            elif args.dl_type == 'source_only':
                self.datasets, self.dataset_test, self.dataset_valid, self.classwise_dataset = domainnet_combined(
                                                                                                target,
                                                                                                self.batch_size,
                                                                                                args.domainnet_directory,
                                                                                                args.seed,
                                                                                                args.num_workers)

            print('load finished!')
            self.entropy_wt = args.entropy_wt
            self.msda_wt = args.msda_wt
            self.kl_wt = args.kl_wt
            self.to_detach = args.to_detach
            num_classes = 345
            num_domains = args.num_domain
            self.num_domains = num_domains
            self.G = Generator_domainnet()
            self.C1 = Classifier_domainnet(num_classes)
            self.C2 = Classifier_domainnet(num_classes)
            self.DP = DP_domainnet(num_domains)

        # print(self.dataset['S1'].shape)
        print('model_loaded')

        self.set_optimizer(which_opt=optimizer, lr=learning_rate)
        print('ARGS EVAL ONLY : ', args.eval_only)
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
            milestones = [900]
        self.sche_g = torch.optim.lr_scheduler.MultiStepLR(self.opt_g, milestones, gamma=0.1)
        self.sche_c1 = torch.optim.lr_scheduler.MultiStepLR(self.opt_c1, milestones, gamma=0.1)
        self.sche_c2 = torch.optim.lr_scheduler.MultiStepLR(self.opt_c2, milestones, gamma=0.1)
        self.sche_dp = torch.optim.lr_scheduler.MultiStepLR(self.opt_dp, milestones, gamma=0.1)

        self.lr = learning_rate
        print('initialize complete')

    def set_optimizer(self, which_opt='momentum', lr=0.001, momentum=0.9):
        if which_opt == 'momentum':
            self.opt_g = optim.SGD(self.G.parameters(),lr=lr, weight_decay=1e-6, momentum=momentum)

            self.opt_c1 = optim.SGD(self.C1.parameters(), lr=lr*10, weight_decay=1e-6, momentum=momentum)
            self.opt_c2 = optim.SGD(self.C2.parameters(), lr=lr*10, weight_decay=1e-6, momentum=momentum)
            self.opt_dp = optim.SGD(self.DP.parameters(), lr=lr/self.args.lr_ratio, weight_decay=1e-6, momentum=momentum)

        if which_opt == 'adam':
            self.opt_g = optim.Adam(self.G.parameters(), lr=lr, weight_decay=1e-6)

            self.opt_c1 = optim.Adam(self.C1.parameters(), lr=lr*10, weight_decay=1e-6)
            self.opt_c2 = optim.Adam(self.C2.parameters(), lr=lr*10, weight_decay=1e-6)
            self.opt_dp = optim.Adam(self.DP.parameters(), lr=lr/self.args.lr_ratio, weight_decay=1e-6)

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
    
    def source_only_loss(self, img_s, label_s, epoch):
        feat_s_comb = self.G(img_s)
        feat_s, conv_feat_s = feat_s_comb
        output_s_c1 = self.C1(feat_s)
        loss_s_c1 = self.softmax_loss_all_domain_soft(output_s_c1, label_s)
        return loss_s_c1
        

    def loss_soft_all_domain(self, img_s, img_t, label_s, epoch, img_s_cl):
        # Takes source images, target images, source labels and returns classifier loss, domain adaptation loss and entropy loss
        feat_s_comb, feat_t_comb = self.feat_soft_all_domain(img_s, img_t)
        feat_s, conv_feat_s = feat_s_comb
        feat_t, conv_feat_t = feat_t_comb
        #with torch.no_grad():
        #    _, conv_feat_cl = self.G(img_s_cl)
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
       
        if self.to_detach:
            loss_msda = msda.msda_regulizer_soft(feat_s, feat_t, 5, domain_prob.detach()) * self.msda_wt 
        else:
            loss_msda = msda.msda_regulizer_soft(feat_s, feat_t, 5, domain_prob) * self.msda_wt
        if (math.isnan(entropy_loss.data.item())):
            raise Exception('entropy loss is nan')
        entropy_loss = entropy_loss * self.entropy_wt

        output_s_c1, output_t_c1 = self.C1_all_domain_soft(feat_s, feat_t)
        output_s_c2, output_t_c2 = self.C2_all_domain_soft(feat_s, feat_t)
        loss_s_c1 = \
            self.softmax_loss_all_domain_soft(output_s_c1, label_s)
        if (math.isnan(loss_s_c1.data.item())):
            raise Exception(' c1 loss is nan')
        loss_s_c2 = \
            self.softmax_loss_all_domain_soft(output_s_c2, label_s)
        #print(loss_s_c1, loss_s_c2, loss_msda, entropy_loss, kl_loss, domain_prob)
        #print(self.DP.fc3.weight)
#        print("loss_s_c1", loss_s_c1, "loss_s_c2", loss_s_c2, "loss_msda", loss_msda, "entropy_loss", entropy_loss, "kl_loss", kl_loss)
        return loss_s_c1, loss_s_c2, loss_msda, entropy_loss, kl_loss, domain_prob


class HLoss(nn.Module):
    def __init__(self):
        super(HLoss, self).__init__()

    def forward(self, x):
        domain_prob = F.softmax(x, dim=1)
        b = F.softmax(x, dim=1) * F.log_softmax(x, dim=1)
        b = -1.0 * b.mean()
        return b, domain_prob + 1e-5

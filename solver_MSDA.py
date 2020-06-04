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
                self.datasets, self.dataset_test, self.dataset_valid = dataset_hard_cluster(target, self.batch_size, args.num_domain, args.office_directory, args.seed,
                        usps_less_data_protocol = args.usps_less_data_protocol)
            elif args.dl_type == 'soft_cluster':
                self.datasets, self.dataset_test, self.dataset_valid, self.classwise_dataset, self.is_multi, self.usps_only = dataset_combined(target, self.batch_size,args.num_domain, args.office_directory, args.seed, 
                        usps_less_data_protocol = args.usps_less_data_protocol)
            elif args.dl_type == 'source_only':
                self.datasets, self.dataset_test, self.dataset_valid, self.classwise_dataset, self.is_multi, self.usps_only = dataset_combined(target, self.batch_size,args.num_domain, args.office_directory, args.seed,
                        usps_less_data_protocol = args.usps_less_data_protocol)
            elif args.dl_type == 'source_target_only':
                self.datasets, self.dataset_test, self.dataset_valid, self.classwise_dataset, self.is_multi, self.usps_only = dataset_combined(target, self.batch_size,args.num_domain, args.office_directory, args.seed,
                        usps_less_data_protocol = args.usps_less_data_protocol)
            else:
                raise Exception('Type of experiment undefined')

            print('load finished!')
            num_classes = 10
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
            num_classes = 163
            num_domains = args.num_domain
            self.num_domains = num_domains
            self.G = Generator_cars()
            self.C1 = Classifier_cars(num_classes)
            self.C2 = Classifier_cars(num_classes)
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
            num_classes = 31
            num_domains = args.num_domain
            self.num_domains = num_domains
            self.G = Generator_office()
            self.C1 = Classifier_office(num_classes)
            self.C2 = Classifier_office(num_classes)
            self.DP = DP_office(num_domains)
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
        #_, conv_s_cl = self.G(img_s_cl)
        #if self.to_detach:
        #    domain_logits, _ = self.DP(conv_feat_s.detach())
        #    cl_s_logits,_ = self.DP(conv_s_cl.detach())
        #else:
        #    domain_logits, _ = self.DP(conv_feat_s)
        #    cl_s_logits,_ = self.DP(conv_s_cl)
        #_, conv_s_cl = self.G(img_s_cl)
        if self.to_detach:
            domain_logits, _ = self.DP(img_s)
            cl_s_logits,_ = self.DP(img_s_cl)
        else:
            domain_logits, _ = self.DP(img_s)
            cl_s_logits,_ = self.DP(img_s_cl)
        entropy_loss, domain_prob = self.entropy_loss(domain_logits)

        _,cl_s_prob = self.entropy_loss(cl_s_logits)

#        if self.args.eval_only:
#            print("domain_prob", domain_prob)
#            print("cl_s_prob", cl_s_prob)
#            return 0, 0, 0, 0, 0, 0

        kl_loss = -self.get_domain_entropy(cl_s_prob)
        kl_loss = kl_loss * self.kl_wt
        
        
#         kl_loss = self.get_kl_loss(domain_prob)
#         kl_loss = kl_loss * self.kl_wt
        
#        kl_loss = torch.zeros(1).cuda()

#         total_domains = domain_prob.size()[1]
#         domains = domain_prob.data.max(1)[1]
#         print(domains)
#         if(epoch % 100 == 0):
#             for i in range(total_domains):
#                 i_index = ((domains == i).nonzero()).squeeze()
#                 img_s_i = img_s[i_index,:,:]
#                 for k in range(img_s_i.size()[0]):
#                     img_ = img_s_i[k, :, :, :].squeeze().permute(1,2,0)
#                     img_ = img_.cpu().detach().numpy()
#                     #print(img_.shape)
#                     #print(img_.size())
#                     index = i_index[k]
#                     plt.imshow(img_) 
#                     plt.savefig(str(i)+'/'+str(index.item())+'.png')
#                 im = Image.fromarray(img_.sum().item())
#                 im.save()
#                 matplotlib.image.imsave(str(i)+'/'+str(index), img_.cpu().sum().item())
#        if (feat_s!=feat_s).any():
#           import pdb
#           pdb.set_trace()
#       if (conv_feat_s!=conv_feat_s).any():
#           import pdb
#           pdb.set_trace()
#   
#       if (domain_prob!=domain_prob).any():
#           import pdb
#           pdb.set_trace()
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


class HLoss(nn.Module):
    def __init__(self):
        super(HLoss, self).__init__()

    def forward(self, x):
        domain_prob = F.softmax(x, dim=1)
        b = F.softmax(x, dim=1) * F.log_softmax(x, dim=1)
        b = -1.0 * b.mean()
        return b, domain_prob + 1e-5

# Takes input tensor of shape (N x num_domains) and computes the entropy loss sum(p * logp)
#class HLoss(nn.Module):
#    def __init__(self):
#        super(HLoss, self).__init__()

#    def forward(self, x):
#        input_ = F.softmax(x, dim=1)
#        mask = input_.ge(0.000001)
#        mask_out = input_*mask + (1-mask.int())*1e-5
#        entropy = -(torch.sum(mask_out * torch.log(mask_out)))
#        loss = entropy/ float(input_.size(0))
#        return loss, mask_out

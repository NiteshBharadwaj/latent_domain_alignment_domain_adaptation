from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import mmd
import msda
from torch.autograd import Variable
from model.build_gen import *
from datasets.dataset_read import dataset_read, dataset_hard_cluster
import numpy as np
from itertools import combinations


# Training settings
class Solver(object):
    def __init__(self, args, batch_size=64, target='mnist', learning_rate=0.0002, interval=100, optimizer='adam'
                 , checkpoint_dir=None, save_epoch=10):
        self.batch_size = batch_size
        self.target = target
        self.checkpoint_dir = checkpoint_dir
        self.save_epoch = save_epoch
        self.use_abs_diff = args.use_abs_diff

        print('dataset loading')
        # self.datasets, self.dataset_test = dataset_read(target, self.batch_size)
        self.datasets, self.dataset_test = dataset_hard_cluster(target, self.batch_size)

        # print(self.dataset['S1'].shape)

        print('load finished!')
        self.G = Generator()
        self.C1 = Classifier()
        self.C2 = Classifier()
        print('model_loaded')

        if args.eval_only:
            self.G.torch.load(
                '%s/%s_to_%s_model_epoch%s_G.pt' % (self.checkpoint_dir, self.source, self.target, args.resume_epoch))
            self.G.torch.load(
                '%s/%s_to_%s_model_epoch%s_G.pt' % (
                    self.checkpoint_dir, self.source, self.target, self.checkpoint_dir, args.resume_epoch))
            self.G.torch.load(
                '%s/%s_to_%s_model_epoch%s_G.pt' % (self.checkpoint_dir, self.source, self.target, args.resume_epoch))

        self.G.cuda()
        self.C1.cuda()
        self.C2.cuda()
        self.interval = interval

        self.set_optimizer(which_opt=optimizer, lr=learning_rate)
        self.lr = learning_rate
        print('initialize complete')

    def set_optimizer(self, which_opt='momentum', lr=0.001, momentum=0.9):
        if which_opt == 'momentum':
            self.opt_g = optim.SGD(self.G.parameters(), lr=lr, weight_decay=0.0005, momentum=momentum)
            self.opt_c1 = optim.SGD(self.C1.parameters(), lr=lr, weight_decay=0.0005, momentum=momentum)
            self.opt_c2 = optim.SGD(self.C2.parameters(), lr=lr, weight_decay=0.0005, momentum=momentum)

        if which_opt == 'adam':
            self.opt_g = optim.Adam(self.G.parameters(), lr=lr, weight_decay=0.0005)
            self.opt_c1 = optim.Adam(self.C1.parameters(), lr=lr, weight_decay=0.0005)
            self.opt_c2 = optim.Adam(self.C2.parameters(), lr=lr, weight_decay=0.0005)

    def reset_grad(self):
        self.opt_g.zero_grad()
        self.opt_c1.zero_grad()
        self.opt_c2.zero_grad()

    def ent(self, output):
        return - torch.mean(output * torch.log(output + 1e-6))

    def discrepancy(self, out1, out2):
        return torch.mean(torch.abs(F.softmax(out1) - F.softmax(out2)))

    def train(self, epoch, record_file=None):
        criterion = nn.CrossEntropyLoss().cuda()
        self.G.train()
        self.C1.train()

        torch.cuda.manual_seed(1)

        for batch_idx, data in enumerate(self.datasets):
            img_s, label_s = [], []
            img_t = Variable(data['T'].cuda())
            num_datasets = len(data['S'])

            for i in range(num_datasets):
                img_s.append(Variable(data['S'][i]).cuda())
                label_s.append(Variable(data['S'][i]).long().cuda())
                
            if any(img_s[i].size()[0] < self.batch_size for i in range(num_datasets)) \
                    and img_t.size()[0] < self.batch_size:
                break

            self.reset_grad()

            feat_s, feat_t = self.feat_all_domain(img_s, img_t)
            output_s, output_t = self.C1_all_domain(feat_s, feat_t)

            loss_s = []
            for i in range(num_datasets):
                loss_s.append(criterion(output_s[i], label_s[i]))

            loss_s_all = sum(loss_s) / num_datasets

            loss_msda = 0.5 * msda.msda_regulizer(feat_s, feat_t, 5) / (self.batch_size * self.batch_size)
            loss = loss_msda + loss_s_all
            loss_s_all.backward()

            self.opt_c1.step()
            self.opt_g.step()
            self.reset_grad()

            if batch_idx > 500:
                return batch_idx

            if batch_idx % self.interval == 0:
                print_val = 'Train Epoch: {} [{}/{} ({:.0f}%)]' + '\tLoss : {:.6f}' * num_datasets + '\t Discrepancy: {:.6f}'
                print(print_val.format(epoch, batch_idx, 100, 100. * batch_idx / 70000, *loss_s.data[0],
                                       loss_msda.data[0]))

                if record_file:
                    record = open(record_file, 'a')
                    record_val = '%s ' * (num_datasets + 1) + '\n'
                    record.write(record_val % (loss_msda.data[0], *loss_s.data[0]))
                    record.close()

        return batch_idx

    def train_merge_baseline(self, epoch, record_file=None):
        criterion = nn.CrossEntropyLoss().cuda()
        self.G.train()
        self.C1.train()

        torch.cuda.manual_seed(1)

        for batch_idx, data in enumerate(self.datasets):
            # if
            img_s = np.concatenate([data['S1'][0:44], data['S2'][44:88], data['S3'][88:]], 0)
            label_s = np.concatenate([data['S1_label'][0:44], data['S2_label'][44:88], data['S3_label'][88:]], 0)
            img_t = Variable(data['T'].cuda())
            img_s = Variable(torch.from_numpy(img_s).cuda())

            label_s = Variable(torch.from_numpy(label_s).long().cuda())

            if img_s.size()[0] < self.batch_size or img_t.size()[0] < self.batch_size:
                break

            self.reset_grad()
            feat_s = self.G(img_s)
            output_s = self.C1(feat_s)

            feat_t = self.G(img_t)
            output_t = self.C1(feat_t)

            loss_s = criterion(output_s, label_s)

            loss_s.backward()

            self.opt_c1.step()
            self.opt_g.step()
            self.reset_grad()
            if batch_idx > 500:
                return batch_idx

            if batch_idx % self.interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss1: {:.6f}\t '.format(
                    epoch, batch_idx, 100,
                    100. * batch_idx / 70000, loss_s.data[0]))
                if record_file:
                    record = open(record_file, 'a')
                    record.write('%s\n' % (loss_s[0]))
                    record.close()
        return batch_idx

    def test(self, epoch, record_file=None, save_model=False):
        self.G.eval()
        self.C1.eval()
        test_loss = 0
        correct1 = 0
        size = 0
        feature_all = np.array([])
        label_all = []
        for batch_idx, data in enumerate(self.dataset_test):
            img = data['T']
            label = data['T_label']

            img, label = img.cuda(), label.long().cuda()
            img, label = Variable(img, volatile=True), Variable(label)
            feat = self.G(img)
            # print('feature.shape:{}'.format(feat.shape))

            if batch_idx == 0:
                label_all = label.data.cpu().numpy().tolist()
                feature_all = feat.data.cpu().numpy()
            else:
                feature_all = np.ma.row_stack((feature_all, feat.data.cpu().numpy()))
                feature_all = feature_all.data
                label_all = label_all + label.data.cpu().numpy().tolist()

            # print(feature_all.shape)

            output1 = self.C1(feat)

            test_loss += F.nll_loss(output1, label).data.item()
            pred1 = output1.data.max(1)[1]
            k = label.data.size()[0]
            correct1 += pred1.eq(label.data).cpu().sum()
            size += k
        np.savez('result_plot_sv_t', feature_all, label_all)
        test_loss = test_loss / size
        print(
            '\nTest set: Average loss: {:.4f}, Accuracy C1: {}/{} ({:.0f}%)  \n'.format(
                test_loss, correct1, size, 100. * correct1 / size))
        if save_model and epoch % self.save_epoch == 0:
            torch.save(self.G,
                       '%s/%s_to_%s_model_epoch%s_G.pt' % (self.checkpoint_dir, self.source, self.target, epoch))
            torch.save(self.C1,
                       '%s/%s_to_%s_model_epoch%s_C1.pt' % (self.checkpoint_dir, self.source, self.target, epoch))
            torch.save(self.C2,
                       '%s/%s_to_%s_model_epoch%s_C2.pt' % (self.checkpoint_dir, self.source, self.target, epoch))
        if record_file:
            record = open(record_file, 'a')
            print('recording %s', record_file)
            record.write('%s\n' % (float(correct1) / size))
            record.close()

    def feat_all_domain(self, img_s, img_t):
        feat_source = []
        for i in range(len(img_s)):
            feat_source.append(self.G(img_s[i]))
        return feat_source, self.G(img_t)

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

        loss_msda = 0.0005 * msda.msda_regulizer(feat_s, feat_t, 5)
        loss_source_C1 = self.softmax_loss_all_domain(C1_feat_s, label_s)
        loss_source_C2 = self.softmax_loss_all_domain(C2_feat_s, label_s)

        return loss_source_C1, loss_source_C2, loss_msda

    def train_MSDA(self, epoch, record_file=None):
        criterion = nn.CrossEntropyLoss().cuda()
        self.G.train()
        self.C1.train()
        self.C2.train()
        torch.cuda.manual_seed(1)

        for batch_idx, data in enumerate(self.datasets):
            img_s, label_s = [], []
            img_t = Variable(data['T'].cuda())
            num_datasets = len(data['S'])

            for i in range(num_datasets):
                img_s.append(Variable(data['S'][i]).cuda())
                label_s.append(Variable(data['S_label'][i]).long().cuda())

            if any(img_s[i].size()[0] < self.batch_size for i in range(num_datasets)) \
                    and img_t.size()[0] < self.batch_size:
                break

            self.reset_grad()
       

            loss_source_C1, loss_source_C2, loss_msda = self.loss_all_domain(img_s, img_t, label_s)

            loss = sum(loss_source_C1) + sum(loss_source_C2) + loss_msda
            loss.backward()

            self.opt_g.step()
            self.opt_c1.step()
            self.opt_c2.step()
            self.reset_grad()

            loss_source_C1, loss_source_C2, loss_msda = self.loss_all_domain(img_s, img_t, label_s)

            feat_t = self.G(img_t)
            output_t1 = self.C1(feat_t)
            output_t2 = self.C2(feat_t)

            loss_s = sum(loss_source_C1) + sum(loss_source_C2) + loss_msda
            loss_dis = self.discrepancy(output_t1, output_t2)
            loss = loss_s - loss_dis
            loss.backward()
            self.opt_c1.step()
            self.opt_c2.step()
            self.reset_grad()

            for i in range(4):
                feat_t = self.G(img_t)
                output_t1 = self.C1(feat_t)
                output_t2 = self.C2(feat_t)
                loss_dis = self.discrepancy(output_t1, output_t2)
                loss_dis.backward()
                self.opt_g.step()
                self.reset_grad()

            if batch_idx > 500:
                return batch_idx

            if batch_idx % self.interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss1: {:.6f}\t Loss2: {:.6f}\t  Discrepancy: {:.6f}'.format(
                    epoch, batch_idx, 100,
                    100. * batch_idx / 70000, sum(loss_source_C1).data.item(), sum(loss_source_C2).data.item(),
                    loss_dis.data.item()))
                if record_file:
                    record = open(record_file, 'a')
                    record.write(
                        '%s %s %s\n' % (loss_dis.data.item(), sum(loss_source_C1).data.item(), sum(loss_source_C2).data.item()))
                    record.close()

        return batch_idx

    def train_MMD(self, epoch, record_file=None):
        criterion = nn.CrossEntropyLoss().cuda()
        self.G.train()
        self.C1.train()

        torch.cuda.manual_seed(1)

        for batch_idx, data in enumerate(self.datasets):
            img_s, label_s = [], []
            img_t = Variable(data['T'].cuda())
            num_datasets = len(data['S'])

            for i in range(num_datasets):
                img_s.append(Variable(data['S'][i]).cuda())
                label_s.append(Variable(data['S'][i]).long().cuda())

            if any(img_s[i].size()[0] < self.batch_size for i in range(num_datasets)) \
                    and img_t.size()[0] < self.batch_size:
                break

            self.reset_grad()

            feat_s, feat_t = self.feat_all_domain(img_s, img_t)
            output_s, output_t = self.C1_all_domain(feat_s, feat_t)

            print('->shape', output_s[0].shape, label_s[0].shape)

            loss_s = []
            for i in range(num_datasets):
                loss_s.append(criterion(output_s[i], label_s[i]))

            loss_s_all = sum(loss_s) / num_datasets

            sigma = [1, 2, 5, 10]
            comb = list(combinations(range(num_datasets), 2))

            loss_msda = 0
            for i in range(num_datasets):
                loss_msda += mmd.mix_rbf_mmd2(feat_s[i], feat_t, sigma)

            for k in range(len(comb)):
                loss_msda += mmd.mix_rbf_mmd2(feat_s[comb[k][0]], feat_s[comb[k][1]])

            loss = 10 * loss_msda + loss_s_all
            loss.backward()

            self.opt_c1.step()
            self.opt_g.step()
            self.reset_grad()
            if batch_idx > 500:
                return batch_idx

            if batch_idx % self.interval == 0:
                print_val = 'Train Epoch: {} [{}/{} ({:.0f}%)]' + '\tLoss : {:.6f}' * num_datasets + '\t Discrepancy: {:.6f}'

                print(print_val.format(epoch, batch_idx, 100, 100. * batch_idx / 70000, *loss_s.data[0],
                                       loss_msda.data[0]))
                if record_file:
                    record = open(record_file, 'a')
                    record_val = '%s ' * (num_datasets + 1) + '\n'
                    record.write(record_val % (loss_msda.data[0], *loss_s.data[0]))
                    record.close()

        return batch_idx

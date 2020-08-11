import torch
import torch.nn as nn
from torch.autograd import Variable
import mmd
import msda
import math


def loss_single_domain(solver, img_s, img_t, label_s. img_s_domain_label):
    feat_s_comb, feat_t_comb = solver.feat_soft_all_domain(img_s, img_t)
    feat_s, conv_feat_s = feat_s_comb
    feat_t, conv_feat_t = feat_t_comb
    output_s_c1, output_t_c1 = solver.C1_all_domain_soft(feat_s, feat_t)
    output_s_c2, output_t_c2 = solver.C2_all_domain_soft(feat_s, feat_t)

    num_clusters = 4
    indices_0 = ((img_s_domain_label == 0).nonzero()).squeeze()
    feat_s_0 = img_s[indices_0,:,:,:]
    indices_1 = ((img_s_domain_label == 1).nonzero()).squeeze()
    feat_s_1 = img_s[indices_1,:,:,:]
    indices_2 = ((img_s_domain_label == 2).nonzero()).squeeze()
    feat_s_2 = img_s[indices_2,:,:,:]
    indices_3 = ((img_s_domain_label == 3).nonzero()).squeeze()
    feat_s_3 = img_s[indices_3,:,:,:]


    loss_msda = msda.msda_regulizer_single(feat_s_0, feat_t, 5) * solver.msda_wt
    loss_msda = loss_msda + msda.msda_regulizer_single(feat_s_1, feat_t, 5) * solver.msda_wt
    loss_msda = loss_msda + msda.msda_regulizer_single(feat_s_2, feat_t, 5) * solver.msda_wt
    loss_msda = loss_msda + msda.msda_regulizer_single(feat_s_3, feat_t, 5) * solver.msda_wt

    if (math.isnan(loss_msda.data.item())):
        raise Exception('msda loss is nan')
    loss_s_c1 = \
        solver.softmax_loss_all_domain_soft(output_s_c1, label_s[:,0])
    if (math.isnan(loss_s_c1.data.item())):
        raise Exception(' c1 loss is nan')
    loss_s_c2 = \
        solver.softmax_loss_all_domain_soft(output_s_c2, label_s[:,0])
    if (math.isnan(loss_s_c2.data.item())):
        raise Exception(' c2 loss is nan')
    return loss_s_c1, loss_s_c2, loss_msda

def train_MSDA_hard_adaptation(solver, epoch, classifier_disc=True, record_file=None):
    solver.G.train()
    solver.C1.train()
    solver.C2.train()
    solver.DP.train()
    #torch.cuda.manual_seed(1)

    batch_idx_g = 0

    for batch_idx, data in enumerate(solver.datasets):
        batch_idx_g = batch_idx
        img_t = Variable(data['T'].cuda())
        img_s = Variable(data['S'].cuda())
        label_s = Variable(data['S_label'].long().cuda())
        img_s_domain_label = data['S_paths']
        if img_s.size()[0] < solver.batch_size or img_t.size()[0] < solver.batch_size:
            break

        solver.reset_grad()

        loss_s_c1, loss_s_c2, loss_msda = loss_single_domain(solver,img_s, img_t, label_s)
        if not classifier_disc:
            loss_s_c2 = loss_s_c1
        loss = loss_s_c1 + loss_msda + loss_s_c2
        #loss = loss_s_c1

        loss.backward()

        solver.opt_g.step()
        solver.opt_c1.step()
        solver.opt_c2.step()
        solver.opt_dp.step()
        loss_dis = loss * 0  # For printing purpose, it's reassigned if classifier_disc=True
        # if classifier_disc:
        #     solver.reset_grad()
        #     loss_s_c1, loss_s_c2, loss_msda = loss_single_domain(solver,img_s, img_t, label_s, img_s_domain_label)

        #     feat_t, conv_feat_t = solver.G(img_t)
        #     output_t1 = solver.C1(feat_t)
        #     output_t2 = solver.C2(feat_t)

        #     loss_s = loss_s_c1 + loss_msda + loss_s_c2
        #     loss_dis = solver.discrepancy(output_t1, output_t2)
        #     loss = loss_s - loss_dis
        #     loss.backward()
        #     solver.opt_c1.step()
        #     solver.opt_c2.step()
        #     solver.reset_grad()

        #     for i in range(solver.args.num_k):
        #         feat_t, conv_feat_t = solver.G(img_t)
        #         output_t1 = solver.C1(feat_t)
        #         output_t2 = solver.C2(feat_t)
        #         loss_dis = solver.discrepancy(output_t1, output_t2)
        #         loss_dis.backward()
        #         solver.opt_g.step()
        #         solver.reset_grad()

        if batch_idx % solver.interval == 0:
            print \
                ('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss1: {:.6f}\t Loss2: {:.6f}\t Loss_mmd: {:.6f}\t Loss_entropy: {:.6f}\t kl_loss: {:.6f}'.format(
                epoch, batch_idx, 100, 100. * batch_idx / 70000, loss_s_c1.data.item(), loss_s_c2.data.item(), loss_msda.data.item(), 0, 0))
            if record_file:
                record = open(record_file, 'a')
                record.write('%s %s %s %s %s %s\n' %
                (0, loss_s_c1.data.item(), loss_s_c2.data.item(), loss_msda.data.item(), 0, 0))
                record.close()
    return batch_idx_g
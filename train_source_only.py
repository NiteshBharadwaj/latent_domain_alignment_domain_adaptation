import torch
import torch.nn as nn
from torch.autograd import Variable
import mmd
import msda
import math


def loss_single_domain(solver, img_s, img_t, label_s):
    feat_s_comb, feat_t_comb = solver.feat_soft_all_domain(img_s, img_t)
    feat_s, conv_feat_s = feat_s_comb
    feat_t, conv_feat_t = feat_t_comb
    output_s_c1, output_t_c1 = solver.C1_all_domain_soft(feat_s, feat_t)
    output_s_c2, output_t_c2 = solver.C2_all_domain_soft(feat_s, feat_t)
    loss_msda = msda.msda_regulizer_single(feat_s, feat_t, 5) * solver.msda_wt

    if (math.isnan(loss_msda.data.item())):
        raise Exception('msda loss is nan')
    loss_s_c1 = \
        solver.softmax_loss_all_domain_soft(output_s_c1, label_s)
    if (math.isnan(loss_s_c1.data.item())):
        raise Exception(' c1 loss is nan')
    loss_s_c2 = \
        solver.softmax_loss_all_domain_soft(output_s_c2, label_s)
    if (math.isnan(loss_s_c2.data.item())):
        raise Exception(' c2 loss is nan')
    return loss_s_c1, loss_s_c2, loss_msda

def train_source_only(solver, epoch, record_file=None):
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
        if img_s.size()[0] < solver.batch_size or img_t.size()[0] < solver.batch_size:
            break
        
        
        solver.reset_grad()
        
        loss_s_c1 = solver.source_only_loss(img_s, label_s, epoch)

        loss_s_c1.backward()

        solver.opt_g.step()
        solver.opt_c1.step()

#         if batch_idx % solver.interval == 0:
#             print \
#                 ('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss1: {:.6f}\t Loss2: {:.6f}\t Loss_mmd: {:.6f}\t Loss_entropy: {:.6f}\t kl_loss: {:.6f}'.format(
#                 epoch, batch_idx, 100, 100. * batch_idx / 70000, loss_s_c1.data.item(), loss_s_c2.data.item(), loss_msda.data.item(), 0, 0))
#             if record_file:
#                 record = open(record_file, 'a')
#                 record.write('%s %s %s %s %s %s\n' %
#                 (0, loss_s_c1.data.item(), loss_s_c2.data.item(), loss_msda.data.item(), 0, 0))
#                 record.close()
    return batch_idx_g

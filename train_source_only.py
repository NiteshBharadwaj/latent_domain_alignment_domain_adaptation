import torch
import torch.nn as nn
from torch.autograd import Variable


def loss_source_only(solver, img_s, label_s):
    # Takes source images, target images, source labels and returns classifier loss, domain adaptation loss and entropy loss
    feat_s, conv_feat_s, feat_da_s = solver.G(img_s)
    output_s_c1 = solver.C1(feat_s)
    output_s_c2 = solver.C2(feat_s)
    loss_s_c1 = solver.softmax_loss_all_domain_soft(output_s_c1, label_s)
    loss_s_c2 = solver.softmax_loss_all_domain_soft(output_s_c2, label_s)
    return loss_s_c1, loss_s_c2


def train_source_only(solver, epoch, record_file=None):
    criterion = nn.CrossEntropyLoss().cuda()
    solver.G.train()
    solver.C1.train()
    solver.C2.train()
    solver.DP.train()
    batch_idx_g = 0

    for batch_idx, data in enumerate(solver.datasets):
        batch_idx_g = batch_idx
        img_t = Variable(data['T'].cuda())
        img_s = Variable(data['S'].cuda())
        label_s = Variable(data['S_label'].long().cuda())
        if img_s.size()[0] < solver.batch_size or img_t.size()[0] < solver.batch_size:
            break

        solver.reset_grad()

        loss_s_c1, loss_s_c2 = loss_source_only(solver, img_s, label_s)

        loss = loss_s_c1

        loss.backward()

        solver.opt_g.step()
        solver.opt_c1.step()

        if batch_idx % solver.interval == 0:
            print(
                'Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss1: {:.6f}\t Loss2: {:.6f}\t Loss_mmd: {:.6f}\t Loss_entropy: {:.6f}\t Discrepancy: {:.6f}'.format(
                    epoch, batch_idx, 100, 100. * batch_idx / 70000, loss_s_c1.data.item(), 0, 0, 0, 0))
            if record_file:
                record = open(record_file, 'a')
                record.write('%s %s %s %s %s %s\n' % (0, loss_s_c1.data.item(), 0, 0, 0, 0))
                record.close()
    return batch_idx_g

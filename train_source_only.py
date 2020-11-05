import torch
import torch.nn as nn
from torch.autograd import Variable

def train_source_only(solver, epoch, record_file=None):
    criterion = nn.CrossEntropyLoss().cuda()
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

        loss_s_c1, loss_s_c2, loss_msda_nc2, loss_msda_nc1, entropy_loss, kl_loss, aux_loss, _ = solver.loss_soft_all_domain(img_s, img_t, label_s,0,img_s)

        loss = loss_s_c1

        loss.backward()

        solver.opt_g.step()
        solver.opt_c1.step()

        if batch_idx % solver.interval == 1:
            print(
                'Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss1: {:.6f}\t Loss2: {:.6f}\t Loss_mmd: {:.6f}\t Loss_entropy: {:.6f}\t Discrepancy: {:.6f}'.format(
                    epoch, batch_idx, 100, 100. * batch_idx / 70000, loss_s_c1.data.item(), 0, 0, 0, 0))
            if record_file:
                record = open(record_file, 'a')
                record.write('%s %s %s %s %s %s\n' % (0, loss_s_c1.data.item(), 0, 0, 0, 0))
                record.close()
    return batch_idx_g

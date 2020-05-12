import torch
import torch.nn as nn
from torch.autograd import Variable

def train_MSDA_soft(solver, epoch, classifier_disc=True, record_file=None):
    solver.G.train()
    solver.C1.train()
    solver.C2.train()
    solver.DP.train()
    #torch.cuda.manual_seed(1)

    batch_idx_g = 0
    classwise_dataset_iterator = iter(solver.classwise_dataset)
    for batch_idx, data in enumerate(solver.datasets):
        batch_idx_g = batch_idx
        img_t = Variable(data['T'].cuda())
        img_s = Variable(data['S'].cuda())
        label_s = Variable(data['S_label'].long().cuda())
        if img_s.size()[0] < solver.batch_size or img_t.size()[0] < solver.batch_size:
            break

        classwise_data = next(classwise_dataset_iterator)
        img_s_cl = Variable(classwise_data['S'].cuda())

        solver.reset_grad()

        loss_s_c1, loss_s_c2, loss_msda, entropy_loss, kl_loss, domain_prob = solver.loss_soft_all_domain(img_s, img_t, label_s, epoch, img_s_cl)
        if not classifier_disc:
            loss_s_c2 = loss_s_c1
        loss = loss_s_c1 + loss_msda + loss_s_c2 + entropy_loss + kl_loss
        
        loss.backward()

        solver.opt_g.step()
        solver.opt_c1.step()
        solver.opt_c2.step()
        solver.opt_dp.step()
        loss_dis = loss * 0  # For printing purpose, it's reassigned if classifier_disc=True
        if classifier_disc:
            solver.reset_grad()
            loss_s_c1, loss_s_c2, loss_msda, entropy_loss, kl_loss, domain_prob = solver.loss_soft_all_domain(img_s, img_t, label_s, epoch, img_s_cl)

            feat_t, conv_feat_t = solver.G(img_t)
            output_t1 = solver.C1(feat_t)
            output_t2 = solver.C2(feat_t)

            loss_s = loss_s_c1 + loss_msda + loss_s_c2 + entropy_loss + kl_loss
            loss_dis = solver.discrepancy(output_t1, output_t2)
            loss = loss_s - loss_dis
            loss.backward()
            solver.opt_c1.step()
            solver.opt_c2.step()
            solver.reset_grad()

            for i in range(solver.args.num_k):
                feat_t, conv_feat_t = solver.G(img_t)
                output_t1 = solver.C1(feat_t)
                output_t2 = solver.C2(feat_t)
                loss_dis = solver.discrepancy(output_t1, output_t2)
                loss_dis.backward()
                solver.opt_g.step()
                solver.reset_grad()

        if batch_idx % solver.interval == 0:
            print \
                ('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss1: {:.6f}\t Loss2: {:.6f}\t Loss_mmd: {:.6f}\t Loss_entropy: {:.6f}\t kl_loss: {:.6f}'.format(
                epoch, batch_idx, 100, 100. * batch_idx / 70000, loss_s_c1.data.item(), loss_s_c2.data.item(), loss_msda.data.item(), entropy_loss.data.item(), kl_loss.data.item()))
            if record_file:
                record = open(record_file, 'a')
                record.write('%s %s %s %s %s %s\n' %
                (0, loss_s_c1.data.item(), loss_s_c2.data.item(), loss_msda.data.item(), entropy_loss.data.item(), kl_loss.data.item()))
                record.close()
    return batch_idx_g

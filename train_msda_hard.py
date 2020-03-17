import torch
import torch.nn as nn
from torch.autograd import Variable

def train_MSDA(solver, epoch, classifier_disc=True, record_file=None):
    solver.G.train()
    solver.C1.train()
    solver.C2.train()
    torch.cuda.manual_seed(1)

    for batch_idx, data in enumerate(solver.datasets):
        img_s, label_s = [], []
        img_t = Variable(data['T'].cuda())
        num_datasets = len(data['S'])

        for i in range(num_datasets):
            img_s.append(Variable(data['S'][i]).cuda())
            label_s.append(Variable(data['S_label'][i]).long().cuda())

        if any(img_s[i].size()[0] < solver.batch_size for i in range(num_datasets)) \
                or img_t.size()[0] < solver.batch_size:
            break

        solver.reset_grad()
        loss_source_C1, loss_source_C2, loss_msda = solver.loss_all_domain(img_s, img_t, label_s)
        loss_s_c1 = sum(loss_source_C1)
        loss_s_c2 = sum(loss_source_C2)
        loss = loss_s_c1+loss_s_c2
        loss += loss_msda

        loss.backward()

        solver.opt_g.step()
        solver.opt_c1.step()
        solver.opt_c2.step()
        solver.reset_grad()
        loss_dis = loss * 0  # For printing purpose, it's reassigned if classifier_disc=True
        if classifier_disc:
            loss_source_C1, loss_source_C2, loss_msda = solver.loss_all_domain(img_s, img_t, label_s)

            feat_t = solver.G(img_t)[0]
            output_t1 = solver.C1(feat_t)
            output_t2 = solver.C2(feat_t)
            loss_s_c1 = sum(loss_source_C1)
            loss_s_c2 = sum(loss_source_C2)

            loss_s = loss_source_C1[0] + loss_source_C2[0] + loss_msda
            loss_dis = solver.discrepancy(output_t1, output_t2)
            loss = loss_s - loss_dis
            loss.backward()
            solver.opt_c1.step()
            solver.opt_c2.step()
            solver.reset_grad()

            for i in range(4):
                feat_t = solver.G(img_t)[0]
                output_t1 = solver.C1(feat_t)
                output_t2 = solver.C2(feat_t)
                loss_dis = solver.discrepancy(output_t1, output_t2)
                loss_dis.backward()
                solver.opt_g.step()
                solver.reset_grad()

        if batch_idx % solver.interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss1: {:.6f}\t Loss2: {:.6f}\t  Discrepancy: {:.6f}'.format(
                epoch, batch_idx, 100,
                100. * batch_idx / 70000, loss_s_c1.data.item(), loss_s_c2.data.item(), loss_dis.data.item()))
            if record_file:
                record = open(record_file, 'a')
                record.write('%s %s %s\n' % (loss_dis.data.item(), loss_s_c1.data.item(), loss_s_c2.data.item()))
                record.close()
    return batch_idx
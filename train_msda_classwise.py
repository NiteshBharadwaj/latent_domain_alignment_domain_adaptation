import torch
import torch.nn as nn
from torch.autograd import Variable

def train_MSDA_classwise(solver, epoch, classifier_disc=True, record_file=None, single_domain_mode=False, summary_writer=None, epoch_start_idx=0):
    solver.G.train()
    solver.C1.train()
    solver.C2.train()
    solver.DP.train()
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
        
        loss_s_c1, loss_s_c2, intra_domain_mmd_loss, inter_domain_mmd_loss, entropy_loss, kl_loss, domain_prob = solver.loss_domain_class_mmd(img_s, img_t, label_s, epoch, img_s_cl, single_domain_mode=single_domain_mode)
        if not classifier_disc:
            loss_s_c2 = loss_s_c1
        if solver.args.saved_model_dir!='na' and epoch < 20:
            loss = entropy_loss + kl_loss
        else:
            if solver.args.alternate_optimization!=-1:
                if batch_idx%solver.args.alternate_optimization==0:
                    loss = loss_s_c1 + intra_domain_mmd_loss + inter_domain_mmd_loss + loss_s_c2
                else:
                    loss = entropy_loss + kl_loss
            else:
                loss = loss_s_c1 + intra_domain_mmd_loss + inter_domain_mmd_loss + loss_s_c2 + entropy_loss + kl_loss
        loss.backward()
        clip_value = 1.0

#        for param_group in solver.G.param_groups:
#            print("LR opt_g", param_group['lr'])
#        for param_group in solver.C1.param_groups:
#            print("LR opt_c1", param_group['lr'])
#        for param_group in solver.C2.param_groups:
#            print("LR opt_c2", param_group['lr'])
#        for param_group in solver.DP.param_groups:
#            print("LR opt_dp", param_group['lr'])

        #torch.nn.utils.clip_grad_norm(solver.G.parameters(), clip_value) 
        #torch.nn.utils.clip_grad_norm(solver.C1.parameters(), clip_value)
        #if classifier_disc:
        #    torch.nn.utils.clip_grad_norm(solver.C2.parameters(), clip_value)
        #torch.nn.utils.clip_grad_norm(solver.DP.parameters(), clip_value)
        solver.opt_g.step()
        solver.opt_c1.step()
        solver.opt_c2.step()
        solver.opt_dp.step()
        if summary_writer is not None:
            summary_writer.add_scalar('Loss/intra_domain_mmd_loss', intra_domain_mmd_loss/(solver.args.msda_wt + 1e-8), epoch_start_idx + batch_idx_g)
            summary_writer.add_scalar('Loss/inter_domain_mmd_loss', inter_domain_mmd_loss/(solver.args.msda_wt + 1e-8), epoch_start_idx + batch_idx_g)
            summary_writer.add_scalar('Loss/loss_s_c1', loss_s_c1, epoch_start_idx + batch_idx_g)
            summary_writer.add_scalar('Loss/loss_s_c2', loss_s_c2, epoch_start_idx + batch_idx_g)
            summary_writer.add_scalar('Loss/loss_entropy', entropy_loss/solver.args.entropy_wt, epoch_start_idx + batch_idx_g)
            summary_writer.add_scalar('Loss/loss_kl', kl_loss/solver.args.kl_wt, epoch_start_idx + batch_idx_g)

        assert not classifier_disc

        if batch_idx % solver.interval == 0:
            print \
                    ('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss1: {:.6f}\t Loss2: {:.6f}\t inter_domain_mmd_loss NC1: {:.6f}\t intra_domain_mmd_loss NC2: {:6f} \t Loss_entropy: {:.6f}\t kl_loss: {:.6f}'.format(
                epoch, batch_idx, 100, 100. * batch_idx / 70000, loss_s_c1.data.item(), loss_s_c2.data.item(), inter_domain_mmd_loss.data.item(), intra_domain_mmd_loss.data.item(), entropy_loss.data.item(), kl_loss.data.item()))
    return batch_idx_g

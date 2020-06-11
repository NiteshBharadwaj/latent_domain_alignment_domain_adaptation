import torch
import torch.nn as nn
from torch.autograd import Variable


def train_MSDA_soft(solver, epoch, classifier_disc=True, record_file=None, single_domain_mode=False,
                    summary_writer=None, epoch_start_idx=0):
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

        loss_s_c1, loss_s_c2, loss_msda_nc2, loss_msda_nc1, entropy_loss, kl_loss, domain_prob = solver.loss_soft_all_domain(
            img_s, img_t, label_s, epoch, img_s_cl, single_domain_mode=single_domain_mode)
        if not classifier_disc:
            loss_s_c2 = loss_s_c1
        else:
            loss_msda_nc1 = loss_msda_nc1 * 0

        if solver.args.adv_mode:
            loss = loss_s_c1 + loss_msda_nc1 + loss_msda_nc2 + loss_s_c2
        else:
            loss = loss_s_c1 + loss_msda_nc1 + loss_msda_nc2 + loss_s_c2 + entropy_loss + kl_loss
        loss.backward()

        solver.opt_g.step()
        solver.opt_c1.step()
        solver.opt_c2.step()
        if summary_writer is not None:
            summary_writer.add_scalar('Loss/loss_nc1', loss_msda_nc1 / (solver.args.msda_wt + 1e-8),
                                      epoch_start_idx + batch_idx_g)
            summary_writer.add_scalar('Loss/loss_nc2', loss_msda_nc2 / (solver.args.msda_wt + 1e-8),
                                      epoch_start_idx + batch_idx_g)
            summary_writer.add_scalar('Loss/loss_s_c1', loss_s_c1, epoch_start_idx + batch_idx_g)
            summary_writer.add_scalar('Loss/loss_s_c2', loss_s_c2, epoch_start_idx + batch_idx_g)
            summary_writer.add_scalar('Loss/loss_entropy', entropy_loss / solver.args.entropy_wt,
                                      epoch_start_idx + batch_idx_g)
            summary_writer.add_scalar('Loss/loss_kl', kl_loss / solver.args.kl_wt, epoch_start_idx + batch_idx_g)
        if not solver.args.adv_mode:
            solver.opt_dp.step()
        else:
            solver.reset_grad()
            _, _, loss_msda, entropy_loss, kl_loss, _ = solver.loss_soft_all_domain(img_s, img_t, label_s, epoch,
                                                                                    img_s_cl, force_attach=True)
            loss_adv = entropy_loss + kl_loss - loss_msda * solver.args.adv_wt
            loss_adv.backward()
            solver.opt_dp.step()

        if classifier_disc:
            solver.reset_grad()
            loss_s_c1, loss_s_c2, _, _, entropy_loss, kl_loss, domain_prob = solver.loss_soft_all_domain(img_s, img_t,
                                                                                                         label_s, epoch,
                                                                                                         img_s_cl,
                                                                                                         single_domain_mode=single_domain_mode)

            feat_t, _, _ = solver.G(img_t)
            output_t1 = solver.C1(feat_t)
            output_t2 = solver.C2(feat_t)

            loss_s = loss_s_c1 + loss_s_c2
            loss_dis = solver.discrepancy(output_t1, output_t2)
            loss = loss_s - loss_dis
            loss.backward()

            solver.opt_c1.step()
            solver.opt_c2.step()
            solver.reset_grad()

            for i in range(solver.args.num_k):
                feat_t, _, _ = solver.G(img_t)
                output_t1 = solver.C1(feat_t)
                output_t2 = solver.C2(feat_t)
                loss_dis = solver.discrepancy(output_t1, output_t2)
                loss_dis.backward()

                solver.opt_g.step()
                solver.reset_grad()

        if batch_idx % solver.interval == 0:
            print \
                    (
                    'Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss1: {:.6f}\t Loss2: {:.6f}\t Loss_mmd NC1: {:.6f}\t Loss_mmd NC2: {:6f} \t Loss_entropy: {:.6f}\t kl_loss: {:.6f}'.format(
                        epoch, batch_idx, 100, 100. * batch_idx / 70000, loss_s_c1.data.item(), loss_s_c2.data.item(),
                        loss_msda_nc1.data.item(), loss_msda_nc2.data.item(), entropy_loss.data.item(),
                        kl_loss.data.item()))
    return batch_idx_g

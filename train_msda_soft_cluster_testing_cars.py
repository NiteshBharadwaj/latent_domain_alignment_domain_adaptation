import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision
import os

cluster_batch = None
classwise_batch = None


# Add batches for each dataset
# data1_batch=None
# data2_batch=None

def switch_bn(model, on):
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
            if on:
                m.train()
            else:
                m.eval()


import time


def train_MSDA_soft(solver, epoch, classifier_disc=True, record_file=None):
    global cluster_batch
    # TODO: add batch for each domain
    # global data1_batch
    # global data2_batch
    global classwise_batch
    solver.G.train()
    solver.C1.train()
    solver.C2.train()
    solver.DP.train()
    # torch.cuda.manual_seed(1)

    batch_idx_g = 0
    print('creating classwise iterator', time.time())
    classwise_dataset_iterator = iter(solver.classwise_dataset)
    print('starting iteration', time.time())
    tt = time.time()
    tot_dataloading_time = 0
    tot_updates_time = 0
    for batch_idx, data in enumerate(solver.datasets):
        batch_idx_g = batch_idx
        img_t = Variable(data['T'].cuda())
        img_s = Variable(data['S'].cuda())
        if (solver.args.clustering_only and cluster_batch is None):
            cluster_batch = img_s
            # data1_batch = Variable(next(iter(solver.dataset_data1))['T'].cuda())
            # data2_batch = Variable(next(iter(solver.dataset_data2))['T'].cuda())

        label_s = Variable(data['S_label'].long().cuda())
        if img_s.size()[0] < solver.batch_size or img_t.size()[0] < solver.batch_size:
            break

        classwise_data = next(classwise_dataset_iterator)
        img_s_cl = Variable(classwise_data['S'].squeeze(0).cuda())
        while (img_s_cl.size()[0] == 1):
            print('Got a batch of 1, trying again')
            classwise_data = next(classwise_dataset_iterator)
            img_s_cl = Variable(classwise_data['S'].cuda())

        # img_s_cl = img_s
        if (solver.args.clustering_only and classwise_batch is None):
            classwise_batch = img_s_cl

        #print('BATCHES DONE!!', time.time() - tt)
        tot_dataloading_time += time.time() - tt
        tt = time.time()

        #        switch_bn(solver.DP,True)
        solver.reset_grad()

        loss_s_c1, loss_s_c2, loss_msda, entropy_loss, kl_loss, aux_loss, domain_prob = solver.loss_soft_all_domain(img_s, img_t,
                                                                                                          label_s,
                                                                                                          epoch,
                                                                                                          img_s_cl)
        if not classifier_disc:
            loss_s_c2 = loss_s_c1
        loss = loss_s_c1 + loss_s_c2 + loss_msda + entropy_loss + kl_loss + aux_loss

        loss.backward()
        clip_value = 1.0

        #torch.nn.utils.clip_grad_norm(solver.G.parameters(), clip_value)
        #torch.nn.utils.clip_grad_norm(solver.C1.parameters(), clip_value)
        #if classifier_disc:
        #    torch.nn.utils.clip_grad_norm(solver.C2.parameters(), clip_value)
        #torch.nn.utils.clip_grad_norm(solver.DP.parameters(), clip_value)

        # solver.opt_g.step()
        if not solver.args.clustering_only:
            solver.opt_g.step()
            solver.opt_c1.step()
            solver.opt_c2.step()
        solver.opt_dp.step()

        #print('GRADIENT UPDATES DONE!!!', time.time() - tt)
        tot_updates_time += time.time() - tt
        tt = time.time()

        #        switch_bn(solver.DP,False)
        #        solver.reset_grad()
        #       _, _, _, _, kl_loss, domain_prob = solver.loss_soft_all_domain(img_s, img_t, label_s, epoch, img_s_cl)

        #        clip_value = 1.0

        #        torch.nn.utils.clip_grad_norm(solver.DP.parameters(), clip_value)
        #        solver.opt_dp.step()
        loss_dis = loss * 0  # For printing purpose, it's reassigned if classifier_disc=True

        if classifier_disc:
            solver.reset_grad()
            loss_s_c1, loss_s_c2, loss_msda, entropy_loss, kl_loss, aux_loss, domain_prob = solver.loss_soft_all_domain(img_s,
                                                                                                              img_t,
                                                                                                              label_s,
                                                                                                              epoch,
                                                                                                              img_s_cl)

            feat_t, conv_feat_t = solver.G(img_t)
            output_t1 = solver.C1(feat_t)
            output_t2 = solver.C2(feat_t)

            loss_s = loss_s_c1 + loss_msda + loss_s_c2 + entropy_loss + kl_loss + aux_loss
            loss_dis = solver.discrepancy(output_t1, output_t2)
            loss = loss_s - loss_dis
            loss.backward()

            #torch.nn.utils.clip_grad_norm(solver.C1.parameters(), clip_value)
            #torch.nn.utils.clip_grad_norm(solver.C2.parameters(), clip_value)
            solver.opt_c1.step()
            solver.opt_c2.step()
            solver.reset_grad()

            for i in range(solver.args.num_k):
                feat_t, conv_feat_t = solver.G(img_t)
                output_t1 = solver.C1(feat_t)
                output_t2 = solver.C2(feat_t)
                loss_dis = solver.discrepancy(output_t1, output_t2)
                loss_dis.backward()
                torch.nn.utils.clip_grad_norm(solver.G.parameters(), clip_value)
                solver.opt_g.step()
                solver.reset_grad()

        # for i in range(img_s.size(0)):
        #   torchvision.utils.save_image(img_s[i, :, :, :], 'source_imgs{}.png'.format(i))
        # for i in range(img_s_cl.size(0)):
        #   torchvision.utils.save_image(img_s_cl[i, :, :, :], 'classwise_imgs{}.png'.format(i))
        if solver.args.clustering_only and epoch % 1 == 0 and batch_idx % 20 == 0:
            solver.G.eval()
            solver.C1.eval()
            solver.C2.eval()
            solver.DP.eval()

            cluster_batchsize = cluster_batch.size()[0]
            classwise_batchsize = classwise_batch.size()[0]
            _, _, _, _, _,_, domain_prob = solver.loss_soft_all_domain(cluster_batch, img_t,
                                                                     label_s[:cluster_batchsize, ], epoch, img_s_cl)
            _, _, _, _, _,_, domain_prob_cw = solver.loss_soft_all_domain(classwise_batch, img_t,
                                                                        label_s[:classwise_batchsize, ], epoch,
                                                                        img_s_cl)

            print(domain_prob)
            print('Classwise Probs', domain_prob_cw.mean(0))
            print('Domain Probs', domain_prob.mean(0))

            directory = "clusters_data-{}-num_domain-{}-batch_size-{}-kl_wt-{}-entropy_wt-{}-lr-{}-seed-{}-target-{}-clustering_only-{}/".format(
                solver.args.data, solver.args.num_domain, solver.args.batch_size, solver.args.kl_wt,
                solver.args.entropy_wt, solver.args.lr, solver.args.seed, solver.args.target,
                solver.args.clustering_only)
            if not os.path.exists(directory):
                os.makedirs(directory)


            torchvision.utils.save_image(img_s[:32, :, :, :],
                                         "{}/source_images_{}_{}.png".format(directory, epoch, batch_idx),
                                         normalize=True)
            torchvision.utils.save_image(img_s_cl[:32, :, :, :],
                                         "{}/classwise_images_{}_{}.png".format(directory, epoch, batch_idx),
                                         normalize=True)

            max_idxs = domain_prob.argmax(1)
            for ii in range(solver.args.num_domain):
                if (max_idxs == ii).any():
                    torchvision.utils.save_image(cluster_batch[max_idxs == ii, :, :, :],
                                                 "{}/source_images_cl{}_{}_{}.png".format(directory, ii, epoch,
                                                                                          batch_idx), normalize=True)
                else:
                    print("No images in Cluster {} _{}_{}".format(ii, epoch, batch_idx))

            #if batch_idx == 0:
            # TODO: Update based on domain_data1 above
            #                _, _, _, _, _, domain_data1 = solver.loss_soft_all_domain(data1_batch, img_t, label_s, epoch, img_s_cl)
            #                print('Domain1 Probs',domain_data1.mean(0))
            #                _, _, _, _, _, domain_data2 = solver.loss_soft_all_domain(data2_batch, img_t, label_s, epoch, img_s_cl)
            #                print('Domain2 Probs',domain_data2.mean(0))

            solver.G.train()
            solver.C1.train()
            solver.C2.train()
            solver.DP.train()
        if batch_idx % solver.interval == 0:
            print \
                    (
                    'Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss1: {:.6f}\t Loss2: {:.6f}\t Loss_mmd: {:.6f}\t Loss_entropy: {:.6f}\t kl_loss: {:.6f} \t aux_loss: {:.6f}'.format(
                        epoch, batch_idx, 100, 100. * batch_idx / 70000, loss_s_c1.data.item(), loss_s_c2.data.item(),
                        loss_msda.data.item(), entropy_loss.data.item(), kl_loss.data.item(), aux_loss.data.item()))
    print('tot_dataloading_time', tot_dataloading_time, 'tot_updates_time', tot_updates_time)
    return batch_idx_g

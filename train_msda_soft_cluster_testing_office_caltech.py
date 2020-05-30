import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision
import os
import numpy as np
import sys

cluster_batch = None
classwise_batch = None

amazon_batch = None
dslr_batch = None
webcam_batch = None
caltech_batch = None


def switch_bn(model, on):
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
            if on:
                m.train()
            else:
                m.eval()


import time


def train_MSDA_soft(solver, epoch, graph_data, classifier_disc=True, record_file=None):
    print('inside function', time.time())
    global cluster_batch
    global amazon_batch
    global dslr_batch
    global webcam_batch
    global caltech_batch
    global classwise_batch
    print('getting train mode', time.time())
    solver.G.train()
    solver.C1.train()
    solver.C2.train()
    solver.DP.train()
    # torch.cuda.manual_seed(1)

    batch_idx_g = 0
    tt = time.time()
    print('creating classwise iterator', time.time())
    solver.classwise_dataset.reset_iter()
    classwise_dataset_iterator = iter(solver.classwise_dataset)

    #     main_dataset_iterator = iter(solver.datasets)
    #     print(sum(1 for _ in main_dataset_iterator))
    main_dataset_iterator = iter(solver.datasets)
    # sys.exit()
    print('starting iteration', time.time())

    tot_dataloading_time = 0
    tot_updates_time = 0
    tot_cuda_time = 0
    tot_classwisedata_time = 0
    tot_main_data_time = 0
    while True:
        try:
            mt0 = time.time()
            data = next(main_dataset_iterator)
            mt = time.time() - mt0
            # print('BATCH Main DATA', mt)
            tot_main_data_time += mt
        except:
            print('End of epoch')
            break
        batch_idx_g += 1
        batch_idx = batch_idx_g
        # print('batch no : ', batch_idx)
        ct1 = time.time()
        img_t = Variable(data['T'].cuda())
        img_s = Variable(data['S'].cuda())
        ct2 = time.time()
        # print('batch size : ', img_s.size()[0])
        #         if(batch_idx > 50):
        #             break
        if solver.args.clustering_only and cluster_batch is None:
            cluster_batch = img_s

            # amazon_batch = Variable(next(iter(solver.dataset_amazon))['T'].cuda())
            # dslr_batch = Variable(next(iter(solver.dataset_dslr))['T'].cuda())
            # webcam_batch = Variable(next(iter(solver.dataset_webcam))['T'].cuda())
            # caltech_batch = Variable(next(iter(solver.dataset_caltech))['T'].cuda())

        ct3 = time.time()
        label_s = Variable(data['S_label'].long().cuda())
        ct4 = time.time()
        if img_s.size()[0] < solver.batch_size or img_t.size()[0] < solver.batch_size:
            print('Breaking because of low batch size')
            break

        classwise_data = next(classwise_dataset_iterator)
        ct5 = time.time()
        img_s_cl = Variable(classwise_data['S'].squeeze(0).float().cuda())
        ct6 = time.time()

        ct = (ct2 - ct1) + (ct4 - ct3) + (ct6 - ct5)
        # print('BATCH CUDA TIME', ct)
        tot_cuda_time += ct
        cl_time = ct5 - ct4
        tot_classwisedata_time += cl_time
        # print('CLASSWISE DATA TIME', cl_time)
        if (img_s_cl.size()[0] <= 1):
            print('CLASS WISE is of size 1. Looping')
            break
            classwise_data = next(classwise_dataset_iterator)
            img_s_cl = Variable(classwise_data['S'].cuda())

        # img_s_cl = img_s

        if (solver.args.clustering_only and classwise_batch is None):
            classwise_batch = img_s_cl

        # print('BATCHES DONE!!', time.time()-tt)
        tot_dataloading_time += time.time() - tt
        tt = time.time()

        #        switch_bn(solver.DP,True)
        solver.reset_grad()
        loss_s_c1, loss_s_c2, loss_msda, entropy_loss, kl_loss, domain_prob = solver.loss_soft_all_domain(img_s, img_t,
                                                                                                          label_s,
                                                                                                          epoch,
                                                                                                          img_s_cl)
        if not classifier_disc:
            loss_s_c2 = loss_s_c1
        #         if(batch_idx % 6 < 3):
        #             loss = loss_s_c1 + loss_s_c2 + loss_msda + 1.1*kl_loss
        #         else:
        #             loss = loss_s_c1 + loss_s_c2 + loss_msda + entropy_loss
        if (epoch % 4 == 1):
            loss = loss_s_c1 + loss_s_c2 + entropy_loss + 0.2 * loss_msda
        else:
            loss = loss_s_c1 + loss_s_c2 + loss_msda + kl_loss

        graph_data['entropy'].append(entropy_loss.data.item())
        graph_data['kl'].append(kl_loss.data.item())
        graph_data['c1'].append(loss_s_c1.data.item())
        graph_data['c2'].append(loss_s_c2.data.item())
        graph_data['msda'].append(loss_msda.data.item())
        graph_data['h'].append(entropy_loss.data.item() + kl_loss.data.item())
        graph_data['total'].append(
            entropy_loss.data.item() + kl_loss.data.item() + loss_s_c1.data.item() + loss_msda.data.item())

        loss.backward()
        clip_value = 1.0

        #         torch.nn.utils.clip_grad_norm(solver.G.parameters(), clip_value)
        #         torch.nn.utils.clip_grad_norm(solver.C1.parameters(), clip_value)
        if classifier_disc:
            torch.nn.utils.clip_grad_norm(solver.C2.parameters(), clip_value)
        #         torch.nn.utils.clip_grad_norm(solver.DP.parameters(), clip_value)

        # solver.opt_g.step()
        if not solver.args.clustering_only:
            solver.opt_g.step()
            solver.opt_c1.step()
            solver.opt_c2.step()
        solver.opt_dp.step()

        # print('GRADIENT UPDATES DONE!!!', time.time()-tt)
        tot_updates_time += time.time() - tt
        tt = time.time()

        if classifier_disc:
            solver.reset_grad()
            loss_s_c1, loss_s_c2, loss_msda, entropy_loss, kl_loss, domain_prob = solver.loss_soft_all_domain(img_s,
                                                                                                              img_t,
                                                                                                              label_s,
                                                                                                              epoch,
                                                                                                              img_s_cl)

            feat_t, conv_feat_t = solver.G(img_t)
            output_t1 = solver.C1(feat_t)
            output_t2 = solver.C2(feat_t)

            loss_s = loss_s_c1 + loss_msda + loss_s_c2 + entropy_loss + kl_loss
            loss_dis = solver.discrepancy(output_t1, output_t2)
            loss = loss_s - loss_dis
            loss.backward()

            torch.nn.utils.clip_grad_norm(solver.C1.parameters(), clip_value)
            torch.nn.utils.clip_grad_norm(solver.C2.parameters(), clip_value)
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

        if batch_idx % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss1: {:.6f}\t Loss2: {:.6f}\t'
                  ' Loss_mmd: {:.6f}\t Loss_entropy: {:.6f}\t kl_loss: {:.6f}\t Combined Entropy: {:.6f}'.format(
                        epoch, batch_idx, 100, 100. * batch_idx / 70000, loss_s_c1.data.item(), loss_s_c2.data.item(),
                        loss_msda.data.item(), entropy_loss.data.item(), kl_loss.data.item(),
                        entropy_loss.data.item() + kl_loss.data.item()))

    print('tot_dataloading_time', tot_dataloading_time, 'tot_updates_time', tot_updates_time)
    print('CUDA Time', tot_cuda_time)
    print('CLDL Total Time', tot_classwisedata_time)
    print('MDL Total Time', tot_main_data_time)
    return batch_idx_g, graph_data

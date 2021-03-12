import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision
import os
import numpy as np
import sys

cluster_batch = None
classwise_batch = None

art_painting_batch = None
cartoon_batch = None
photo_batch = None
sketch_batch = None


def switch_bn(model, on):
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
            if on:
                m.train()
            else:
                m.eval()


import time


def train_MSDA_soft(solver, epoch, graph_data, classifier_disc=True, record_file=None, max_it=10000, prev_count=0):
    #print('inside function', time.time())
    global cluster_batch
    global art_painting_batch
    global cartoon_batch
    global photo_batch
    global sketch_batch
    global classwise_batch

    #print('getting train mode', time.time())
    solver.G.train()
    solver.C1.train()
    solver.C2.train()
    if solver.args.pretrained_clustering=="yes":
        solver.DP.eval()
    else:
        solver.DP.train()
    # torch.cuda.manual_seed(1)

    batch_idx_g = 0
    tt = time.time()
    #print('creating classwise iterator', time.time())
    solver.classwise_dataset.reset_iter()
    classwise_dataset_iterator = iter(solver.classwise_dataset)

    #     main_dataset_iterator = iter(solver.datasets)
    #     print(sum(1 for _ in main_dataset_iterator))
    main_dataset_iterator = iter(solver.datasets)
    # sys.exit()
    #print('starting iteration', time.time())

    tot_dataloading_time = 0
    tot_updates_time = 0
    tot_cuda_time = 0
    tot_classwisedata_time = 0
    tot_main_data_time = 0
    tot_clustering_loss = 0
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
        if (batch_idx_g>max_it): 
            break
        batch_idx = batch_idx_g
        # print('batch no : ', batch_idx)
        ct1 = time.time()
        img_t = Variable(data['T'].cuda())
        img_s = Variable(data['S'].cuda())
        img_s_dl = Variable(data['SD_label'].long().cuda())
        ct2 = time.time()


        ct3 = time.time()
        label_s = Variable(data['S_label'].long().cuda())
        label_t = Variable(data['T_label'].long().cuda())
        ct4 = time.time()
        if img_s.size()[0] < solver.batch_size or img_t.size()[0] < solver.batch_size:
            #print('Breaking because of low batch size')
            break
        ct5 = time.time()
        if not solver.args.pretrained_clustering:
            classwise_data = next(classwise_dataset_iterator)
            img_s_cl = Variable(classwise_data['S'].squeeze(0).float().cuda())
            if (img_s_cl.size()[0] <= 1):
                # print('CLASS WISE is of size 1. Looping')
                break
        else:
            img_s_cl = None
        ct6 = time.time()

        ct = (ct2 - ct1) + (ct4 - ct3) + (ct6 - ct5)
        # print('BATCH CUDA TIME', ct)
        tot_cuda_time += ct
        cl_time = ct5 - ct4
        tot_classwisedata_time += cl_time
        # print('CLASSWISE DATA TIME', cl_time)
        # print('BATCHES DONE!!', time.time()-tt)
        tot_dataloading_time += time.time() - tt
        tt = time.time()

        #        switch_bn(solver.DP,True)
        solver.reset_grad(prev_count+batch_idx)
        start = time.time()
        loss_s_c1, loss_s_c2, loss_msda, entropy_loss, kl_loss, domain_prob = solver.loss_soft_all_domain(img_s, img_t,
                                                                                                          label_s, 
                                                                                                          epoch,
                                                                                                          img_s_cl, img_s_dl)
        end = time.time()
        #print("Time taken in training batch : ", end-start)
        if not classifier_disc:
            loss_s_c2 = loss_s_c1
        if solver.args.pretrained_clustering=="yes":
            loss = loss_s_c1 + loss_s_c2 + loss_msda
        elif solver.args.pretrained_source=="yes":
            loss = entropy_loss + kl_loss
        else:
            loss = loss_s_c1 + loss_s_c2 + entropy_loss + loss_msda + kl_loss
        tot_clustering_loss += entropy_loss.data.item() + kl_loss.data.item()
        graph_data['entropy'].append(entropy_loss.data.item())
        graph_data['kl'].append(kl_loss.data.item())
        graph_data['c1'].append(loss_s_c1.data.item())
        graph_data['c2'].append(loss_s_c2.data.item())
        graph_data['msda'].append(loss_msda.data.item())
        graph_data['h'].append(entropy_loss.data.item() + kl_loss.data.item())
        graph_data['total'].append(
            entropy_loss.data.item() + kl_loss.data.item() + loss_s_c1.data.item() + loss_msda.data.item())

        loss.backward()
        if not solver.args.clustering_only and not solver.args.pretrained_source=="yes":
            solver.opt_g.step()
            solver.opt_c1.step()
            solver.opt_c2.step()
        if not solver.args.pretrained_clustering=="yes":
            solver.opt_dp.step()

        # print('GRADIENT UPDATES DONE!!!', time.time()-tt)
        tot_updates_time += time.time() - tt
        tt = time.time()


        if batch_idx % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss1: {:.6f}\t Loss2: {:.6f}\t'
                  ' Loss_mmd: {:.6f}\t Loss_entropy: {:.6f}\t kl_loss: {:.6f}\t Combined Entropy: {:.6f}'.format(
                        epoch, batch_idx, 100, 100. * batch_idx / 70000, loss_s_c1.data.item(), loss_s_c2.data.item(),
                        loss_msda.data.item(), entropy_loss.data.item(), kl_loss.data.item(),
                        entropy_loss.data.item() + kl_loss.data.item()))
    tot_clustering_loss = tot_clustering_loss/(batch_idx_g + 1e-8)
    return batch_idx_g, graph_data, tot_clustering_loss

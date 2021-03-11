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


def train_MSDA_classwise(solver, epoch, graph_data, classifier_disc=True, record_file=None, max_it=10000, single_domain_mode=False, prev_count=0):
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
    #solver.classwise_dataset.reset_iter()
    #classwise_dataset_iterator = iter(solver.classwise_dataset)

    #     main_dataset_iterator = iter(solver.datasets)
    #     print(sum(1 for _ in main_dataset_iterator))
    #main_dataset_iterator = iter(solver.datasets)
    # sys.exit()
    #print('starting iteration', time.time())

    tot_dataloading_time = 0
    tot_updates_time = 0
    tot_cuda_time = 0
    tot_classwisedata_time = 0
    tot_main_data_time = 0
    for batch_idx, data in enumerate(solver.datasets):
        batch_idx_g = batch_idx
        if (batch_idx>max_it): 
            break
        batch_idx = batch_idx_g
        # print('batch no : ', batch_idx)
        ct1 = time.time()
        img_t = Variable(data['T'].cuda())
        idx_t = Variable(data['T_idx'].cuda())
        img_s = Variable(data['S'].cuda())
        img_s_dl = Variable(data['SD_label'].long().cuda())
        ct2 = time.time()
        # print('batch size : ', img_s.size()[0])
        #         if(batch_idx > 50):
        #             break
        if solver.args.clustering_only and cluster_batch is None:
            cluster_batch = img_s


        ct3 = time.time()
        label_s = Variable(data['S_label'].long().cuda())
        label_t = Variable(data['T_label'].long().cuda())
        ct4 = time.time()
        if img_s.size()[0] < solver.batch_size or img_t.size()[0] < solver.batch_size:
            #print('Breaking because of low batch size')
            break


        ct = (ct2 - ct1) + (ct4 - ct3)
        # print('BATCH CUDA TIME', ct)
        tot_cuda_time += ct
        #cl_time = ct5 - ct4
        #tot_classwisedata_time += cl_time
        # print('CLASSWISE DATA TIME', cl_time)
        #if (img_s_cl.size()[0] <= 1):
            #print('CLASS WISE is of size 1. Looping')
        #    break
        #    classwise_data = next(classwise_dataset_iterator)
        #    img_s_cl = Variable(classwise_data['S'].cuda())

        # img_s_cl = img_s

        #if solver.args.clustering_only and classwise_batch is None:
        #    classwise_batch = img_s_cl

        # print('BATCHES DONE!!', time.time()-tt)
        tot_dataloading_time += time.time() - tt
        tt = time.time()

        #        switch_bn(solver.DP,True)
        solver.reset_grad(prev_count+batch_idx)
        start = time.time()
        loss_s_c1, loss_s_c2, intra_domain_mmd_loss, inter_domain_mmd_loss, entropy_loss, kl_loss, class_tear_apart_loss = solver.loss_domain_class_mmd(img_s, img_t, label_s, label_t, epoch, None, img_s_dl,idx_t, single_domain_mode=single_domain_mode) 
        end = time.time()
        class_tear_apart_loss = class_tear_apart_loss*solver.args.class_tear_apart_wt
        loss_msda = intra_domain_mmd_loss + inter_domain_mmd_loss + class_tear_apart_loss
        #print("Time taken in training batch : ", end-start)
        if not classifier_disc:
            loss_s_c2 = loss_s_c1
        if solver.args.pretrained_clustering=="yes":
            loss = loss_s_c1 + loss_s_c2 + loss_msda
        elif solver.args.pretrained_source=="yes":
            loss = entropy_loss + kl_loss
        else:
            if 0:#solver.args.load_ckpt!='' and epoch < 20:
                loss = entropy_loss + kl_loss
            else:
                loss = loss_s_c1 + loss_s_c2 + entropy_loss + loss_msda + kl_loss

        graph_data['entropy'].append(entropy_loss.data.item())
        graph_data['kl'].append(kl_loss.data.item())
        graph_data['c1'].append(loss_s_c1.data.item())
        graph_data['c2'].append(loss_s_c2.data.item())
        #graph_data['inter_domain_mmd_loss'].append(inter_domain_mmd_loss.data.item())
        #graph_data['intra_domain_mmd_loss'].append(intra_domain_mmd_loss.data.item())
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
                  ' Loss_mmd_inter: {:.6f}\t Loss_mmd_intra: {:.6f}\t Loss_entropy: {:.6f}\t class_tear_apart_loss: {:.6f}\t kl_loss: {:.6f}\t Combined Entropy: {:.6f}'.format(
                        epoch, batch_idx, 100, 100. * batch_idx / 70000, loss_s_c1.data.item(), loss_s_c2.data.item(),
                        inter_domain_mmd_loss.data.item(), intra_domain_mmd_loss.data.item(),entropy_loss.data.item(), class_tear_apart_loss.data.item(), kl_loss.data.item(),
                        entropy_loss.data.item() + kl_loss.data.item()))
    
    print('End of epoch')
    return batch_idx_g, graph_data

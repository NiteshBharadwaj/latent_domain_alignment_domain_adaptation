import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import time

def generate_pseudo(solver, model_G, model_C1, dataset, split="train", reject_quantile=0.25, logits_criteria=True):
    model_G.eval()
    model_C1.eval()
    test_loss = 0
    correct1 = 0
    size = 0
    feature_all = np.array([])
    label_all = []

    classwise_acc = [0]*solver.num_classes
    classwise_sum = [0]*solver.num_classes
    logits_list = []
    labels_list = []
    dset_size = len(dataset.data_loader_t.dataset)
    logits_map = torch.zeros((dset_size,solver.num_classes),dtype=torch.float32)
    softmax = torch.nn.Softmax(dim=1)
    total_batches = len(dataset.data_loader_t)
    with torch.no_grad():
        for batch_idx, data in enumerate(dataset.data_loader_t):
            start = time.time()
            img, label, _, idxes = data

            img, label,idxes = img.cuda(), label.long().cuda(), idxes.long().cuda()
            img, label = Variable(img, volatile=True), Variable(label)
            feat, _, _ = model_G(img)

            # print('feature.shape:{}'.format(feat.shape))

            if batch_idx == 0:
                label_all = label.data.cpu().numpy().tolist()

                # feature_all = feat.data.cpu().numpy()
            else:
                # feature_all = np.ma.row_stack((feature_all, feat.data.cpu().numpy()))
                # feature_all = feature_all.data
                label_all = label_all + label.data.cpu().numpy().tolist()

            # print(feat.shape)

            output1 = solver.C1(feat)
            logits_list.append(output1)
            labels_list.append(label)

            logits_map[idxes] = output1.cpu().detach()

            test_loss += nn.CrossEntropyLoss()(output1, label).data.item()
            pred1 = output1.data.max(1)[1]
            k = label.data.size()[0]
            correct1 += pred1.eq(label.data).cpu().sum()
            size += k
            for class_id in range(solver.num_classes):
                idxes = label==class_id
                classwise_acc[class_id] += pred1[idxes].eq(label[idxes].data).cpu().sum()
                classwise_sum[class_id] += (idxes*1==1).sum()
            end = time.time()
            if batch_idx%10==0:
                print("{}/{} done".format(batch_idx, total_batches))
            #print("Time taken for testing batch : ", end-start)
    # np.savez('result_plot_sv_t', feature_all, label_all )
    test_loss = test_loss / (size + 1e-6)

    print('\n{} set: Average loss: {:.4f}, Accuracy C1: {}/{} ({:.06f}%)  \n'.format(split,test_loss, correct1, size,
                                                                                           100. * correct1 / (
                                                                                                       size + 1e-6)))
    for class_id in range(solver.num_classes):
        class_acc = 100. * classwise_acc[class_id]/(classwise_sum[class_id]+1e-6)
        print('{} set: Average Accuracy Class Id {}: {}'.format(split,class_id,class_acc))

    probits_map = softmax(solver.T_scaling(logits_map, cpu=True)).detach()
    if logits_criteria:
        logits_map_max,_ = logits_map.max(dim=1)
        _, sorted_indices = torch.sort(logits_map_max,dim=0)
        reject_indices = sorted_indices[:int((sorted_indices.shape[0])*reject_quantile)]
    else:
        probits_map_max,_ = probits_map.max(dim=1)
        _, sorted_indices = torch.sort(probits_map_max, dim=0)
        reject_indices = sorted_indices[:int((sorted_indices.shape[0]) * reject_quantile)]
    accept_mask = torch.ones((probits_map.shape[0]),dtype=torch.int32)
    accept_mask[reject_indices] = 0
    accept_mask = accept_mask==1
    return probits_map, accept_mask


def generate_empty_pseudo(solver, dataset):
    dset_size = len(dataset.data_loader_t.dataset)
    logits_map = torch.zeros((dset_size,solver.num_classes),dtype=torch.float32)
    probits_map = logits_map
    reject_indices = torch.arange(0,probits_map.shape[0], dtype=torch.int32).long()
    accept_mask = torch.ones((probits_map.shape[0]),dtype=torch.int32)
    accept_mask[reject_indices] = 0
    accept_mask = accept_mask==1
    return probits_map, accept_mask

def get_one_hot(labels, num_classes):
    y = torch.eye(num_classes)
    return y[labels]

def generate_perfect_pseudo(solver, dataset):
    data_label_symbol = 'T_label'
    dset_size = len(dataset.data_loader_t.dataset)
    probits_map = torch.zeros((dset_size,solver.num_classes),dtype=torch.float32)
    with torch.no_grad():
        for batch_idx, data in enumerate(dataset.data_loader_t): 
            _, label, _, idxes = data
            probits_map[idxes] = get_one_hot(label.long(), solver.num_classes)

    accept_mask = torch.ones((probits_map.shape[0]),dtype=torch.int32)
    accept_mask = accept_mask==1
    return probits_map, accept_mask



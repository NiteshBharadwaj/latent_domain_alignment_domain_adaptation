import torch
import torch.nn as nn
import torchvision
from torch.autograd import Variable
import sys
import csv
import os
from collections import Counter

def view_clusters(solver,clusters_file_class,probs_csv_class,epoch):
    if(solver.dl_type != 'soft_cluster'):
        print('no clusters for dl type : ', solver.dl_type)
        return
    solver.G.eval()
    solver.C1.eval()
    solver.C2.eval()
    solver.DP.eval()
    #torch.cuda.manual_seed(1)

    batch_idx_g = 0
    prev = solver.batch_size
    arrayOfClusterstorchClass = []
    arrayOfClustersboolClass = []
    arrayOfClustersprobsClass = []
    arrayOfClustersDomainsClass = []
    for c in range(solver.num_classes):
        arrayOfClusterstorch = []
        for i in range(solver.num_domains):
            arrayOfClusterstorch.append(0)
        arrayOfClustersbool = []
        for i in range(solver.num_domains):
            arrayOfClustersbool.append(False)
        arrayOfClustersprobs = []
        for i in range(solver.num_domains):
            arrayOfClustersprobs.append(0)
        arrayOfClustersDomains = []
        for i in range(solver.num_domains):
            arrayOfClustersDomains.append(0)
        arrayOfClusterstorchClass.append(arrayOfClusterstorch)
        arrayOfClustersboolClass.append(arrayOfClustersbool)
        arrayOfClustersprobsClass.append(arrayOfClustersprobs)
        arrayOfClustersDomainsClass.append(arrayOfClustersDomains)

    
    for batch_idx, data in enumerate(solver.datasets):
        if(batch_idx > 14):
            break
        batch_idx_g = batch_idx
        img_t = data['T'].cuda()
        img_s = data['S'].cuda()
        label_s = data['S_label'].long().cuda()
        actual_domain_s = data['SD_label'].long()

#         if img_s.size()[0] < solver.batch_size or img_t.size()[0] < solver.batch_size:
#             break
        if(img_s.size()[0] > prev):
            break
        prev = img_s.size()[0]
        #solver.reset_grad()

        loss_s_c1, loss_s_c2, intra_domain_mmd_loss, inter_domain_mmd_loss, entropy_loss, kl_loss, domain_prob = solver.loss_domain_class_mmd(img_s, img_t, label_s, epoch, img_s, actual_domain_s, single_domain_mode=False)
        domains_max = domain_prob.data.max(2)
        best_domain_probs = domains_max[0]
        best_domains = domains_max[1]
        for cur_class in range(solver.num_classes):
            cur_class_idxes = label_s == cur_class
            for i in range(solver.num_domains):
                i_index = ((best_domains[cur_class_idxes][cur_class] == i).nonzero()).squeeze()
                #print(i_index)
                img_s_i = img_s[cur_class_idxes][i_index,:,:,:]
                cur_probs = best_domain_probs[cur_class_idxes][cur_class][i_index]
                cur_real_domains = actual_domain_s[cur_class_idxes][i_index]
                if(img_s_i.size()[0] > 0):
                    try:
                        a = img_s_i.size()[3]
                    except:
                        img_s_i = torch.unsqueeze(img_s_i, 0)
                        cur_probs = torch.unsqueeze(cur_probs, 0)
                        cur_real_domains = torch.unsqueeze(cur_real_domains, 0)
                    if(arrayOfClustersboolClass[cur_class][i] == False):
                        arrayOfClusterstorchClass[cur_class][i] = img_s_i
                        arrayOfClustersboolClass[cur_class][i] = True
                        arrayOfClustersprobsClass[cur_class][i] = cur_probs
                        arrayOfClustersDomainsClass[cur_class][i] = cur_real_domains
                    else:
                        #print(arrayOfClusterstorch[i].size())
                        arrayOfClusterstorchClass[cur_class][i] = torch.cat((arrayOfClusterstorchClass[cur_class][i], img_s_i), 0)
                        arrayOfClustersprobsClass[cur_class][i] = torch.cat((arrayOfClustersprobsClass[cur_class][i], cur_probs), 0)
                        arrayOfClustersDomainsClass[cur_class][i] = torch.cat((arrayOfClustersDomainsClass[cur_class][i], cur_real_domains), 0)

    topk = 10
    arrayOfProbsClass = []
    for c in range(solver.num_classes):
        probs_csv = probs_csv_class[c]
        arrayOfProbsClass.append([])
        clusters_file = clusters_file_class[c]
        arrayOfProbs = arrayOfProbsClass[c]
        for i in range(solver.num_domains):
            arrayOfProbs.append([])
            try:
                os.remove(clusters_file[i])
                os.remove(clusters_file[i][:-4] + '_probs_descending.png')
            except:
                pass
            if(arrayOfClustersboolClass[c][i] == True):
                print("cluster no : ", str(i))
                arrayOfClustersprobsClass[c][i] = arrayOfClustersprobsClass[c][i].data.cpu().numpy()
    #             print(arrayOfClustersprobs[i])
    #             print(arrayOfClustersprobs[i].shape)
                maxProbIndices = arrayOfClustersprobsClass[c][i].argsort()[-min(arrayOfClustersprobsClass[c][i].shape[0],topk):][::-1]
                maxProbs = arrayOfClustersprobsClass[c][i][maxProbIndices].tolist()
                #maxProbIndices = torch.from_numpy(maxProbIndices.copy()).long().cuda()
                arrayOfProbs.append(maxProbs)
                #print(arrayOfClusterstorch[i].size())
                topImages = arrayOfClusterstorchClass[c][i][maxProbIndices.copy(),:,:,:]
                #torchvision.utils.save_image(arrayOfClusterstorch[i], clusters_file[i],nrow=7)
                torchvision.utils.save_image(topImages, clusters_file[i][:-4] + '_probs_descending_'+str(epoch)+'.png',nrow=2, normalize=True)
                print('Total Num images in this cluster : ', arrayOfClusterstorchClass[c][i].size()[0])
                cc = Counter(arrayOfClustersDomainsClass[c][i].cpu().tolist())
                for key in cc:
                    print('Num images of domain', key, '----', cc[key])
                print('bestProbs in this cluster : ', maxProbs[:min(topk,5)])
        try:
            os.remove(probs_csv)
        except:
            pass
        with open(probs_csv,'w') as myfile:
            wr = csv.writer(myfile)
            wr.writerows(arrayOfProbs)
            


import torch
import torch.nn as nn
import torchvision
from torch.autograd import Variable
import sys
import csv
import os

def view_clusters(solver,clusters_file,probs_csv):
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
    arrayOfClusterstorch = []
    for i in range(solver.num_domains):
        arrayOfClusterstorch.append(0)
    arrayOfClustersbool = []
    for i in range(solver.num_domains):
        arrayOfClustersbool.append(False)
    arrayOfClustersprobs = []
    for i in range(solver.num_domains):
        arrayOfClustersprobs.append(0)
    
    
    for batch_idx, data in enumerate(solver.datasets):
        batch_idx_g = batch_idx
        img_t = data['T'].cuda()
        img_s = data['S'].cuda()
        label_s = data['S_label'].long().cuda()
#         if img_s.size()[0] < solver.batch_size or img_t.size()[0] < solver.batch_size:
#             break
        if(img_s.size()[0] > prev):
            break
        prev = img_s.size()[0]
        #solver.reset_grad()

        loss_s_c1, loss_s_c2, loss_msda, entropy_loss, kl_loss, domain_prob = solver.loss_soft_all_domain(img_s, img_t, label_s, 0)
#         print(domain_prob.size())
#         print(domain_prob[0])
        domains_max = domain_prob.data.max(1)
#         print(domains_max[0][0])
#         print(domains_max[1][0])
#         print(domains_max)
#         sys.exit()
        
        best_domain_probs = domains_max[0]
        best_domains = domains_max[1]
        
        for i in range(solver.num_domains):
            i_index = ((best_domains == i).nonzero()).squeeze()
            #print(i_index)
            img_s_i = img_s[i_index,:,:,:]
            cur_probs = best_domain_probs[i_index]
            if(img_s_i.size()[0] > 0):
                try:
                    a = img_s_i.size()[3]
                except:
                    img_s_i = torch.unsqueeze(img_s_i, 0)
                    cur_probs = torch.unsqueeze(cur_probs, 0)
                if(arrayOfClustersbool[i] == False):
                    arrayOfClusterstorch[i] = img_s_i
                    arrayOfClustersbool[i] = True
                    arrayOfClustersprobs[i] = cur_probs
                else:
                    #print(arrayOfClusterstorch[i].size())
                    arrayOfClusterstorch[i] = torch.cat((arrayOfClusterstorch[i], img_s_i), 0)
                    arrayOfClustersprobs[i] = torch.cat((arrayOfClustersprobs[i], cur_probs), 0)
                    
    topk = 50
    arrayOfProbs = []
    for i in range(solver.num_domains):
        arrayOfProbs.append([])
        try:
            os.remove(clusters_file[i])
            os.remove(clusters_file[i][:-4] + '_probs_descending.png')
        except:
            pass
        if(arrayOfClustersbool[i] == True):
            print("cluster no : ", str(i))
            arrayOfClustersprobs[i] = arrayOfClustersprobs[i].data.cpu().numpy()
#             print(arrayOfClustersprobs[i])
#             print(arrayOfClustersprobs[i].shape)
            maxProbIndices = arrayOfClustersprobs[i].argsort()[-min(arrayOfClustersprobs[i].shape[0],topk):][::-1]
            maxProbs = arrayOfClustersprobs[i][maxProbIndices].tolist()
            #maxProbIndices = torch.from_numpy(maxProbIndices.copy()).long().cuda()
            arrayOfProbs.append(maxProbs)
            #print(arrayOfClusterstorch[i].size())
            topImages = arrayOfClusterstorch[i][maxProbIndices.copy(),:,:,:]
            torchvision.utils.save_image(arrayOfClusterstorch[i], clusters_file[i],nrow=7)
            torchvision.utils.save_image(topImages, clusters_file[i][:-4] + '_probs_descending.png',nrow=7)
            print('images in this cluster : ', arrayOfClusterstorch[i].size()[0])
            print('bestProbs in this cluster : ', maxProbs[:min(topk,2)])
    try:
        os.remove(probs_csv)
    except:
        pass
    with open(probs_csv,'w') as myfile:
        wr = csv.writer(myfile)
        wr.writerows(arrayOfProbs)
            
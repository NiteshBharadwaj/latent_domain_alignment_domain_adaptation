import torch
import torch.nn as nn
import torchvision
from torch.autograd import Variable

def view_clusters(solver,clusters_file):
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
        best_domains = domain_prob.data.max(1)[1]
        for i in range(solver.num_domains):
            i_index = ((best_domains == i).nonzero()).squeeze()
            img_s_i = img_s[i_index,:,:,:]
            if(img_s_i.size()[0] > 0):
                try:
                    a = img_s_i.size()[3]
                except:
                    img_s_i = torch.unsqueeze(img_s_i, 0)
                if(arrayOfClustersbool[i] == False):
                    arrayOfClusterstorch[i] = img_s_i
                    arrayOfClustersbool[i] = True
                else:
                    #print(arrayOfClusterstorch[i].size())
                    arrayOfClusterstorch[i] = torch.cat((arrayOfClusterstorch[i], img_s_i), 0)
    for i in range(solver.num_domains):
        if(arrayOfClustersbool[i] == True):
            torchvision.utils.save_image(arrayOfClusterstorch[i], clusters_file[i],nrow=7)
            print('clustering images saved in : ', clusters_file[i])
            print('total images in this cluster :  ', arrayOfClusterstorch[i].size()[0])
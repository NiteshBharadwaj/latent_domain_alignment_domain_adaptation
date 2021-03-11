import torch
import torch.nn as nn
import torchvision
from torch.autograd import Variable
import sys
import csv
import os


def view_clusters(solver, clusters_file, probs_csv):
    if (solver.dl_type != 'soft_cluster'):
        print('no clusters for dl type : ', solver.dl_type)
        return
    solver.G.eval()
    solver.C1.eval()
    solver.C2.eval()
    solver.DP.eval()

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

    interval = 10
    cluster_viz_res = {}
    max_bucket_size = 25
    for cluster_id in range(solver.num_domains):
        cluster_viz_res[cluster_id] = []
        for i in range(0, 101, interval):
            cluster_viz_res[cluster_id].append([])
    classwise_dataset_iterator = iter(solver.classwise_dataset)
    with torch.no_grad(): 
        for batch_idx, data in enumerate(solver.datasets):
            batch_idx_g = batch_idx
            img_t = data['T'].cuda()
            img_s = data['S'].cuda()
            label_s = data['S_label'].long().cuda()
            if img_s.size()[0] < solver.batch_size or img_t.size()[0] < solver.batch_size:
                continue
            if (img_s.size()[0] > prev):
                break
            prev = img_s.size()[0]
            # solver.reset_grad()
            try:
                classwise_data = next(classwise_dataset_iterator)
            except:
                break
            img_s_cl = Variable(classwise_data['S'].squeeze(0).cuda())
            if img_s_cl.shape[0] < 2:
                break
            loss_s_c1, loss_s_c2, loss_msda, entropy_loss, kl_loss, domain_prob = solver.loss_soft_all_domain(img_s, img_s,
                                                                                                              label_s, 0,
                                                                                                              img_s, label_s*0)
            bucket_ids = (domain_prob * 100).int() // len(cluster_viz_res[0])
            for cluster_id in range(solver.num_domains):
                clbucket_ids = bucket_ids[:, cluster_id]
                for myidx in range(clbucket_ids.shape[0]):
                    bucket_id = clbucket_ids[myidx]
                    if len(cluster_viz_res[cluster_id][bucket_id]) > max_bucket_size:
                        continue
                    else:
                        cluster_viz_res[cluster_id][bucket_id].append(img_s[myidx].unsqueeze(0).detach().cpu())
            
            if batch_idx > 500:
                break

    import os
    
    for cluster_id in range(solver.num_domains):
        clusters = cluster_viz_res[cluster_id]
        for bucket_id in range(len(clusters)):
            img_list = cluster_viz_res[cluster_id][bucket_id]
            if len(img_list) > 0:
                img_list = torch.cat(img_list)
                if not os.path.exists("clus/clus_{}".format(cluster_id)):
                    os.makedirs("clus/clus_{}".format(cluster_id))
                torchvision.utils.save_image(img_list, "clus/clus_{}/bucket_{}.png".format(cluster_id, bucket_id),
                                             normalize=True)

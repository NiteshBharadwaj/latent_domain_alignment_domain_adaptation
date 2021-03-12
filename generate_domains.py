import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import time

def worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)

def generate_domains(solver, model_G, model_DP, dataset, split="train"):
    model_G.eval()
    model_DP.eval()
    dataset_s = dataset.data_loader_s.dataset
    dset_size = len(dataset_s)
    is_classwise = solver.is_classwise
    is_classaware = solver.classaware_dp
    dataloader = torch.utils.data.DataLoader(dataset_s, batch_size=solver.args.batch_size,
                                                    num_workers=solver.args.num_workds, worker_init_fn=worker_init_fn,
                                                    pin_memory=True)
    if is_classwise:
        probs_map = torch.zeros((dset_size,solver.num_classes,solver.num_domains),dtype=torch.float32)
    else:
        probs_map = torch.zeros((dset_size, solver.num_classes), dtype=torch.float32)
    total_batches = len(dataset.data_loader_t)
    with torch.no_grad():
        for batch_idx, data in enumerate(dataloader):
            img, label, _, idxes = data

            img, label,idxes = img.cuda(), label.long().cuda(), idxes.long().cuda()
            img, label = Variable(img, volatile=True), Variable(label)
            if is_classaware:
                feat, _, _ = model_G(img)
                _, domain_prob = solver.DP(feat.clone().detach())
            else:
                domain_logits, _ = solver.DP(img)
                _, domain_prob = solver.entropy_loss(domain_logits)
            domain_prob = domain_prob.clone().detach().cpu()
            if is_classwise:
                domain_prob = domain_prob.reshape(domain_prob.shape[0], solver.num_classes,
                                                            solver.num_domains)
            probs_map[idxes] = domain_prob
            if batch_idx%10==0:
                print("{}/{} done".format(batch_idx, total_batches))

    return probs_map




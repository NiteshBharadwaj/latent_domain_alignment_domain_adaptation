import torch
from itertools import combinations


def euclidean(x1, x2):
    return torch.sqrt(((x1 - x2) ** 2).sum() + 1e-8)


def k_moment(source_output, target_output, k):
    num_sources = len(source_output)
    source_output_ = []
    for i in range(num_sources):
        source_output_.append((source_output[i] ** k).mean(0))
    target_output = (target_output ** k).mean(0)

    kth_moment = 0
    for i in range(num_sources):
        kth_moment += euclidean(source_output_[i], target_output)

    comb = list(combinations(range(num_sources), 2))

    for k in range(len(comb)):
        kth_moment += euclidean(source_output_[comb[k][0]], source_output_[comb[k][1]])

    return kth_moment


def moment_soft(output_s, output_t, class_prob_s, class_prob_t):
    output_s = output_s.reshape(output_s.shape[0], output_s.shape[1],1)
    class_prob_s = class_prob_s.reshape(class_prob_s.shape[0], 1, class_prob_s.shape[1]) # SHAPE -> N x 1 x c
    output_s_times_cp = torch.matmul(output_s, class_prob_s) #SHAPE -> N x e x c
    class_prob_sum_s = class_prob_s.sum(0) + 1e-6
    class_prob_sum_s = class_prob_sum_s.reshape(1, -1) #SHAPE -> 1 x c
    output_prob_s = (output_s_times_cp.sum(0)/class_prob_sum_s) #SHAPE -> e x c

    output_t = output_t.reshape(output_t.shape[0], output_t.shape[1],1)
    class_prob_t = class_prob_t.reshape(class_prob_t.shape[0], 1, class_prob_t.shape[1]) # SHAPE -> N x 1 x c
    output_t_times_cp = torch.matmul(output_t, class_prob_t) #SHAPE -> N x e x c
    class_prob_sum_t = class_prob_t.sum(0) + 1e-6
    class_prob_sum_t = class_prob_sum_t.reshape(1, -1) #SHAPE -> 1 x c
    output_prob_t = (output_t_times_cp.sum(0)/class_prob_sum_t) #SHAPE -> e x c

    classwise_da_loss = 0
    for cc in range(output_prob_s.shape[1]):
        classwise_da_loss += class_prob_sum_s[0,cc]*class_prob_sum_t[0,cc]*euclidean(output_prob_s[:,cc], output_prob_t[:,cc])/(output_s.shape[0]**2)
    return classwise_da_loss


def k_moment_soft(output_s, output_t, k, class_prob_s, class_prob_t):
    output_s_k = output_s**k
    output_t_k = output_t**k
    return moment_soft(output_s_k, output_t_k, class_prob_s, class_prob_t)

def class_da_regulizer_soft(output_s, output_t, belta_moment, class_prob_s, class_prob_t):
    # print('s1:{}, s2:{}, s3:{}, s4:{}'.format(output_s1.shape, output_s2.shape, output_s3.shape, output_t.shape))        
    classwise_da_loss = 0
    for i in range(0,belta_moment):
        classwise_da_loss += k_moment_soft(output_s, output_t, i + 1, class_prob_s, class_prob_t)
    return classwise_da_loss


import torch
from itertools import combinations


def euclidean(x1, x2):
    return ((x1 - x2) ** 2).sum().sqrt()


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


def msda_regulizer(source_output, target_output, beta_moment):
    num_sources = len(source_output)
    s_mean = []
    source_output_ = []
    for i in range(num_sources):
        s_mean.append(source_output[i].mean(0))

    t_mean = target_output.mean(0)

    for i in range(num_sources):
        source_output_.append(source_output[i] - s_mean[i])

    target_output = target_output - t_mean

    # Compute first moment for nC2 combinations
    moment1 = 0
    for i in range(num_sources):
        moment1 += euclidean(source_output_[i], target_output)

    comb = list(combinations(range(num_sources), 2))

    for k in range(len(comb)):
        moment1 += euclidean(source_output_[comb[k][0]], source_output_[comb[k][1]])

    reg_info = moment1
    # print(reg_info)

    for i in range(beta_moment - 1):
        reg_info += k_moment(source_output_, target_output, i + 2)

    return reg_info / 6
# return euclidean(output_s1, output_t)

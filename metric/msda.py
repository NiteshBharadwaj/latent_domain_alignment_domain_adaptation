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

def moment_soft(output_s, domain_prob, output_t, k):
    output_s = output_s**k
    domain_prob_sum = domain_prob.sum(0)
    domain_prob_normalized = domain_prob/domain_prob_sum
    domain_embeddings = torch.matmul(domain_prob_normalized.t(), output_s)
    loss1 = 0
    cross_domain_terms = 0
    for i in range(len(domain_embeddings)):
        for j in range(len(domain_embeddings)):
            if(i != j and j > i):
                loss1 += euclidean(domain_embeddings[i,:], domain_embeddings[j,:])
                cross_domain_terms += 1
    loss1 /= cross_domain_terms
    loss2 = 0
    for i in range(len(domain_embeddings)):
        loss2 += euclidean(domain_embeddings[i,:], output_t)
    loss2 /= len(domain_embeddings)
    #loss2 = euclidean(output_s.mean(0), output_t)

    return loss1 + loss2



# def moment_soft(output_s, domain_prob):
#     output_s = output_s.reshape(output_s.shape[0], output_s.shape[1],1)
#     domain_prob = domain_prob.reshape(domain_prob.shape[0], 1, domain_prob.shape[1])
#     #print('shape of domain_prob : ', domain_prob.size())
#     output_prob = torch.matmul(output_s, domain_prob)
#     output_prob_sum = domain_prob.sum(0)
#     #print('shape of output prob sum : ', output_prob_sum.size())
#     output_prob = output_prob/output_prob_sum.reshape(1, 1, domain_prob.shape[2])
#     loss = 0
#     # print('shape of output prob: ', output_prob.size())
#     # print(output_prob[0,0,:])
#     # print(output_prob[0,1,:])
#     for i in range(output_prob.shape[2]):
#     	for j in range(i+1,output_prob.shape[2]):
#     		loss += euclidean(output_prob[:,i,:], output_prob[:,j,:])
#     return loss


def k_moment_soft(output_s, output_t, k, domain_prob):
    output_s_k = (output_s**k)
    output_s_mean = output_s_k.mean(0)
    output_t = (output_t**k).mean(0)
    # print('shape of output_s_mean : ', output_s_mean.size())
    # print('shape of output_s : ', output_s.size())
    #return euclidean(output_s_mean, output_t) + moment_soft(output_s, domain_prob)
    return euclidean(output_s_mean, output_t)
    return moment_soft(output_s, domain_prob, output_t, k)

def msda_regulizer_soft(output_s, output_t, belta_moment, domain_prob):
	# print('s1:{}, s2:{}, s3:{}, s4:{}'.format(output_s1.shape, output_s2.shape, output_s3.shape, output_t.shape))
	reg_info = 0
	# print(reg_info)
	for i in range(belta_moment):
		reg_info += k_moment_soft(output_s, output_t, i + 1, domain_prob)

	return reg_info / 6
# return euclidean(output_s1, output_t)

from __future__ import print_function
import argparse
import torch
import random
import numpy as np
import sys
import os
from time import time

sys.path.append('./model')
sys.path.append('./datasets')
sys.path.append('./metric');
from solver_MSDA_cluster_testing import Solver
import os
from train_msda_hard import train_MSDA as train_MSDA_hard
from train_msda_soft_cluster_testing import train_MSDA_soft
from train_msda_soft_cluster_testing_office import train_MSDA_soft as train_MSDA_soft_office
from train_msda_soft_cluster_testing_office_caltech import train_MSDA_soft as train_MSDA_soft_office_caltech
from train_msda_soft_cluster_testing_pacs import train_MSDA_soft as train_MSDA_soft_pacs
from train_msda_single import train_MSDA_single
from train_msda_classwise import train_MSDA_classwise
from train_ssda_classwise import train_SSDA_classwise
from test import test
from test2 import test2
from view_clusters import view_clusters
from view_clusters_ldada_unsorted import view_clusters as view_clusters_ldada_unsorted
from train_source_only import train_source_only
from matplotlib import pyplot as plt
from generate_pseudo import generate_pseudo, generate_empty_pseudo, generate_perfect_pseudo
from generate_domains import generate_domains
from plot_tsne import plot_tsne1,plot_tsne2, plot_tsne3


# Training settings
parser = argparse.ArgumentParser(description='PyTorch MSDA Implementation')
parser.add_argument('--all_use', type=str, default='no', metavar='N',
                    help='use all training data? in usps adaptation')
parser.add_argument('--model_sel_acc', type=int, default=1, metavar='n',
                    help='model selection accuracy? 1/0?')
parser.add_argument('--to_detach', type=str, default='yes', metavar='n',
                    help='classifier_discrepancy? yes/no')
parser.add_argument('--class_disc', type=str, default='no', metavar='N',
                    help='classifier_discrepancy? yes/no')
parser.add_argument('--clustering_only', type=int, default=1, metavar='N',
                    help='only kl/entropy loss? 1/0')
parser.add_argument('--load_ckpt', type=str, default="", metavar='N',
                    help='load checkpoint')

parser.add_argument('--known_domains', type=int, default=-1, metavar='N',
                    help='Domains known?')
parser.add_argument('--pretrained_clustering', type=str, default="no", metavar='N',
                    help='pretrained_clustering')
parser.add_argument('--pretrained_source', type=str, default="no", metavar='N',
                    help='pretrained_source')
parser.add_argument('--dl_type', type=str, default='', metavar='N',
                    help='original, hard_cluster, combined, soft_cluster')
parser.add_argument('--num_domain', type=int, default=4, metavar='N',
                    help='input latent domains')
parser.add_argument('--data', type=str, default='', metavar='N',
                    help='digits,cars,pacs')
parser.add_argument('--record_folder', type=str, default='record', metavar='N',
                    help='record folder')
parser.add_argument('--office_directory', type=str, default='.', metavar='N',
                    help='directory for office data')
parser.add_argument('--office_caltech_directory', type=str, default='.', metavar='N',
                    help='directory for office_caltech data')
parser.add_argument('--pacs_directory', type=str, default='.', metavar='N',
                    help='directory for pacs data')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--checkpoint_dir', type=str, default='checkpoint', metavar='N',
                    help='source only or not')
# parser.add_argument('--eval_only', action='store_true', default=False,
#                     help='evaluation only option')
parser.add_argument('--eval_only', type=str, default='no', metavar='N',
                    help='Evaluate only? yes/no')
parser.add_argument('--kl_wt', type=float, default=0.0, metavar='LR',
                    help='KL_wt (default: 0)')
parser.add_argument('--msda_wt', type=float, default=0.00001, metavar='LR',
                    help='msda_wt (default: 0)')
parser.add_argument('--lr_ratio', type=float, default=1.0, metavar='LR',
                    help='lr for domain predictor will be divided by this (default: 0)')
parser.add_argument('--entropy_wt', type=float, default=0.01, metavar='LR',
                    help='entropy_wt (default: 0)')
parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                    help='learning rate (default: 0.0002)')
parser.add_argument('--max_epoch', type=int, default=200, metavar='N',
                    help='how many epochs')
parser.add_argument('--target_baseline_pre', type=str, default="",
                     help='target baseline pretrained model')
parser.add_argument('--baseline_effect', action='store_true', default=False,
                     help='baseine effect')
parser.add_argument('--temperature_scaling', action='store_true', default=False,
                     help='temperature scaling')
parser.add_argument('--class_tear_apart_wt', type=float, default=1.,
                     help='class tear apart weight')
parser.add_argument('--class_tear_apart', action='store_true', default=False,
                     help='want class_tear_apart loss ?')
parser.add_argument('--target_clustering', action='store_true', default=False,
                     help='want to cluster target?')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--num_k', type=int, default=4, metavar='N',
                    help='hyper paremeter for generator update')
parser.add_argument('--num_workers', type=int, default=4, metavar='N',
                    help='dataloader num_workers')
parser.add_argument('--classaware_dp', type=str, default='no', metavar='N',
                    help='yes/no')
parser.add_argument('--one_step', action='store_true', default=False,
                    help='one step training with gradient reversal layer')
parser.add_argument('--optimizer', type=str, default='adam', metavar='N', help='which optimizer')
parser.add_argument('--resume_epoch', type=int, default=100, metavar='N',
                    help='epoch to resume')
parser.add_argument('--save_epoch', type=int, default=1, metavar='N',
                    help='when to restore the model')
parser.add_argument('--save_model', action='store_true', default=True,
                    help='save_model or not')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--source', type=str, default='svhn', metavar='N',
                    help='source dataset')
parser.add_argument('--target', type=str, default='mnist', metavar='N', help='target dataset')
parser.add_argument('--network', type=str, default='', metavar='N', help='network')
parser.add_argument('--use_abs_diff', action='store_true', default=False,
                    help='use absolute difference value as a measurement')
parser.add_argument('--pseudo_label_mode', type=str, default='init_only', metavar='N',
                    help='gen_epoch, gen_best_epoch, perfect, init_only')
parser.add_argument('--pseudo_logits_criteria', action='store_true', default=False,
                    help='Use raw logits to sort or probabilities?')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)
print(args)


def plot_data(graph_data, loss_plot):
    fig, ax = plt.subplots(len(graph_data), sharex=True)
    i = 0
    for k in graph_data:
        # print(len(graph_data[k]))
        ax[i].plot(graph_data[k])
        ax[i].yaxis.tick_right()
        ax[i].set_ylabel(k[:3])
        i += 1
    plt.savefig(loss_plot, dpi=200)


def main():
    # if not args.one_step:
    boolDict = {
        'yes': True,
        'no': False
    }

    args.eval_only = boolDict[args.eval_only]
    seed = args.seed
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    # torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.enabled = False
    record_num = 0

    record_train = '%s/%s_%s.txt' % (args.record_folder, args.target, record_num)
    record_test = '%s/%s_%s_test.txt' % (args.record_folder, args.target, record_num)
    record_val = '%s/%s_%s_val.txt' % (args.record_folder, args.target, record_num)
    checkpoint_dir = '%s/%s_%s' % (args.record_folder, args.target, record_num)
    plot_before_source = '%s/%s_%s_plot_before_source.png' % (args.record_folder, args.target, record_num)
    plot_before_target = '%s/%s_%s_plot_before_target.png' % (args.record_folder, args.target, record_num)
    plot_after_source = '%s/%s_%s_plot_after_source.png' % (args.record_folder, args.target, record_num)
    plot_after_target = '%s/%s_%s_plot_after_target.png' % (args.record_folder, args.target, record_num)
    all_plots = '%s/%s_%s_source_target_plots.png' % (args.record_folder, args.target, record_num)
    plot_domain1 = '%s/%s_%s_latent_domain_plot_conv.png' % (args.record_folder, args.target, record_num)
    plot_domain2 = '%s/%s_%s_latent_domain_plot_last.png' % (args.record_folder, args.target, record_num)
    plot_domain3 = '%s/%s_%s_latent_domain_plots.png' % (args.record_folder, args.target, record_num)
    plot_domains = [plot_domain1, plot_domain2, plot_domain3]
    loss_plot = '%s/%s_%s_loss_plots.png' % (args.record_folder, args.target, record_num)
    clusters_file_class = []
    probs_csv_class = []
    for c in range(7):
        clusters_file = []
        for i in range(args.num_domain):
            clusters_file.append('%s/cluster_cl%s_d%s.png' % (args.record_folder, str(c), str(i)))
        clusters_file_class.append(clusters_file)
        probs_csv = '%s/class_%s_best_%s.csv' % (args.record_folder,str(c),'probs')
        probs_csv_class.append(probs_csv)
    while os.path.exists(record_train):
        record_num += 1
        record_train = '%s/%s_%s.txt' % (args.record_folder, args.target, record_num)
        record_test = '%s/%s_%s_test.txt' % (args.record_folder, args.target, record_num)
        # record_val = '%s/%s_%s_val.txt' % (args.record_folder, args.target, record_num)

        checkpoint_dir = '%s/%s_%s' % (args.record_folder, args.target, record_num)
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    if not os.path.exists(args.record_folder):
        os.makedirs(args.record_folder)
    args.checkpoint_dir = checkpoint_dir
    classifier_disc = True if args.class_disc == 'yes' else False
    args.to_detach = True if args.to_detach == 'yes' else False
    if args.eval_only:
        checkpoint_dir = '%s/%s_%s' % (args.record_folder, args.target, record_num - 1)
        args.checkpoint_dir = checkpoint_dir
        solver = Solver(args, target=args.target, learning_rate=args.lr, batch_size=args.batch_size,
                        optimizer=args.optimizer,
                        checkpoint_dir=args.checkpoint_dir,
                        save_epoch=args.save_epoch)

        # train_MSDA_soft(solver,0,classifier_disc)

        #test2(solver, 0, 'test', record_file=None, save_model=False, temperature_scaling=True)
        #view_clusters(solver, clusters_file_class, probs_csv_class,0)
        view_clusters_ldada_unsorted(solver, clusters_file_class, probs_csv_class)
        #plot_tsne1(solver, plot_before_source, plot_before_target, plot_after_source, plot_after_target, all_plots,
        #            plot_domains, args.data)
        #plot_tsne3(solver, plot_before_source, plot_before_target, plot_after_source, plot_after_target, all_plots,
        #            plot_domains, args.data)
        #
        # solver = Solver(args, target=args.target, learning_rate=args.lr, batch_size=args.batch_size,
        #                 optimizer=args.optimizer,
        #                 checkpoint_dir=args.checkpoint_dir,
        #                 save_epoch=args.save_epoch)
        #
        #plot_tsne2(solver, plot_before_source, plot_before_target, plot_after_source, plot_after_target, all_plots,
        #            plot_domains, args.data)

    else:

        solver = Solver(args, target=args.target, learning_rate=args.lr, batch_size=args.batch_size,
                        optimizer=args.optimizer,
                        checkpoint_dir=args.checkpoint_dir,
                        save_epoch=args.save_epoch)
        if args.target_baseline_pre!="":
            #test2(solver, 0, 'target', record_file=record_test, save_model=args.save_model)
            print("Setting pseudo label temperature")
            _,val_acc = test(solver, 0, 'val', record_file=record_val, save_model=args.save_model, temperature_scaling=True, use_g_t=True)
        if args.pretrained_clustering=='yes':
            solver.clusters = generate_domains(solver,solver.G, solver.DP,solver.datasets)
        if solver.is_classwise:
            if args.target_baseline_pre!="":
                print("Generating pseudo labels using pretrained model")
                solver.pseudo_labels, solver.pseudo_accept_mask = generate_pseudo(solver,solver.G_T,solver.C1_T,solver.datasets,logits_criteria=args.pseudo_logits_criteria, split="target",reject_quantile=1-val_acc/100.+0.1)
            else:
                if args.pseudo_label_mode=="perfect":
                    print("Initializing perfect pseudo labels")
                    solver.pseudo_labels, solver.pseudo_accept_mask = generate_perfect_pseudo(solver, solver.datasets)
                else:
                    print("Initializing pseudo labels to all zeros, no classwise adaptation can be performed initially")
                    solver.pseudo_labels, solver.pseudo_accept_mask = generate_empty_pseudo(solver, solver.datasets)
        count = 0
        solver.best_clus_loss = 1e9
        graph_data = {}
        keys = ['entropy', 'kl', 'c1', 'c2', 'h', 'total', 'msda']
        for k in keys:
            graph_data[k] = []
        total_it = 120000
        for t in range(args.max_epoch):
            start = time()
            print(t)
            if (count>=total_it):
                break
            if not args.one_step:
                # num = solver.train_merge_baseline(t, record_file=record_train)
                if args.dl_type == 'soft_cluster':
                    torch.cuda.empty_cache()
                    if args.data == 'digits':
                        num = train_MSDA_soft(solver, t, classifier_disc, record_file=record_train)
                    elif args.data == 'office':
                        num, graph_data = train_MSDA_soft_office(solver, t, graph_data, classifier_disc,
                                                                 record_file=record_train)
                    elif args.data == 'office_caltech':
                        num, graph_data = train_MSDA_soft_office_caltech(solver, t, graph_data, classifier_disc,
                                                                         record_file=record_train)
                    elif args.data == 'pacs':
                        num, graph_data, clus_loss = train_MSDA_soft_pacs(solver, t, graph_data, classifier_disc,
                                                                         record_file=record_train, max_it=total_it-count)
                    elif args.data == 'birds':
                        num, graph_data, clus_loss = train_MSDA_soft_pacs(solver, t, graph_data, classifier_disc,
                                                               record_file=record_train, max_it=total_it - count)
                    else:
                        print("WTF Noob")
                elif args.dl_type == 'source_only':
                    torch.cuda.empty_cache()
                    num = train_source_only(solver, t, record_file=record_train, max_it = total_it-count, prev_count = count)
                elif args.dl_type=='classwise_msda':
                    num, graph_data = train_MSDA_classwise(solver,t, graph_data,classifier_disc,record_file=record_train, single_domain_mode=True, max_it=total_it-count, prev_count = count)
                elif args.dl_type=='classwise_ssda':
                    num, graph_data = train_SSDA_classwise(solver,t,graph_data, classifier_disc,record_file=record_train, single_domain_mode=True, max_it=total_it-count, prev_count=count)
                elif args.dl_type == 'source_target_only':
                    torch.cuda.empty_cache()
                    num = train_MSDA_single(solver, t, classifier_disc, record_file=record_train, max_it=total_it-count, prev_count=count)
                else:
                    num = train_MSDA_hard(solver, t, classifier_disc, record_file=record_train)
            else:
                raise Exception('One step solver not defined')
            end = time()
            print("Time taken for training epoch : ", end-start)
            if 0:#not solver.args.clustering_only:
                solver.sche_g.step()
                solver.sche_c1.step()
                solver.sche_c2.step()
            if 0:
                solver.sche_dp.step()
            count += num
            if args.msda_wt<1e-8:
                if solver.best_clus_loss > clus_loss and num>10:
                    print("Current best clustering loss {}".format(clus_loss))
                    solver.best_clus_loss = clus_loss
                    print('Saving best clustering model','%s/%s_model_best.pth' % (solver.checkpoint_dir, solver.target))
                    checkpoint = {}
                    checkpoint['G_state_dict'] = solver.G.module.state_dict()
                    checkpoint['C1_state_dict'] = solver.C1.state_dict()
                    checkpoint['C2_state_dict'] = solver.C2.state_dict()
                    checkpoint['DP_state_dict'] = solver.DP.module.state_dict()

                    checkpoint['G_state_dict_opt'] = solver.opt_g.state_dict()
                    checkpoint['C1_state_dict_opt'] = solver.opt_c1.state_dict()
                    checkpoint['C2_state_dict_opt'] = solver.opt_c2.state_dict()
                    checkpoint['DP_state_dict_opt'] = solver.opt_dp.state_dict()
                    torch.save(checkpoint, '%s/%s_model_best.pth' % (solver.checkpoint_dir, solver.target))



            if (t % 1 == 0 or count>=total_it) and args.msda_wt>1e-8:
                # print('testing now')
                #if (args.dl_type == 'soft_cluster' or args.dl_type == 'classwise_ssda' or args.dl_type == 'classwise_msda'):
                    #plot_data(graph_data, loss_plot)
                    #view_clusters(solver, clusters_file_class, probs_csv_class, t)
                if args.data == 'cars':
                    test2(solver, t, 'train', record_file=record_test, save_model=args.save_model)
                best, val_acc = test2(solver, t, 'val', record_file=record_val, save_model=args.save_model, temperature_scaling=args.temperature_scaling)
                if args.pseudo_label_mode=="gen_epoch":
                    solver.pseudo_labels, solver.pseudo_accept_mask = generate_pseudo(solver,solver.G,solver.C1,solver.datasets,logits_criteria=args.pseudo_logits_criteria, split="target", reject_quantile=1-val_acc/100.+0.5)
                if best:
                    if t>2 and args.pseudo_label_mode=="gen_best_epoch":
                        solver.pseudo_labels, solver.pseudo_accept_mask = generate_pseudo(solver,solver.G,solver.C1,solver.datasets,logits_criteria=args.pseudo_logits_criteria, split="target", reject_quantile=1-val_acc/100.+0.5)
                    print('best epoch : ', t)
                    #start = time()
                    #test2(solver, t, 'test', record_file=record_test, save_model=args.save_model)
                    #end = time()
                    #print("Time taken for testing epoch : ", end-start)
                #view_clusters(solver, clusters_file, probs_csv, t)
                # print('clustering images saved in!')

        # generate_plots(solver, 0, 'test', plot_before_source, plot_before_target, plot_after_source, plot_after_target, False)


if __name__ == '__main__':
    main()

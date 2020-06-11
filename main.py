from __future__ import print_function
import argparse
import torch
import random
import numpy as np
import sys

sys.path.append('./model')
sys.path.append('./datasets')
sys.path.append('./metric');
from solver_MSDA import Solver
import os
from train_msda_hard import train_MSDA as train_MSDA_hard
from train_msda_soft import train_MSDA_soft
from test import test
from view_clusters import view_clusters
from train_source_only import train_source_only
from plot_tsne import plot_tsne
from torch.utils.tensorboard import SummaryWriter

# Training settings
parser = argparse.ArgumentParser(description='PyTorch MSDA Implementation')
parser.add_argument('--all_use', type=str, default='no', metavar='N',
                    help='use all training data? in usps adaptation')
parser.add_argument('--to_detach', type=str, default='no', metavar='N',
                    help='classifier_discrepancy? yes/no')
parser.add_argument('--class_disc', type=str, default='no', metavar='N',
                    help='classifier_discrepancy? yes/no')
parser.add_argument('--dl_type', type=str, default='', metavar='N',
                    help='original, hard_cluster, combined, soft_cluster')
parser.add_argument('--num_domain', type=int, default=4, metavar='N',
                    help='input latent domains')
parser.add_argument('--data', type=str, default='', metavar='N',
                    help='digits,cars')
parser.add_argument('--usps_less_data_protocol', type=int, default=0, metavar='N',
                    help='usps less data protocol? usps=1800, mnist=2000')
parser.add_argument('--record_folder', type=str, default='record', metavar='N',
                    help='record folder')
parser.add_argument('--office_directory', type=str, default='.', metavar='N',
                    help='directory for office data')
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
parser.add_argument('--adv_wt', type=float, default=1.0, metavar='LR',
                    help='adv_wt (default: 0)')
parser.add_argument('--entropy_wt', type=float, default=0.01, metavar='LR',
                    help='entropy_wt (default: 0)')
parser.add_argument('--adv_mode', type=int, default=0, metavar='LR',
                    help='adversarial mode? (default: 0)')
parser.add_argument('--lr', type=float, default=0.0001, metavar='LR',
                    help='learning rate (default: 0.0002)')
parser.add_argument('--max_epoch', type=int, default=200, metavar='N',
                    help='how many epochs')
parser.add_argument('--model_sel_acc', type=int, default=0, metavar='N',
                    help='if want to do model selection using acc => 1')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--num_k', type=int, default=4, metavar='N',
                    help='hyper paremeter for generator update')
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
parser.add_argument('--use_abs_diff', action='store_true', default=False,
                    help='use absolute difference value as a measurement')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)
print(args)


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
    torch.backends.cudnn.deterministic = True
    record_num = 0
    summary_writer = SummaryWriter(args.record_folder)
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
    clusters_file = []
    for i in range(args.num_domain):
        clusters_file.append('%s/cluster_%s.png' % (args.record_folder, str(i)))
    probs_csv = '%s/best_%s.csv' % (args.record_folder, 'probs')
    while os.path.exists(record_train):
        record_num += 1
        record_train = '%s/%s_%s.txt' % (args.record_folder, args.target, record_num)
        record_test = '%s/%s_%s_test.txt' % (args.record_folder, args.target, record_num)
        record_val = '%s/%s_%s_val.txt' % (args.record_folder, args.target, record_num)

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

        test(solver, 0, 'test', record_file=None, save_model=False)
        view_clusters(solver, clusters_file, probs_csv)
        plot_tsne(solver, plot_before_source, plot_before_target, plot_after_source, plot_after_target, all_plots,
                   plot_domains, args.data)

    else:

        solver = Solver(args, target=args.target, learning_rate=args.lr, batch_size=args.batch_size,
                        optimizer=args.optimizer,
                        checkpoint_dir=args.checkpoint_dir,
                        save_epoch=args.save_epoch, class_disc=classifier_disc)
        count = 0
        for t in range(args.max_epoch):
            print(t)
            if not args.one_step:
                if args.dl_type == 'soft_cluster':
                    torch.cuda.empty_cache()
                    num = train_MSDA_soft(solver, t, classifier_disc, record_file=record_train,
                                          summary_writer=summary_writer, epoch_start_idx=count)
                elif args.dl_type == 'source_only':
                    torch.cuda.empty_cache()
                    num = train_source_only(solver, t, record_file=record_train)
                elif args.dl_type == 'source_target_only':
                    torch.cuda.empty_cache()
                    num = train_MSDA_soft(solver, t, classifier_disc, record_file=record_train, single_domain_mode=True)
                else:
                    num = train_MSDA_hard(solver, t, classifier_disc, record_file=record_train)
            else:
                raise Exception('One step solver not defined')
            solver.sche_g.step()
            solver.sche_c1.step()
            solver.sche_c2.step()
            solver.sche_dp.step()
            count += num
            if t % 1 == 0:
                best = test(solver, t, 'val', record_file=record_val, save_model=args.save_model,
                            summary_writer=summary_writer)
                if best:
                    test(solver, t, 'test', record_file=record_test, save_model=args.save_model,
                         summary_writer=summary_writer)

        # generate_plots(solver, 0, 'test', plot_before_source, plot_before_target, plot_after_source, plot_after_target, False)


if __name__ == '__main__':
    main()

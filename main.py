from __future__ import print_function
import argparse
import torch

import sys
sys.path.append('./model')
sys.path.append('./datasets')
sys.path.append('./metric');
from solver_MSDA import Solver
import os
from train_msda_hard import train_MSDA as train_MSDA_hard
from train_msda_soft import train_MSDA_soft
from test import test
from train_source_only import train_source_only


# Training settings
parser = argparse.ArgumentParser(description='PyTorch MSDA Implementation')
parser.add_argument('--all_use', type=str, default='no', metavar='N',
                    help='use all training data? in usps adaptation')
parser.add_argument('--class_disc', type=str, default='yes', metavar='N',
                    help='classifier_discrepancy? yes/no')
parser.add_argument('--dl_type', type=str, default='', metavar='N',
                    help='original, hard_cluster, combined, soft_cluster')
parser.add_argument('--num_domain', type=int, default=4, metavar='N',
                    help='input latent domains')
parser.add_argument('--data', type=str, default='', metavar='N',
                    help='digits,cars')
parser.add_argument('--record_folder', type=str, default='record', metavar='N',
                    help='record folder')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--checkpoint_dir', type=str, default='checkpoint', metavar='N',
                    help='source only or not')
parser.add_argument('--eval_only', action='store_true', default=False,
                    help='evaluation only option')
parser.add_argument('--lr', type=float, default=0.0001, metavar='LR',
                    help='learning rate (default: 0.0002)')
parser.add_argument('--max_epoch', type=int, default=200, metavar='N',
                    help='how many epochs')
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

    record_num = 0
    
    record_train = '%s/%s_%s.txt' % (args.record_folder, args.target,record_num)
    record_test = '%s/%s_%s_test.txt' % (args.record_folder, args.target, record_num)
    record_val = '%s/%s_%s_val.txt' % (args.record_folder, args.target, record_num)
    checkpoint_dir = '%s/%s_%s' % (args.record_folder, args.target, record_num)
    while os.path.exists(record_train):
        record_num += 1
        record_train = '%s/%s_%s.txt' % (args.record_folder, args.target, record_num)
        record_test = '%s/%s_%s_test.txt' % (args.record_folder, args.target, record_num)
        record_val = '%s/%s_%s_val.txt' % (args.record_folder, args.target, record_num)
    
        checkpoint_dir = '%s/%s_%s' % (args.record_folder, args.target, record_num)
    if not os.path.exists(checkpoint_dir):
        os.mkdir(checkpoint_dir)
    if not os.path.exists(args.record_folder):
        os.mkdir(args.record_folder)
    args.checkpoint_dir = checkpoint_dir
    classifier_disc = True if args.class_disc=='yes' else False
    
    solver = Solver(args, target=args.target, learning_rate=args.lr, batch_size=args.batch_size,
                    optimizer=args.optimizer, 
                    checkpoint_dir=args.checkpoint_dir,
                    save_epoch=args.save_epoch)
    if args.eval_only:
        solver.test(0)
    else:
        count = 0
        for t in range(args.max_epoch):
            print(t)
            if not args.one_step:
                # num = solver.train_merge_baseline(t, record_file=record_train)
                if args.dl_type=='soft_cluster':
                    torch.cuda.empty_cache()
                    num= train_MSDA_soft(solver,t,classifier_disc,record_file=record_train)
                elif args.dl_type=='source_only':
                    torch.cuda.empty_cache()
                    num= train_source_only(solver,t,record_file=record_train)
                else:
                    num = train_MSDA_hard(solver,t, classifier_disc, record_file=record_train)
            else:
                raise Exception('One step solver not defined')
            count += num
            if t % 1 == 0:
                if args.data=='cars':
                    test(solver, t, 'train', record_file=record_test, save_model=args.save_model)
                best = test(solver, t, 'val', record_file=record_val, save_model=args.save_model)
                if best:
                    test(solver, t, 'test', record_file=record_test, save_model=args.save_model)


if __name__ == '__main__':
    main()

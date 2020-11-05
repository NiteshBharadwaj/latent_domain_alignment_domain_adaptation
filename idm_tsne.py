import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
from sklearn.manifold import TSNE
import sys
import sklearn
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects
import matplotlib
from PIL import Image
#%matplotlib inline
import os
import seaborn as sns
sns.set_style('darkgrid')
sns.set_palette('muted')
sns.set_context("notebook", font_scale=1.5,
                rc={"lines.linewidth": 2.5})

RS = 20150101

def get_concat_h(im1, im2):
    dst = Image.new('RGB', (im1.width + im2.width, im1.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (im1.width, 0))
    return dst

def get_concat_v(im1, im2):
    dst = Image.new('RGB', (im1.width, im1.height + im2.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (0, im1.height))
    return dst

def plot_tsne():
    solver.G.eval()
    solver.C1.eval()
    solver.C2.eval()
    solver.DP.eval()

    if(dataset == 'digits'):
        labelsOfInterest = [1,4,6,9]

    with torch.no_grad():
        before_source_torch = 0
        before_source_bool = False
        after_source_torch = 0
        after_source_bool = False
        source_label_torch = []
        source_label_bool = False

        for batch_idx, data in enumerate(solver.datasets):
            img = data['S']
            label = data['S_label'].long()
            img_transformed, _ = solver.G(img.cuda())

            for l in labelsOfInterest:
                l_index = ((label == l).nonzero()).squeeze()
                img_l = img[l_index,:,:,:]
                img_transformed_l = img_transformed[l_index,:]
                if(img_l.size()[0] > 0):
                    try:
                        a = img_l.size()[3]
                    except:
                        img_l = torch.unsqueeze(img_l, 0)
                        img_transformed_l = torch.unsqueeze(img_transformed_l, 0)
                    img_l = img_l.view(img_l.size()[0], -1)
                    img_transformed_l = img_transformed_l.view(img_transformed_l.size()[0], -1)
                    cur_y = torch.zeros([img_l.size()[0]]).fill_(l)
                    source_label_torch.append(cur_y)

                    if(before_source_bool == False):
                        before_source_torch = img_l
                        before_source_bool = True
                        after_source_torch = img_transformed_l
                        after_source_bool = True
                    else:
                        before_source_torch = torch.cat((before_source_torch,img_l),0)
                        after_source_torch = torch.cat((after_source_torch,img_transformed_l),0)

        source_label_torch = tuple(source_label_torch)
        source_label_torch = torch.cat(source_label_torch,axis=0)
        source_label_torch = source_label_torch.data.cpu().numpy()
        before_source_torch = before_source_torch.data.cpu().numpy()
        after_source_torch = after_source_torch.data.cpu().numpy()


        before_target_torch = 0
        before_target_bool = False
        after_target_torch = 0
        after_target_bool = False
        target_label_torch = []
        target_label_bool = False
        total_iterations = 0
        for batch_idx, data in enumerate(solver.dataset_test):
            img = data['T']
            label = data['T_label'].long()
            img_transformed, _ = solver.G(img.cuda())
            for l in labelsOfInterest:
                l_index = ((label == l).nonzero()).squeeze()
                img_l = img[l_index,:,:,:]
                img_transformed_l = img_transformed[l_index,:]
                if(img_l.size()[0] > 0):
                    try:
                        a = img_l.size()[3]
                    except:
                        img_l = torch.unsqueeze(img_l, 0)
                        img_transformed_l = torch.unsqueeze(img_transformed_l, 0)
                    img_l = img_l.view(img_l.size()[0], -1)
                    img_transformed_l = img_transformed_l.view(img_transformed_l.size()[0], -1)
                    cur_y = torch.zeros([img_l.size()[0]]).fill_(l)
                    target_label_torch.append(cur_y)
                    if(before_target_bool == False):
                        before_target_torch = img_l
                        before_target_bool = True
                        after_target_torch = img_transformed_l
                        after_target_bool = True
                    else:
                        before_target_torch = torch.cat((before_target_torch,img_l),0)
                        after_target_torch = torch.cat((after_target_torch,img_transformed_l),0)
        target_label_torch = tuple(target_label_torch)
        target_label_torch = torch.cat(target_label_torch,axis=0)
        target_label_torch = target_label_torch.data.cpu().numpy()
        before_target_torch = before_target_torch.data.cpu().numpy()
        after_target_torch = after_target_torch.data.cpu().numpy()

        before_torch = np.concatenate((before_source_torch,before_target_torch), axis=0)
        after_torch = np.concatenate((after_source_torch,after_target_torch), axis=0)



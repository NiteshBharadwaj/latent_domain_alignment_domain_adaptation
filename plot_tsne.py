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



def scatter(x, colors, labelsOfInterest, figSize):
    # We choose a color palette with seaborn.
    palette = np.array(sns.color_palette("hls", max(labelsOfInterest)+1))

    # We create a scatter plot.
    f = plt.figure(figsize=figSize)
    ax = plt.subplot(aspect='equal')
    sc = ax.scatter(x[:,0], x[:,1], lw=0, s=40,
                    c=palette[colors.astype(np.int)])
    plt.xlim(-25, 25)
    plt.ylim(-25, 25)
    ax.axis('off')
    ax.axis('tight')

#     # We add the labels for each digit.
#     txts = []
#     for i in range(len(labelsOfInterest)):
#         # Position of each label.
#         a = np.median(x[colors == labelsOfInterest[i], :], axis=0)
#         print(a)
# #         txt = ax.text(xtext, ytext, str(i), fontsize=24)
# #         txt.set_path_effects([
# #             PathEffects.Stroke(linewidth=5, foreground="w"),
# #             PathEffects.Normal()])
# #         txts.append(txt)

#     return f, ax, sc, txts

# X = np.vstack([digits.data[digits.target==i]
#                for i in range(10)])
# y = np.hstack([digits.target[digits.target==i]
#                for i in range(10)])
# digits_proj = TSNE(random_state=RS).fit_transform(X)
def plot_tsne(solver,plot_before_source, plot_before_target, plot_after_source, plot_after_target, all_plots, plot_domains, dataset):
    solver.G.eval()
    solver.C1.eval()
    solver.C2.eval()
    solver.DP.eval()
    
    if(dataset == 'office'):
        labelsOfInterest = [2,8,14,22,30]
    if(dataset == 'digits'):
        labelsOfInterest = [1,4,6,9]
    prev = solver.batch_size
    
    with torch.no_grad():
        
        before_source_torch = 0
        before_source_bool = False
        after_source_torch = 0
        after_source_bool = False
        source_label_torch = []
        source_label_bool = False
        total_iterations = 0
        prev = solver.batch_size
        for batch_idx, data in enumerate(solver.datasets):
            img = data['S']
            label = data['S_label'].long()
            img_transformed, _ = solver.G(img.cuda())
            
            if(img.size()[0] > prev):
                total_iterations += 1
            prev = img.size()[0]
            if(total_iterations > 0):
                break
            for l in labelsOfInterest:
                l_index = ((label == l).nonzero()).squeeze()
                img_l = img[l_index,:,:,:]
                img_transformed_l = img_transformed[l_index,:]
                #print(img_transformed_l.size())
                #print(img_l.size())
                #cur_y = label[l_index,]
                if(img_l.size()[0] > 0):
                    try:
                        a = img_l.size()[3]
                    except:
                        img_l = torch.unsqueeze(img_l, 0)
                        img_transformed_l = torch.unsqueeze(img_transformed_l, 0)
                        #cur_y = torch.unsqueeze(cur_y,0)
                    #print(img_transformed_l.size())
                    img_l = img_l.view(img_l.size()[0], -1)
                    img_transformed_l = img_transformed_l.view(img_transformed_l.size()[0], -1)
                    #source_label_torch.append(cur_y)
                    cur_y = torch.zeros([img_l.size()[0]]).fill_(l)
                    source_label_torch.append(cur_y)
                    
                    if(before_source_bool == False):
                        before_source_torch = img_l
                        #print(before_source_torch.size())
                        before_source_bool = True
                        after_source_torch = img_transformed_l
                        after_source_bool = True
                    else:
                        before_source_torch = torch.cat((before_source_torch,img_l),0)
                        after_source_torch = torch.cat((after_source_torch,img_transformed_l),0)
        source_label_torch = tuple(source_label_torch)
        source_label_torch = torch.cat(source_label_torch,axis=0)
        source_label_torch = source_label_torch.data.cpu().numpy()
        #print(source_label_torch.shape)
        before_source_torch = before_source_torch.data.cpu().numpy()
        after_source_torch = after_source_torch.data.cpu().numpy()
        
        scatter(TSNE(random_state=RS).fit_transform(before_source_torch),source_label_torch,labelsOfInterest,(5,5))
        plt.savefig(plot_before_source, dpi=120)
        print('source before plot saved in : ', plot_before_source)
        
        scatter(TSNE(random_state=RS).fit_transform(after_source_torch),source_label_torch,labelsOfInterest,(5,5))
        plt.savefig(plot_after_source, dpi=120)
        print('source after plot saved in : ', plot_after_source)
        
        
        
        before_target_torch = 0
        before_target_bool = False
        after_target_torch = 0
        after_target_bool = False
        target_label_torch = []
        target_label_bool = False
        total_iterations = 0
        prev = solver.batch_size
        for batch_idx, data in enumerate(solver.dataset_test):
            img = data['T']
            label = data['T_label'].long()
            img_transformed, _ = solver.G(img.cuda())
            
            if(img.size()[0] > prev):
                total_iterations += 1
            prev = img.size()[0]
            if(total_iterations > 0):
                break
            for l in labelsOfInterest:
                l_index = ((label == l).nonzero()).squeeze()
                img_l = img[l_index,:,:,:]
                img_transformed_l = img_transformed[l_index,:]
                #cur_y = label[l_index,]
                if(img_l.size()[0] > 0):
                    try:
                        a = img_l.size()[3]
                    except:
                        img_l = torch.unsqueeze(img_l, 0)
                        img_transformed_l = torch.unsqueeze(img_transformed_l, 0)
                        #cur_y = torch.unsqueeze(cur_y,0)
                    #print(img_transformed_l.size())
                    img_l = img_l.view(img_l.size()[0], -1)
                    img_transformed_l = img_transformed_l.view(img_transformed_l.size()[0], -1)
                    cur_y = torch.zeros([img_l.size()[0]]).fill_(l)
                    target_label_torch.append(cur_y)
                    if(before_target_bool == False):
                        before_target_torch = img_l
                        #print(before_source_torch.size())
                        before_target_bool = True
                        after_target_torch = img_transformed_l
                        after_target_bool = True
                    else:
                        before_target_torch = torch.cat((before_target_torch,img_l),0)
                        after_target_torch = torch.cat((after_target_torch,img_transformed_l),0)
        target_label_torch = tuple(target_label_torch)
        target_label_torch = torch.cat(target_label_torch,axis=0)
        target_label_torch = target_label_torch.data.cpu().numpy()
        #print(source_label_torch.shape)
        before_target_torch = before_target_torch.data.cpu().numpy()
        after_target_torch = after_target_torch.data.cpu().numpy()
        
        scatter(TSNE(random_state=RS).fit_transform(before_target_torch),target_label_torch,labelsOfInterest,(5,5))
        plt.savefig(plot_before_target, dpi=120)
        print('target before plot saved in : ', plot_before_target)
        
        scatter(TSNE(random_state=RS).fit_transform(after_target_torch),target_label_torch,labelsOfInterest,(5,5))
        plt.savefig(plot_after_target, dpi=120)
        print('target after plot saved in : ', plot_after_source)
        
        before_source = Image.open(plot_before_source)
        after_source = Image.open(plot_after_source)
        before_target = Image.open(plot_before_target)
        after_target = Image.open(plot_after_target)
        v1 = get_concat_v(before_source, before_target)
        v2 = get_concat_v(after_source, after_target)
        final = get_concat_h(v1,v2)
        final.save(all_plots)
        
        
        
        
#############################################################################
        
        
        prev = solver.batch_size
        domain_x_torch = 0
        domain_x_bool = False
        domain_y_torch = []
        domain_y_bool = False
        
        if(solver.dl_type == 'soft_cluster'):
            
            for batch_idx, data in enumerate(solver.datasets):
                
                img_t = data['T'].cuda()
                img = data['S'].cuda()
                label = data['S_label'].long().cuda()
                _, img_transformed = solver.G(img.cuda())
                
                #print(img.size()[0])
                if(img.size()[0] > prev):
                    break
                prev = img.size()[0]
                #solver.reset_grad()

                loss_s_c1, loss_s_c2, loss_msda, entropy_loss, kl_loss, domain_prob = solver.loss_soft_all_domain(img, img_t, label, 0)
                best_domains = domain_prob.data.max(1)[1]
                for i in range(solver.num_domains):
                    i_index = ((best_domains == i).nonzero()).squeeze()
                    img_i = img[i_index,:,:,:]
                    img_transformed_i = img_transformed[i_index,:,:,:]
                    if(img_i.size()[0] > 0):
                        try:
                            a = img_i.size()[3]
                        except:
                            img_i = torch.unsqueeze(img_i, 0)
                            img_transformed_i = torch.unsqueeze(img_transformed_i, 0)
                        #cur_y = torch.zeros([img_i.size()[0]], dtype=torch.int32)
                        img_transformed_i = img_transformed_i.view(img_transformed_i.size()[0], -1)
                        cur_y = torch.zeros([img_i.size()[0]]).fill_(i)
                        domain_y_torch.append(cur_y)
                        if(domain_x_bool == False):
                            domain_x_torch = img_transformed_i
                            domain_x_bool = True
                        else:
                            domain_x_torch = torch.cat((domain_x_torch, img_transformed_i), 0)
            domain_y_torch = tuple(domain_y_torch)
            domain_y_torch = torch.cat(domain_y_torch,axis=0)
            domain_y_torch = domain_y_torch.data.cpu().numpy()

            domain_x_torch = domain_x_torch.data.cpu().numpy()

            scatter(TSNE(random_state=RS).fit_transform(domain_x_torch),domain_y_torch,[i for i in range(solver.num_domains)], (8,8))
            plt.savefig(plot_domains, dpi=120)
            
            print('domain tsne plot saved in : ', plot_domains)
                
                
                
                            
        
        
        
        











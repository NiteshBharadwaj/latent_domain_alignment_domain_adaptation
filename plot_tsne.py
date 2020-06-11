import torch
import torch.nn as nn
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

from PIL import Image
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


def scatter(x, colors, labelsOfInterest, figSize, plotName):
    # We choose a color palette with seaborn.
    palette = np.array(sns.color_palette("hls", max(labelsOfInterest) + 1))

    # We create a scatter plot.
    f = plt.figure(figsize=figSize)
    ax = plt.subplot(aspect='equal')
    sc = ax.scatter(x[:, 0], x[:, 1], lw=0, s=40,
                    c=palette[colors.astype(np.int)])
    plt.xlim(-25, 25)
    plt.ylim(-25, 25)
    plt.title(plotName)
    ax.axis('off')
    ax.axis('tight')


def plot_tsne(solver, plot_before_source, plot_before_target, plot_after_source, plot_after_target, all_plots,
               plot_domains, dataset):
    solver.G.eval()
    solver.C1.eval()
    solver.C2.eval()
    solver.DP.eval()
    with torch.no_grad():
        prev = solver.batch_size
        domain_x1_torch = 0
        domain_x1_bool = False
        domain_x2_torch = 0
        domain_x2_bool = False
        domain_y1_torch = []
        domain_y1_bool = False
        domain_y2_torch = []
        domain_y2_bool = False
        label_y_s_torch = []
        label_y_t_torch = []

        total_s = 0
        total_t = 0

    num_imgs = 1000
    if solver.dl_type == 'soft_cluster':
        for batch_idx, data in enumerate(solver.datasets):
            img_t = data['T'].cuda()
            img = data['S'].cuda()
            label_s = data['S_label'].long().cuda()
            label_t = data['T_label'].long().cuda()

            if dataset == 'digits':
                img_transformed1, _, _ = solver.G(img.cuda())
                img_transformed_t, _, _ = solver.G(img_t.cuda())
                _, img_transformed2 = solver.DP(img.cuda())

                # total_s += img.size()[0]
            if (total_s > num_imgs) or (img.size()[0] > prev):
                break

            prev = img.size()[0]

            domain_prob = solver.get_domain_probs(img)
            best_domains = domain_prob.data.max(1)[1]
            best_probs = domain_prob.data.max(1)[0]
            for i in range(solver.num_domains):
                i_index = ((best_domains == i).nonzero()).squeeze()
                img_i = img[i_index, :, :, :]
                label_si = label_s[i_index]
                total_s += img_i.size()[0]
                img_transformed1_i = img_transformed1[i_index, :]
                img_transformed2_i = img_transformed2[i_index, :]
                if (img_i.size()[0] > 0):
                    try:
                        a = img_i.size()[3]
                    except:
                        img_i = torch.unsqueeze(img_i, 0)
                        img_transformed1_i = torch.unsqueeze(img_transformed1_i, 0)
                        img_transformed2_i = torch.unsqueeze(img_transformed2_i, 0)
                    img_transformed1_i = img_transformed1_i.view(img_transformed1_i.size()[0], -1)
                    img_transformed2_i = img_transformed2_i.view(img_transformed2_i.size()[0], -1)
                    cur_y = torch.zeros([img_i.size()[0]]).fill_(i)
                    domain_y1_torch.append(cur_y)
                    domain_y2_torch.append(cur_y)
                    label_y_s_torch.append(label_si)
                    if (domain_x1_bool == False):
                        domain_x1_torch = img_transformed1_i
                        domain_x1_bool = True
                        domain_x2_torch = img_transformed2_i
                        domain_x2_bool = True
                    else:
                        domain_x1_torch = torch.cat((domain_x1_torch, img_transformed1_i), 0)
                        domain_x2_torch = torch.cat((domain_x2_torch, img_transformed2_i), 0)

            total_t += img_t.size()[0]
            if (total_t < num_imgs // 4 and img_t.size()[0] > 0):
                try:
                    a = img_t.size()[3]
                except:
                    img_t = torch.unsqueeze(img_t, 0)
                    img_transformed_t = torch.unsqueeze(img_transformed_t, 0)
                img_transformed_t = img_transformed_t.view(img_transformed_t.size()[0], -1)
                cur_y = torch.zeros([img_t.size()[0]]).fill_(solver.num_domains)
                label_y_t_torch.append(label_t)

                domain_y1_torch.append(cur_y)
                if domain_x1_bool == False:
                    assert (False)
                else:
                    domain_x1_torch = torch.cat((domain_x1_torch, img_transformed_t), 0)

    print(domain_x1_torch.size()[0], domain_x2_torch.size()[0])
    domain_y1_torch = tuple(domain_y1_torch)
    domain_y1_torch = torch.cat(domain_y1_torch, axis=0)
    domain_y1_torch = domain_y1_torch.data.cpu().numpy()

    domain_y2_torch = tuple(domain_y2_torch)
    domain_y2_torch = torch.cat(domain_y2_torch, axis=0)
    domain_y2_torch = domain_y2_torch.data.cpu().numpy()

    domain_x1_torch = domain_x1_torch.data.cpu().numpy()
    domain_x2_torch = domain_x2_torch.data.cpu().numpy()

    label_y_s_torch = tuple(label_y_s_torch)
    label_y_s_torch = torch.cat(label_y_s_torch, axis=0)
    label_y_t_torch = tuple(label_y_t_torch)
    label_y_t_torch = torch.cat(label_y_t_torch, axis=0)

    scatter(TSNE(random_state=RS).fit_transform(domain_x1_torch), domain_y1_torch,
            [i for i in range(solver.num_domains + 1)], (8, 8), 'Main network features')
    plt.savefig(plot_domains[0], dpi=120)

    scatter(TSNE(random_state=RS).fit_transform(domain_x2_torch), domain_y2_torch,
            [i for i in range(solver.num_domains + 1)], (8, 8), 'DP network layer')
    plt.savefig(plot_domains[1], dpi=120)

    final = get_concat_h(Image.open(plot_domains[0]), Image.open(plot_domains[1]))
    final.save(plot_domains[2])
    print('latent domain plots saved in : ', plot_domains[2])

    os.remove(plot_domains[0])
    os.remove(plot_domains[1])

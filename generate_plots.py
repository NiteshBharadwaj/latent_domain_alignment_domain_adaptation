import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
from sklearn.manifold import TSNE
import sys
#torchvision.utils.save_image(((images + 1)/2).detach(), f'./data/generated.png')




def plot_tsne1(X, labels, source, target, labelsOfInterest, source_examples):
    labels = [element.item() for element in labels.flatten()]
    print(source_examples)
    print(X.size())
    #print(set(labels))
    import seaborn as sns
    X = X.view(X.size()[0], -1)
    #print(labels)
    tsne = TSNE(perplexity=30, n_components=2, init="pca", n_iter=5000)

    X_embedded = tsne.fit_transform(X.cpu().numpy())
    import seaborn as sns1
    sns1.set(rc={'figure.figsize':(5,5)})
    palette = sns1.color_palette("bright", len(labelsOfInterest))
    sns_plot1 = sns1.scatterplot(X_embedded[:source_examples,0], X_embedded[:source_examples,1], hue=labels[:source_examples], palette=palette, sizes=(2, 2), legend=False)
    sns_plot1.figure.savefig(source)
    import seaborn as sns2
    sns2.set(rc={'figure.figsize':(5,5)})
    palette = sns2.color_palette("bright", len(labelsOfInterest))
    sns_plot2 = sns2.scatterplot(X_embedded[source_examples:,0], X_embedded[source_examples:,1], hue=labels[source_examples:], palette=palette, sizes=(2, 2), legend=False)
    sns_plot2.figure.savefig(target)


def plot_tsne2(X, labels, tags, source_location, target_location, labelsDict):
    d = {
        '0' : 's',
        '1' : 't'
    }
    labels = [element.item() for element in labels.flatten()]
    X = X.view(X.size()[0], -1)
    tsne = TSNE(perplexity=30, n_components=2, init="pca", n_iter=3000)

    X = tsne.fit_transform(X.cpu().numpy())
    x_min, x_max = np.min(X, 0), np.max(X, 0)
    X = (X - x_min) / (x_max - x_min)

    # Plot colors numbers
    from matplotlib import pyplot as plt
    plt.figure(figsize=(10,10))
    ax = plt.subplot(111)

    for i in range(X.shape[0]):
        # plot colored number
        domain = d[str(tags[i])]
        if(domain == 's'):
            plt.text(X[i, 0], X[i, 1], 'o',
                     color=labelsDict[labels[i]],
                     fontdict={'weight': 'bold', 'size': 9})

    plt.xticks([]), plt.yticks([])
    plt.savefig(source_location)

    from matplotlib import pyplot as plt
    plt.figure(figsize=(10,10))
    ax = plt.subplot(111)

    for i in range(X.shape[0]):
        # plot colored number
        domain = d[str(tags[i])]
        if(domain == 't'):
            plt.text(X[i, 0], X[i, 1], 'o',
                     color=labelsDict[labels[i]],
                     fontdict={'weight': 'bold', 'size': 9})

    plt.xticks([]), plt.yticks([])
    plt.savefig(target_location)

def generate_plots(solver, epoch, split, plot_before_source, plot_before_target, plot_after_source, plot_after_target, save_model=False):
    solver.G.eval()
    solver.C1.eval()
    test_loss = 0
    correct1 = 0
    size = 0
    feature_all = np.array([])
    label_all = []


    # if split=='val':
    #     which_dataset = solver.dataset_valid
    #     data_symbol = 'T'
    #     data_label_symbol = 'T_label'
    # elif split=='train':
    #     which_dataset = solver.datasets
    #     data_symbol = 'S'
    #     data_label_symbol = 'S_label'
    # elif split=='test':
    #     which_dataset = solver.datasets
    #     data_symbol = 'S'
    #     data_label_symbol = 'S_label'

    which_dataset = solver.datasets

    before = []
    after = []
    all_labels = []
    s_tags = []
    tags = []
    #labelsOfInterest = [2,8,15,25]
    labelsDict = {
        2 : 'r',
        8 : 'g',
        15 : 'b',
        25 : 'y'
    }
    labelsOfInterest = list(labelsDict.keys())
    source_examples = 0

    with torch.no_grad():
        for batch_idx, data in enumerate(which_dataset):

            img = data['S']
            label = data['S_label'].long()

            # img_t = Variable(data['T'].cuda())
            # img_s = Variable(data['S'].cuda())
            # label_s = Variable(data['S_label'].long().cuda())
            # solver.generate_cluster_images(img_s, img_t, label_s)
            # continue

            #source_examples += label.size()[0]
            #print(label.size())

            for l in labelsOfInterest:
                #[element.item() for element in labels.flatten()]
                idx = (label == l).nonzero().flatten()
                #idx = [element.item() for element in (label == l).nonzero().flatten()] 
                if(len(idx) > 0):
                    cur_x = img[idx, :, :, :]
                    cur_y = label[idx,]
                    before.append(cur_x)
                    all_labels.append(cur_y)
                    source_examples += cur_x.size()[0]
                    tags.append(torch.zeros((cur_x.size()[0])).type(torch.LongTensor))

            # before.append(img)
            # all_labels.append(label)

            img, label = img.cuda(), label.long().cuda()
            img, label = Variable(img, volatile=True), Variable(label)
            feat, _ = solver.G(img)

            for l in labelsOfInterest:
                #[element.item() for element in labels.flatten()]
                idx = (label == l).nonzero().flatten()
                #idx = [element.item() for element in (label == l).nonzero().flatten()] 
                if(len(idx) > 0):
                    cur_x = feat[idx, :]
                    cur_y = label[idx,]
                    after.append(cur_x)
                    #all_labels.append(cur_y)            


            #after.append(feat)
            # print('feature.shape:{}'.format(feat.shape))

            if batch_idx == 0:
                label_all = label.data.cpu().numpy().tolist()

                # feature_all = feat.data.cpu().numpy()
            else:
                # feature_all = np.ma.row_stack((feature_all, feat.data.cpu().numpy()))
                # feature_all = feature_all.data
                label_all = label_all + label.data.cpu().numpy().tolist()

            # print(feat.shape)

            output1 = solver.C1(feat)

            test_loss += nn.CrossEntropyLoss()(output1, label).data.item()
            pred1 = output1.data.max(1)[1]
            k = label.data.size()[0]
            correct1 += pred1.eq(label.data).cpu().sum()
            size += k

    #sys.exit()
    with torch.no_grad():
        for batch_idx, data in enumerate(which_dataset):

            img = data['T']
            label = data['T_label'].long()
            #print(label.size())

            for l in labelsOfInterest:
                #[element.item() for element in labels.flatten()]
                idx = (label == l).nonzero().flatten()
                #idx = [element.item() for element in (label == l).nonzero().flatten()] 
                if(len(idx) > 0):
                    cur_x = img[idx, :, :, :]
                    cur_y = label[idx,]
                    before.append(cur_x)
                    all_labels.append(cur_y)
                    tags.append(torch.ones((cur_x.size()[0])).type(torch.LongTensor))

            # before.append(img)
            # all_labels.append(label)

            img, label = img.cuda(), label.long().cuda()
            img, label = Variable(img, volatile=True), Variable(label)
            feat, _ = solver.G(img)

            for l in labelsOfInterest:
                #[element.item() for element in labels.flatten()]
                idx = (label == l).nonzero().flatten()
                #idx = [element.item() for element in (label == l).nonzero().flatten()] 
                if(len(idx) > 0):
                    cur_x = feat[idx, :]
                    cur_y = label[idx,]
                    after.append(cur_x)
                    #all_labels.append(cur_y)            


            #after.append(feat)
            # print('feature.shape:{}'.format(feat.shape))

            if batch_idx == 0:
                label_all = label.data.cpu().numpy().tolist()

                # feature_all = feat.data.cpu().numpy()
            else:
                # feature_all = np.ma.row_stack((feature_all, feat.data.cpu().numpy()))
                # feature_all = feature_all.data
                label_all = label_all + label.data.cpu().numpy().tolist()

            # print(feat.shape)

            output1 = solver.C1(feat)

            test_loss += nn.CrossEntropyLoss()(output1, label).data.item()
            pred1 = output1.data.max(1)[1]
            k = label.data.size()[0]
            correct1 += pred1.eq(label.data).cpu().sum()
            size += k

    #sys.exit()
    before = tuple(before)
    after = tuple(after)
    all_labels = tuple(all_labels)
    before = torch.cat(before,axis=0)
    after = torch.cat(after,axis=0)
    all_labels = torch.cat(all_labels,axis=0)
    tags = tuple(tags)
    tags = torch.cat(tags,axis=0)
    print(tags.size())
    tags = [element.item() for element in tags.flatten()]

    # plot_tsne(before,all_labels,plot_before_source,plot_before_target,labelsOfInterest,source_examples)
    # plot_tsne(after,all_labels,plot_after_source,plot_after_target,labelsOfInterest,source_examples)
    #plot_tsne2(before,all_labels,tags,plot_before_target)
    plot_tsne2(before,all_labels,tags,plot_before_source,plot_before_target,labelsDict)
    plot_tsne2(after,all_labels,tags,plot_after_source,plot_after_target,labelsDict)

#     # np.savez('result_plot_sv_t', feature_all, label_all )
#     test_loss = test_loss / (size + 1e-6)

#     print('\n{} set: Average loss: {:.4f}, Accuracy C1: {}/{} ({:.0f}%)  \n'.format(split,test_loss, correct1, size,
#                                                                                            100. * correct1 / (
#                                                                                                        size + 1e-6)))
#     best = False
#     if split=='val' and size!=0:
#         if save_model and epoch % solver.save_epoch == 0 and test_loss < solver.best_loss:
#             print('Saving best model','%s/%s_model_best.pth' % (solver.checkpoint_dir, solver.target))
#             checkpoint = {}
#             checkpoint['G_state_dict'] = solver.G.state_dict()
#             checkpoint['C1_state_dict'] = solver.C1.state_dict()
#             checkpoint['C2_state_dict'] = solver.C2.state_dict()
#             checkpoint['DP_state_dict'] = solver.DP.state_dict()

#             checkpoint['G_state_dict_opt'] = solver.opt_g.state_dict()
#             checkpoint['C1_state_dict_opt'] = solver.opt_c1.state_dict()
#             checkpoint['C2_state_dict_opt'] = solver.opt_c2.state_dict()
#             checkpoint['DP_state_dict_opt'] = solver.opt_dp.state_dict()
#             torch.save(checkpoint, '%s/%s_model_best.pth' % (solver.checkpoint_dir, solver.target))

#         if test_loss < solver.best_loss and size!=0:
#             solver.best_loss = test_loss
#             best = True

#         if record_file:
#             record = open(record_file, 'a')
#             print('recording %s', record_file)
#             record.write('%s\n' % (float(correct1) / (size + 1e-6)))
#             record.close()
#     elif split=='test' and size!=0:
#         if record_file:
#             record = open(record_file, 'a')
#             print('recording %s', record_file)
#             record.write('%s\n' % (float(correct1) / (size + 1e-6)))
#             record.close()
#     return best

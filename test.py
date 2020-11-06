import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np

def test(solver, epoch, split, record_file=None, save_model=False):
    solver.G.eval()
    solver.C1.eval()
    test_loss = 0
    correct1 = 0
    size = 0
    feature_all = np.array([])
    label_all = []


    if split=='val':
        which_dataset = solver.dataset_valid
        data_symbol = 'T'
        data_label_symbol = 'T_label'
    elif split=='train':
        which_dataset = solver.datasets
        data_symbol = 'S'
        data_label_symbol = 'S_label'
    elif split=='test':
        which_dataset = solver.dataset_test
        data_symbol = 'T'
        data_label_symbol = 'T_label'

    with torch.no_grad():
        for batch_idx, data in enumerate(which_dataset):

            img = data[data_symbol]
            label = data[data_label_symbol]

            img, label = img.cuda(), label.long().cuda()
            img, label = Variable(img, volatile=True), Variable(label)
            domain_labels_target = torch.zeros(img.shape[0],solver.num_domains,dtype=torch.float32).cuda()
            domain_labels_target = torch.cat((domain_labels_target,torch.ones(img.shape[0],1,dtype=torch.float32).cuda()),1) 
            feat, _ = solver.G(img,domain_labels_target)
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
    # np.savez('result_plot_sv_t', feature_all, label_all )
    test_loss = test_loss / (size + 1e-6)
    
    print('\n{} set: Average loss: {:.4f}, Accuracy C1: {}/{} ({:.06f}%)  \n'.format(split,test_loss, correct1, size,
                                                                                           100. * correct1 / (
                                                                                                       size + 1e-6)))
    test_acc =  100. * correct1 / (size + 1e-6)
    best = False

    bool_to_check = (test_loss < solver.best_loss)
    if solver.args.model_sel_acc == 1:
        bool_to_check = (test_acc > solver.best_acc)

    if split=='val' and size!=0:
#         if save_model and epoch % solver.save_epoch == 0 and test_acc > solver.best_acc:
        if save_model and epoch % solver.save_epoch == 0 and bool_to_check:
            print('Saving best model','%s/%s_model_best.pth' % (solver.checkpoint_dir, solver.target))
            checkpoint = {}
            checkpoint['G_state_dict'] = solver.G.state_dict()
            checkpoint['C1_state_dict'] = solver.C1.state_dict()
            checkpoint['C2_state_dict'] = solver.C2.state_dict()
            checkpoint['DP_state_dict'] = solver.DP.state_dict()

            checkpoint['G_state_dict_opt'] = solver.opt_g.state_dict()
            checkpoint['C1_state_dict_opt'] = solver.opt_c1.state_dict()
            checkpoint['C2_state_dict_opt'] = solver.opt_c2.state_dict()
            checkpoint['DP_state_dict_opt'] = solver.opt_dp.state_dict()
            torch.save(checkpoint, '%s/%s_model_best.pth' % (solver.checkpoint_dir, solver.target))

#         if test_acc > solver.best_acc and size!=0:
#             solver.best_acc = test_acc
#             best = True
#        if test_loss < solver.best_loss and size!=0:
#            solver.best_loss = test_loss
#            best = True

        if bool_to_check and size!=0:
            if solver.args.model_sel_acc == 1:
                solver.best_acc = test_acc
            else:
                solver.best_loss = test_loss
#           solver.best_loss = test_loss
            best = True

        if record_file:
            record = open(record_file, 'a')
            print('recording %s', record_file)
            record.write('%s\n' % (float(correct1) / (size + 1e-6)))
            record.close()
    elif split=='test' and size!=0:
        if record_file:
            record = open(record_file, 'a')
            print('recording %s', record_file)
            record.write('%s\n' % (float(correct1) / (size + 1e-6)))
            record.close()
    return best

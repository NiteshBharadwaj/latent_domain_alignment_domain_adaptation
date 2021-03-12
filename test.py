import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import time

def test(solver, epoch, split, record_file=None, save_model=False, temperature_scaling = False, use_g_t=False):
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
    classwise_acc = [0]*solver.num_classes
    classwise_sum = [0]*solver.num_classes
    logits_list = []
    labels_list = []
    with torch.no_grad():
        for batch_idx, data in enumerate(which_dataset):
            start = time.time()
            img = data[data_symbol]
            label = data[data_label_symbol]

            img, label = img.cuda(), label.long().cuda()
            img, label = Variable(img, volatile=True), Variable(label)
            feat, _, _ = solver.G(img)
            if use_g_t:
                feat,_,_ = solver.G_T(img)

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
            if use_g_t:
                output1 = solver.C1_T(feat)
            logits_list.append(output1)
            labels_list.append(label)

            test_loss += nn.CrossEntropyLoss()(output1, label).data.item()
            pred1 = output1.data.max(1)[1]
            k = label.data.size()[0]
            correct1 += pred1.eq(label.data).cpu().sum()
            size += k
            for class_id in range(solver.num_classes):
                idxes = label==class_id
                classwise_acc[class_id] += pred1[idxes].eq(label[idxes].data).cpu().sum()
                classwise_sum[class_id] += (idxes*1==1).sum()
            end = time.time()
            #print("Time taken for testing batch : ", end-start)
    # np.savez('result_plot_sv_t', feature_all, label_all )
    test_loss = test_loss / (size + 1e-6)
    if temperature_scaling:
        criterion = nn.CrossEntropyLoss()
        logits_list = torch.cat(logits_list).cuda()
        labels_list = torch.cat(labels_list).cuda()
        def _eval():
            loss = criterion(solver.T_scaling(logits_list), labels_list)
            loss.backward()
            return loss
        solver.temperature_optim.zero_grad()
        solver.temperature_optim.step(_eval)
        print("Temperature updated to {}".format(solver.temperature.cpu().item()))
    else:
        del logits_list
        del labels_list


    print('\n{} set: Average loss: {:.4f}, Accuracy C1: {}/{} ({:.06f}%)  \n'.format(split,test_loss, correct1, size,
                                                                                           100. * correct1 / (
                                                                                                       size + 1e-6)))
    test_acc =  100. * correct1 / (size + 1e-6)
    for class_id in range(solver.num_classes):
        class_acc = 100. * classwise_acc[class_id]/(classwise_sum[class_id]+1e-6)
        print('{} set: Average Accuracy Class Id {}: {}'.format(split,class_id,class_acc))
    best = False
    bool_to_check = (test_loss < solver.best_loss)
    if solver.args.model_sel_acc == 1:
        bool_to_check = (test_acc > solver.best_acc)
    
    if split=='val' and size!=0:
#         if save_model and epoch % solver.save_epoch == 0 and test_acc > solver.best_acc:
        if save_model and epoch % solver.save_epoch == 0 and bool_to_check and not use_g_t:
            print('Saving best model','%s/%s_model_best.pth' % (solver.checkpoint_dir, solver.target))
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

#         if test_acc > solver.best_acc and size!=0:
#             solver.best_acc = test_acc
#             best = True
#        if test_loss < solver.best_loss and size!=0:
#            solver.best_loss = test_loss
#            best = True

        if bool_to_check and size!=0 and not use_g_t:
            if solver.args.model_sel_acc == 1:
                solver.best_acc = test_acc
            else:
                solver.best_loss = test_loss
            #solver.best_loss = test_loss
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
    return best, test_acc

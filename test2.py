import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import time

def test2(solver, epoch, split, record_file=None, save_model=False, temperature_scaling = False, use_g_t=False):
    solver.G.eval()
    solver.C1.eval()
    loader = solver.dataset_test10
    with torch.no_grad():
        start_test = True
        iter_val = [iter(loader['val'+str(i)]) for i in range(10)]
        for i in range(len(loader['val0'])):
            data = [iter_val[j].next() for j in range(10)]
            inputs = [data[j][0] for j in range(10)]
            labels = data[0][1]
            for j in range(10):
                inputs[j] = inputs[j].cuda()
            labels = labels.cuda()
            #labels = labels[:, 0]
            outputs = []
            for j in range(10):
                feat,_,_ = solver.G(inputs[j])
                output = solver.C1(feat)
                outputs.append(output)
            outputs = sum(outputs)
            if start_test:
                all_output = outputs.data.float()
                all_label = labels.data.float()
                start_test = False
            else:
                all_output = torch.cat((all_output, outputs.data.float()), 0)
                all_label = torch.cat((all_label, labels.data.float()), 0)
        _, predict = torch.max(all_output, 1)
        test_acc = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])
    print("{} epoch {}: Test Acc: {}".format(split,epoch, test_acc))
    best = False
    bool_to_check = (test_acc > solver.best_acc)

    if split=='val':
#         if save_model and epoch % solver.save_epoch == 0 and test_acc > solver.best_acc:
        if save_model and epoch % solver.save_epoch == 0 and bool_to_check and not use_g_t:
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

        if bool_to_check and not use_g_t:
            solver.best_acc = test_acc

            #solver.best_loss = test_loss
            best = True

        if record_file:
            record = open(record_file, 'a')
            print('recording %s', record_file)
            record.write('%s\n' % test_acc)
            record.close()
    elif split=='test':
        if record_file:
            record = open(record_file, 'a')
            print('recording %s', record_file)
            record.write('%s\n' % test_acc)
            record.close()
    return best, test_acc

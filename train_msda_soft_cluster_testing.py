import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision

def train_MSDA_soft(solver, epoch, classifier_disc=True, record_file=None):
    solver.G.train()
    solver.C1.train()
    solver.C2.train()
    solver.DP.train()
    #torch.cuda.manual_seed(1)

    batch_idx_g = 0
    classwise_dataset_iterator = iter(solver.classwise_dataset)
    for batch_idx, data in enumerate(solver.datasets):
        batch_idx_g = batch_idx
        img_t = Variable(data['T'].cuda())
        img_s = Variable(data['S'].cuda())
        label_s = Variable(data['S_label'].long().cuda())
        if img_s.size()[0] < solver.batch_size or img_t.size()[0] < solver.batch_size:
            break

        classwise_data = next(classwise_dataset_iterator)
        img_s_cl = Variable(classwise_data['S'].cuda())

        solver.reset_grad()

        loss_s_c1, loss_s_c2, loss_msda, entropy_loss, kl_loss, domain_prob = solver.loss_soft_all_domain(img_s, img_t, label_s, epoch, img_s_cl)
        if not classifier_disc:
            loss_s_c2 = loss_s_c1
#        if epoch > 2:
#            loss = entropy_loss + kl_loss
#        else:
#            loss = loss_s_c1 + loss_msda + loss_s_c2 + entropy_loss + kl_loss
        loss = loss_s_c1 + loss_s_c2 + loss_msda + entropy_loss + kl_loss

        #torchvision.utils.save_image(img_s, "source_images.png")
        #torchvision.utils.save_image(img_s_cl, "classwise_images.png")

        loss.backward()
        clip_value = 1.0

#        for param_group in solver.G.param_groups:
#            print("LR opt_g", param_group['lr'])
#        for param_group in solver.C1.param_groups:
#            print("LR opt_c1", param_group['lr'])
#        for param_group in solver.C2.param_groups:
#            print("LR opt_c2", param_group['lr'])
#        for param_group in solver.DP.param_groups:
#            print("LR opt_dp", param_group['lr'])

        torch.nn.utils.clip_grad_norm(solver.G.parameters(), clip_value) 
        torch.nn.utils.clip_grad_norm(solver.DP.parameters(), clip_value)
        solver.opt_g.step()
        solver.opt_dp.step()
        loss_dis = loss * 0  # For printing purpose, it's reassigned if classifier_disc=True
        if classifier_disc:
            solver.reset_grad()
            loss_s_c1, loss_s_c2, loss_msda, entropy_loss, kl_loss, domain_prob = solver.loss_soft_all_domain(img_s, img_t, label_s, epoch, img_s_cl)

            feat_t, conv_feat_t = solver.G(img_t)
            output_t1 = solver.C1(feat_t)
            output_t2 = solver.C2(feat_t)

            loss_s = loss_s_c1 + loss_msda + loss_s_c2 + entropy_loss + kl_loss
            loss_dis = solver.discrepancy(output_t1, output_t2)
            loss = loss_s - loss_dis
            loss.backward()

            torch.nn.utils.clip_grad_norm(solver.C1.parameters(), clip_value)
            torch.nn.utils.clip_grad_norm(solver.C2.parameters(), clip_value)
            solver.opt_c1.step()
            solver.opt_c2.step()
            solver.reset_grad()

            for i in range(solver.args.num_k):
                feat_t, conv_feat_t = solver.G(img_t)
                output_t1 = solver.C1(feat_t)
                output_t2 = solver.C2(feat_t)
                loss_dis = solver.discrepancy(output_t1, output_t2)
                loss_dis.backward()
                torch.nn.utils.clip_grad_norm(solver.G.parameters(), clip_value)
                solver.opt_g.step()
                solver.reset_grad()

        #for i in range(img_s.size(0)):
        #   torchvision.utils.save_image(img_s[i, :, :, :], 'source_imgs{}.png'.format(i))
        #for i in range(img_s_cl.size(0)):
        #   torchvision.utils.save_image(img_s_cl[i, :, :, :], 'classwise_imgs{}.png'.format(i))
        if epoch%3==0 and batch_idx%100==0:
            #print(domain_prob)
            #torchvision.utils.save_image(img_s[:32,:,:,:], "clus/source_images_{}_{}.png".format(epoch,batch_idx), normalize=True)
            #torchvision.utils.save_image(img_s_cl[:32,:,:,:], "clus/classwise_images_{}_{}.png".format(epoch,batch_idx), normalize=True)
            max_idxs = domain_prob.argmax(1)
            if (max_idxs==0).any():
                torchvision.utils.save_image(img_s[max_idxs==0,:,:,:], "clus/source_images_cl0_{}_{}.png".format(epoch,batch_idx), normalize=True)
            else:
                print("No images in Cluster 0 _{}_{}".format(epoch,batch_idx))
            if (max_idxs==1).any():
                torchvision.utils.save_image(img_s[max_idxs==1,:,:,:], "clus/source_images_cl1_{}_{}.png".format(epoch, batch_idx), normalize=True)
            else:
                print("No images in Cluster 1 _{}_{}".format(epoch,batch_idx))
        if batch_idx % solver.interval == 0:
            print \
                ('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss1: {:.6f}\t Loss2: {:.6f}\t Loss_mmd: {:.6f}\t Loss_entropy: {:.6f}\t kl_loss: {:.6f}'.format(
                epoch, batch_idx, 100, 100. * batch_idx / 70000, loss_s_c1.data.item(), loss_s_c2.data.item(), loss_msda.data.item(), entropy_loss.data.item(), kl_loss.data.item()))
    return batch_idx_g

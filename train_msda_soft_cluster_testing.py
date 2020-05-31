import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision
import os 

cluster_batch = None
svhn_batch=None
mnist_batch=None
usps_batch =None
syn_batch=None
classwise_batch = None

def switch_bn(model, on):
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d) or isinstance(m,nn.BatchNorm1d):
            if on:
                m.train()
            else:
                m.eval()

def train_MSDA_soft(solver, epoch, classifier_disc=True, record_file=None):
    global cluster_batch
    global svhn_batch
    global mnist_batch
    global usps_batch
    global syn_batch
    global classwise_batch
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
        if (solver.args.clustering_only and cluster_batch is None):
            cluster_batch = img_s
            svhn_batch = Variable(next(iter(solver.dataset_svhn))['T'].cuda())
            usps_batch = Variable(next(iter(solver.dataset_usps))['T'].cuda())
            syn_batch = Variable(next(iter(solver.dataset_syn))['T'].cuda())
            mnist_batch = Variable(next(iter(solver.dataset_mnist))['T'].cuda())

        label_s = Variable(data['S_label'].long().cuda())
        if img_s.size()[0] < solver.batch_size or img_t.size()[0] < solver.batch_size:
            break

        classwise_data = next(classwise_dataset_iterator)
        img_s_cl = Variable(classwise_data['S'].cuda())
        if (solver.args.clustering_only and classwise_batch is None):
            classwise_batch = img_s_cl

#        switch_bn(solver.DP,True)
        solver.reset_grad()

        loss_s_c1, loss_s_c2, loss_msda_nc2, loss_msda_nc1, entropy_loss, kl_loss, domain_prob = solver.loss_soft_all_domain(img_s, img_t, label_s, epoch, img_s_cl)
        if not classifier_disc:
            loss_s_c2 = loss_s_c1
        loss = loss_s_c1 + loss_s_c2 + loss_msda_nc2 + loss_msda_nc1 + entropy_loss + kl_loss

        loss.backward()
        clip_value = 1.0

        torch.nn.utils.clip_grad_norm(solver.G.parameters(), clip_value)
        torch.nn.utils.clip_grad_norm(solver.C1.parameters(), clip_value)
        if classifier_disc:
            torch.nn.utils.clip_grad_norm(solver.C2.parameters(), clip_value)
        torch.nn.utils.clip_grad_norm(solver.DP.parameters(), clip_value)

        #solver.opt_g.step()
        if not solver.args.clustering_only:
            solver.opt_g.step()
            solver.opt_c1.step()
            solver.opt_c2.step()
        solver.opt_dp.step()

 #       switch_bn(solver.DP,False)
 #       solver.reset_grad()
 #       _, _, _, _, kl_loss, domain_prob = solver.loss_soft_all_domain(img_s, img_t, label_s, epoch, img_s_cl)
        
 #       clip_value = 1.0

 #       torch.nn.utils.clip_grad_norm(solver.DP.parameters(), clip_value)
 #       solver.opt_dp.step()
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
        if solver.args.clustering_only and epoch%3==0 and batch_idx%100==0:
            solver.G.eval()
            solver.C1.eval()
            solver.C2.eval()
            solver.DP.eval()
            _, _, _, _, _,_, domain_prob = solver.loss_soft_all_domain(cluster_batch, img_t, label_s, epoch, img_s_cl)
            _, _, _, _, _,_, domain_prob_cw = solver.loss_soft_all_domain(classwise_batch, img_t, label_s, epoch, img_s_cl)
            print('Classwise Probs',domain_prob_cw.mean(0))
            
            directory = "clusters_data-{}-num_domain-{}-batch_size-{}-kl_wt-{}-entropy_wt-{}-lr-{}-seed-{}-target-{}-clustering_only-{}/".format(solver.args.data, solver.args.num_domain, solver.args.batch_size, solver.args.kl_wt, solver.args.entropy_wt, solver.args.lr, solver.args.seed, solver.args.target, solver.args.clustering_only)
            if not os.path.exists(directory):            
                os.makedirs(directory)

            if batch_idx==0:
                print(domain_prob)

            torchvision.utils.save_image(img_s[:32,:,:,:], "{}/source_images_{}_{}.png".format(directory,epoch,batch_idx), normalize=True)
            torchvision.utils.save_image(img_s_cl[:32,:,:,:], "{}/classwise_images_{}_{}.png".format(directory,epoch,batch_idx), normalize=True)
            max_idxs = domain_prob.argmax(1)
            for ii in range(solver.args.num_domain):
                if (max_idxs==ii).any():
                    torchvision.utils.save_image(cluster_batch[max_idxs==ii,:,:,:], "{}/source_images_cl{}_{}_{}.png".format(directory,ii,epoch,batch_idx), normalize=True)
                else:
                    print("No images in Cluster {} _{}_{}".format(ii,epoch,batch_idx))

            if batch_idx==0:
                torchvision.utils.save_image(mnist_batch, "{}/mnist_images_{}_{}.png".format(directory,epoch,batch_idx), normalize=True)
                _, _, _, _, _,_, domain_prob_svhn = solver.loss_soft_all_domain(svhn_batch, img_t, label_s, epoch, img_s_cl)
                print('SVHN Probs',domain_prob_svhn.mean(0))
                _, _, _, _, _,_, domain_prob_usps = solver.loss_soft_all_domain(usps_batch, img_t, label_s, epoch, img_s_cl)
                print('USPS Probs',domain_prob_usps.mean(0))
                _, _, _, _, _,_, domain_prob_syn = solver.loss_soft_all_domain(syn_batch, img_t, label_s, epoch, img_s_cl)
                print('SYN Probs',domain_prob_syn.mean(0))
                _, _, _, _, _,_, domain_prob_mnist = solver.loss_soft_all_domain(mnist_batch, img_t, label_s, epoch, img_s_cl)
                print('MNIST Probs',domain_prob_mnist.mean(0))
            solver.G.train()
            solver.C1.train()
            solver.C2.train()
            solver.DP.train()
        if batch_idx % solver.interval == 0:
            print \
                ('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss1: {:.6f}\t Loss2: {:.6f}\t Loss_mmd: {:.6f}\t Loss_entropy: {:.6f}\t kl_loss: {:.6f}'.format(
                epoch, batch_idx, 100, 100. * batch_idx / 70000, loss_s_c1.data.item(), loss_s_c2.data.item(), loss_msda_nc2.data.item(), entropy_loss.data.item(), kl_loss.data.item()))
    return batch_idx_g


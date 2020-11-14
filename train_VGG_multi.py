#coding:utf-8
'''Train datasets with PyTorch.'''
from __future__ import print_function

import numpy 
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms
from tensorboardX import SummaryWriter
import os
import argparse
from PIL import Image
import random
#from torch.models import *
#from torch.utils import progress_bar
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader

from nets import vgg19_bn_fx as VGG
#from nets import vgg19_bn as VGG
#from resnet import resnet50 as VGG
from nets import Encoder
from dataset_nature_multi import config_VGG, SingleDataset_VGG, SingleDataset_GAN
from dataset_nature_multi import config as config_ele



class Classify(object):
    def __init__(self,args,config=config_VGG):
        self.args = args
        self.config = config_VGG
        self.attributes = args.attributes
        self.n_attributes = len(self.attributes)
        image_f = open("../datasets/multipie/images.list")

        if not os.path.isdir(self.config.checkpoint):
            os.mkdir(self.config.checkpoint)
        self.f_acc = open(self.config.checkpoint + '/acc.txt', 'w')

        # deal with the dataloader
        self.im_names = []
        self.labels = []
        self.test_im_names = []
        self.test_labels = []

        for line in image_f:
            pic_name = line.strip().split()[0]
            if pic_name[:4] == 'trai': #train
                self.im_names.append(line.strip().split()[0])#[:-4]+'.png')
                self.labels.append(int(line.strip().split()[1]))
            else:
                self.test_im_names.append(line.strip().split()[0])#[:-4]+'.png')
                self.test_labels.append(int(line.strip().split()[1]))

        self.dataset_train_raw = SingleDataset_VGG(self.im_names, self.labels, self.config, 'train', 'raw')
        self.dataset_test = SingleDataset_VGG(self.test_im_names, self.test_labels, self.config, 'test','raw') # test 图片只用真实图片？？
        
        self.train_loader_raw = DataLoader(dataset = self.dataset_train_raw, batch_size = self.config.ncwh[0], shuffle = self.config.shuffle, num_workers = self.config.num_workers) 
        self.test_loader = DataLoader(dataset = self.dataset_test, batch_size = self.config.ncwh[0], shuffle = self.config.shuffle, num_workers = self.config.num_workers)

        self.gpu = args.gpu
        os.environ["CUDA_VISIBLE_DEVICES"] = self.gpu[0]
        self.mode = args.mode
        self.gan_raw = self.config.gan_raw
        self.resume = args.resume
        self.resume_dir = args.resume_dir
        self.use_cuda = torch.cuda.is_available()
        self.checkpoint = self.config.checkpoint

        if self.resume:  
            print('==> Resuming from checkpoint..')
            if os.path.isdir(self.resume_dir):
                print('Checkpoint directory found!')
            else:
                print('no Checkpoint directory found!')  
            checkpoint = torch.load(self.resume_dir+'/max_ckpt.t7')

            self.net = VGG(6,pretrained=False)
            self.net.load_state_dict(checkpoint['net'])
            self.best_acc = float(checkpoint['acc'])

            print("Success resume the max checkpoing")
            print("Bese acc is:",self.best_acc)
            self.start_iter = checkpoint['step']
        else:
            print('==> Building model..')
            self.net = VGG(6,pretrained=True)

        if self.mode == 'train':
            self.best_acc = 0.  # best test accuracy
            self.start_iter = 0
        if self.use_cuda:
            with torch.cuda.device(0):
                # move param and buffer to GPU
                self.net.cuda()
                    # parallel use GPU
            if len(self.args.gpu)>1 :
                self.net = torch.nn.DataParallel(self.net, device_ids=range(len(self.gpu)))
             # speed up slightly
            cudnn.benchmark = True

        self.criterion = nn.CrossEntropyLoss(size_average=True)
        self.optimizer = torch.optim.SGD(self.net.parameters(), lr=self.config.lr, momentum=0.9, weight_decay=5e-4)
        self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=self.config.step_size, gamma=self.config.gamma)

        self.sum_step = 0
        self.writer = SummaryWriter(self.config.log_dir)


    def tensor2var(self, tensors, volatile=False):
        if not hasattr(tensors, '__iter__'): tensors = [tensors]
        out = []
        for tensor in tensors:
            if len(self.gpu):
                tensor = tensor.cuda(0)
            out.append(tensor)
        if len(out) == 1:
            return out[0]
        else:
            return out

    def get_attr_chs(self, encodings, attribute_id):
        num_chs = encodings.size(1)
        per_chs = float(num_chs) / self.n_attributes
        start = int(numpy.rint(per_chs * attribute_id))
        end = int(numpy.rint(per_chs * (attribute_id + 1)))
        # return encodings[:, start:end]
        return encodings.narrow(1, start, end-start)

    def img_denorm(self, img, scale=255):
        return (img + 1) * scale / 2.

    def train(self, epoch):
        # switch to train mode
        self.net.train()

        gan_num = 0  # train gan data times (every 6)
        self.gan_im_names = []
        self.gan_labels = []

        # load new gan-pic every time
        f_gan = open("../datasets/multipie/images_gan_nature.list")
        if self.args.ablation: # ablation study
            ablation_num = 2
            f_gan_lines = f_gan.readlines()
            random.shuffle(f_gan_lines)
            for item in f_gan_lines[:int(ablation_num*36744/6)]:
                self.gan_im_names.append(item.strip().split()[0])
                label = item.strip().split()[1]
                self.gan_labels.append(int(label))
        else:
            for item in f_gan:
                self.gan_im_names.append(item.strip().split()[0])
                label = item.strip().split()[1]
                self.gan_labels.append(int(label))
        self.dataset_train_gan = SingleDataset_VGG(self.gan_im_names, self.gan_labels, self.config, 'train', 'gan')
        self.train_loader_gan = DataLoader(dataset=self.dataset_train_gan, batch_size=self.config.ncwh[0],shuffle=self.config.shuffle, num_workers=self.config.num_workers)

        
        # load temp Encoder
        self.Enc = Encoder()
        ckpt_file_enc = os.path.join(config_ele.model_dir, 'Enc_iter_temp.pth')
        print(ckpt_file_enc)
        assert os.path.exists(ckpt_file_enc)
        self.Enc.load_state_dict(torch.load(ckpt_file_enc), strict=False)
        self.Enc.eval()
        self.Enc.cuda()
        print("Load Encoder successful!")

        if 1==1:
            for train_step, (inputs_, _ , targets_) in enumerate(self.train_loader_raw):
                if self.args.multi_add_gan:
                    if epoch < 2.5*self.config.train_vgg_times:
                        self.train_loader_add = self.train_loader_raw
                        self.gan_add = 1
                    if epoch >=2.5*self.config.train_vgg_times and epoch < 5*self.config.train_vgg_times:
                        self.train_loader_add = self.train_loader_gan
                        self.gan_add = self.config.gan_raw/3
                    if epoch >= 5*self.config.train_vgg_times and epoch < 7.5*self.config.train_vgg_times:
                        self.train_loader_add = self.train_loader_gan
                        self.gan_add = self.config.gan_raw/2
                    if epoch >= 7.5*self.config.train_vgg_times:
                        self.train_loader_add = self.train_loader_gan
                        self.gan_add = self.config.gan_raw
                else:
                    self.train_loader_add = self.train_loader_gan
                    self.gan_add = self.config.gan_raw

                self.inputs = inputs_ #Variable(inputs_)
                self.targets = targets_ #Variable(torch.Tensor(targets_))
                if self.use_cuda:
                    self.inputs, self.targets = self.inputs.cuda(), self.targets.cuda()

                self.lr_scheduler.step()
                self.optimizer.zero_grad()
                fx = self.Enc(self.inputs,return_skip=False)
                fx = torch.cat([fx, fx], 1)
                outputs = self.net(self.inputs, fx)

                loss = self.criterion(outputs, self.targets.long())
                loss.backward()
                self.optimizer.step()
                
                #----eval----
                self.step = train_step + self.gan_add*gan_num
                
                train_loss = loss.item()
                _, predicted = torch.max(outputs.data, 1)
            
                total = self.targets.size(0)
                correct = predicted.eq(self.targets.long().data).cpu().sum()

                # TENSORBOARD
                self.sum_step = self.sum_step + 1
                self.writer.add_scalar("train-loss", train_loss, self.sum_step)
                self.writer.add_scalar("train-Acc", 100. * float(correct) / float(total), self.sum_step)
                self.writer.add_scalar("lr*1000", 1000 * self.lr_scheduler.get_lr()[0], self.sum_step)

                if self.step % 50 == 0:
                    print('    Epoch:%d/%d  Step: %d/%d  Sum_step:%d   Lr: %f    Loss: %.3f | Acc: %.3f%% (%d/%d) GAN:%d --raw'% \
                          (epoch, 20, self.step, (self.gan_add+1)*(int(len(self.dataset_train_raw)/self.config.ncwh[0])+1),self.sum_step,\
                           self.optimizer.param_groups[0]['lr'], train_loss, 100.*float(correct)/float(total), correct, total,self.gan_add))

                if self.step % 200 == 0:
                   self.test()

                # ============training the gan data=============
                # shuffle every time
                for gan_step, (inputs_gan_, fx_gan, targets_gan_) in enumerate(self.train_loader_add):
                    self.inputs_gan = inputs_gan_
                    self.fx_gan = fx_gan
                    self.targets_gan = targets_gan_

                    if self.use_cuda:
                        self.inputs_gan, self.fx_gan, self.targets_gan = self.inputs_gan.cuda(), self.fx_gan.cuda(), self.targets_gan.cuda()

                    self.lr_scheduler.step()

                    self.optimizer.zero_grad()
                    if self.train_loader_add == self.train_loader_raw:
                        self.fx_gan = self.Enc(self.inputs_gan, return_skip=False)
                        self.fx_gan = torch.cat([self.fx_gan, self.fx_gan], 1)
                        outputs_gan = self.net(self.inputs_gan, self.fx_gan)
                        #outputs_gan = self.net(self.inputs_gan)
                    else:
                        outputs_gan = self.net(self.inputs_gan, self.fx_gan)
                        #outputs_gan = self.net(self.inputs_gan)

                    loss = self.criterion(outputs_gan, self.targets_gan.long())
                    loss.backward()
                    self.optimizer.step()

                    # ----eval----
                    self.step = self.step + 1

                    train_loss = loss.item()

                    _, predicted_gan = torch.max(outputs_gan.data, 1)

                    total = self.targets_gan.size(0)
                    correct = predicted_gan.eq(self.targets_gan.long().data).cpu().sum()

                    # TENSORBOARD
                    self.sum_step = self.sum_step + 1
                    self.writer.add_scalar("train-loss", train_loss, self.sum_step)
                    self.writer.add_scalar("train-Acc", 100. * float(correct) / float(total), self.sum_step)
                    self.writer.add_scalar("lr*1000", 1000 * self.lr_scheduler.get_lr()[0], self.sum_step)

                    if self.step % 50 == 0:
                        print('    Epoch:%d/%d  Step: %d/%d  Sum_step: %d  Lr:%f   Loss: %.3f | Acc: %.3f%% (%d/%d)  GAN:%d --gan' % (
                        epoch, 20, self.step,
                        (self.gan_add + 1) * (int(len(self.dataset_train_raw) / self.config.ncwh[0]) + 1),self.sum_step,
                        self.optimizer.param_groups[0]['lr'], train_loss, 100. * float(correct) / float(total), correct,
                        total, self.gan_add))

                    if self.step % 200 == 0:
                        self.test()

                    if (gan_step + 1) % self.gan_add == 0:
                        # have a look at input
                        #img = numpy.transpose(self.img_denorm(self.inputs_gan.data.cpu().numpy()), (0, 2, 3, 1)).astype(numpy.uint8)[0]
                        #Image.fromarray(img).save("./input_sample/" + str(epoch) + '.jpg')
                        break

                gan_num += 1

            print('VGG: Finished Training for ',epoch,'th Epoch!')
            #self.test()
            if epoch == 4:
                torch.save(self.Enc.state_dict(), self.checkpoint+'/Enc_base.pth')
                torch.save(self.net.state_dict(), self.checkpoint+'/vgg_base.pth')
                print("********* save base model ***********")
            print("--------------------------------------")

    def test(self):
        self.net.eval()

        test_loss = 0
        correct = 0
        total = 0
        for test_iter,(inputs_,_,targets_) in enumerate(self.test_loader):
            inputs = Variable(inputs_)
            targets_ = list(targets_)
            targets = Variable(torch.Tensor(targets_))
            if self.use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()

            fx_test = self.Enc(inputs, return_skip=False)
            fx_test = torch.cat([fx_test, fx_test],1)
            outputs = self.net(inputs, fx_test)
            #outputs = self.net(inputs)

            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += predicted.eq(targets.long().data).cpu().sum()

        # Save checkpoint.
        acc = 100.*float(correct)/float(total)

        self.f_acc.write(str(acc)+'\n')
        self.f_acc.flush()

        if acc > self.best_acc:
            print('Saving..')
            state = {
                'net': self.net.state_dict(),
                'acc': acc,
                'step': self.step,
            }
            if not os.path.isdir(self.checkpoint):
                os.mkdir(self.checkpoint)
            torch.save(state, self.checkpoint+'/max_ckpt.t7')
            self.best_acc = acc
            torch.save(self.Enc.state_dict(), self.checkpoint+'/Enc_max.pth')
            print("save max Encoder")

        self.writer.add_scalar("Test-Acc", acc, self.sum_step)

        print(('        Test     Loss: %.3f | Acc: %.3f%% (%d/%d) | max_Acc: %.3f%%' % (test_loss/len(self.dataset_test), 100.*float(correct)/float(total), correct, total, self.best_acc)))


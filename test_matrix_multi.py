#coding:utf-8
'''Train datasets with PyTorch.'''
from __future__ import print_function

import numpy 
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

import os
import argparse

from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader

from nets import vgg19_bn_fx as VGG
from nets import Encoder
from dataset_nature_multi import config_VGG, SingleDataset_VGG

class Classify(object):
    def __init__(self,args,config=config_VGG):
        self.args = args
        self.config = config

        if self.args.data == 2:#multipie
            image_f = open("dataset/multipie/images.list")
            self.labels = ['DI', 'SC', 'SM', 'SQ', 'SU', 'NE']
        if self.args.data == 3:#raf
            image_f = open("dataset/RAF/basic/images_vgg.list")
            self.labels = ['SU', 'FE', 'DI', 'HA', 'SA','AN','NE']
        if self.args.data == 1:#mmi
            image_f = open("dataset/mmi/images_vgg.list")
            self.labels = ['AN', 'DI', 'FE', 'HA', 'SA','SU']

        self.test_im_names = []
        self.test_labels = []

        for line in image_f:
            pic_name = line.strip().split()[0]
            if 1==1:#pic_name[:4] == 'test':
                if self.args.data == 1:
                    if int(line.strip().split()[1])!=6:
                        self.test_im_names.append(line.strip().split()[0])
                        self.test_labels.append(int(line.strip().split()[1]))
                else:
                    self.test_im_names.append(line.strip().split()[0])
                    self.test_labels.append(int(line.strip().split()[1]))
        self.dataset_test = SingleDataset_VGG(self.test_im_names, self.test_labels, self.config, 'test','raw')
        self.test_loader = DataLoader(dataset = self.dataset_test, batch_size = 1, shuffle = False, num_workers = 4)
        
        self.gpu = args.gpu
        os.environ["CUDA_VISIBLE_DEVICES"] = self.gpu[0]

        self.resume_dir = args.resume_dir
        self.use_cuda = torch.cuda.is_available()
 
        print('==> Resuming from checkpoint..')
        if os.path.isdir(self.resume_dir):
            print('Checkpoint directory found!')
        else:
            print('no Checkpoint directory found!')  
        checkpoint = torch.load(self.resume_dir+'/max_ckpt.t7')

        if self.args.data==2 or 1:
            self.net = VGG(6,pretrained=False) # multipie
        else:
            self.net = VGG(7,pretrained=False) # mmi and raf
        self.net.load_state_dict(checkpoint['net'])
        self.best_acc = float(checkpoint['acc'])

        print("Success resume the max checkpoing")
        print("Bese acc is:",self.best_acc)
        self.start_iter = checkpoint['step']

        if self.use_cuda:
            with torch.cuda.device(0):
                # move param and buffer to GPU
                self.net.cuda()
                    # parallel use GPU
            if len(self.args.gpu)>1 :
                self.net = torch.nn.DataParallel(self.net, device_ids=range(len(self.gpu)))
             # speed up 
            cudnn.benchmark = True


    def tensor2var(self, tensors, volatile=False):
        if not hasattr(tensors, '__iter__'): tensors = [tensors]
        out = []
        for tensor in tensors:
            if len(self.gpu):
                tensor = tensor.cuda(0)
            #var = torch.autograd.Variable(tensor,volatile=volatile)
            out.append(tensor)
        if len(out) == 1:
            return out[0]
        else:
            return out


    def test(self):
        self.net.eval()

        self.Enc = Encoder()
        ckpt_file_enc = os.path.join(self.resume_dir, 'Enc_max.pth')
        print(ckpt_file_enc)
        assert os.path.exists(ckpt_file_enc)
        self.Enc.load_state_dict(torch.load(ckpt_file_enc), strict=False)
        self.Enc.eval()
        self.Enc.cuda()

        test_loss = 0
        correct = 0
        total = 0
        self.test_predict=[]
        for test_iter,(inputs_,_,targets_) in enumerate(self.test_loader):
            inputs = Variable(inputs_)
            targets_ = list(targets_)
            targets = Variable(torch.Tensor(targets_))
            if self.use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()

            fx_test = self.Enc(inputs, return_skip=False)
            fx_test = torch.cat([fx_test, fx_test],1)
            outputs = self.net(inputs, fx_test)
            # outputs = self.net(inputs)

            _, predicted = torch.max(outputs.data, 1)
            #self.test_predict.append(predicted[0])
            self.test_predict.append(predicted[0].cpu().numpy())
            total += targets.size(0)
            correct += predicted.eq(targets.long().data).cpu().sum()

        print(('        Test     Loss: %.3f | Acc: %.3f%% (%d/%d) | max_Acc: %.3f%%' % (test_loss/len(self.dataset_test), 100.*float(correct)/float(total), correct, total, self.best_acc)))

    def Confusion_matrix(self):
        labels = self.labels
        cm = confusion_matrix(self.test_labels,self.test_predict)
        cm = cm.astype(numpy.float32)
        if self.args.data == 2 or 1:
            sums=[numpy.sum(cm[0]),numpy.sum(cm[1]),numpy.sum(cm[2]),numpy.sum(cm[3]),numpy.sum(cm[4]),numpy.sum(cm[5])]
        else:
            sums=[numpy.sum(cm[0]),numpy.sum(cm[1]),numpy.sum(cm[2]),numpy.sum(cm[3]),numpy.sum(cm[4]),numpy.sum(cm[5]),numpy.sum(cm[6])]
        for i in range(len(sums)):
            for j in range(len(sums)):
                cm[i][j]=round(float(cm[i][j])/float(sums[i]),2)#*100

        print(cm)
        print(labels)
        sns.set()
        f,ax=plt.subplots()
        sns.heatmap(cm,annot=True,annot_kws={'size':10},ax=ax,cmap=plt.cm.Blues)
        ax.set_xticklabels(labels)
        ax.set_yticklabels(labels, rotation=0)
        ax.tick_params(axis='y',labelsize=10)
        ax.tick_params(axis='x',labelsize=10)
        plt.savefig("Con_Matrix.png")
        print("The confusion matrix is saved in Con_Matrix.png.")
    
    def Per_expression(self):
        if self.args.data == 2:
            results = [0,0,0,0,0,0]
        else:
            results = [0,0,0,0,0,0,0]
        for item in self.test_predict:
            results[item]+=1
        print(self.labels)
        print(results/len(self.test_predict))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume_dir',type=str,help='The model to be test:/checkpoint/VGG')
    parser.add_argument('--gpu',type=str,help='the gpu id')
    parser.add_argument('--data',type=int,help="1:mmi,2:multipie,3:raf")
    args = parser.parse_args()

    print(args)
    VGG_model = Classify(args)
    VGG_model.test()
    VGG_model.Confusion_matrix()





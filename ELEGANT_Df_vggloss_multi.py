from dataset_nature_multi import config, MultiCelebADataset, config_VGG, SingleDataset_GAN

from nets import Encoder, Decoder, Decoder_label, Discriminator, VGGLoss

from train_VGG_multi import Classify
import os
import argparse
import torch
from torchvision import transforms
from PIL import Image
import numpy as np
from itertools import chain
import random
from torch.utils.data import Dataset, DataLoader
import time


class ELEGANT(object):
    def __init__(self, args,
                 config=config, dataset=MultiCelebADataset, \
                 encoder=Encoder, decoder=Decoder, decoder_label=Decoder_label, discriminator=Discriminator):

        self.args = args
        self.attributes = args.attributes
        self.n_attributes = len(self.attributes)
        self.gpu = args.gpu
        self.mode = args.mode
        self.restore = args.restore
        self.train_dis = True

        # init dataset and networks
        self.config = config
       
        self.dataset = dataset(self.attributes)
        
        self.Enc = encoder()

        if self.args.decoder_label == True:
            self.Dec = decoder_label(self.args.label_copy, self.n_attributes)
        elif self.args.exp_label == True:
            self.Dec = decoder_label(self.args.label_copy, self.n_attributes + 1)
        else:
            self.Dec = decoder()
        

        self.D1  = discriminator(self.n_attributes, self.config.nchw[-1])
        self.D2  = discriminator(self.n_attributes, self.config.nchw[-1]//2)

        self.adv_criterion = torch.nn.BCELoss()
        self.recon_criterion = torch.nn.MSELoss() #l2
        self.fakerecon_criterion = torch.nn.L1Loss() #l1

        self.criterionVGG = VGGLoss(6,self.gpu[0]) # perceptual loss

        self.restore_from_file()
        self.set_mode_and_gpu()

        self.vgg_loss = 0
        self.l1_loss = 0
        self.l2_loss = 0
        self.tv_loss = 0

    def restore_from_file(self):
        if self.restore is not None:
            ckpt_file_enc = os.path.join(self.config.model_dir, 'Enc_iter_temp.pth')#.format(self.restore))
            assert os.path.exists(ckpt_file_enc)
            ckpt_file_dec = os.path.join(self.config.model_dir, 'Dec_iter_temp.pth')#.format(self.restore))
            assert os.path.exists(ckpt_file_dec)
            if self.gpu:
                self.Enc.load_state_dict(torch.load(ckpt_file_enc), strict=False)
                self.Dec.load_state_dict(torch.load(ckpt_file_dec), strict=False)
            else:
                self.Enc.load_state_dict(torch.load(ckpt_file_enc, map_location='cpu'), strict=False)
                self.Dec.load_state_dict(torch.load(ckpt_file_dec, map_location='cpu'), strict=False)

            if self.mode == 'train':
                ckpt_file_d1  = os.path.join(self.config.model_dir, 'D1_iter_temp.pth')#.format(self.restore))
                assert os.path.exists(ckpt_file_d1)
                ckpt_file_d2  = os.path.join(self.config.model_dir, 'D2_iter_temp.pth')#.format(self.restore))
                assert os.path.exists(ckpt_file_d2)
                if self.gpu:
                    self.D1.load_state_dict(torch.load(ckpt_file_d1), strict=False)
                    self.D2.load_state_dict(torch.load(ckpt_file_d2), strict=False)
                else:
                    self.D1.load_state_dict(torch.load(ckpt_file_d1, map_location='cpu'), strict=False)
                    self.D2.load_state_dict(torch.load(ckpt_file_d2, map_location='cpu'), strict=False)
                print("load restore model")

            self.start_step = self.restore + 1
        else:
            self.start_step = 1

    def set_mode_and_gpu(self):
        if self.mode == 'train':
            self.Enc.train()
            self.Dec.train()
            self.D1.train()
            self.D2.train()

            self.optimizer_G = torch.optim.Adam(chain(self.Enc.parameters(), self.Dec.parameters()),
                                           lr=self.config.G_lr, betas=(0.5, 0.999),
                                           weight_decay=self.config.weight_decay)

            self.optimizer_D = torch.optim.Adam(chain(self.D1.parameters(), self.D2.parameters()),
                                           lr=self.config.D_lr, betas=(0.5, 0.999),
                                           weight_decay=self.config.weight_decay)

            self.G_lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer_G, step_size=self.config.step_size, gamma=self.config.gamma)
            self.D_lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer_D, step_size=self.config.step_size, gamma=self.config.gamma)
            if self.restore is not None:
                for _ in range(self.restore):
                    self.D_lr_scheduler.step()
                    self.D_lr_scheduler.step()

            if self.gpu:
                with torch.cuda.device(0):
                    self.Enc.cuda()
                    self.Dec.cuda()
                    self.D1.cuda()
                    self.D2.cuda()
                    self.adv_criterion.cuda()
                    self.recon_criterion.cuda()
                    self.fakerecon_criterion.cuda()

            if len(self.gpu) > 1:
                self.Enc = torch.nn.DataParallel(self.Enc, device_ids=list(range(len(self.gpu))))
                self.Dec = torch.nn.DataParallel(self.Dec, device_ids=list(range(len(self.gpu))))
                self.D1  = torch.nn.DataParallel(self.D1, device_ids=list(range(len(self.gpu))))
                self.D2  = torch.nn.DataParallel(self.D2, device_ids=list(range(len(self.gpu))))

        elif self.mode == 'test':
            self.Enc.eval()
            self.Dec.eval()

            if self.gpu:
                with torch.cuda.device(0):
                    self.Enc.cuda()
                    self.Dec.cuda()

            if len(self.gpu) > 1:
                self.Enc = torch.nn.DataParallel(self.Enc, device_ids=list(range(len(self.gpu))))
                self.Dec = torch.nn.DataParallel(self.Dec, device_ids=list(range(len(self.gpu))))

        else:
            raise NotImplementationError()

    def tensor2var(self, tensors, volatile=False):
        if not hasattr(tensors, '__iter__'): tensors = [tensors]
        out = []
        for tensor in tensors:
            if len(self.gpu):
                tensor = tensor.cuda(0)
            #var = torch.autograd.Variable(tensor, volatile=volatile)
            out.append(tensor)
        if len(out) == 1:
            return out[0]
        else:
            return out

    def get_attr_chs(self, encodings, attribute_id):
        num_chs = encodings.size(1)
        per_chs = float(num_chs) / self.n_attributes
        start = int(np.rint(per_chs * attribute_id))
        end = int(np.rint(per_chs * (attribute_id + 1)))
        # return encodings[:, start:end]
        return encodings.narrow(1, start, end-start)

    def get_attr_chs_B(self, encodings, attribute_id):
        num_chs = encodings.size(0)
        per_chs = float(num_chs) / self.n_attributes
        start = int(np.rint(per_chs * attribute_id))
        end = int(np.rint(per_chs * (attribute_id + 1)))
        # return encodings[:, start:end]
        return encodings.narrow(0, start, end-start)

    def forward_G(self):
        self.z_A, self.A_skip = self.Enc(self.A, return_skip=True)
        self.z_B, self.B_skip = self.Enc(self.B, return_skip=True)

        self.z_C = torch.cat([self.get_attr_chs(self.z_A, i) if i != self.attribute_id \
                              else self.get_attr_chs(self.z_B, i)  for i in range(self.n_attributes)], 1)

        self.z_D = torch.cat([self.get_attr_chs(self.z_B, i) if i != self.attribute_id \
                              else self.get_attr_chs(self.z_A, i)  for i in range(self.n_attributes)], 1)
        for idx in range(self.z_C.size(0)):
            self.z_C[idx] = torch.cat([self.get_attr_chs_B(self.z_C[idx], i) if i != self.attribute_y_B[idx] \
                                  else self.get_attr_chs_B(self.z_B[idx], i) for i in range(self.n_attributes)], 0)

            self.z_D[idx] = torch.cat([self.get_attr_chs_B(self.z_D[idx], i) if i != self.attribute_y_B[idx] \
                                  else self.get_attr_chs_B(self.z_A[idx], i) for i in range(self.n_attributes)], 0)

        if self.args.decoder_label == True:# input switch labels
            self.change_vector_A = torch.FloatTensor(self.z_A.shape[0],self.n_attributes).zero_()
            self.change_vector_C = torch.FloatTensor(self.z_C.shape[0],self.n_attributes).zero_()
            for i in range(self.z_A.shape[0]):
                self.change_vector_C[i][self.attribute_id] = 1
                self.change_vector_C[i][self.attribute_y_B[i]] = 1

            self.change_vector_C_new = self.change_vector_C
            self.change_vector_A_new = self.change_vector_A

            for i in range(self.args.label_copy-1):
                self.change_vector_C_new = torch.cat([self.change_vector_C_new, self.change_vector_C], 1)
                self.change_vector_A_new = torch.cat([self.change_vector_A_new, self.change_vector_A], 1)
          
            if self.gpu:
                self.change_vector_C = self.change_vector_C_new.cuda(0)
                self.change_vector_A = self.change_vector_A_new.cuda(0)

            self.R_A = self.Dec(self.z_A, self.z_A, self.change_vector_A, skip=self.A_skip)
            self.R_B = self.Dec(self.z_B, self.z_B, self.change_vector_A, skip=self.B_skip)
            self.R_C = self.Dec(self.z_C, self.z_A, self.change_vector_C, skip=self.A_skip)
            self.R_D = self.Dec(self.z_D, self.z_B, self.change_vector_C, skip=self.B_skip)
            
            if self.args.alpha5:# alpha5
                self.z_C_5 = 0.5*(self.z_C-self.z_A)+self.z_A
                self.z_D_5 = 0.5*(self.z_D-self.z_B)+self.z_B
                self.R_C_5 = self.Dec(self.z_C_5, self.z_A, self.change_vector_C, skip=self.A_skip)
                self.R_D_5 = self.Dec(self.z_D_5, self.z_B, self.change_vector_C, skip=self.B_skip)

        else: # input nothing
            self.R_A = self.Dec(self.z_A, self.z_A, skip=self.A_skip)
            self.R_B = self.Dec(self.z_B, self.z_B, skip=self.B_skip)
            self.R_C = self.Dec(self.z_C, self.z_A, skip=self.A_skip)
            self.R_D = self.Dec(self.z_D, self.z_B, skip=self.B_skip)
            
            if self.args.alpha5:#alpha5
                self.z_C_5 = 0.5*(self.z_C-self.z_A)+self.z_A
                self.z_D_5 = 0.5*(self.z_D-self.z_B)+self.z_B
                self.R_C_5 = self.Dec(self.z_C_5, self.z_A, skip=self.A_skip)
                self.R_D_5 = self.Dec(self.z_D_5, self.z_B, skip=self.B_skip)

        self.A1 = torch.clamp(self.A + self.R_A, -1, 1)
        self.B1 = torch.clamp(self.B + self.R_B, -1, 1)
        self.C  = torch.clamp(self.A + self.R_C, -1, 1)
        self.D  = torch.clamp(self.B + self.R_D, -1, 1)
        if self.args.alpha5:
            self.C_5 = torch.clamp(self.A + self.R_C_5,-1,1)
            self.D_5 = torch.clamp(self.B + self.R_D_5,-1,1)

    def forward_D_real_sample(self):
        self.d1_A = self.D1(self.A, self.y_A)
        self.d1_B = self.D1(self.B, self.y_B)
        self.d2_A = self.D2(self.A, self.y_A)
        self.d2_B = self.D2(self.B, self.y_B)

    def forward_D_fake_sample(self, detach):
        self.y_C, self.y_D = self.y_A.clone(), self.y_B.clone()
        self.y_C.data[:, self.attribute_id] = self.y_B.data[:, self.attribute_id]
        self.y_D.data[:, self.attribute_id] = self.y_A.data[:, self.attribute_id]
        for idx in range(self.y_D.size(0)):
            self.y_C.data[idx, self.attribute_y_B[idx]] = self.y_B.data[idx, self.attribute_y_B[idx]]
            self.y_D.data[idx, self.attribute_y_B[idx]] = self.y_A.data[idx, self.attribute_y_B[idx]]
        
        if detach:
            self.d1_C = self.D1(self.C.detach(), self.y_C)
            self.d1_D = self.D1(self.D.detach(), self.y_D)
            self.d2_C = self.D2(self.C.detach(), self.y_C)
            self.d2_D = self.D2(self.D.detach(), self.y_D)
            if self.args.alpha5:# 交换一半表情
                self.d1_C_5 = self.D1(self.C_5.detach(), self.y_C)
                self.d2_C_5 = self.D2(self.C_5.detach(), self.y_C)
                self.d1_D_5 = self.D1(self.D_5.detach(), self.y_D)
                self.d2_D_5 = self.D2(self.D_5.detach(), self.y_D)
        else:
            self.d1_C = self.D1(self.C, self.y_C)
            self.d1_D = self.D1(self.D, self.y_D)
            self.d2_C = self.D2(self.C, self.y_C)
            self.d2_D = self.D2(self.D, self.y_D)
            if self.args.alpha5:# 交换一半表情
                self.d1_C_5 = self.D1(self.C_5, self.y_C)
                self.d2_C_5 = self.D2(self.C_5, self.y_C)
                self.d1_D_5 = self.D1(self.D_5, self.y_D)
                self.d2_D_5 = self.D2(self.D_5, self.y_D)

    def compute_loss_D(self):
        self.D_loss = {
            'D1':   self.adv_criterion(self.d1_A, torch.normal(mean=torch.ones_like(self.d1_A), std=self.args.std)) + \
                    self.adv_criterion(self.d1_B, torch.normal(mean=torch.ones_like(self.d1_A), std=self.args.std))  + \
                    self.adv_criterion(self.d1_C, torch.normal(mean=torch.zeros_like(self.d1_A), std=self.args.std)) + \
                    self.adv_criterion(self.d1_D, torch.normal(mean=torch.zeros_like(self.d1_A), std=self.args.std)),

            'D2':   self.adv_criterion(self.d2_A, torch.normal(mean=torch.ones_like(self.d1_A), std=self.args.std))  + \
                    self.adv_criterion(self.d2_B, torch.normal(mean=torch.ones_like(self.d1_A), std=self.args.std))  + \
                    self.adv_criterion(self.d2_C, torch.normal(mean=torch.zeros_like(self.d1_A), std=self.args.std)) + \
                    self.adv_criterion(self.d2_D, torch.normal(mean=torch.zeros_like(self.d1_A), std=self.args.std)),
        }
        if self.args.alpha5:
            self.D_loss['D1'] += self.adv_criterion(self.d1_C_5, torch.normal(mean=torch.zeros_like(self.d1_A), std=self.args.std)) + \
                    self.adv_criterion(self.d1_D_5, torch.normal(mean=torch.zeros_like(self.d1_A), std=self.args.std))
            self.D_loss['D2'] += self.adv_criterion(self.d2_C_5, torch.normal(mean=torch.zeros_like(self.d1_A), std=self.args.std)) + \
                    self.adv_criterion(self.d2_D_5, torch.normal(mean=torch.zeros_like(self.d1_A), std=self.args.std))

        self.loss_D = (self.D_loss['D1'] + 0.5 * self.D_loss['D2']) * self.args.lambda_D

    def compute_loss_G(self):
        C_l = self.C.detach()
        D_l = self.D.detach()
        self.G_loss = {
            'reconstruction': self.recon_criterion(self.A1, self.A) + self.recon_criterion(self.B1, self.B), #l2
            'fake_reconstruction': self.fakerecon_criterion(self.A + self.B,self.C + self.D), #l1

            'adv1': self.adv_criterion(self.d1_C, torch.normal(mean=torch.ones_like(self.d1_A), std=self.args.std)) + \
                    self.adv_criterion(self.d1_D, torch.normal(mean=torch.ones_like(self.d1_A), std=self.args.std)),
            'adv2': self.adv_criterion(self.d2_C, torch.normal(mean=torch.ones_like(self.d1_A), std=self.args.std))  + \
                    self.adv_criterion(self.d2_D, torch.normal(mean=torch.ones_like(self.d1_A), std=self.args.std)),
            
            'tv_C': (self.recon_criterion(C_l[:, :, 1:, :] , C_l[:, :, :self.config.nchw[2] - 1, :]) / self.config.nchw[2]) + \
                  (self.recon_criterion(C_l[:, :, :,1:] , C_l[:, :, :, :self.config.nchw[3] - 1]) / self.config.nchw[3]),

            'tv_D': ((self.recon_criterion(D_l[:, :, 1:, :] , D_l[:, :, :self.config.nchw[2] - 1, :]) / self.config.nchw[2]) + \
                  (self.recon_criterion(D_l[:, :, :,1:] , D_l[:, :, :, :self.config.nchw[3] - 1]) / self.config.nchw[3])),
        }
        if self.args.alpha5:
            self.G_loss['adv1'] += self.adv_criterion(self.d1_C_5, torch.normal(mean=torch.ones_like(self.d1_A), std=self.args.std)) + \
                    self.adv_criterion(self.d1_D_5, torch.normal(mean=torch.ones_like(self.d1_A), std=self.args.std))
            self.G_loss['adv2'] += self.adv_criterion(self.d2_C_5, torch.normal(mean=torch.ones_like(self.d1_A), std=self.args.std)) + \
                    self.adv_criterion(self.d2_D_5, torch.normal(mean=torch.ones_like(self.d1_A), std=self.args.std))
       
        self.loss_G = self.G_loss['adv1'] + 0.5 * self.G_loss['adv2'] 
        if self.args.vgg_loss:
            self.vgg_loss = self.criterionVGG(self.D, self.A) + self.criterionVGG(self.B, self.C)
            self.loss_G += self.args.lambda_vgg_loss * self.vgg_loss
        if self.args.l1_loss:
            self.l1_loss = self.G_loss['fake_reconstruction']
            self.loss_G += self.args.lambda_l1_loss * self.l1_loss
        if self.args.l2_loss:
            self.l2_loss = self.G_loss['reconstruction']
            self.loss_G += self.args.lambda_l2_loss * self.l2_loss
        if self.args.tv_loss:
            self.tv_loss = self.G_loss['tv_C'] + self.G_loss['tv_D']
            self.loss_G += self.args.lambda_tv_loss * self.tv_loss


    def backward_D(self):
        self.loss_D.backward()
        self.optimizer_D.step()

    def backward_G(self):
        self.loss_G.backward()
        self.optimizer_G.step()

    def img_denorm(self, img, scale=255):
        return (img + 1) * scale / 2.

    def save_image_log(self, save_num=20):
        image_info = {
            'A/img'   : self.img_denorm(self.A.data.cpu(), 1)[:save_num],
            'B/img'   : self.img_denorm(self.B.data.cpu(), 1)[:save_num],
            'C/img'   : self.img_denorm(self.C.data.cpu(), 1)[:save_num],
            'D/img'   : self.img_denorm(self.D.data.cpu(), 1)[:save_num],
            'A1/img'  : self.img_denorm(self.A1.data.cpu(), 1)[:save_num],
            'B1/img'  : self.img_denorm(self.B1.data.cpu(), 1)[:save_num],
            'R_A/img' : self.img_denorm(self.R_A.data.cpu(), 1)[:save_num],
            'R_B/img' : self.img_denorm(self.R_B.data.cpu(), 1)[:save_num],
            'R_C/img' : self.img_denorm(self.R_C.data.cpu(), 1)[:save_num],
            'R_D/img' : self.img_denorm(self.R_D.data.cpu(), 1)[:save_num],
        }
        #for tag, images in image_info.items():
        #    for idx, image in enumerate(images):
        #        self.writer.add_image(tag+'/{}_{:02d}'.format(self.attribute_id, idx), image, self.step)

    def save_sample_images(self, epoch, save_num=10):
        canvas = torch.cat((self.A, self.B, self.C, self.D, self.A1, self.B1), -1)
        img_array = np.transpose(self.img_denorm(canvas.data.cpu().numpy()), (0,2,3,1)).astype(np.uint8)
        for i in range(save_num):
            Image.fromarray(img_array[i]).save(os.path.join(self.config.img_dir, 'Epoch_{:03d}_attr_{}_{:02d}.jpg'.format(epoch, self.attribute_id, i)))

    def save_scalar_log(self):
        scalar_info = {
            'loss_D': self.loss_D.data.cpu().numpy()[0],
            'loss_G': self.loss_G.data.cpu().numpy()[0],
            'G_lr'  : self.G_lr_scheduler.get_lr()[0],
            'D_lr'  : self.D_lr_scheduler.get_lr()[0],
        }

        for key, value in self.G_loss.items():
            scalar_info['G_loss/' + key] = value.data[0]

        for key, value in self.D_loss.items():
            scalar_info['D_loss/' + key] = value.data[0]

        #for tag, value in scalar_info.items():
        #    self.writer.add_scalar(tag, value, self.step)

    def save_model(self, epoch):
        if epoch == -1: # newest model
            reduced = lambda key: key[7:] if key.startswith('module.') else key
            torch.save({reduced(key): val.cpu() for key, val in self.Enc.state_dict().items()}, os.path.join(self.config.model_dir, 'Enc_iter_temp.pth'))
            torch.save({reduced(key): val.cpu() for key, val in self.Dec.state_dict().items()}, os.path.join(self.config.model_dir, 'Dec_iter_temp.pth'))
            torch.save({reduced(key): val.cpu() for key, val in self.D1.state_dict().items()},  os.path.join(self.config.model_dir, 'D1_iter_temp.pth'))
            torch.save({reduced(key): val.cpu() for key, val in self.D2.state_dict().items()},  os.path.join(self.config.model_dir, 'D2_iter_temp.pth'))
        else:
            reduced = lambda key: key[7:] if key.startswith('module.') else key
            torch.save({reduced(key): val.cpu() for key, val in self.Enc.state_dict().items()}, os.path.join(self.config.model_dir, 'Enc_iter_{:03d}.pth'.format(epoch)))
            torch.save({reduced(key): val.cpu() for key, val in self.Dec.state_dict().items()}, os.path.join(self.config.model_dir, 'Dec_iter_{:03d}.pth'.format(epoch)))
            torch.save({reduced(key): val.cpu() for key, val in self.D1.state_dict().items()},  os.path.join(self.config.model_dir, 'D1_iter_{:03d}.pth'.format(epoch)))
            torch.save({reduced(key): val.cpu() for key, val in self.D2.state_dict().items()},  os.path.join(self.config.model_dir, 'D2_iter_{:03d}.pth'.format(epoch)))

    def train(self, epoch): # train for 1 epoch
        self.Enc.train()
        self.Dec.train()
        self.D1.train()
        self.D2.train()
        
        for self.step in range(0, 1+int(self.dataset.datanum / self.config.nchw[0] )):
            
            self.G_lr_scheduler.step()
            self.D_lr_scheduler.step()

            self.attribute_id = random.randint(0,len(self.attributes)-1)
            A, y_A = next(self.dataset.gen(self.attribute_id, True))
            B, y_B = next(self.dataset.gen(self.attribute_id, False))
            #print(y_B)
            self.attribute_y_B = torch.IntTensor(y_B.size(0)).zero_()
            for idx, p in enumerate(y_B):
                for idx_q, q in enumerate(p):
                    if q == 1:
                        self.attribute_y_B[idx] = idx_q

            self.A, self.y_A, self.B, self.y_B = self.tensor2var([A, y_A, B, y_B])

            # forward
            self.forward_G()

            # update D
            if self.train_dis:
                self.forward_D_real_sample()
                self.forward_D_fake_sample(detach=True)
                self.compute_loss_D()
                self.optimizer_D.zero_grad()
                self.backward_D()
            else:
                print("    stop training Dis")

            # update G
            self.forward_D_fake_sample(detach=False)
            self.compute_loss_G()
            self.optimizer_G.zero_grad()
            self.backward_G()

            if self.step % 20 == 0:
                print('    step: %06d, D: %.4f, G: %.4f, l2: %.4f, l1: %.4f ,tv:%.4f ,vgg_loss: %.4f, attri:%d' % (self.step, self.loss_D.data.cpu().numpy(), self.loss_G.data.cpu().numpy(), self.args.lambda_l2_loss * self.l2_loss.data.cpu().numpy(), self.args.lambda_l1_loss*self.l1_loss, self.args.lambda_tv_loss*self.tv_loss.data.cpu().numpy(), self.args.lambda_vgg_loss*self.vgg_loss.data.cpu().numpy(),self.attribute_id))

            if self.loss_G.data.cpu().numpy() >= self.args.G_max:#4:
                self.train_dis = False
            else:
                self.train_dis = True

        # save sample images
        if epoch % 1 == 0:
            print("save sample_images")
            self.save_sample_images(epoch)
        
        # save models
        if epoch % 10 ==0:
            self.save_model(epoch)

        self.save_model(-1)

        print('Finished Training for 1 Epoch!')
        #self.writer.close()


    def generate(self):
        f_gan = open(self.config.data_dir+'/images_gan_nature.list')
        lines = f_gan.readlines()

        for attr in range(len(self.attributes)):
            print("attr:", attr)
            if attr == len(self.attributes)-1:
                continue
            input_names = []
            target_names = []
            for line in lines:
                if line.strip().split()[1] == str(attr):
                    line_name = line.strip().split()[0]
                    input_names.append(line_name.split('_')[0] + '_' + line_name.split('_')[1] + '.jpg')  # exp
                    target_names.append(line_name.split('_')[2] + '_' + line_name.split('_')[3])  # nature

            Dataset_gan = SingleDataset_GAN(input_names, target_names, self.config)
            Dataloader_gan = DataLoader(dataset=Dataset_gan, batch_size=self.config.nchw[0], shuffle=False,
                                        num_workers=self.config.num_workers)

            print("Attr ", str(attr), " generate:", len(Dataset_gan))
            for step, (input_, target_, input_name, target_name) in enumerate(Dataloader_gan):
                i = 0

                if len(self.gpu):
                    self.A, self.B = input_.cuda(0), target_.cuda(0)
                self.attribute_id = int(attr)
                self.attribute_y_B = torch.IntTensor(self.A.size(0)).zero_()
                for idx in range(self.A.size(0)):
                    self.attribute_y_B[idx] = len(self.attributes) - 1  # nature

                self.forward_G()

                img = self.D
                img = np.transpose(self.img_denorm(img.data.cpu().numpy()), (0, 2, 3, 1)).astype(np.uint8)
                for idx, item in enumerate(img):
                    name = input_name[idx].split('.')[0] + '_' + target_name[idx]
                    Image.fromarray(item).save(self.config.gan_dir + '/' + name)
        print('Attr ',len(self.attributes)-1)
        for line in lines:
            if line.strip().split()[1] == str(len(self.attributes)-1):
                line_name = line.strip().split()[0]
                input_name=line_name.split('_')[0] + '_' + line_name.split('_')[1] + '.jpg'  # exp
                target_name=line_name.split('_')[2] + '_' + line_name.split('_')[3]  # nature
                nature_img = Image.open(self.config.data_dir+"/align_5p/" + target_name)
                name = input_name.split('.')[0] + '_' + target_name
                nature_img.save(self.config.gan_dir+'/'+name)
        f_gan.close()


    def generate_fx(self):
        ckpt_file_enc = os.path.join(self.config.model_dir, 'Enc_iter_temp.pth')
        assert os.path.exists(ckpt_file_enc)
        self.Enc.load_state_dict(torch.load(ckpt_file_enc), strict=False)
        self.Enc.eval()
        self.Enc.cuda()

        f_gan = open(self.config.data_dir+'/images_gan_nature.list')
        lines = f_gan.readlines()
        for attr in range(len(self.attributes)):
            print("attr:",attr)
            input_names = []
            target_names = []
            for line in lines:
                if line.strip().split()[1] == str(attr):
                    line_name = line.strip().split()[0]
                    input_names.append(line_name.split('_')[0] +'_'+ line_name.split('_')[1] +'.jpg') #exp
                    target_names.append(line_name.split('_')[2] +'_'+ line_name.split('_')[3] )# nature

            Dataset_gan = SingleDataset_GAN(input_names, target_names, self.config)
            Dataloader_gan = DataLoader(dataset = Dataset_gan, batch_size = self.config.nchw[0], shuffle = False, num_workers = self.config.num_workers) 
        
            print("Attr ",str(attr), " generate:",len(Dataset_gan))
            for step ,(input_, target_, input_name, target_name) in enumerate(Dataloader_gan):
                i=0
                
                if len(self.gpu):
                    self.A, self.B = input_.cuda(0), target_.cuda(0)

                self.attribute_id = int(attr)
                self.attribute_y_B = torch.IntTensor(self.A.size(0)).zero_()
                for idx in range(self.A.size(0)):
                    self.attribute_y_B[idx] = len(self.attributes) - 1  # nature

                self.z_A, self.A_skip = self.Enc(self.A, return_skip=True)
                self.z_B, self.B_skip = self.Enc(self.B, return_skip=True)


                self.z_C = torch.cat([self.get_attr_chs(self.z_A, i) if i != self.attribute_id \
                              else self.get_attr_chs(self.z_B, i)  for i in range(self.n_attributes)], 1)
                self.z_D = torch.cat([self.get_attr_chs(self.z_B, i) if i != self.attribute_id \
                              else self.get_attr_chs(self.z_A, i)  for i in range(self.n_attributes)], 1)
                for idx in range(self.z_C.size(0)):
                    self.z_C[idx] = torch.cat([self.get_attr_chs_B(self.z_C[idx], i) if i != self.attribute_y_B[idx] \
                                                   else self.get_attr_chs_B(self.z_B[idx], i) for i in range(self.n_attributes)], 0)

                    self.z_D[idx] = torch.cat([self.get_attr_chs_B(self.z_D[idx], i) if i != self.attribute_y_B[idx] \
                                                   else self.get_attr_chs_B(self.z_A[idx], i) for i in range(self.n_attributes)], 0)

                for idx, item in enumerate(self.z_D):
                    fx_name = input_name[idx].split('.')[0] + '_' + target_name[idx].split('.')[0] + '.npy'
                    if attr == len(self.attributes)-1: # nature
                        fx = torch.cat([self.z_B[idx], self.z_B[idx]], 0)
                    else:
                        fx = torch.cat([item, self.z_B[idx]],0)
                    np.save(file = self.config.fx_dir + '/' +fx_name, arr=fx.detach().cpu().numpy())
        f_gan.close()
    

    def transform(self, *images):
        transform1 = transforms.Compose([
            transforms.Resize(self.config.nchw[-2:]),
            transforms.ToTensor(),
        ])
        transform2 = lambda x: x.view(1, *x.size()) * 2 - 1
        out = [transform2(transform1(image)) for image in images]
        return out


    def swap(self,test_idx,attribute_id, attribute_id_b,test_input, test_target):
        '''
        swap attributes of two images.
        '''
        self.attribute_id = attribute_id
        self.attribute_y_B = torch.IntTensor(1).zero_()

        self.attribute_y_B[0] = attribute_id_b
        self.B, self.A = self.tensor2var(self.transform(Image.open(test_input), Image.open(test_target)), volatile=True)

        self.forward_G()
        img = torch.cat((self.B, self.A, self.D, self.C, self.R_D, self.R_C), -1)
        img = np.transpose(self.img_denorm(img.data.cpu().numpy()), (0,2,3,1)).astype(np.uint8)[0]
        Image.fromarray(img).save(os.path.join(self.config.test_dir ,str(attribute_id)+'_'+str(attribute_id_b)+'_'+str(test_idx)+'.jpg'))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-a', '--attributes', nargs='+', type=str, help='Specify attribute names.')
    parser.add_argument('-g', '--gpu', default=[], nargs='+', type=str, help='Specify GPU ids.')
    parser.add_argument('-m', '--mode', default='train', type=str, choices=['train', 'test'])
    parser.add_argument('-r', '--restore', default=None, action='store', type=int, help='Specify checkpoint id to restore')

    # test parameters
    parser.add_argument('--swap', action='store_true', help='Swap attributes.')
    parser.add_argument('--swap_list', default=[], nargs='+', type=int, help='Specify the attributes ids for swapping.')
    parser.add_argument('-i', '--input', type=str, help='Specify the input image.')
    parser.add_argument('-t', '--target', nargs='+', type=str, help='Specify target images.')
    parser.add_argument('-s', '--size', nargs='+', type=int, help='Specify the interpolation size.')

    # loss parameters
    parser.add_argument('--vgg_loss', action='store_true', default = False, help='use vgg_loss or not')
    parser.add_argument('--lambda_vgg_loss', type=float, default=10.0, help="weith for vgg_loss")
    parser.add_argument('--l1_loss', action='store_true', default = False, help='use l1_loss or not')
    parser.add_argument('--lambda_l1_loss', type=float, default=0.0, help="weith for l1_loss")
    parser.add_argument('--tv_loss', action='store_true', default = False, help='use tv_loss or not')
    parser.add_argument('--lambda_tv_loss', type=float, default=5.0, help="weith for tv_loss")
    parser.add_argument('--l2_loss', action='store_true', default = True, help='use l2_loss or not')
    parser.add_argument('--lambda_l2_loss', type=float, default=5.0, help="weith for l2_loss")
    parser.add_argument('--lambda_D', type=float, default=0.25, help="weith for Discriminator loss")
    parser.add_argument('--G_max', type=float, default=5.0, help="G max loss")

    parser.add_argument('--label_copy', type=int, default=20, help="times to copy label")
    parser.add_argument('--decoder_label', action='store_true', default = False, help="add decoder label or not")
    parser.add_argument('--exp_label', action='store_true', default = False, help="add exp label or not")

    parser.add_argument('--std', type=float, default=0.1, help="smooth loss for GAN")
    parser.add_argument('--alpha5',action='store_true',default = False,help='add the half switch or not')

    # classify parameters
    parser.add_argument('--VGG',default = False,type = bool, help='whether train the classify net VGG')
    parser.add_argument('--resume_dir',default='checkpoint',help = 'resume dir from checkpoint')  
    parser.add_argument('--resume', action='store_true', help='resume from checkpoint')
    parser.add_argument('--vgg_mode', default='train', type=str, choices=['train','test'],help='resume from checkpoint')
    parser.add_argument('--multi_add_gan',action='store_true', help='whther use multi-add gan picture')
    parser.add_argument('--ablation',action='store_true', help='whther in the ablation mode')

    args = parser.parse_args()
    print(args)
    all_start = time.clock()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu[0]
    if args.mode == 'test':
        assert args.swap + args.linear + args.matrix == 1
        assert args.restore is not None
        assert args.VGG is False

    model = ELEGANT(args)

    if args.VGG == True: # train the VGG
        VGG_model = Classify(args)

    epoch_vgg = 0
    if args.mode == 'train':
        for epoch in range(1, 1 + model.config.max_epoch):
            print("========== Training ELEGANT for 1 epoch ========")
            start = time.clock()
            model.train(epoch)
            print("Use time:", str((time.clock() - start) / 60), " min")
            print(epoch, "++++++++++++++")

            if epoch % config_VGG.train_gan_times == 0:
                print("========== Generating pictures ============")
                model.generate()
                print("========== End Generating pictures ===========")

                print("========== Generating fx ============")
                model.generate_fx()
                print("========== End Generating fx ===========")

                for i in range(1, 1 + config_VGG.train_vgg_times):
                    print("=========== Training VGG for ", epoch_vgg, "th epoch ============")
                    start = time.clock()
                    VGG_model.train(epoch_vgg)
                    print("Use time:", str((time.clock() - start) / 60), " min")
                    epoch_vgg += 1


    elif args.mode == 'test' and args.swap:
        assert len(args.swap_list) == 2 #and args.input and len(args.target) == 1
        attribute_test = args.attributes[int(args.swap_list[0])] # happy
        attribute_test_b = args.attributes[int(args.swap_list[1])]  # sad

        test_exp_list = []
        test_exp_list_b = []
        # test_natural_list = []
        f = open(config.data_dir+"/list_attr_multipie_test.txt")
        lines = f.readlines()
        for idx, item in enumerate(lines[1].strip().split()):
            if item == attribute_test:
                attribute_id = idx
            if item == attribute_test_b:
                attribute_id_b = idx

        for line in lines[2:]:
            label = line.strip().split()[int(attribute_id) + 1]
            name = line.strip().split()[0]
            if label == '1':
                test_exp_list.append(name)
            label_b = line.strip().split()[int(attribute_id_b) + 1]
            name_b = line.strip().split()[0]
            if label_b == '1':
                test_exp_list_b.append(name_b)

        test_idx = 0
        random.shuffle(test_exp_list)
        random.shuffle(test_exp_list_b)
        for item in test_exp_list[:10]:
            test_input = config.data_dir+'/align_5p/' + item
            for exp in test_exp_list_b[:10]:
                test_target = config.data_dir+'/align_5p/' + exp
                model.swap(test_idx, attribute_id, attribute_id_b, test_input, test_target)
                print(test_idx)
                test_idx = test_idx + 1

    else:
        raise NotImplementationError()

    print("All Use time:",str((time.clock()-all_start)/60)," min")


if __name__ == "__main__":
    main()

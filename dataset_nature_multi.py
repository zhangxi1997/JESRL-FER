import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import os,time
from PIL import Image


class Config:
    @property
    def data_dir(self):
        data_dir = '../dataset/multipie'
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
        return data_dir

    @property
    def exp_dir(self):
        exp_dir = os.path.join('./checkpoint/checkpoint_multipie/')
        if not os.path.exists(exp_dir):
            os.makedirs(exp_dir)
        return exp_dir

    @property
    def model_dir(self):
        model_dir = os.path.join(self.exp_dir, 'model')
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        return model_dir

    @property
    def log_dir(self):
        log_dir = os.path.join(self.exp_dir, 'log')
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        return log_dir

    @property
    def img_dir(self):
        img_dir = os.path.join(self.exp_dir, 'img')
        if not os.path.exists(img_dir):
            os.makedirs(img_dir)
        return img_dir

    @property
    def test_dir(self):
        test_dir = os.path.join(self.exp_dir,'test_result')
        if not os.path.exists(test_dir):
            os.makedirs(test_dir)
        return test_dir

    @property
    def fx_dir(self):
        fx_dir = os.path.join(self.data_dir,'GAN_data_multipie_fx')
        if not os.path.exists(fx_dir):
            os.makedirs(fx_dir)
        return fx_dir

    @property
    def gan_dir(self):
        gan_dir = os.path.join(self.data_dir, 'GAN_data_multipie')
        if not os.path.exists(gan_dir):
            os.makedirs(gan_dir)
        return gan_dir

    nchw = [10,3,256,256]

    G_lr = 2e-4

    D_lr = 2e-4

    betas = [0,5, 0.999]

    weight_decay = 1e-5

    step_size = 300

    gamma = 0.97

    shuffle = True

    num_workers = 4

    max_epoch = 50

    attr = ['surp','fear','disg','happy','sad','angry']

config = Config()

class Config_VGG:
    @property
    def checkpoint(self):
        exp_dir = os.path.join('./checkpoint/checkpoint_multipie_twostage/VGG')
        if not os.path.exists(exp_dir):
            os.makedirs(exp_dir)
        return exp_dir

    @property
    def log_dir(self):
        log_dir = os.path.join(self.checkpoint, 'log')
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        return log_dir

    ncwh = [12,3,256,256]
    len_trainset = 12771
    len_testset = 3068
    num_workers = 4
    shuffle = True
    step_size = 1800
    gamma = 0.97 # lr down param
    gan_raw = 6 # generated data is 6 times to the raw data
    train_gan_times = 5
    train_vgg_times = 4
    lr = 0.001

config_VGG = Config_VGG()

class SingleCelebADataset(Dataset):
    def __init__(self, im_names, labels, config):
        self.im_names = im_names
        self.labels = labels
        self.config = config

    def __len__(self):
        return len(self.im_names)

    def __getitem__(self, idx):
        image = Image.open(self.im_names[idx])
        image = self.transform(image) * 2 - 1
        label = (self.labels[idx] + 1) / 2
        return image, label

    @property
    def transform(self):
        transform = transforms.Compose([
            transforms.Resize(self.config.nchw[-2:]),
            transforms.ToTensor(),
        ])
        return transform

    
    def gen(self):
        dataloader = DataLoader(self, batch_size=self.config.nchw[0], shuffle=False, num_workers=self.config.num_workers, drop_last=True)
        while True:
            for data in dataloader:
                yield data
    

class MultiCelebADataset(object):
    def __init__(self, attributes, config=config):
        self.attributes = attributes
        self.config = config
        self.datanum = 0
        im_names_list = []
        all_labels_list = []
        # use all pic(train+test) to train the GAN
        # only use the train_pic to generate pic used for train VGG
        with open(os.path.join(self.config.data_dir, 'list_attr_multipie_train.txt'), 'r') as f:
            lines = f.read().strip().split('\n')
            col_ids = [lines[1].split().index(attribute) + 1 for attribute in self.attributes]
            for x in lines[2:]:
                all_labels_list.append([int(x.split()[col_id]) for col_id in col_ids])

            for idx in range(len(all_labels_list)):
                im_names_list.append(os.path.join(self.config.data_dir, 'align_5p/train_{:04d}.jpg'.format(idx+1)))

        self.im_names = np.array(im_names_list)
        self.all_labels = np.array(all_labels_list, dtype=np.float32)
            
        self.dict = {i: {True: None, False: None} for i in range(len(self.attributes))}
        for attribute_id in range(len(self.attributes)):
            #for is_positive in [True, False]:
            idxs = np.where(self.all_labels[:,attribute_id] == (int(True)*2 - 1))[0]
            im_names = self.im_names[idxs]
            labels = self.all_labels[idxs]
            data_gen = SingleCelebADataset(im_names, labels, self.config).gen()
            self.dict[attribute_id][True] = data_gen
            self.datanum += len(im_names)
        self.datanum = self.datanum 

        # last att: natural att   to be negative
        for attribute_id in range(len(self.attributes)):
            #idxs = np.where(self.all_labels_nature[:,len(self.attributes)] == (int(True)*2 - 1))[0]
            idxs = np.where(self.all_labels[:,attribute_id] == (int(False)*2 - 1))[0]
            im_names = self.im_names[idxs]
            labels = self.all_labels[idxs]
            data_gen = SingleCelebADataset(im_names, labels, self.config).gen()
            self.dict[attribute_id][False] = data_gen
    def gen(self, attribute_id, is_positive):
        data_gen = self.dict[attribute_id][is_positive]
        return data_gen

class SingleDataset_VGG(Dataset):
    def __init__(self, im_names, labels, config, mode, img_type):
        self.im_names = im_names
        self.labels = labels
        self.config = config
        self.mode = mode
        self.img_type = img_type

    def __len__(self):
        return len(self.im_names)

    def __getitem__(self,idx):
        if self.img_type == 'raw':
            image = Image.open('../datasets/multipie/align_5p/'+self.im_names[idx])
            fx = torch.FloatTensor(2,4).zero_()
            
        if self.img_type == 'gan':
            image = Image.open(config.gan_dir + '/'+ self.im_names[idx])
            fx = np.load(file = config.fx_dir + '/'+ self.im_names[idx].split('.')[0] + '.npy')
            fx = torch.from_numpy(fx)

        if self.mode == 'train':
            image = self.transform(image)
        if self.mode == 'test':
            image = self.transform_test(image)
        
        label = (self.labels[idx])
        return image, fx, label

    @property
    def transform(self):
        transform = transforms.Compose([
            transforms.Resize(self.config.ncwh[-2:]),
            transforms.RandomHorizontalFlip(),
        
            transforms.ToTensor(),
            transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5)),
        ])
        return transform

    @property
    def transform_test(self):
        transform_test = transforms.Compose([
            transforms.Resize(self.config.ncwh[-2:]),
            transforms.ToTensor(),
            transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5)),
        ])
        return transform_test

class SingleDataset_GAN(Dataset):
    def __init__(self, input_names, target_names, config):
        self.input_names = input_names  # exp pic
        self.target_names = target_names # nature pic
        self.config = config

    def __len__(self):
        return len(self.target_names)

    def __getitem__(self,idx):
        input_image_ = Image.open(self.config.data_dir+'/align_5p/'+self.input_names[idx])
        target_image_ = Image.open(self.config.data_dir+'/align_5p/'+self.target_names[idx])

        input_image = self.transform(input_image_)
        target_image = self.transform(target_image_)

        return input_image, target_image, self.input_names[idx], self.target_names[idx]

    @property
    def transform(self):
        transform1 = transforms.Compose([
            transforms.Resize(self.config.nchw[-2:]),
            transforms.ToTensor(),
            transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5)),
        ])
        return transform1
        
def test():
    dataset = MultiCelebADataset(['Bangs', 'Smiling'])

    import cProfile
    pr = cProfile.Profile()
    pr.enable()
    for i in range(10):
        if i % 4 == 0:
            images, labels = next(dataset.gen(0, True))
        elif i % 4 == 1:
            images, labels = next(dataset.gen(0,False))
        elif i % 4 == 2:
            images, labels = next(dataset.gen(1,True))
        elif i % 4 == 3:
            images, labels = next(dataset.gen(1,False))

    pr.disable()
    from IPython import embed; embed(); exit()
    # pr.print_stats(sort='tottime')

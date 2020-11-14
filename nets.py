import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.utils.model_zoo as model_zoo

model_urls = {
    'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth',
    'vgg19_bn': 'https://download.pytorch.org/models/vgg19_bn-c79401a0.pth',

}

class NTimesTanh(nn.Module):
    def __init__(self, N):
        super(NTimesTanh, self).__init__()
        self.N = N
        self.tanh = nn.Tanh()

    def forward(self, x):
        return self.tanh(x) * self.N

class Normalization(nn.Module):
    def __init__(self):
        super(Normalization, self).__init__()
        self.alpha = Parameter(torch.ones(1))
        self.beta  = Parameter(torch.zeros(1))

    def forward(self, x):
        x = torch.nn.functional.normalize(x, dim=1)
        return x * self.alpha + self.beta


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.main = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(3,64,3,2,1,bias=True),
                Normalization(),
                nn.LeakyReLU(negative_slope=0.2),
            ),
            nn.Sequential(
                nn.Conv2d(64,128,3,2,1,bias=True),
                Normalization(),
                nn.LeakyReLU(negative_slope=0.2),
            ),
            nn.Sequential(
                nn.Conv2d(128,256,3,2,1,bias=True),
                Normalization(),
                nn.LeakyReLU(negative_slope=0.2),
            ),
            nn.Sequential(
                nn.Conv2d(256,512,3,2,1,bias=True),
                Normalization(),
                nn.LeakyReLU(negative_slope=0.2),
            ),
            nn.Sequential(
                nn.Conv2d(512,512,3,2,1,bias=True),
                Normalization(),
                nn.LeakyReLU(negative_slope=0.2),
            ),
        ])

        # init weight
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, 0, 0.02)
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.normal_(m.weight, 0, 0.02)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.normal_(m.weight, 1, 0.02)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.02)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, return_skip=True):
        skip = []
        for i in range(len(self.main)):
            x = self.main[i](x)
            if i < len(self.main) - 1:
                skip.append(x)
        if return_skip:
            return x, skip
        else:
            return x


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.main = nn.ModuleList([
            nn.Sequential(
                nn.ConvTranspose2d(1024,512,3,2,1,1,bias=True),
                Normalization(),
                nn.ReLU(),
            ),
            nn.Sequential(
                nn.ConvTranspose2d(512,256,3,2,1,1,bias=True),
                Normalization(),
                nn.ReLU(),
            ),
            nn.Sequential(
                nn.ConvTranspose2d(256,128,3,2,1,1,bias=True),
                Normalization(),
                nn.ReLU(),
            ),
            nn.Sequential(
                nn.ConvTranspose2d(128,64,3,2,1,1,bias=True),
                Normalization(),
                nn.ReLU(),
            ),
            nn.Sequential(
                nn.ConvTranspose2d(64,3,3,2,1,1,bias=True),
            ),
        ])
        self.activation = NTimesTanh(2)

        # init weight
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, 0, 0.02)
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.normal_(m.weight, 0, 0.02)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.normal_(m.weight, 1, 0.02)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.02)
                nn.init.constant_(m.bias, 0)

    def forward(self, enc1, enc2, skip=None):
        x = torch.cat([enc1, enc2], 1)
        for i in range(len(self.main)):
            x = self.main[i](x)
            if skip is not None and i < len(skip):
                x += skip[-i-1]
        return self.activation(x)

class Decoder_label(nn.Module):
    def __init__(self, label_copy, label_length):
        super(Decoder_label, self).__init__()
        self.label_copy = label_copy
        self.label_length = label_length

        self.main = nn.ModuleList([
            nn.Sequential(
                nn.ConvTranspose2d(1024 + self.label_length*self.label_copy,512,3,2,1,1,bias=True),
                Normalization(),
                nn.ReLU(),
            ),
            nn.Sequential(
                nn.ConvTranspose2d(512,256,3,2,1,1,bias=True),
                Normalization(),
                nn.ReLU(),
            ),
            nn.Sequential(
                nn.ConvTranspose2d(256,128,3,2,1,1,bias=True),
                Normalization(),
                nn.ReLU(),
            ),
            nn.Sequential(
                nn.ConvTranspose2d(128,64,3,2,1,1,bias=True),
                Normalization(),
                nn.ReLU(),
            ),
            nn.Sequential(
                nn.ConvTranspose2d(64,3,3,2,1,1,bias=True),
            ),
        ])
        self.activation = NTimesTanh(2)

        # init weight
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, 0, 0.02)
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.normal_(m.weight, 0, 0.02)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.normal_(m.weight, 1, 0.02)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.02)
                nn.init.constant_(m.bias, 0)

    def forward(self, enc1, enc2, label, skip=None):
        new_label = label.view((enc1.shape[0], self.label_length*self.label_copy, 1, 1)).expand((enc1.shape[0], self.label_length*self.label_copy, enc1.shape[2], enc1.shape[3]))
        x = torch.cat([enc1, enc2, new_label], 1)
        for i in range(len(self.main)):
            x = self.main[i](x)
            if skip is not None and i < len(skip):
                x += skip[-i-1]
        return self.activation(x)


class Discriminator(nn.Module):
    def __init__(self, n_attributes, img_size):
        super(Discriminator, self).__init__()
        self.n_attributes = n_attributes
        self.img_size = img_size
        self.conv = nn.Sequential(
            nn.Conv2d(3+n_attributes,64,3,2,1,bias=True),
            Normalization(),
            nn.LeakyReLU(negative_slope=0.2),

            nn.Conv2d(64,128,3,2,1,bias=True),
            Normalization(),
            nn.LeakyReLU(negative_slope=0.2),

            nn.Conv2d(128,256,3,2,1,bias=True),
            Normalization(),
            nn.LeakyReLU(negative_slope=0.2),

            nn.Conv2d(256,512,3,2,1,bias=True),
            Normalization(),
            nn.LeakyReLU(negative_slope=0.2),
        )
        self.linear = nn.Sequential(
            #nn.Conv2d(512,1,1,1,0,bias=True),
            nn.Linear(512*(self.img_size//16)*(self.img_size//16), 1),
            nn.Sigmoid(),
        )
        self.downsample = torch.nn.AvgPool2d(2, stride=2)

        # init weight
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, 0, 0.02)
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.normal_(m.weight, 0, 0.02)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.normal_(m.weight, 1, 0.02)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.02)
                nn.init.constant_(m.bias, 0)

    def forward(self, image, label):
        '''
        image: (n * c * h * w)
        label: (n * n_attributes)
        '''
        while image.shape[-1] != self.img_size or image.shape[-2] != self.img_size:
            image = self.downsample(image)
        new_label = label.view((image.shape[0], self.n_attributes, 1, 1)).expand((image.shape[0], self.n_attributes, image.shape[2], image.shape[3]))
        x = torch.cat([image, new_label], 1)
        output = self.conv(x)
        output = output.view(output.shape[0], -1)
        output = self.linear(output)
        return output


class VGG_fx(nn.Module):
    def __init__(self, features, num_classes=7, init_weights=True):
        super(VGG_fx, self).__init__()
        self.features = features
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            #nn.Linear(4096, num_classes),
        )
        self.fx_linear = nn.Sequential(
            nn.Conv2d(1024,512,1,1,0,bias=True),
            Normalization(),
            nn.LeakyReLU(negative_slope=0.2),
        )
        self.new_fc = nn.Linear(4096, num_classes)
        if init_weights:
            self._initialize_weights()

    def forward(self, x, fx):
        x = self.features(x)# 512*8*8

        fx = self.fx_linear(fx) # maybe delete 1024->512
        
        x = torch.cat([x,fx],1) # 1024*8*8
        x = self.fx_linear(x) # 512*8*8
        x = self.avgpool(x) # x输出为512*7*7
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        x = self.new_fc(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')#, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

class VGG(nn.Module):

    def __init__(self, features, num_classes=7, init_weights=True):
        super(VGG, self).__init__()
        self.features = features
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            #nn.Linear(4096, num_classes),
        )
        self.new_fc = nn.Linear(4096, num_classes)
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        x = self.new_fc(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')#, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


def vgg19(num_class,pretrained=False,**kwargs):
    """VGG 19-layer model (configuration "E")
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG(make_layers(cfg['E']) ,num_class, **kwargs)
    if pretrained:
        pretrained_dict = model_zoo.load_url(model_urls['vgg19'])
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        print("Success load the pretrained vgg19 model\n")
    return model


def vgg19_bn_fx(num_class, pretrained=False, **kwargs):
    """VGG 19-layer model (configuration 'E') with batch normalization
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG_fx(make_layers(cfg['E'], batch_norm=True) ,num_class, **kwargs)
    if pretrained:
        pretrained_dict=model_zoo.load_url(model_urls['vgg19_bn'])
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        print("Success load the pretrained vgg model\n")
    return model

def vgg19_bn(num_class,pretrained=False, **kwargs):
    """VGG 19-layer model (configuration 'E') with batch normalization
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG(make_layers(cfg['E'], batch_norm=True), num_class,**kwargs)
    if pretrained:
        pretrained_dict=model_zoo.load_url(model_urls['vgg19_bn'])
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        print("Success load the pretrained vgg model\n")
    return model

def vgg_bn_loss(num_class):
    vgg_bn = vgg19_bn(num_class,pretrained = True)
    checkpoint = torch.load('./VGG_loss/'+str(num_class)+'_max_ckpt.t7')
    vgg_bn.load_state_dict(checkpoint['net'])
    return vgg_bn

class Vgg19_for_loss(torch.nn.Module):
    def __init__(self, num_class,requires_grad=False):
        super(Vgg19_for_loss, self).__init__()
        vgg_pretrained_features = vgg_bn_loss(num_class).features #models.vgg19_bn(pretrained=True).features
        #exit()
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)        
        h_relu3 = self.slice3(h_relu2)        
        h_relu4 = self.slice4(h_relu3)        
        h_relu5 = self.slice5(h_relu4)                
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        #print(out)
        return out

class VGGLoss(nn.Module):
    def __init__(self, num_class,gpu_ids):
        super(VGGLoss, self).__init__()
        self.vgg = Vgg19_for_loss(num_class).cuda()
        self.criterion = nn.L1Loss()
        self.weights = [1.0/32, 1.0/16, 1.0/8, 1.0/4, 1.0]        

    def forward(self, x, y):              
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        loss = 0
        for i in range(len(x_vgg)):
            loss += self.weights[i] * self.criterion(x_vgg[i], y_vgg[i].detach())        
        return loss


if __name__ == "__main__":
    '''
    enc = Encoder()
    dec = Decoder()
    D1 = Discriminator(3, 256)
    D2 = Discriminator(3, 128)

    imgs = Variable(torch.rand(32,3,256,256))
    labels = Variable(torch.ones(32,3))

    out, skip = enc(imgs)
    rec = dec(enc1=out, enc2=out, skip=skip)

    fake1 = D1(imgs, labels)
    fake2 = D2(imgs, labels)
    '''

    from IPython import embed; embed(); exit()

import cv2
import toml
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from torch.autograd.function import once_differentiable
from torch.utils.data import DataLoader
#from prefetch_generator import BackgroundGenerator
import flow_vis


H = 128

def show_img_from_tensor(tensor, path):
    tensor_ = tensor.cpu().view(H, H).numpy() * 255
    cv2.imwrite(path, tensor_)

def show_att_from_tensor(tensor, path):
    tensor_ = (tensor.cpu().view(H, H).numpy() + 1) * 127
    cv2.imwrite(path, tensor_)

def show_df_from_tensor(tensor, path):
    """
    tensor = torch.sqrt(torch.abs(tensor.cpu()))
    temp = torch.zeros(1, H, H)
    tensor_ = torch.cat([tensor, temp], 0).permute(1, 2, 0)
    tensor_ = tensor_.numpy() * 255
    cv2.imwrite(path, tensor_)
    """
    tensor = tensor.cpu().permute(1,2,0).numpy()
    #tensor = np.rollaxis(tensor, 2, 0)
    flow_color = flow_vis.flow_to_color(tensor, convert_to_bgr=True)
    cv2.imwrite(path, flow_color)

def deform(source_imgs, df):
    #source_imgs.shape (B, 1, H, W)
    #df.shape (B, 2, H, W)
    #output.shape (B, 1, H, W)
    batch_size = source_imgs.size(0)
    xs = np.linspace(-1, 1, df.size(2))
    xs = np.meshgrid(xs, xs)
    xs = np.stack(xs, 2)
    xs = torch.tensor(xs).unsqueeze(0).repeat(batch_size, 1,1,1).cuda()
    xs  = xs.float()
    sampler_xs = (df.permute(0,2,3,1) + xs).clamp(min=-1,max=1)
    output_img = F.grid_sample(source_imgs.detach(), sampler_xs)
    return output_img

class Maskloss(nn.Module):
    def __init__(self, k):
        super().__init__()
        self.k = k
    
    def forward(self, X, Y):
        assert(X.shape == Y.shape)
        batch_size = X.size(0)
        frame_size = X.size(1)
        H = X.size(4)
        loss1 = torch.sum(torch.abs(X - Y))
        loss2 = torch.sum((self.k -1) * torch.abs(X[:,:,:,7:40,7:104] - Y[:,:,:,7:40,7:104]))
        loss3 = torch.sum((self.k -1) * torch.abs(X[:,:,:,65:90,23:87] - Y[:,:,:,65:90,23:87]))
        loss = (loss1 + loss2 + loss3) / (batch_size * frame_size * H * H)
        
        return loss

class DCL(nn.Module):
    def __init__(self):
        super(DCL, self).__init__()

    def loss_func(self, X, Y):
        temp = torch.sub(X, Y)
        temp = torch.pow(temp, 2)
        loss = torch.mean(torch.add(temp[:,0], temp[:,1]))

        return loss

    def forward(self, X):
        # X shape (B, 2, H, W)
        loss = self.loss_func(X[:,:,:,:-1], X[:,:,:,1:])
        loss += self.loss_func(X[:,:,:-1,:], X[:,:,1:,:])
        loss += self.loss_func(X[:,:,:-1,:-1], X[:,:,1:,1:])
        loss += self.loss_func(X[:,:,:-1,1:], X[:,:,1:,:-1])

        return loss * 2

class PyramidLoss(nn.Module):
    def __init__(self, config):
        super(PyramidLoss, self).__init__()
        self.eta = config["hyperparameters"]["eta"] # TODO add eta to config file i.e. toml config file
        self.size = config["hyperparameters"]["size"]
        self.list = []
        for s in self.size:
            self.list.append(nn.Upsample(size=s, mode="bilinear"))
        self.list = nn.ModuleList(self.list)
        self.criterion_l1 = nn.L1Loss()

    def forward(self, df, src, tar):
        # df of shape like (B, 2, ., .) where . can be 112, 56, 28, 14, 7, 7
        # src of shape (B, 1, H, W)
        # tar of shape (B, 1, H, W)
        loss = 0
        for (i,s) in enumerate(self.size):
            loss = loss + self.eta[i] * self.compare(df[i], self.list[i](src), self.list[i](tar))
        return loss

    def compare(self, df, source_imgs, target_imgs):
        #df.shape (B, 2, H, W)
        #source_imgs.shape (B, 1, H, W)
        #target_imgs.shape (B, 1, H, W)
        batch_size = source_imgs.size(0)
        xs = np.linspace(-1, 1, df.size(2))
        xs = np.meshgrid(xs, xs)
        xs = np.stack(xs, 2)
        xs = torch.tensor(xs).unsqueeze(0).repeat(batch_size, 1,1,1).cuda()
        xs  = xs.float()
        sampler_xs = (df.permute(0,2,3,1) + xs).clamp(min=-1,max=1)
        output_imgs = F.grid_sample(source_imgs.detach(), sampler_xs)

        loss = self.criterion_l1(target_imgs, output_imgs)

        return loss


def init(net):
    for m in net.modules():
        if isinstance(m, nn.Conv3d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
            if m.bias is not None:
                m.bias.data.zero_()

        elif isinstance(m, nn.Conv2d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
            if m.bias is not None:
                m.bias.data.zero_()

        elif isinstance(m, nn.Conv1d):
            n = m.kernel_size[0] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
            if m.bias is not None:
                m.bias.data.zero_()

        elif isinstance(m, nn.BatchNorm3d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()

        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()

        elif isinstance(m, nn.BatchNorm1d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()

class DataLoaderX(DataLoader):

    def __iter__(self):
        return BackgroundGenerator(super().__iter__())


class Distill(nn.Module):
    def __init__(self, eta=1, T=20):
        super(Distill, self).__init__()
        self.eta = eta
        self.T = T
        self.softmax = nn.Softmax(dim=1)
        self.kl = nn.KLDivLoss(reduction='batchmean')
        self.cel = nn.CrossEntropyLoss()

    def forward(self, df, gray, idx):
        # cel
        loss_df = self.cel(df, idx)
        loss_gray = self.cel(gray, idx)
        loss_ce = loss_df + loss_gray

        # kld
        df = self.softmax(df / self.T)
        gray = self.softmax(gray / self.T)
        loss_df2gray = self.kl(torch.log(df), gray)
        loss_gray2df = self.kl(torch.log(gray), df)
        loss_kld = loss_df2gray + loss_gray2df

        loss = loss_ce + self.eta * loss_kld

        return loss, loss_ce, loss_kld


class AdjustLR(object):
    def __init__(self, optimizer, init_lr, gamma=0.5, sleep_epoch=10, half=5, verbose=False):
        super(AdjustLR, self).__init__()
        self.optimizer = optimizer
        self.init_lr = init_lr
        self.gamma = gamma
        self.sleep_epoch = sleep_epoch - 1
        self.half = half
        self.verbose = verbose

    def step(self, epoch):
        if epoch >= self.sleep_epoch:
            for param_group in self.optimizer.param_groups:
                new_lr = self.init_lr * math.pow(self.gamma, (epoch-self.sleep_epoch+1)/self.half)
                param_group["lr"] = new_lr
            if self.verbose:
                print(">>>adjust learning rate to {}<<<".format(new_lr))

def save_checkpoint(state, path):
    direc = os.path.split(path)[0]
    if not os.path.exists(direc):
        os.makedirs(direc)
    torch.save(state, path)
'''
Creator: Odd
Date: 2022-11-05 00:58:53
LastEditTime: 2022-11-05 11:33:39
FilePath: \torch_captcha\train.py
Description: 
'''
import torch
from torch import nn
from loss_function import MaskedSoftmaxCELoss
from tqdm import tqdm
from dataset import CaptchaDataset
from torchvision import transforms
from model import CRNN
from torch.utils.data import DataLoader
from params import *


def try_gpu(i=0):
    if torch.cuda.device_count() >= i + 1:
        return torch.device(f'cuda:{i}')
    return torch.device('cpu')

def eval():
    pass


def train(net, data_iter, lr, num_epochs, device):
    def xavier_init_weights(m):
        if type(m) == nn.Linear:
            nn.init.xavier_uniform_(m.weight)
        if type(m) == nn.GRU:
            for param in m._flat_weights_names:
                # weight初始化，bias初值为0
                if "weight" in param:
                    nn.init.xavier_uniform_(m._parameters[param])

    net.apply(xavier_init_weights)
    net.to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    loss = MaskedSoftmaxCELoss()
    net.train()
    
    
    losses = []
    accuracy = []
    loop = tqdm((data_iter), total = len(data_iter))

    for epoch in range(num_epochs):
        for batch in loop:
            optimizer.zero_grad()
            X,  Y, Y_valid_len = [x.to(device) for x in batch]
            y_hat = net(X)
            l = loss(y_hat, Y, Y_valid_len)
            l.sum().backward()
            optimizer.step()
            
            loop.set_description(f'Epoch_[{epoch+1}/{num_epochs}]')
            loop.set_postfix(loss = l.sum().item())


if __name__ == '__main__':

    batch_size = 64
    device = try_gpu()
    learning_rate = 1e-3
    num_epochs = 100

    train_dataset = CaptchaDataset(img_dir='train', transform=transforms.Compose([
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.2, 0.2, 0.2))
    ]), one_hot=False)
    
    train_dataloder = DataLoader(train_dataset, batch_size, shuffle=True)
    net = CRNN(charset_len=len(charset))

    train(net, train_dataloder, learning_rate, num_epochs, device)

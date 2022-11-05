'''
Creator: Odd
Date: 2022-11-04 20:10:31
LastEditTime: 2022-11-05 11:32:32
FilePath: \torch_captcha\dataset.py
Description: 
'''
from torch.utils.data import Dataset
from torchvision.io import read_image
import torch.nn.functional as F
from torchvision import transforms
import os
import torch
from torch import nn
from params import *


class CaptchaDataset(Dataset):
    def __init__(self, img_dir, time_step=10, transform=None, one_hot=False):
        self.img_dir = img_dir
        self.time_step = time_step
        self.imgs_name = os.listdir(img_dir)
        self.transform = transform
        self.one_hot = one_hot
        self.char_to_idx = {ch: idx for idx, ch in enumerate(charset)}

    def __len__(self):
        return len(self.imgs_name)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.imgs_name[idx])
        image = read_image(img_path)/255
        label = self.imgs_name[idx].split('.')[0]
        length = len(label)
        if self.transform:
            image = self.transform(image.float())
        labels = [self.char_to_idx[c] for c in label] + [self.char_to_idx['<eos>']] + [
            self.char_to_idx['<pad>'] for _ in range(self.time_step-length-1)]
        labels = torch.tensor(labels, dtype=torch.long)
        if self.one_hot:
            label = F.one_hot(label, num_classes=10)
            label = label.view(-1).float()
        return image, labels, length


if __name__ == '__main__':
    # 测试代码
    test_data = CaptchaDataset(img_dir='test', time_step=10, transform=transforms.Compose([
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.2, 0.2, 0.2))
    ]), one_hot=False)
    for i, j, k in test_data:
        print(i.shape)
        print(j)
        print(k)
        break

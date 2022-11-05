'''
Creator: Odd
Date: 2022-11-04 18:05:01
LastEditTime: 2022-11-05 10:57:20
FilePath: \torch_captcha\model.py
Description: Model for Captcha
'''
import torch
from torch import nn
from params import *

class CRNN(nn.Module):
    '''
    CRNN网络
    '''

    def __init__(self, charset_len, max_len=10) -> None:
        super().__init__()
        self.charset_len = charset_len

        self.max_len = max_len

        # cnn提取图像特征
        self.cnn = nn.Sequential(
            # in: 3*60*160, out: 32*60*160
            nn.Conv2d(3, 32, (3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(),
            nn.MaxPool2d((2, 2), stride=2),  # out: 32*30*80
            nn.Conv2d(32, 64, (3, 3), stride=(1, 1),
                      padding=(1, 1)),  # out: 64*30*80
            nn.ReLU(),
            nn.MaxPool2d((2, 2), stride=2),  # out: 64*15*40
            nn.Conv2d(64, 128, (3, 3), stride=(1, 1),
                      padding=(1, 1)),  # out: 128*15*40
            nn.ReLU(),
            nn.MaxPool2d((5, 2), stride=2),  # out: 128*6*20
            nn.Conv2d(128, 256, (3, 3), stride=(1, 1),
                      padding=(1, 1)),  # out: 256*6*20
            nn.ReLU(),
            nn.MaxPool2d((4, 2), stride=2),  # out: 256*2*10
            nn.Conv2d(256, 256, (3, 3), stride=(1, 1),
                      padding=(1, 1)),  # out: 256*2*10
            nn.ReLU(),
            nn.MaxPool2d((2, 1), stride=(2, 1)),  # out: 256*1*10
        )

        # rnn 抽出时序特征
        self.rnn = nn.Sequential(
            nn.GRU(256, 256, bidirectional=True),
        )

        self.dense = nn.Linear((max_len+1)*512, max_len*charset_len)

    def forward(self, X):
        X = self.cnn(X)
        X = X.squeeze(2)
        X = X.permute(2, 0, 1)

        out, state = self.rnn(X)
        state = state.reshape(-1, out.shape[1], out.shape[-1])
        out = torch.concat((out, state), dim=0)

        t, b, c = out.shape
        out = out.reshape(b, t*c)

        out = self.dense(out)

        return out.reshape(-1, 10, self.charset_len)
        # return state


if __name__ == '__main__':
    net = CRNN(charset_len=len(charset))
    out = net(torch.randn((64, 3, 60, 160)))
    print(out.shape)

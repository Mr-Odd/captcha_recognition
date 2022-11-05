'''
Creator: Odd
Date: 2022-11-05 01:32:34
LastEditTime: 2022-11-05 01:53:18
FilePath: \torch_captcha\loss_function.py
Description: 
'''
from torch import nn
from torch import Tensor
import torch


class MaskedSoftmaxCELoss(nn.CrossEntropyLoss):
    def forward(self, input: Tensor, target: Tensor, valid_len) -> Tensor:

        def sequence_mask(X, valid_len, value=0):
            maxlen = X.size(1)

            # None会把向量从一维变成二维
            mask = torch.arange((maxlen), dtype=torch.float32,
                                device=X.device)[None, :] < valid_len[:, None]
            # ~相当于取非
            X[~mask] = value
            return X

        weights = torch.ones_like(target)
        weights = sequence_mask(weights, valid_len)
        # 无需其他求mean/sum操作
        self.reduction = 'none'

        # pytorch实现中，需要把预测的维度放在中间
        unweighted_loss = super(MaskedSoftmaxCELoss, self).forward(
            input.permute(0, 2, 1), target
        )
        # 相乘表示有效的长度留下值，其他都为0，并对每个句子取平均，返回的shape是[batch]
        weighted_loss = (unweighted_loss * weights).mean(dim=1)
        return weighted_loss
    
if __name__ == '__main__':
    target = torch.empty(3, dtype=torch.long).random_(5)
    print(target)

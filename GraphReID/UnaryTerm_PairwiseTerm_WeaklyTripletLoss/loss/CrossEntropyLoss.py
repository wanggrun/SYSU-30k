#!/usr/bin/env python
# -*- coding: utf-8 -*-
import torch
from torch import nn
from torch.nn import functional as F

class CrossEntropyLoss(nn.Module):
    """Cross entropy loss with soft label.
    """
    def __init__(self, use_gpu=True):
        super(CrossEntropyLoss, self).__init__()
        self.use_gpu = use_gpu
        self.logsoftmax = nn.LogSoftmax(dim=1)
        self.softmax = nn.Softmax(dim=1)
        self.cse_loss = nn.CrossEntropyLoss()

    def forward(self, inputs, targets, low_level_feat):
        """
        """
        # log_probs = self.logsoftmax(inputs)

        # targets = torch.zeros(log_probs.size()).scatter_(1, targets.unsqueeze(1).data.cpu(), 1)
        if self.use_gpu: 
            targets = targets.cuda()
            targets = targets.type(torch.cuda.FloatTensor)
        targets = torch.autograd.Variable(targets, requires_grad=False)

        # targets = 0.9 * targets + 0.1 / 751
        # loss = (- targets * log_probs).mean(0).sum()
        input_softmax = self.softmax(inputs)
        targets = torch.argmax(targets * input_softmax, -1)

        n = inputs.size(0)
        mask = targets.expand(n, n).eq(targets.expand(n, n).t())
        mask = 1.0 - mask.float()

        # print(low_level_feat.size())
        dist = torch.pow(low_level_feat, 2).sum(dim=1, keepdim=True).expand(n, n)
        dist = dist + dist.t()
        dist.addmm_(1, -2, low_level_feat, low_level_feat.t())
        # dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
        sigma = 1.
        dist = torch.exp(- dist / sigma)

        # inner_prod = -torch.matmul(input_softmax, torch.log(input_softmax.t()))

        inner_prod = -torch.matmul(input_softmax, input_softmax.t())

        loss_pairwise = dist * inner_prod * mask
        loss_pairwise = loss_pairwise.mean()

        # for i in range(n):
        #     dist_ap.append(dist[i][mask[i]].max().unsqueeze(0))
        #     dist_an.append(dist[i][mask[i] == 0].min().unsqueeze(0))
        loss = self.cse_loss(inputs, targets) + 2. * loss_pairwise
        return loss

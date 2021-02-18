# coding: utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F


class SoftCrossEntropy(nn.Module):
    def __init__(self):
        super(SoftCrossEntropy, self).__init__()
        return

    def forward(self, inputs, target):
        log_likelihood = - F.log_softmax(inputs, dim=1)
        loss = torch.sum(torch.mul(log_likelihood, target)) / inputs.shape[0]

        return loss

    def forward_each_example(self, inputs, target):
        log_likelihood = - F.log_softmax(inputs, dim=1)
        loss = torch.sum(torch.mul(log_likelihood, target), dim=1)

        return loss


class SmoothCrossEntropyLoss(nn.Module):
    def __init__(self, label_smoothing=0.0, size_average=True):
        super(SmoothCrossEntropyLoss. self).__init__()
        self.label_smoothing = label_smoothing
        self.size_average = size_average

    def forward(self, input, target):
        if len(target.size()) == 1:
            target = torch.nn.functional.one_hot(target, num_classes=input.size(-1))
            target = target.float().cuda()
        if self.label_smoothing > 0.0:
            s_by_c = self.label_smoothing / len(input[0])
            smooth = torch.zeros_like(target)
            smooth = smooth + s_by_c
            target = target * (1. - s_by_c) + smooth

        return cross_entropy(input, target, self.size_average)


def cross_entropy(input, target, size_average=True):
    logsoftmax = torch.nn.LogSoftmax(dim=1)
    if size_average:
        return torch.mean(torch.sum(-target * logsoftmax(input), dim=1))
    else:
        return torch.sum(torch.sum(-target * logsoftmax(input), dim=1))

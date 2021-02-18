# coding: utf-8
import torch.nn as nn
import numpy as np
import util


class ConvNet(nn.Module):
    def __init__(self, num_classes, num_channel, size_after_cnn, n_aug):
        super(ConvNet, self).__init__()
        self.num_classes = num_classes
        self.size_after_cnn = size_after_cnn
        self.n_aug = n_aug

        self.layer1 = nn.Sequential(
            nn.Conv2d(num_channel, 32, kernel_size=3),
            nn.BatchNorm2d(32),
            nn.ReLU())
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(32),
            nn.ReLU())
        self.layer3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3),
            nn.BatchNorm2d(64),
            nn.ReLU())
        self.layer4 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(64),
            nn.ReLU())
        self.fc = nn.Linear(size_after_cnn * size_after_cnn * 64, num_classes)

    def forward(self, x, y, flag_spa=0, flag_noise=0, index=0):
        if flag_spa == 1:
            x, y = util.self_paced_augmentation(images=x,
                                                labels=y,
                                                flag_noise=flag_noise,
                                                index=index,
                                                n_aug=self.n_aug,
                                                num_classes=self.num_classes)
            out = self.layer1(x)
            out = self.layer2(out)
            out = self.layer3(out)
            out = self.layer4(out)
            out = out.reshape(out.size(0), -1)
            out = self.fc(out)
        else:
            if self.n_aug != 0:
                x, y = util.run_n_aug(x, y, self.n_aug, self.num_classes)
            out = self.layer1(x)
            out = self.layer2(out)
            out = self.layer3(out)
            out = self.layer4(out)
            out = out.reshape(out.size(0), -1)
            out = self.fc(out)

        return out, y

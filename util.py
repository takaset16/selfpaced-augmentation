# coding: utf-8
import torch
from torch.autograd import Variable
import sklearn.utils
import augmentation
import numpy as np
import cv2


def to_var(x):
    if torch.cuda.is_available():
        x = x.cuda()

    return Variable(x)


def to_var_grad(x):
    if torch.cuda.is_available():
        x = x.cuda()

    return Variable(x, requires_grad=True)


def shuffle(x, i):

    return sklearn.utils.shuffle(x, random_state=i)


def make_training_data(x, num, loop):
    x = shuffle(x, 1001)
    x_training = x[0:num]

    return x_training


def make_training_test_data(x, num, loop):
    x = shuffle(x, 1001)
    x_test = x[0:num]
    x_training = x[num:]

    return x_training, x_test


def run_n_aug(x, y, n_aug, num_classes):
    # x = augmentation.horizontal_flip(x)
    # x = augmentation.vertical_flip(x)
    # x = augmentation.random_crop(x)
    # x = augmentation.random_transfer(x)
    # x = augmentation.random_rotation(x)
    # x, y = augmentation.mixup(image=x, label=y, num_classes=num_classes)
    # x = augmentation.cutout(x)
    # x = augmentation.random_erasing(x)
    # x, y = augmentation.ricap(image_batch=x, label_batch=y, num_classes=num_classes)

    if n_aug == 1:
        x = augmentation.horizontal_flip(x)
        # x = augmentation.vertical_flip(x)
    elif n_aug == 2:
        x = augmentation.random_crop(x)
    elif n_aug == 3:
        x = augmentation.random_transfer(x)
    elif n_aug == 4:
        x = augmentation.random_rotation(x)
    elif n_aug == 5:
        x, y = augmentation.mixup(image=x, label=y, num_classes=num_classes)
    elif n_aug == 6:
        x = augmentation.cutout(x)
    elif n_aug == 7:
        x = augmentation.random_erasing(x)
    elif n_aug == 8:
        x, y = augmentation.ricap(image_batch=x, label_batch=y, num_classes=num_classes)
        x = to_var(x)
        y = to_var(y)
    elif n_aug == 12:
        x = augmentation.horizontal_flip(x)
        x = augmentation.random_crop(x)
    elif n_aug == 17:
        x = augmentation.horizontal_flip(x)
        x = augmentation.random_erasing(x)
    elif n_aug == 34:
        x = augmentation.random_transfer(x)
        x = augmentation.random_rotation(x)

    return x, y


def self_paced_augmentation(images, labels, flag_noise, index, n_aug, num_classes):
    x, y = run_n_aug(images, labels, n_aug, num_classes)

    images = np.array(images.data.cpu())
    labels = np.array(labels.data.cpu())
    x = np.array(x.data.cpu())
    y = np.array(y.data.cpu())
    x = np.where(flag_noise[index].reshape(-1, 1, 1, 1) < 1, images, x)

    if labels.ndim > 1:
        y = np.where(flag_noise[index].reshape(-1, 1) < 1, labels, y)

    x = to_var(torch.from_numpy(x).float())
    y = to_var(torch.from_numpy(y))

    return x, y


def flag_update(loss, judge_noise):
    flag_noise = np.where(loss < judge_noise, 0, 1)
    # flag_noise = np.where(loss > judge_noise, 0, 1)

    return flag_noise


def accuracy(output, target, topk=(1,)):
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))

        return res


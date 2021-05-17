# coding: utf-8
import torch
import torchvision
from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image

import numpy as np
import util
from augmentations import *


_IMAGENET_PCA = {
    'eigval': [0.2175, 0.0188, 0.0045],
    'eigvec': [
        [-0.5675,  0.7192,  0.4009],
        [-0.5808, -0.0045, -0.8140],
        [-0.5836, -0.6948,  0.4203],
    ]
}


class DataSetXY(Dataset):
    def __init__(self, x, y, transform=None):
        self.x = x
        self.y = y
        self.transform = transform

    def __getitem__(self, index):
        x = self.x[index]
        y = self.y[index]

        if self.transform is not None:
            x = self.transform(x)
        return x, y

    def __len__(self):
        return len(self.x)


class MyDataset_training(Dataset):
    def __init__(self, n_data, num_data, seed, flag_randaug, rand_n, rand_m, cutout):
        self.sampler = None

        """Preprocessing"""
        transform_train = None

        if n_data == 'MNIST':
            transform_train = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.1307,), std=(0.3081,))
            ])
        elif n_data == 'CIFAR-10' or n_data == 'CIFAR-100':
            if num_data != 0:
                transform_train = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010))
                ])
            else:
                transform_train = transforms.Compose([
                    # transforms.RandomCrop(32, padding=4),
                    # transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010))
                ])
        elif n_data == 'SVHN':
            if num_data != 0:
                transform_train = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(mean=(0.4376821, 0.4437697, 0.47280442), std=(0.19803012, 0.20101562, 0.19703614))  # 画像保存するときはコメントアウトしたほうがよい
                ])
            else:
                transform_train = transforms.Compose([
                    # transforms.RandomCrop(32, padding=4),
                    # transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=(0.4376821, 0.4437697, 0.47280442), std=(0.19803012, 0.20101562, 0.19703614))  # 画像保存するときはコメントアウトしたほうがよい
                ])
        if n_data == 'Fashion-MNIST':
            transform_train = transforms.Compose([
                transforms.ToTensor()
            ])
        elif n_data == 'STL-10':
            transform_train = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
            ])
        elif n_data == 'TinyImageNet':
            transform_train = transforms.Compose([
                # transforms.RandomResizedCrop(224, scale=(0.08, 1.0), interpolation=Image.BICUBIC),
                # transforms.Scale(256),
                transforms.Resize(256, interpolation=Image.BICUBIC),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

        """Cutout"""
        if cutout == 1:
            transform_train.transforms.append(CutoutDefault(cutout))

        """RandAugment"""
        if flag_randaug == 1:
            transform_train.transforms.insert(0, RandAugment(rand_n, rand_m, n_data))

        if n_data == 'MNIST':
            self.mydata = torchvision.datasets.MNIST(root='../../datasets/mnist', train=True, transform=transform_train, download=True)
        elif n_data == 'CIFAR-10':
            self.mydata = torchvision.datasets.CIFAR10(root='../../datasets/cifar10', train=True, transform=transform_train, download=True)
        elif n_data == 'SVHN':
            if num_data != 0:
                self.mydata = torchvision.datasets.SVHN(root='../../datasets/svhn', split='train', transform=transform_train, download=True)  # only train data
            else:
                self.mydata = torchvision.datasets.SVHN(root='../../datasets/svhn', split='train', transform=transform_train, download=True)
                # trainset = torchvision.datasets.SVHN(root='../../datasets/svhn', split='train', transform=transform_train, download=True)
                # extraset = torchvision.datasets.SVHN(root='../../datasets/svhn', split='extra', transform=transform_train, download=True)
                # self.mydata = ConcatDataset([trainset, extraset])
        elif n_data == 'STL-10':
            self.mydata = torchvision.datasets.STL10(root='../../datasets/stl10', split='train', transform=transform_train, download=True)
        elif n_data == 'CIFAR-100':
            self.mydata = torchvision.datasets.CIFAR100(root='../../datasets/cifar100', train=True, transform=transform_train, download=True)
        elif n_data == 'Fashion-MNIST':
            self.mydata = torchvision.datasets.FashionMNIST(root='../../datasets/FashionMNIST', train=True, transform=transform_train, download=True)
        elif n_data == 'TinyImageNet':
            self.mydata = torchvision.datasets.ImageFolder(root='../../datasets/tiny-imagenet-200/train', transform=transform_train)

            weights = self.mydata.make_weights_for_balanced_classes()
            self.sampler = WeightedRandomSampler(weights, len(weights))

        if num_data != 0:
            self.mydata = util.make_training_data(self.mydata, num_data, seed)

    def __getitem__(self, index):
        x, y = self.mydata[index]

        return x, y, index

    def __len__(self):
        return len(self.mydata)

    def get_info(self, n_data):
        num_channel = 3
        num_classes = 10
        size_after_cnn = 4
        input_size = 0
        hidden_size = 0

        if n_data == 'MNIST':
            num_channel = 1
            num_classes = 10
            size_after_cnn = 4
            input_size = 28 * 28 * 1
            # num_training_data = 60000
            # num_test_data = 10000
        elif n_data == 'CIFAR-10':
            num_channel = 3
            num_classes = 10
            size_after_cnn = 5
            input_size = 32 * 32 * 3
            # num_training_data = 50000
            # num_test_data = 10000
        elif n_data == 'SVHN':
            num_channel = 3
            num_classes = 10
            size_after_cnn = 5
            input_size = 32 * 32 * 3
            # num_training_data = 73257
            # num_training_data = 73257 + 531131
            # num_test_data = 26032
        elif n_data == 'STL-10':
            num_channel = 3
            num_classes = 10
            size_after_cnn = 8  # cnn_stl
            input_size = 96 * 96 * 3
            # size_after_cnn = 21  # small CNN
            # num_training_data = 5000
            # num_test_data = 8000
        elif n_data == 'CIFAR-100':
            num_channel = 3
            num_classes = 100
            size_after_cnn = 5
            input_size = 32 * 32 * 3
            # num_training_data = 50000
            # num_test_data = 10000
        elif n_data == 'Fashion-MNIST':
            num_channel = 1
            num_classes = 10
            size_after_cnn = 4
            input_size = 28 * 28 * 1
            # num_training_data = 60000
            # num_test_data = 10000
        elif n_data == 'TinyImageNet':
            num_channel = 3
            num_classes = 200
            input_size = 224 * 224 * 3
            # num_training_data = 100000
            # num_test_data = 10000

        return num_channel, num_classes, size_after_cnn, input_size, hidden_size


class MyDataset_test(Dataset):
    def __init__(self, n_data):
        self.sampler = None

        """Preprocessing"""
        transform_test = None
        if n_data == 'MNIST':
            transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.1307, ), std=(0.3081, ))
            ])
        elif n_data == 'CIFAR-10' or n_data == 'CIFAR-100':
            transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010)),
            ])
        elif n_data == 'SVHN':
            transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.4376821, 0.4437697, 0.47280442), std=(0.19803012, 0.20101562, 0.19703614))
            ])
        if n_data == 'Fashion-MNIST':
            transform_test = transforms.Compose([
                transforms.ToTensor()
            ])
        elif n_data == 'STL-10':
            transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
            ])
        elif n_data == 'TinyImageNet':
            transform_test = transforms.Compose([
                # transforms.RandomResizedCrop(224, scale=(0.08, 1.0), interpolation=Image.BICUBIC),
                # transforms.Scale(256),
                transforms.Resize(256, interpolation=Image.BICUBIC),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

        if n_data == 'MNIST':
            self.mydata = torchvision.datasets.MNIST(root='../../datasets/mnist', train=False, transform=transform_test, download=True)
        elif n_data == 'CIFAR-10':
            self.mydata = torchvision.datasets.CIFAR10(root='../../datasets/cifar10', train=False, transform=transform_test, download=True)
        elif n_data == 'SVHN':
            self.mydata = torchvision.datasets.SVHN(root='../../datasets/svhn', split='test', transform=transform_test, download=True)
        elif n_data == 'STL-10':
            self.mydata = torchvision.datasets.STL10(root='../../datasets/stl10', split='test', transform=transform_test, download=True)
        elif n_data == 'CIFAR-100':
            self.mydata = torchvision.datasets.CIFAR100(root='../../datasets/cifar100', train=False, transform=transform_test, download=True)
        elif n_data == 'Fashion-MNIST':
            self.mydata = torchvision.datasets.FashionMNIST(root='../../datasets/FashionMNIST', train=False, transform=transform_test, download=True)
        elif n_data == 'TinyImageNet':
            self.mydata = torchvision.datasets.ImageFolder(root='../../datasets/tiny-imagenet-200/val', transform=transform_test)

    def __getitem__(self, index):
        x, y = self.mydata[index]

        return x, y

    def __len__(self):
        return len(self.mydata)

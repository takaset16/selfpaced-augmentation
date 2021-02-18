# coding: utf-8
import torch.nn
import timeit
import wandb
from warmup_scheduler import GradualWarmupScheduler

import objective
import dataset
from models import *
from RandAugment import RandAugment


class MainNN(object):
    def __init__(self, loop, n_data, gpu_multi, hidden_size, num_samples, num_epochs, batch_size_training, batch_size_test,
                 n_model, opt, save_file, flag_wandb, show_params, save_images, cutout, flag_traintest, n_aug, flag_transfer,
                 flag_randaug, rand_n, rand_m, flag_lars, lb_smooth, flag_lr_schedule, flag_warmup, flag_acc5, flag_spa, judge_noise, flag_variance):
        self.loop = loop
        self.seed = 1001 + loop
        self.train_loader = None
        self.test_loader = None
        self.n_data = n_data
        self.gpu_multi = gpu_multi
        self.input_size = 0
        self.hidden_size = hidden_size
        self.num_classes = 10
        self.num_channel = 0
        self.size_after_cnn = 0
        self.num_training_data = num_samples
        self.num_test_data = 0
        self.num_epochs = num_epochs
        self.batch_size_training = batch_size_training
        self.batch_size_test = batch_size_test
        self.n_model = n_model
        self.loss_training_batch = None
        self.opt = opt
        self.save_file = save_file
        self.flag_wandb = flag_wandb
        self.show_params = show_params
        self.save_images = save_images
        self.cutout = cutout
        self.flag_traintest = flag_traintest
        self.flag_acc5 = flag_acc5
        self.n_aug = n_aug
        self.flag_transfer = flag_transfer
        self.flag_shake = 0
        self.flag_randaug = flag_randaug
        self.rand_n = rand_n
        self.rand_m = rand_m
        self.flag_lars = flag_lars
        self.lb_smooth = lb_smooth
        self.flag_lr_schedule = flag_lr_schedule
        self.flag_warmup = flag_warmup
        self.flag_spa = flag_spa
        self.flag_noise = None
        self.judge_noise = judge_noise
        self.flag_variance = flag_variance

    def run_main(self):
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)

        if self.flag_randaug == 1:
            if self.rand_n == 0 and self.rand_m == 0:
                if self.n_model == 'ResNet':
                    self.rand_n = 2
                    self.rand_m = 9
                elif self.n_model == 'WideResNet':
                    if self.n_data == 'CIFAR-10':
                        self.rand_n = 3
                        self.rand_m = 5
                    elif self.n_data == 'CIFAR-100':
                        self.rand_n = 2
                        self.rand_m = 14
                    elif self.n_data == 'SVHN':
                        self.rand_n = 3
                        self.rand_m = 7

        traintest_dataset = dataset.MyDataset_training(n_data=self.n_data, num_data=self.num_training_data, seed=self.seed,
                                                       flag_randaug=self.flag_randaug, rand_n=self.rand_n, rand_m=self.rand_m, cutout=self.cutout)
        self.num_channel, self.num_classes, self.size_after_cnn, self.input_size, self.hidden_size = traintest_dataset.get_info(n_data=self.n_data)

        n_samples = len(traintest_dataset)
        if self.num_training_data == 0:
            self.num_training_data = n_samples

        if self.flag_traintest == 1:
            # train_size = self.num_classes * 100
            train_size = int(n_samples * 0.65)
            test_size = n_samples - train_size
            train_dataset, test_dataset = torch.utils.data.random_split(traintest_dataset, [train_size, test_size])

            train_sampler = None
            test_sampler = None
        else:
            train_dataset = traintest_dataset
            test_dataset = dataset.MyDataset_test(n_data=self.n_data)

            train_sampler = train_dataset.sampler
            test_sampler = test_dataset.sampler

        num_workers = 16
        train_shuffle = True
        test_shuffle = False

        if train_sampler:
            train_shuffle = False
        if test_sampler:
            test_shuffle = False
        self.train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=self.batch_size_training, sampler=train_sampler,
                                                        shuffle=train_shuffle, num_workers=num_workers, pin_memory=True)
        self.test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=self.batch_size_test, sampler=test_sampler,
                                                       shuffle=test_shuffle, num_workers=num_workers, pin_memory=True)

        if self.flag_transfer == 1:
            pretrained = True
            num_classes = 1000
        else:
            pretrained = False
            num_classes = self.num_classes

        model = None
        if self.n_model == 'CNN':
            model = cnn.ConvNet(num_classes=self.num_classes, num_channel=self.num_channel, size_after_cnn=self.size_after_cnn, n_aug=self.n_aug)
        elif self.n_model == 'ResNet':
            model = resnet.ResNet(n_data=self.n_data, depth=50, num_classes=self.num_classes, num_channel=self.num_channel, n_aug=self.n_aug, bottleneck=True)
        elif self.n_model == 'WideResNet':
            model = wideresnet.WideResNet(depth=28, widen_factor=10, dropout_rate=0.0, num_classes=self.num_classes, num_channel=self.num_channel, n_aug=self.n_aug)

        if self.flag_transfer == 1:
            for param in model.parameters():
                param.requires_grad = False

            num_features = model.fc.in_features
            model.fc = nn.Linear(num_features, self.num_classes)

        if self.show_params == 1:
            params = 0
            for p in model.parameters():
                if p.requires_grad:
                    params += p.numel()
            print(params)

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = model.to(device)
        if device == 'cuda':
            if self.gpu_multi == 1:
                model = torch.nn.DataParallel(model)
            torch.backends.cudnn.benchmark = True
            print('GPU={}'.format(torch.cuda.device_count()))

        if self.lb_smooth > 0.0:
            criterion = objective.SmoothCrossEntropyLoss(self.lb_smooth)
        else:
            criterion = objective.SoftCrossEntropy()

        optimizer = 0

        if self.opt == 0:
            if self.flag_transfer == 1:
                optimizer = torch.optim.Adam(model.fc.parameters(), lr=0.001)
            else:
                optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        elif self.opt == 1:
            if self.n_model == 'ResNet':
                lr = 0.1
                weight_decay = 0.0001
            elif self.n_model == 'WideResNet':
                if self.n_data == 'SVHN':
                    lr = 0.005
                    weight_decay = 0.001
                else:
                    lr = 0.1
                    weight_decay = 0.0005
            else:
                lr = 0.1
                weight_decay = 0.0005

            optimizer = torch.optim.SGD(
                model.parameters(),
                lr=lr,
                momentum=0.9,
                weight_decay=weight_decay,
                nesterov=True
            )
        if self.flag_lars == 1:
            from torchlars import LARS
            optimizer = LARS(optimizer)

        scheduler = None
        if self.flag_lr_schedule == 2:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.num_epochs, eta_min=0.)
        elif self.flag_lr_schedule == 3:
            if self.num_epochs == 90:
                scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [30, 60, 80])
            elif self.num_epochs == 180:
                scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [60, 120, 160])
            elif self.num_epochs == 270:
                scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [90, 180, 240])

        if self.flag_warmup == 1:
            if self.n_model == 'ResNet':
                multiplier = 2
                total_epoch = 3
            elif self.n_model == 'WideResNet':
                multiplier = 2
                if self.n_data == 'SVHN':
                    total_epoch = 3
                else:
                    total_epoch = 5
            else:
                multiplier = 2
                total_epoch = 3

            scheduler = GradualWarmupScheduler(
                optimizer,
                multiplier=multiplier,
                total_epoch=total_epoch,
                after_scheduler=scheduler
            )

        self.flag_noise = np.random.randint(0, 2, self.num_training_data)

        if self.flag_acc5 == 1:
            results = np.zeros((self.num_epochs, 6))
        else:
            results = np.zeros((self.num_epochs, 5))
        start_time = timeit.default_timer()

        t = 0
        # fixed_interval = 10
        fixed_interval = 1
        loss_fixed_all = np.zeros(self.num_epochs // fixed_interval)
        self.loss_training_batch = np.zeros(int(self.num_epochs * np.ceil(self.num_training_data / self.batch_size_training)))

        for epoch in range(self.num_epochs):
            model.train()
            start_epoch_time = timeit.default_timer()

            loss_each_all = np.zeros(self.num_training_data)
            loss_training_all = 0
            loss_test_all = 0

            if self.flag_variance == 1:
                if epoch % fixed_interval == 0:
                    loss_fixed = np.zeros(self.num_training_data // self.batch_size_training)

                    for i, (images, labels, index) in enumerate(self.train_loader):
                        if np.array(images.data.cpu()).ndim == 3:
                            images = images.reshape(images.shape[0], 1, images.shape[1], images.shape[2]).to(device)
                        else:
                            images = images.to(device)
                        labels = labels.to(device)

                        flag_onehot = 0
                        if self.flag_spa == 1:
                            outputs_fixed = model.forward(images)
                            if flag_onehot == 0:
                                labels = np.identity(self.num_classes)[labels]
                        else:
                            outputs_fixed = model(images)
                            labels = np.identity(self.num_classes)[np.array(labels.data.cpu())]
                        labels = util.to_var(torch.from_numpy(labels).float())

                        loss_fixed[i] = criterion.forward(outputs_fixed, labels)
                    loss_fixed_all[t] = np.var(loss_fixed)
                    t += 1

            total_steps = len(self.train_loader)
            steps = 0
            num_training_data = 0
            for i, (images, labels, index) in enumerate(self.train_loader):
                steps += 1
                if np.array(images.data.cpu()).ndim == 3:
                    images = images.reshape(images.shape[0], 1, images.shape[1], images.shape[2]).to(device)
                else:
                    images = images.to(device)
                labels = labels.to(device)

                if self.save_images == 1:
                    util.save_images(images)

                if self.flag_spa == 1:
                    outputs = model(images)
                    labels_spa = labels.clone()

                    if labels_spa.ndim == 1:
                        labels_spa = torch.eye(self.num_classes, device='cuda')[labels_spa].clone()

                    loss_each = criterion.forward_each_example(outputs, labels_spa)
                    loss_each_all[index] = np.array(loss_each.data.cpu())

                    self.flag_noise = util.flag_update(loss_each_all, self.judge_noise)

                if self.flag_spa == 1:
                    images, labels = util.self_paced_augmentation(images=images,
                                                                  labels=labels,
                                                                  flag_noise=self.flag_noise,
                                                                  index=np.array(index.data.cpu()),
                                                                  n_aug=self.n_aug,
                                                                  num_classes=self.num_classes)
                else:
                    images, labels = util.run_n_aug(x=images,
                                                    y=labels,
                                                    n_aug=self.n_aug,
                                                    num_classes=self.num_classes)

                outputs = model(images)

                if labels.ndim == 1:
                    labels = torch.eye(self.num_classes, device='cuda')[labels].clone()

                loss_training = criterion.forward(outputs, labels)
                loss_training_all += loss_training.item() * outputs.shape[0]
                # self.loss_training_batch[int(i + epoch * np.ceil(self.num_training_data / self.batch_size_training))] = loss_training * outputs.shape[0]
                num_training_data += images.shape[0]

                optimizer.zero_grad()
                loss_training.backward()
                optimizer.step()

            loss_training_each = loss_training_all / num_training_data
            # np.random.shuffle(self.flag_noise)

            model.eval()

            with torch.no_grad():
                if self.flag_acc5 == 1:
                    top1 = list()
                    top5 = list()
                else:
                    correct = 0
                    total = 0

                num_test_data = 0
                for images, labels in self.test_loader:
                    if np.array(images.data).ndim == 3:
                        images = images.reshape(images.shape[0], 1, images.shape[1], images.shape[2]).to(device)
                    else:
                        images = images.to(device)
                    labels = labels.to(device)

                    outputs = model(x=images)

                    if self.flag_acc5 == 1:
                        acc1, acc5 = util.accuracy(outputs.data, labels.long(), topk=(1, 5))
                        top1.append(acc1[0].item())
                        top5.append(acc5[0].item())
                    else:
                        _, predicted = torch.max(outputs.data, 1)
                        correct += (predicted == labels.long()).sum().item()
                        total += labels.size(0)

                    if labels.ndim == 1:
                        labels = torch.eye(self.num_classes, device='cuda')[labels]

                    loss_test = criterion.forward(outputs, labels)
                    loss_test_all += loss_test.item() * outputs.shape[0]
                    num_test_data += images.shape[0]

            top1_avg = 0
            top5_avg = 0
            test_accuracy = 0

            if self.flag_acc5 == 1:
                top1_avg = sum(top1) / float(len(top1))
                top5_avg = sum(top5) / float(len(top5))
            else:
                test_accuracy = 100.0 * correct / total

            loss_test_each = loss_test_all / num_test_data

            end_epoch_time = timeit.default_timer()
            epoch_time = end_epoch_time - start_epoch_time
            num_flag = np.sum(self.flag_noise == 1)

            if self.flag_lr_schedule > 1 and scheduler is not None:
                scheduler.step(epoch - 1 + float(steps) / total_steps)

            flag_log = 1
            if flag_log == 1:
                if self.flag_acc5 == 1:
                    print('Epoch [{}/{}], Train Loss: {:.4f}, Top1 Test Acc: {:.3f} %, Top5 Test Acc: {:.3f} %, Test Loss: {:.4f}, Epoch Time: {:.2f}s, Num_flag: {}'.
                          format(epoch + 1, self.num_epochs, loss_training_each, top1_avg, top5_avg, loss_test_each, epoch_time, num_flag))
                else:
                    print('Epoch [{}/{}], Train Loss: {:.4f}, Test Acc: {:.3f} %, Test Loss: {:.4f}, Epoch Time: {:.2f}s, Num_flag: {}'.
                          format(epoch + 1, self.num_epochs, loss_training_each, test_accuracy, loss_test_each, epoch_time, num_flag))

            if self.flag_wandb == 1:
                wandb.log({"loss_training_each": loss_training_each})
                wandb.log({"test_accuracy": test_accuracy})
                wandb.log({"loss_test_each": loss_test_each})
                wandb.log({"num_flag": num_flag})
                wandb.log({"epoch_time": epoch_time})

            if self.save_file == 1:
                if flag_log == 1:
                    if self.flag_acc5 == 1:
                        results[epoch][0] = loss_training_each
                        results[epoch][1] = top1_avg
                        results[epoch][2] = top5_avg
                        results[epoch][3] = loss_test_each
                        results[epoch][4] = num_flag
                        results[epoch][5] = epoch_time
                    else:
                        results[epoch][0] = loss_training_each
                        results[epoch][1] = test_accuracy
                        results[epoch][2] = loss_test_each
                        results[epoch][3] = num_flag
                        results[epoch][4] = epoch_time

        end_time = timeit.default_timer()

        flag_log = 1
        if flag_log == 1:
            print(' ran for %.4fm' % ((end_time - start_time) / 60.))

        top1_avg_max = np.max(results[:, 1])
        print(top1_avg_max)

        if flag_log == 1 and self.flag_acc5 == 1:
            top5_avg_max = np.max(results[:, 2])
            print(top5_avg_max)

        if self.save_file == 1:
            if self.flag_randaug == 1:
                np.savetxt('results/data_%s_model_%s_num_%s_randaug_%s_n_%s_m_%s_seed_%s_acc_%s.csv'
                           % (self.n_data, self.n_model, self.num_training_data, self.flag_randaug,
                              self.rand_n, self.rand_m, self.seed, top1_avg_max), results, delimiter=',')
            else:
                if self.flag_spa == 1:
                    np.savetxt('results/data_%s_model_%s_judge_%s_aug_%s_num_%s_seed_%s_acc_%s.csv'
                               % (self.n_data, self.n_model, self.judge_noise, self.n_aug, self.num_training_data,
                                  self.seed, top1_avg_max), results, delimiter=',')
                else:
                    np.savetxt('results/data_%s_model_%s_aug_%s_num_%s_seed_%s_acc_%s.csv'
                               % (self.n_data, self.n_model, self.n_aug, self.num_training_data,
                                  self.seed, top1_avg_max), results, delimiter=',')

            # np.savetxt('results/loss_training_batch_judge_%s_aug_%s_seed_%s_acc_%s.csv'
            #            % (self.judge_noise, self.n_aug, self.seed, top1_avg_max),
            #            self.loss_training_batch, delimiter=',')

            if self.flag_variance == 1:
                np.savetxt('results/loss_variance_judge_%s_aug_%s_acc_%s.csv'
                           % (self.judge_noise, self.n_aug, top1_avg_max), loss_fixed_all, delimiter=',')

# coding: utf-8
import argparse
import main_nn
import wandb

'''################################################################################################
     n_data: 'MNIST', 'CIFAR-10', 'SVHN', 'STL-10', 'CIFAR-100', 'EMNIST', 
             'COIL-20', 'Fashion-MNIST', 'ImageNet', 'TinyImageNet', 
             'Letter Recognition', 'Car Evaluation', 'Epileptic Seizure'
     n_aug: 0(random_noise), 1(flips), 2(crop), 3(transfer), 4(rotation), 5(mixup), 6(cutout), 
            7(random erasing), 8(ricap), 9(shake-shake)
#################################################################################################'''


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--loop', type=int, default=0)
    parser.add_argument('--n_data', default='CIFAR-10')
    parser.add_argument('--gpu_multi', type=int, default=0)
    parser.add_argument('--hidden_size', type=int, default=100)
    parser.add_argument('--num_samples', type=int, default=0)
    parser.add_argument('--num_epochs', type=int, default=200)
    parser.add_argument('--batch_size_training', type=int, default=256)
    parser.add_argument('--batch_size_test', type=int, default=1000)
    parser.add_argument('--n_model', default='CNN')
    parser.add_argument('--opt', type=int, default=1)
    parser.add_argument('--save_file', type=int, default=1)
    parser.add_argument('--flag_wandb', type=int, default=0)
    parser.add_argument('--show_params', type=int, default=0)
    parser.add_argument('--save_images', type=int, default=0)
    parser.add_argument('--cutout', type=int, default=0)
    parser.add_argument('--flag_traintest', type=int, default=0)
    parser.add_argument('--n_aug', type=int, default=12)
    parser.add_argument('--flag_transfer', type=int, default=0)
    parser.add_argument('--flag_randaug', type=int, default=0)
    parser.add_argument('--rand_n', type=int, default=0)
    parser.add_argument('--rand_m', type=int, default=0)
    parser.add_argument('--flag_lars', type=int, default=0)
    parser.add_argument('--lb_smooth', type=float, default=0.0)
    parser.add_argument('--flag_lr_schedule', type=int, default=2)
    parser.add_argument('--flag_warmup', type=int, default=1)
    parser.add_argument('--flag_acc5', type=int, default=1)
    parser.add_argument('--flag_spa', type=int, default=1)
    parser.add_argument('--judge_noise', type=float, default=0)
    parser.add_argument('--flag_variance', type=int, default=0)
    args = parser.parse_args()

    if args.flag_wandb == 1:
        wandb.init(project="SPA", config=args)
        config = wandb.config

        main_model = main_nn.MainNN(loop=config.loop,
                                    n_data=config.n_data,
                                    gpu_multi=config.gpu_multi,
                                    hidden_size=config.hidden_size,
                                    num_samples=config.num_samples,
                                    num_epochs=config.num_epochs,
                                    batch_size_training=config.batch_size_training,
                                    batch_size_test=config.batch_size_test,
                                    n_model=config.n_model,
                                    opt=config.opt,
                                    save_file=config.save_file,
                                    flag_wandb=config.flag_wandb,
                                    show_params=config.show_params,
                                    save_images=config.save_images,
                                    cutout=config.cutout,
                                    flag_traintest=config.flag_traintest,
                                    n_aug=config.n_aug,
                                    flag_transfer=config.flag_transfer,
                                    flag_randaug=config.flag_randaug,
                                    rand_n=config.rand_n,
                                    rand_m=config.rand_m,
                                    flag_lars=config.flag_lars,
                                    lb_smooth=config.lb_smooth,
                                    flag_lr_schedule=config.flag_lr_schedule,
                                    flag_warmup=config.flag_warmup,
                                    flag_acc5=config.flag_acc5,
                                    flag_spa=config.flag_spa,
                                    judge_noise=config.judge_noise,
                                    flag_variance=config.flag_variance,)

    else:
        main_model = main_nn.MainNN(loop=args.loop,
                                    n_data=args.n_data,
                                    gpu_multi=args.gpu_multi,
                                    hidden_size=args.hidden_size,
                                    num_samples=args.num_samples,
                                    num_epochs=args.num_epochs,
                                    batch_size_training=args.batch_size_training,
                                    batch_size_test=args.batch_size_test,
                                    n_model=args.n_model,
                                    opt=args.opt,
                                    save_file=args.save_file,
                                    flag_wandb=args.flag_wandb,
                                    show_params=args.show_params,
                                    save_images=args.save_images,
                                    cutout=args.cutout,
                                    flag_traintest=args.flag_traintest,
                                    n_aug=args.n_aug,
                                    flag_transfer=args.flag_transfer,
                                    flag_randaug=args.flag_randaug,
                                    rand_n=args.rand_n,
                                    rand_m=args.rand_m,
                                    flag_lars=args.flag_lars,
                                    lb_smooth=args.lb_smooth,
                                    flag_lr_schedule=args.flag_lr_schedule,
                                    flag_warmup=args.flag_warmup,
                                    flag_acc5=args.flag_acc5,
                                    flag_spa=args.flag_spa,
                                    judge_noise=args.judge_noise,
                                    flag_variance=args.flag_variance)
    main_model.run_main()


if __name__ == '__main__':
    main()

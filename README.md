# Self-paced augmentation

Demo code for the Self-paced augmentation method proposed in "[Self-paced data augmentation for training neural networks](https://www.sciencedirect.com/science/article/pii/S0925231221003374)" by Tomoumi Takase, Ryo Karakida, and Hideki Asoh.

# Demo

## Requirements
pytorch 1.6.0

## Execution
```
python main.py --n_data 'CIFAR-10' --n_model 'WideResNet' --n_aug 1 --num_epochs 200 --flag_spa 0 --judge_noise 0 --gpu_multi 1 --loop 0
```

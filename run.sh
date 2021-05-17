source /etc/profile.d/modules.sh
module load python/3.6/3.6.5 cuda/10.1/10.1.243 cudnn/7.6/7.6.5 nccl/2.5/2.5.6-1 openmpi/2.1.6 gcc/7.4.0
source ~/venv/pytorch/bin/activate

python main.py --n_data 'CIFAR-10' --n_model 'CNN' --n_aug 1 --num_epochs 200 --flag_spa 0 --judge_noise 0 --gpu_multi 1 --loop 0


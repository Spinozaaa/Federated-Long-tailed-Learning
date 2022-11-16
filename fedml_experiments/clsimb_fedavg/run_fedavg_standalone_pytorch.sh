#!/usr/bin/env bash

wandb offline
# wandb login f0ef3de910b8ba85213c87fed9484bd1c14c7b9f

nohup python3 -u ./main_fedavg.py \
--gpu 0 --dataset cifar10 --data_dir ./../../../data/cifar10 --model resnet18_gn \
--partition_method global_class_imba --client_num_in_total 100 --client_num_per_round 10 \
--comm_round 190 --epochs 5 --batch_size 64 --client_optimizer adam --lr 0.0001 --ci 0 \
--imb_factor 0.01 --imb_type global --num_imba_client 0 --method global_data+randaugment > out5

# Federated-Long-tailed-Learning

This repository is the official implementation of "Label-distribution-agnostic Ensemble Learning on Federated Long-tailed Data"

## Installation

```
conda env create -f ldae.yaml
```

## Run

```
cd fedml_experiments/clsimb_fedavg/
```
For ldae on cifar100-lt, run:
```
python -u ./main_fedavg.py \
--comm_round 2000 --epochs 2 --batch_size 64 --client_optimizer sgd --lr 0.6 --lr_decay 0.05 \
--imb_factor 0.01 --partition_alpha 0.1 --method ldae_train_exp_esti_global --frequency_of_the_test 50 --beta 0.8
```

Other re-balance strategies are available: focal, ldam, lade, blsm, ride.
They can use different class priors like: local re-balance: ldam; global re-balance: ldam_real_global; GPI: ldam_esti_global.

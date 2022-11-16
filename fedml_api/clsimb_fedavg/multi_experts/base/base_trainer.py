import torch
import torch.nn
from abc import abstractmethod
from numpy import inf
import argparse
import collections
import logging

from .. import loss as module_loss
from ..utils import load_state_dict, rename_parallel_state_dict
from ..parse_config import ConfigParser


class BaseTrainer:
    """
    Base class for all trainers
    """
    def __init__(self, model, optimizer, args, cls_num_list):

        self.args = args
        self.cls_num_list = cls_num_list
        config, criterion = self.init_()

        self.config, self.criterion = config, criterion
        self.metric = None

        # setup GPU device if available, move model into configured device
        self.device, device_ids = self._prepare_device(config['n_gpu'])
        self.device_ids = device_ids
        self.model = model

        self.real_model = self.model

        self.criterion = self.criterion.to(self.device)

        self.optimizer = optimizer

        cfg_trainer = config['trainer']
        self.epochs = args.epochs
        self.save_period = cfg_trainer['save_period']
        self.monitor = cfg_trainer.get('monitor', 'off')

        # configuration to monitor model performance and save best
        if self.monitor == 'off':
            self.mnt_mode = 'off'
            self.mnt_best = 0
        else:
            self.mnt_mode, self.mnt_metric = self.monitor.split()
            assert self.mnt_mode in ['min', 'max']

            self.mnt_best = inf if self.mnt_mode == 'min' else -inf
            self.early_stop = cfg_trainer.get('early_stop', inf)

        self.start_epoch = 1

    def init_(self):

        CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
        args = [
            CustomArgs(['--lr', '--learning_rate'], type=float, target='optimizer;args;lr'),
            CustomArgs(['--bs', '--batch_size'], type=int, target='data_loader;args;batch_size'),
            CustomArgs(['--name'], type=str, target='name'),
            CustomArgs(['--epochs'], type=int, target='trainer;epochs'),
            CustomArgs(['--step1'], type=int, target='lr_scheduler;args;step1'),
            CustomArgs(['--step2'], type=int, target='lr_scheduler;args;step2'),
            CustomArgs(['--warmup'], type=int, target='lr_scheduler;args;warmup_epoch'),
            CustomArgs(['--gamma'], type=float, target='lr_scheduler;args;gamma'),
            CustomArgs(['--save_period'], type=int, target='trainer;save_period'),
            CustomArgs(['--reduce_dimension'], type=int, target='arch;args;reduce_dimension'),
            CustomArgs(['--layer2_dimension'], type=int, target='arch;args;layer2_output_dim'),
            CustomArgs(['--layer3_dimension'], type=int, target='arch;args;layer3_output_dim'),
            CustomArgs(['--layer4_dimension'], type=int, target='arch;args;layer4_output_dim'),
            CustomArgs(['--num_experts'], type=int, target='arch;args;num_experts'),
            CustomArgs(['--distribution_aware_diversity_factor'], type=float,
                       target='loss;args;additional_diversity_factor'),
            CustomArgs(['--pos_weight'], type=float, target='arch;args;pos_weight'),
            CustomArgs(['--collaborative_loss'], type=int, target='loss;args;collaborative_loss'),
            CustomArgs(['--distill_checkpoint'], type=str, target='distill_checkpoint')
        ]

        if "ride" in self.args.method:
            config_file = "/config_imbalance_cifar100_ride.json"
        elif "ldae" in self.args.method:
            config_file = "/config_imbalance_cifar100_ldae.json"

        config = ConfigParser.from_args(args, config_file=config_file)

        # get function handles of loss and metrics
        criterion = config.init_obj('loss', module_loss, cls_num_list=self.cls_num_list, num_experts=config["arch"]["args"]["num_experts"])

        return config, criterion

    @abstractmethod
    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Current epoch number
        """
        raise NotImplementedError

    def train(self):
        for epoch in range(self.start_epoch, self.epochs + 1):
            self._train_epoch(epoch)

    def _prepare_device(self, n_gpu_use):
        """
        setup GPU device if available, move model into configured device
        """
        n_gpu = torch.cuda.device_count()
        if n_gpu_use > 0 and n_gpu == 0:
            n_gpu_use = 0
        if n_gpu_use > n_gpu:
            n_gpu_use = n_gpu
        device = torch.device('cuda:0' if n_gpu_use > 0 else 'cpu')
        list_ids = list(range(n_gpu_use))
        return device, list_ids

import numpy as np
import torch
from torchvision.utils import make_grid
from .base import BaseTrainer
from .utils import inf_loop, MetricTracker, load_state_dict, rename_parallel_state_dict, autocast, use_fp16


class Trainer(BaseTrainer):
    """
    Trainer class
    """
    def __init__(self, model, optimizer=None, data_loader=None, args=None, cls_num_list=None, training_exp=None):
        super().__init__(model, optimizer, args, cls_num_list)

        self.distill = False
        
        # add_extra_info will return info about individual experts. This is crucial for individual loss. If this is false, we can only get a final mean logits.
        self.add_extra_info = self.config._config.get('add_extra_info', False)
        if self.args.num_experts == 1:
            self.add_extra_info = False

        self.data_loader = data_loader
        self.len_epoch = len(self.data_loader)
        self.scaler = None

        # self.lr_scheduler = lr_scheduler
        self.log_step = int(np.sqrt(data_loader.batch_size))
        self.training_exp = training_exp
        self.device = args.gpu
        self.model = self.model.to(self.device)

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """

        self.model.train()

        if hasattr(self.criterion, "_hook_before_epoch"):
            self.criterion._hook_before_epoch(epoch)

        for batch_idx, data in enumerate(self.data_loader):
            if self.distill and len(data) == 4:
                data, target, idx, contrast_idx = data
            else:
                data, target = data

            data, target = data.to(self.device), target.to(self.device)

            self.optimizer.zero_grad()

            with autocast():
                if self.real_model.requires_target:
                    output = self.model(data, target=target)
                    output, loss = output
                else:
                    extra_info = {}
                    output = self.model(data)

                    if self.add_extra_info:
                        if isinstance(output, dict):
                            logits = output["logits"]
                            extra_info.update({
                                "logits": logits.transpose(0, 1)
                            })
                        else:
                            extra_info.update({
                                "logits": self.real_model.backbone.logits
                            })

                    if isinstance(output, dict):
                        output = output["output"]

                    # if self.distill:
                    #     loss = self.criterion(student=output, target=target, teacher=teacher, extra_info=extra_info)
                    if self.add_extra_info and "ldae" in self.args.method:
                        loss = self.criterion(output_logits=output, target=target, extra_info=extra_info, training_exp=self.training_exp, device=self.device)
                    else:
                        loss = self.criterion(output_logits=output, target=target)

            if not use_fp16:
                loss.backward()
                self.optimizer.step()
            else:
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()

        return

    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        self.model.eval()

        with torch.no_grad():
            if hasattr(self.model, "confidence_model") and self.model.confidence_model:
                cumulative_sample_num_experts = torch.zeros((self.model.backbone.num_experts, ), device=self.device)
                num_samples = 0
                confidence_model = True
            else:
                confidence_model = False
            for batch_idx, (data, target) in enumerate(self.valid_data_loader):
                data, target = data.to(self.device), target.to(self.device)

                if confidence_model:
                    output, sample_num_experts = self.model(data)
                    num, count = torch.unique(sample_num_experts, return_counts=True)
                    cumulative_sample_num_experts[num - 1] += count
                    num_samples += data.size(0)
                else:
                    output = self.model(data)
                if isinstance(output, dict):
                    output = output["output"]
                loss = self.criterion(output, target)

                self.writer.set_step((epoch - 1) * len(self.valid_data_loader) + batch_idx, 'valid')
                self.valid_metrics.update('loss', loss.item())
                for met in self.metric_ftns:
                    self.valid_metrics.update(met.__name__, met(output, target, return_length=True))
                self.writer.add_image('input', make_grid(data.cpu(), nrow=8, normalize=True))

            if confidence_model:
                print("Samples with num_experts:", *[('%.2f'%item) for item in (cumulative_sample_num_experts * 100 / num_samples).tolist()])

        # add histogram of model parameters to the tensorboard
        for name, p in self.model.named_parameters():
            self.writer.add_histogram(name, p, bins='auto')
        return self.valid_metrics.result()

    def _progress(self, batch_idx):
        base = '[{}/{} ({:.0f}%)]'
        if hasattr(self.data_loader, 'n_samples'):
            current = batch_idx * self.data_loader.batch_size
            total = self.data_loader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)

    def update_mode(self, model_para):
        self.model.load_state_dict(model_para)
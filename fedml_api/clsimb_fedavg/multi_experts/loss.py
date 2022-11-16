import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

eps = 1e-7


class LDAMLoss(nn.Module):
    def __init__(self, cls_num_list=None, max_m=0.5, s=30, reweight_epoch=-1):
        super().__init__()
        if cls_num_list is None:
            # No cls_num_list is provided, then we cannot adjust cross entropy with LDAM.
            self.m_list = None
        else:
            self.reweight_epoch = reweight_epoch
            m_list = 1.0 / np.sqrt(np.sqrt(cls_num_list))
            m_list = m_list * (max_m / np.max(m_list))
            m_list = torch.tensor(m_list, dtype=torch.float, requires_grad=False)
            self.m_list = m_list
            assert s > 0
            self.s = s
            if reweight_epoch != -1:
                idx = 1 # condition could be put in order to set idx
                betas = [0, 0.9999]
                effective_num = 1.0 - np.power(betas[idx], cls_num_list)
                per_cls_weights = (1.0 - betas[idx]) / np.array(effective_num)
                per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * len(cls_num_list)
                self.per_cls_weights_enabled = torch.tensor(per_cls_weights, dtype=torch.float, requires_grad=False)
            else:
                self.per_cls_weights_enabled = None
                self.per_cls_weights = None

    def to(self, device):
        super().to(device)
        if self.m_list is not None:
            self.m_list = self.m_list.to(device)

        if self.per_cls_weights_enabled is not None:
            self.per_cls_weights_enabled = self.per_cls_weights_enabled.to(device)

        return self

    def _hook_before_epoch(self, epoch):
        if self.reweight_epoch != -1:
            self.epoch = epoch

            if epoch > self.reweight_epoch:
                self.per_cls_weights = self.per_cls_weights_enabled
            else:
                self.per_cls_weights = None

    def get_final_output(self, output_logits, target):
        x = output_logits

        index = torch.zeros_like(x, dtype=torch.uint8, device=x.device)
        index.scatter_(1, target.data.view(-1, 1), 1)

        index_float = index.float()
        batch_m = torch.matmul(self.m_list[None, :], index_float.transpose(0,1))

        batch_m = batch_m.view((-1, 1))
        x_m = x - batch_m * self.s

        final_output = torch.where(index, x_m, x)
        return final_output

    def forward(self, output_logits, target):
        if self.m_list is None:
            return F.cross_entropy(output_logits, target)

        final_output = self.get_final_output(output_logits, target)
        return F.cross_entropy(final_output, target, weight=self.per_cls_weights)


class RIDELoss(nn.Module):
    # def __init__(self, cls_num_list=None, base_diversity_temperature=1.0, max_m=0.8, s=30, reweight=True, reweight_epoch=800,
    #     base_loss_factor=1.0, additional_diversity_factor=-0.2, reweight_factor=0.05):#cifar10

    def __init__(self, cls_num_list=None, base_diversity_temperature=1.0, max_m=0.8, s=30, reweight=True,
                 reweight_epoch=200, base_loss_factor=1.0, additional_diversity_factor=-0.1, reweight_factor=0.01, num_experts=3):#cifar100
        super().__init__()
        self.base_loss = F.cross_entropy
        self.base_loss_factor = base_loss_factor
        if not reweight:
            self.reweight_epoch = -1
        else:
            self.reweight_epoch = reweight_epoch

        # LDAM is a variant of cross entropy and we handle it with self.m_list.
        if cls_num_list is None:
            # No cls_num_list is provided, then we cannot adjust cross entropy with LDAM.

            self.m_list = None
            self.per_cls_weights_enabled = None
            self.per_cls_weights_enabled_diversity = None
        else:
            # We will use LDAM loss if we provide cls_num_list.

            # avoid dividing by zero
            cls_num_list = [i if i != 0 else 0.1 for i in cls_num_list]

            m_list = 1.0 / np.sqrt(np.sqrt(cls_num_list))
            m_list = m_list * (max_m / np.max(m_list))
            m_list = torch.tensor(m_list, dtype=torch.float, requires_grad=False)
            self.m_list = m_list
            self.s = s
            assert s > 0
            
            if reweight_epoch != -1:
                idx = 1 # condition could be put in order to set idx
                betas = [0, 0.9999]
                effective_num = 1.0 - np.power(betas[idx], cls_num_list)
                per_cls_weights = (1.0 - betas[idx]) / np.array(effective_num)
                per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * len(cls_num_list)
                self.per_cls_weights_enabled = torch.tensor(per_cls_weights, dtype=torch.float, requires_grad=False)
            else:
                self.per_cls_weights_enabled = None

            cls_num_list = np.array(cls_num_list) / np.sum(cls_num_list)
            C = len(cls_num_list)
            per_cls_weights = C * cls_num_list * reweight_factor + 1 - reweight_factor

            # Experimental normalization: This is for easier hyperparam tuning, the effect can be described in the learning rate so the math formulation keeps the same.
            # At the same time, the 1 - max trick that was previously used is not required since weights are already adjusted.
            per_cls_weights = per_cls_weights / np.max(per_cls_weights)

            assert np.all(per_cls_weights > 0), "reweight factor is too large: out of bounds"
            # save diversity per_cls_weights
            self.per_cls_weights_enabled_diversity = torch.tensor(per_cls_weights, dtype=torch.float, requires_grad=False).cuda()

        self.base_diversity_temperature = base_diversity_temperature
        self.additional_diversity_factor = additional_diversity_factor

    def to(self, device):
        super().to(device)
        if self.m_list is not None:
            self.m_list = self.m_list.to(device)
        
        if self.per_cls_weights_enabled is not None:
            self.per_cls_weights_enabled = self.per_cls_weights_enabled.to(device)

        if self.per_cls_weights_enabled_diversity is not None:
            self.per_cls_weights_enabled_diversity = self.per_cls_weights_enabled_diversity.to(device)

        return self

    def _hook_before_epoch(self, epoch):
        if self.reweight_epoch != -1:
            self.epoch = epoch

            if epoch > self.reweight_epoch:
                self.per_cls_weights_base = self.per_cls_weights_enabled
                self.per_cls_weights_diversity = self.per_cls_weights_enabled_diversity
            else:
                self.per_cls_weights_base = None
                self.per_cls_weights_diversity = None

    def get_final_output(self, output_logits, target):
        x = output_logits

        index = torch.zeros_like(x, dtype=torch.uint8, device=x.device)
        index.scatter_(1, target.data.view(-1, 1), 1)
        
        index_float = index.float()
        batch_m = torch.matmul(self.m_list[None, :], index_float.transpose(0,1))
        
        batch_m = batch_m.view((-1, 1))
        x_m = x - batch_m * self.s

        final_output = torch.where(index, x_m, x)
        return final_output

    def forward(self, output_logits, target, extra_info=None, moe=False):
        if extra_info is None:
            return self.base_loss(output_logits, target)

        loss = 0

        # Adding RIDE Individual Loss for each expert
        for logits_item in extra_info['logits']:
            ride_loss_logits = output_logits if self.additional_diversity_factor == 0 else logits_item
            if self.m_list is None:
                loss += self.base_loss_factor * self.base_loss(ride_loss_logits, target)
            else:
                final_output = self.get_final_output(ride_loss_logits, target)
                loss += self.base_loss_factor * self.base_loss(final_output, target, weight=self.per_cls_weights_base)
            
            base_diversity_temperature = self.base_diversity_temperature

            if self.per_cls_weights_diversity is not None:
                diversity_temperature = base_diversity_temperature * self.per_cls_weights_diversity.view((1, -1))
                temperature_mean = diversity_temperature.mean().item()
            else:
                diversity_temperature = base_diversity_temperature
                temperature_mean = base_diversity_temperature
            
            output_dist = F.log_softmax(logits_item / diversity_temperature, dim=1)
            with torch.no_grad():
                # Using the mean takes only linear instead of quadratic time in computing and has only a slight difference so using the mean is preferred here
                mean_output_dist = F.softmax(output_logits / diversity_temperature, dim=1)
            
            loss += self.additional_diversity_factor * temperature_mean * temperature_mean * F.kl_div(output_dist, mean_output_dist, reduction='batchmean')
        
        return loss


class RIDELossWithDistill(nn.Module):
    def __init__(self, cls_num_list=None, additional_distill_loss_factor=1.0, distill_temperature=1.0, ride_loss_factor=1.0, **kwargs):
        super().__init__()
        self.ride_loss = RIDELoss(cls_num_list=cls_num_list, **kwargs)
        self.distill_temperature = distill_temperature

        self.ride_loss_factor = ride_loss_factor
        self.additional_distill_loss_factor = additional_distill_loss_factor

    def to(self, device):
        super().to(device)
        self.ride_loss = self.ride_loss.to(device)
        return self

    def _hook_before_epoch(self, epoch):
        self.ride_loss._hook_before_epoch(epoch)

    def forward(self, student, target=None, teacher=None, extra_info=None):
        output_logits = student
        if extra_info is None:
            return self.ride_loss(output_logits, target)

        loss = 0
        num_experts = len(extra_info['logits'])
        for logits_item in extra_info['logits']:
            loss += self.ride_loss_factor * self.ride_loss(output_logits, target, extra_info)
            distill_temperature = self.distill_temperature

            student_dist = F.log_softmax(student / distill_temperature, dim=1)
            with torch.no_grad():
                teacher_dist = F.softmax(teacher / distill_temperature, dim=1)
            
            distill_loss = F.kl_div(student_dist, teacher_dist, reduction='batchmean')
            distill_loss = distill_temperature * distill_temperature * distill_loss
            loss += self.additional_distill_loss_factor * distill_loss
        return loss


class DiverseExpertLoss(nn.Module):
    def __init__(self, cls_num_list=None, max_m=0.5, s=30, tau=2, moe=False, num_experts=3):
        super().__init__()
        self.base_loss = F.cross_entropy

        prior = np.array(cls_num_list) / np.sum(cls_num_list)
        self.prior = torch.tensor(prior)
        self.C_number = len(cls_num_list)  # class number
        self.s = s
        self.tau = tau
        self.moe = moe

    def inverse_prior(self, prior):
        value, idx0 = torch.sort(prior)
        _, idx1 = torch.sort(idx0)
        idx2 = prior.shape[0] - 1 - idx1  # reverse the order
        inverse_prior = value.index_select(0, idx2)

        return inverse_prior

    def forward(self, output_logits, target, extra_info=None, moe=False, training_exp=None, device=0):
        if extra_info is None:
            return self.base_loss(output_logits, target)  # output_logits indicates the final prediction
        self.prior = self.prior.to(device)
        loss = 0

        # if moe:
        #     moe_weight = extra_info['moe_weight']
        #     # output_logits = output_logits + torch.log(self.prior + 1e-9)
        #     loss = self.base_loss(output_logits, target)
        #     regular = torch.sum(-torch.log(moe_weight.squeeze(-1)))/target.shape[0]
        #     loss += regular
        #     return loss

        # Obtain logits from each expert
        expert1_logits = extra_info['logits'][0]
        expert2_logits = extra_info['logits'][1]
        expert3_logits = extra_info['logits'][2]
        target = target.long()
        # Softmax loss for expert 1
        if training_exp is None or '0' in training_exp:
            loss += self.base_loss(expert1_logits, target)

        # Balanced Softmax loss for expert 2
        if training_exp is None or '1' in training_exp:
            expert2_logits = expert2_logits + torch.log(self.prior + 1e-9)
            loss += self.base_loss(expert2_logits, target)

        # Inverse Softmax loss for expert 3
        if training_exp is None or '2' in training_exp:
            inverse_prior = self.inverse_prior(self.prior)
            expert3_logits = expert3_logits + torch.log(self.prior + 1e-9) - self.tau * torch.log(inverse_prior + 1e-9)
            # expert3_logits = expert3_logits + torch.log(self.prior + 1e-9)
            loss += self.base_loss(expert3_logits, target)

        return loss


class FDGELoss(nn.Module):
    def __init__(self, cls_num_list=None, s=30, tau=2, moe=False, num_experts=3):
        super().__init__()
        self.base_loss = F.cross_entropy
        prior = np.array(cls_num_list) / np.sum(cls_num_list)
        self.prior = torch.tensor(prior)
        self.C_number = len(cls_num_list)  # class number
        self.num_experts = num_experts

    def forward(self, output_logits, target, extra_info=None, training_exp=None, device=None):
        if extra_info is None:
            return self.base_loss(output_logits, target)  # output_logits indicates the final prediction

        loss = 0

        hard_label = torch.zeros([len(self.prior)]).to(target.get_device())
        self.prior = self.prior.to(target.get_device())

        for exp_id, expert_logit in enumerate(extra_info['logits']):

            # Obtain logits from each expert

            target = target.long()

            func = self.func(exp_id)

            # print("#####################", idx, "####################")
            if training_exp is None or str(exp_id) in training_exp: ###use one experts
                loss += self.base_loss(expert_logit + func, target)
                # loss += self.base_loss(expert_logit, target)

            # _, predicted = torch.max(expert_logit, -1)
            # hard_sample = ~ predicted.eq(target)
            #
            # for idx, i in enumerate(hard_sample):
            #     # hard_label[target[idx]] += int(i)
            #     if exp_id == 0:
            #         hard_label[target[idx]] = int(i)
            #     else:
            #         hard_label[target[idx]] = int(i) * hard_label[target[idx]]
            # hard_label = hard_label * (exp_id+1)/2

        # if training_exp is None:
        #     loss = loss/self.num_experts

        if training_exp is not None:
            loss = loss * self.num_experts

        return loss

    def func(self, expert_id, device=None, hard_label=None):
        # func = np.array(list(range(len(self.prior))))
        prior = self.prior
        func = torch.log(prior + 1e-9) #balance softmax base

        return func

import copy
import logging

import torch
from collections import Counter
from torch import nn, autograd
from torch.utils.data import DataLoader, Dataset, Sampler
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable
from sklearn.preprocessing import label_binarize


class CrossEntropy(nn.Module):
    def __init__(self, para_dict=None):
        super(CrossEntropy, self).__init__()
        self.para_dict = para_dict
        self.num_classes = self.para_dict['num_classes']
        self.num_class_list = self.para_dict['num_class_list']
        self.device = self.para_dict['device']

        self.weight_list = None
        #setting about defferred re-balancing by re-weighting (DRW)
        self.drw = self.para_dict['if_drw']
        self.drw_start_epoch = self.para_dict['drw_start_round']


    def forward(self, inputs, targets, **kwargs):
        """
        Args:
            inputs: prediction matrix (before softmax) with shape (batch_size, num_classes)
            targets: ground truth labels with shape (batch_size)
        """
        loss = F.cross_entropy(inputs, targets, weight=self.weight_list)
        return loss

    def update(self, epoch):
        """
        Adopt cost-sensitive cross-entropy as the default
        Args:
            epoch: int. starting from 1.
        """
        start = (epoch-1) // self.drw_start_epoch
        if start and self.drw:
            self.weight_list = torch.FloatTensor(np.array([min(self.num_class_list) / N for N in self.num_class_list])).to(self.device)


class FocalLoss(nn.Module):
    def __init__(self, gamma=0, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim() > 2:
            input = input.view(input.size(0), input.size(1), -1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1, )    # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1, input.size(2))   # N,H*W,C => N*H*W,C
        target = target.view(-1, 1)

        logpt = F.log_softmax(input, dim=-1)
        logpt = logpt.gather(1, target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type() != input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0, target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1-pt)**self.gamma * logpt
        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()


class LDAMLoss(nn.Module):
    def __init__(self, cls_num_list=None, max_m=0.5, s=30, reweight_ep=0):
        super().__init__()

        self.cls_num_list = [i if i != 0 else 0.1 for i in cls_num_list]
        m_list = 1.0 / np.sqrt(np.sqrt(self.cls_num_list))
        m_list = m_list * (max_m / np.max(m_list))
        m_list = torch.tensor(m_list, dtype=torch.float, requires_grad=False)
        self.m_list = m_list
        self.s = s
        self.reweight_ep = reweight_ep
        self.per_cls_weights = None

    def forward(self, x, target, round_idx=0, device='0'):

        idx = int(round_idx // self.reweight_ep)

        betas = [0, 0.9999]
        effective_num = 1.0 - np.power(betas[idx], self.cls_num_list)
        per_cls_weights = (1.0 - betas[idx]) / np.array(effective_num)
        per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * len(self.cls_num_list)
        self.per_cls_weights = torch.tensor(per_cls_weights, dtype=torch.float, requires_grad=False)

        index = torch.zeros_like(x, dtype=torch.uint8)
        index.scatter_(1, target.data.view(-1, 1), 1)

        self.m_list = self.m_list.to(device)
        index_float = index.type(torch.FloatTensor).to(device)
        self.per_cls_weights = self.per_cls_weights.to(device)

        batch_m = torch.matmul(self.m_list[None, :], index_float.transpose(0, 1))
        batch_m = batch_m.view((-1, 1))
        x_m = x - self.s * batch_m

        output = torch.where(index, x_m, x)
        return F.cross_entropy(output, target, weight=self.per_cls_weights)


class GHMC(nn.Module):
    """GHM Classification Loss.

    Details of the theorem can be viewed in the paper
    "Gradient Harmonized Single-stage Detector".
    https://arxiv.org/abs/1811.05181

    Args:
        bins (int): Number of the unit regions for distribution calculation.
        momentum (float): The parameter for moving average.
        use_sigmoid (bool): Can only be true for BCE based loss now.
        loss_weight (float): The weight of the total GHM-C loss.
    """
    def __init__(
            self,
            bins=30,
            momentum=0,
            use_sigmoid=True,
            loss_weight=1.0):
        super(GHMC, self).__init__()
        self.bins = bins
        self.momentum = momentum
        self.edges = torch.arange(bins + 1).float().cuda() / bins
        self.edges[-1] += 1e-6
        if momentum > 0:
            self.acc_sum = torch.zeros(bins).cuda()
        self.use_sigmoid = use_sigmoid
        if not self.use_sigmoid:
            raise NotImplementedError
        self.loss_weight = loss_weight

    def forward(self, pred, target, label, *args, **kwargs):
        """Calculate the GHM-C loss.

        Args:
            pred (float tensor of size [batch_num, class_num]):
                The direct prediction of classification fc layer.
            target (float tensor of size [batch_num, class_num]):
                Binary class target for each sample.
            label_weight (float tensor of size [batch_num, class_num]):
                the value is 1 if the sample is valid and 0 if ignored.
        Returns:
            The gradient harmonized loss.
        """
        # the target should be binary class label
        edges = self.edges
        mmt = self.momentum
        self.edges = self.edges.to(target.get_device())
        self.acc_sum = self.acc_sum.to(target.get_device())

        N = pred.size(0)
        C = pred.size(1)
        P = F.softmax(pred)

        class_mask = pred.data.new(N, C).fill_(0)
        class_mask = Variable(class_mask)
        ids = label.view(-1, 1)
        class_mask.scatter_(1, ids.data, 1.)

        # gradient length
        g = torch.abs(pred.sigmoid().detach() - target)
        g = (g * class_mask).sum(1).view(-1, 1)
        weights = torch.zeros_like(g)

        # valid = label_weight > 0
        # tot = max(valid.float().sum().item(), 1.0)
        tot = pred.size(0)
        n = 0  # n valid bins
        for i in range(self.bins):
            inds = (g >= edges[i]) & (g < edges[i+1]) 
            num_in_bin = inds.sum().item()
            # print(num_in_bin)
            if num_in_bin > 0:
                if mmt > 0:
                    self.acc_sum[i] = mmt * self.acc_sum[i] \
                        + (1 - mmt) * num_in_bin
                    weights[inds] = tot / self.acc_sum[i]
                else:
                    weights[inds] = tot / num_in_bin
                n += 1
        if n > 0:
            weights = weights / n

        
        # print(pred)
        # pred = P * weights
        # print(pred)

        probs = (P * class_mask).sum(1).view(-1, 1)

        log_p = probs.log()
        # print(probs)
        # print(probs)
        # print(log_p.size(), weights.size())

        batch_loss = -log_p * weights / tot
        # print(batch_loss)
        loss = batch_loss.sum()
        # print(loss)
        return loss


class CrossEntropyLabelAwareSmooth(CrossEntropy):
    r"""Cross entropy loss with label-aware smoothing regularizer.

    Reference:
        Zhong et al. Improving Calibration for Long-Tailed Recognition. CVPR 2021. https://arxiv.org/abs/2104.00466

    For more details of label-aware smoothing, you can see Section 3.2 in the above paper.

    Args:
        shape (str): the manner of how to get the params of label-aware smoothing.
        smooth_head (float): the largest  label smoothing factor
        smooth_tail (float): the smallest label smoothing factor
    """
    def __init__(self, para_dict=None):
        super(CrossEntropyLabelAwareSmooth, self).__init__(para_dict)

        smooth_head = self.para_dict['SMOOTH_HEAD']
        smooth_tail = self.para_dict['SMOOTH_TAIL']
        shape = self.para_dict['SHAPE']

        n_1 = max(self.num_class_list)
        n_K = min(self.num_class_list)
        if n_1 == n_K:
            n_1 = n_K + 1
        if shape == 'concave':
            self.smooth = smooth_tail + (smooth_head - smooth_tail) * np.sin((np.array(self.num_class_list) - n_K) * np.pi / (2 * (n_1 - n_K)))
        elif shape == 'linear':
            self.smooth = smooth_tail + (smooth_head - smooth_tail) * (np.array(self.num_class_list) - n_K) / (n_1 - n_K)
        elif shape == 'convex':
            self.smooth = smooth_head + (smooth_head - smooth_tail) * np.sin(1.5 * np.pi + (np.array(self.num_class_list) - n_K) * np.pi / (2 * (n_1 - n_K)))
        else:
            raise AttributeError

        self.smooth = torch.from_numpy(self.smooth)
        self.smooth = self.smooth.float()
        if torch.cuda.is_available():
            self.smooth = self.smooth.cuda()

    def forward(self, inputs, targets, **kwargs):
        smoothing = self.smooth[targets].to(inputs.get_device())
        confidence = 1. - smoothing
        logprobs = F.log_softmax(inputs, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=targets.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = confidence * nll_loss + smoothing * smooth_loss
        return loss.mean()


class Weighted_Cross_Entropy(nn.Module):

    def __init__(self, args, class_num, alpha=None, size_average=True):
        self.args = args
        super(Weighted_Cross_Entropy, self).__init__()
        if alpha is None:
            self.alpha = Variable(torch.ones(class_num, 1))
        else:
            if isinstance(alpha, Variable):
                self.alpha = alpha
            else:
                self.alpha = Variable(alpha)
        self.class_num = class_num
        self.size_average = size_average

    def forward(self, inputs, targets):
        N = inputs.size(0)
        C = inputs.size(1)

        inputs = inputs.float()
        P = F.log_softmax(inputs, dim=-1)

        class_mask = inputs.data.new(N, C).fill_(0)
        class_mask = Variable(class_mask)
        ids = targets.view(-1, 1)
        class_mask.scatter_(1, ids.data, 1.)

        if inputs.is_cuda and not self.alpha.is_cuda:
            self.alpha = self.alpha.to(targets.get_device())
        alpha = self.alpha[ids.data.view(-1)]

        probs = (P * class_mask).sum(1).view(-1, 1)

        log_p = probs

        alpha = alpha.to(torch.device(self.args.gpu))
        probs = probs.to(torch.device(self.args.gpu))
        log_p = log_p.to(torch.device(self.args.gpu))

        batch_loss = - alpha * log_p
        # batch_loss = -log_p

        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss.sum()
        return loss


class CB_loss(nn.Module):
    def __init__(self, samples_per_cls, no_of_classes, device, beta=0.99, gamma=0.5):
        super().__init__()
        samples_per_cls = [i if i != 0 else 0.1 for i in samples_per_cls]
        effective_num = 1.0 - np.power(beta, samples_per_cls)
        weights = (1.0 - beta) / np.array(effective_num)
        weights = weights / np.sum(weights) * no_of_classes

        self.weights = weights
        self.beta = beta
        self.gamma = gamma
        self.device = device
        self.no_of_classes = no_of_classes

    def forward(self, logits, labels):

        labels_one_hot = F.one_hot(labels, self.no_of_classes).float().to(self.device)

        weights = torch.tensor(self.weights).float().to(self.device)
        weights = weights.unsqueeze(0)
        weights = weights.repeat(labels_one_hot.shape[0], 1) * labels_one_hot
        weights = weights.sum(1)
        weights = weights.unsqueeze(1)
        weights = weights.repeat(1, self.no_of_classes)

        BCLoss = F.binary_cross_entropy_with_logits(input=logits, target=labels_one_hot, reduction="none")

        if self.gamma == 0.0:
            modulator = 1.0
        else:
            modulator = torch.exp(-self.gamma * labels_one_hot * logits - self.gamma * torch.log(1 + torch.exp(-1.0 * logits))).to(self.device)

        loss = modulator * BCLoss

        weighted_loss = weights * loss
        loss = torch.sum(weighted_loss)

        loss /= torch.sum(labels_one_hot)

        return loss


class DROLoss(nn.Module):
    def __init__(self, temperature=0.07, base_temperature=0.07, class_weights=None, epsilons=None):
        super().__init__()
        self.temperature = temperature
        self.base_temperature = base_temperature
        self.class_weights = class_weights
        self.epsilons = epsilons

    def pairwise_euaclidean_distance(self, x, y):
        return torch.cdist(x, y)

    def pairwise_cosine_sim(self, x, y):
        x = x / x.norm(dim=1, keepdim=True)
        y = y / y.norm(dim=1, keepdim=True)
        return torch.matmul(x, y.T)

    def forward(self, batch_feats, batch_targets, centroid_feats, centroid_targets):
        device = (torch.device('cuda')
                  if centroid_feats.is_cuda
                  else torch.device('cpu'))

        classes, positive_counts = torch.unique(batch_targets, return_counts=True)
        centroid_classes = torch.unique(centroid_targets)
        train_prototypes = torch.stack([centroid_feats[torch.where(centroid_targets == c)[0]].mean(0)
                                        for c in centroid_classes])
        pairwise = -1 * self.pairwise_euaclidean_distance(train_prototypes, batch_feats)

        # epsilons
        if self.epsilons is not None:
            mask = torch.eq(centroid_classes.contiguous().view(-1, 1), batch_targets.contiguous().view(-1, 1).T).to(
                device)
            a = pairwise.clone()
            pairwise[mask] = a[mask] - self.epsilons[batch_targets].to(device)

        logits = torch.div(pairwise, self.temperature)

        # compute log_prob
        log_prob = logits - torch.log(torch.exp(logits).sum(1, keepdim=True))
        log_prob = torch.stack([log_prob[:, torch.where(batch_targets == c)[0]].mean(1) for c in classes], dim=1)

        # compute mean of log-likelihood over positive
        mask = torch.eq(centroid_classes.contiguous().view(-1, 1), classes.contiguous().view(-1, 1).T).float().to(
            device)
        log_prob_pos = (mask * log_prob).sum(1)

        # loss
        loss = - (self.temperature / self.base_temperature) * log_prob_pos
        # weight by class weight
        if self.class_weights is not None:
            weights = self.class_weights[centroid_classes]
            weighted_loss = loss * weights
            loss = weighted_loss.sum() / weights.sum()
        else:
            loss = loss.sum() / len(classes)

        return loss


###LADE###
class LADELoss(nn.Module):
    def __init__(self, num_classes=10, img_max=None, prior=None, prior_txt=None, remine_lambda=0.1):
        super().__init__()
        if img_max is not None or prior_txt is not None:
            self.img_num_per_cls = torch.Tensor(prior_txt)
            self.prior = self.img_num_per_cls / self.img_num_per_cls.sum()
        else:
            self.prior = None

        self.balanced_prior = torch.tensor(1. / num_classes)
        self.remine_lambda = remine_lambda

        self.num_classes = num_classes
        self.cls_weight = (self.img_num_per_cls.float() / torch.sum(self.img_num_per_cls.float()))

    def mine_lower_bound(self, x_p, x_q, num_samples_per_cls):
        N = x_p.size(-1)
        first_term = torch.sum(x_p, -1) / (num_samples_per_cls + 1e-8)
        second_term = torch.logsumexp(x_q, -1) - np.log(N)

        return first_term - second_term, first_term, second_term

    def remine_lower_bound(self, x_p, x_q, num_samples_per_cls):
        loss, first_term, second_term = self.mine_lower_bound(x_p, x_q, num_samples_per_cls)
        reg = (second_term ** 2) * self.remine_lambda
        return loss - reg, first_term, second_term

    def forward(self, y_pred, target, q_pred=None):
        """
        y_pred: N x C
        target: N
        """
        self.prior = self.prior.to(y_pred.get_device())
        self.balanced_prior = self.balanced_prior.to(y_pred.get_device())
        self.cls_weight = self.cls_weight.to(y_pred.get_device())
        per_cls_pred_spread = y_pred.T * (target == torch.arange(0, self.num_classes).view(-1, 1).type_as(target))  # C x N
        pred_spread = (y_pred - torch.log(self.prior + 1e-9) + torch.log(self.balanced_prior + 1e-9)).T  # C x N

        num_samples_per_cls = torch.sum(target == torch.arange(0, self.num_classes).view(-1, 1).type_as(target), -1).float()  # C
        estim_loss, first_term, second_term = self.remine_lower_bound(per_cls_pred_spread, pred_spread, num_samples_per_cls)

        loss = -torch.sum(estim_loss * self.cls_weight)
        return loss


class PriorCELoss(nn.Module):
    # Also named as LADE-CE Loss
    def __init__(self, num_classes, img_max=None, prior=None, prior_txt=None):
        super().__init__()
        self.img_num_per_cls = torch.Tensor(prior_txt)
        self.prior = self.img_num_per_cls / self.img_num_per_cls.sum()
        self.criterion = nn.CrossEntropyLoss()
        self.num_classes = num_classes

    def forward(self, x, y):
        self.prior = self.prior.to(x.get_device())
        logits = x + torch.log(self.prior + 1e-9)
        loss = self.criterion(logits, y)
        return loss


class PaCoLoss(nn.Module):
    def __init__(self, alpha, beta=1.0, gamma=1.0, supt=1.0, temperature=1.0, base_temperature=None, K=128,
                 num_classes=100, device=None):
        super(PaCoLoss, self).__init__()
        self.temperature = temperature
        self.base_temperature = temperature if base_temperature is None else base_temperature
        self.K = K
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.supt = supt
        self.num_classes = num_classes
        self.device = device

    def cal_weight_for_classes(self, cls_num_list):
        cls_num_list = torch.Tensor(cls_num_list).view(1, self.num_classes)
        self.weight = cls_num_list / cls_num_list.sum()
        self.weight = self.weight.to(self.device)

    def forward(self, features, labels=None, sup_logits=None):
        device = self.device

        ss = features.shape[0]
        batch_size = (features.shape[0] - self.K) // 2

        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels[:batch_size], labels.T).float().to(device)

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(features[:batch_size], features.T),
            self.temperature)

        # add supervised logits
        anchor_dot_contrast = torch.cat(((sup_logits + torch.log(self.weight + 1e-9)) / self.supt, anchor_dot_contrast),
                                        dim=1)

        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # add ground truth
        one_hot_label = torch.nn.functional.one_hot(labels[:batch_size, ].view(-1, ), num_classes=self.num_classes).to(
            torch.float32)
        mask = torch.cat((one_hot_label * self.beta, mask * self.alpha), dim=1)

        # compute log_prob
        logits_mask = torch.cat((torch.ones(batch_size, self.num_classes).to(device), self.gamma * logits_mask), dim=1)
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-12)

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.mean()

        return loss


def AccuarcyCompute(pred, label):
    pred = pred.cpu().data.numpy()
    label = label.cpu().data.numpy()
    test_np = np.argmax(pred, 1)
    count = 0
    for i in range(len(test_np)):
        if test_np[i] == label[i]:
            count += 1
    return np.sum(count), len(test_np)


class DatasetSplit(Dataset):
    def __init__(self, dataset, labels, idxs, transform=None, target_transform=None):
        self.dataset = dataset
        self.labels = labels
        self.idxs = list(idxs)
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        if len(self.labels) != 0:
            image = self.dataset[self.idxs[item]]
            label = self.labels[self.idxs[item]]
        else:
            image, label = self.dataset[self.idxs[item]]

        if self.transform is not None:
            image = self.transform(image)

        if self.target_transform is not None:
            label = self.target_transform(label)

        return image, label
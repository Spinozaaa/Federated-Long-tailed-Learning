import logging

import numpy as np
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import copy

from collections import Counter
from .cifar_lt import IMBALANCECIFAR100
from .datasets import CIFAR100_truncated

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)

CLASS_NUM = 100


def record_net_data_stats(y_train, net_dataidx_map):
    net_cls_counts = {}

    for net_i, dataidx in net_dataidx_map.items():
        unq, unq_cnt = np.unique(y_train[dataidx], return_counts=True)
        tmp = {unq[i]: unq_cnt[i] for i in range(len(unq))}
        net_cls_counts[net_i] = tmp
    logging.debug('Data statistics: %s' % str(net_cls_counts))
    return net_cls_counts


class Cutout(object):
    def __init__(self, length):
        self.length = length

    def __call__(self, img):
        h, w = img.size(1), img.size(2)
        mask = np.ones((h, w), np.float32)
        y = np.random.randint(h)
        x = np.random.randint(w)

        y1 = np.clip(y - self.length // 2, 0, h)
        y2 = np.clip(y + self.length // 2, 0, h)
        x1 = np.clip(x - self.length // 2, 0, w)
        x2 = np.clip(x + self.length // 2, 0, w)

        mask[y1: y2, x1: x2] = 0.
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img *= mask
        return img


def _data_transforms_cifar100():
    CIFAR_MEAN = [0.5071, 0.4865, 0.4409]
    CIFAR_STD = [0.2673, 0.2564, 0.2762]

    train_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])

    train_transform.transforms.append(Cutout(16))

    valid_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])

    return train_transform, valid_transform


def load_cifar100_data(args):
    train_transform, test_transform = _data_transforms_cifar100()

    cifar100_train_ds = CIFAR100_truncated(args.data_dir, train=True, download=True, transform=train_transform, contrast=contrast)
    cifar100_test_ds = CIFAR100_truncated(args.data_dir, train=False, download=True, transform=test_transform)

    return cifar100_train_ds, cifar100_test_ds


def load_cifar100_lt_data(args):
    train_transform, test_transform = _data_transforms_cifar100()

    cifar100_train_ds = IMBALANCECIFAR100(imbalance_ratio=args.imb_factor, root=args.data_dir,  train=True, download=False, transform=train_transform, contrast=contrast)
    cifar100_test_ds = IMBALANCECIFAR100(imbalance_ratio=args.imb_factor, root=args.data_dir, train=False, download=False, transform=test_transform)

    return cifar100_train_ds, cifar100_test_ds


def partition_data(args, n_nets=0):
    logging.info("*********partition data***************")
    partition = args.partition_method

    if "cifar100_lt" in args.dataset:
        cifar100_train_ds, cifar100_test_ds = load_cifar100_lt_data(args)
    else:
        cifar100_train_ds, cifar100_test_ds = load_cifar100_data(args)

    X_train, y_train = cifar100_train_ds.data, cifar100_train_ds.target
    n_train = X_train.shape[0]

    if partition == "homo":
        total_num = n_train
        idxs = np.random.permutation(total_num)
        batch_idxs = np.array_split(idxs, n_nets)
        net_dataidx_map = {i: batch_idxs[i] for i in range(n_nets)}

    elif partition == "hetero":
        min_size = 0
        K = CLASS_NUM
        N = n_train
        logging.info("N = " + str(N))
        net_dataidx_map = {}

        while min_size < 10:
            idx_batch = [[] for _ in range(n_nets)]
            # for each class in the dataset
            for k in range(K):
                idx_k = np.where(y_train == k)[0]
                np.random.shuffle(idx_k)
                proportions = np.random.dirichlet(np.repeat(args.partition_alpha, n_nets))
                ## Balance
                proportions = np.array([p * (len(idx_j) < N / n_nets) for p, idx_j in zip(proportions, idx_batch)])
                proportions = proportions / proportions.sum()
                proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
                idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]
                min_size = min([len(idx_j) for idx_j in idx_batch])

        for j in range(n_nets):
            np.random.shuffle(idx_batch[j])
            net_dataidx_map[j] = idx_batch[j]

    traindata_cls_counts = record_net_data_stats(y_train, net_dataidx_map)

    return cifar100_train_ds, cifar100_test_ds, net_dataidx_map, traindata_cls_counts


def get_dataloader_CIFAR100(dataset, bs, shuffle, dataidxs=None, drop_last=False):

    ds = copy.deepcopy(dataset)
    if dataidxs is not None:
        ds.data = ds.data[dataidxs]
        ds.target = ds.target[dataidxs]

    train_data = ds.data

    redundacy = train_data.shape[0] % bs
    if redundacy == 1:
        logging.info("Delete one sample to avoid bug in batch_norm for client")
        ds.data = np.delete(train_data, 0, axis=0)
        ds.target = np.delete(ds.target, 0, axis=0)

    dl = data.DataLoader(dataset=ds, batch_size=bs, shuffle=shuffle, drop_last=drop_last)

    return dl


def load_partition_data_cifar100(args):

    client_number = args.client_num_in_total

    cifar100_train_ds, cifar100_test_ds, net_dataidx_map, traindata_cls_counts = partition_data(args, n_nets=client_number)

    X_train, y_train = cifar100_train_ds.data, cifar100_train_ds.target
    X_test, y_test = cifar100_test_ds.data, cifar100_test_ds.target
    total_cls_num = Counter(y_train)
    if "lt" in args.dataset:
        print("================== class num:", total_cls_num, "=======================================")
    class_num = len(np.unique(y_train))

    train_data_num = sum([len(net_dataidx_map[r]) for r in range(client_number)])

    train_data_global = get_dataloader_CIFAR100(cifar100_train_ds, args.batch_size, True)
    test_data_global = get_dataloader_CIFAR100(cifar100_test_ds, args.batch_size, False)

    logging.info("train_dl_global number = " + str(len(train_data_global)))
    logging.info("test_dl_global number = " + str(len(train_data_global)))
    test_data_num = len(test_data_global)

    # get local dataset
    data_local_num_dict = dict()
    train_data_local_dict = dict()
    test_data_local_dict = dict()

    #test dataidx
    total_num = len(y_test)
    idxs = np.random.permutation(total_num)
    batch_idxs = np.array_split(idxs, client_number)
    test_dataidx_map = {i: batch_idxs[i] for i in range(client_number)}

    for client_idx in range(client_number):
        dataidxs = net_dataidx_map[client_idx]
        test_data_idx = test_dataidx_map[client_idx]
        local_data_num = len(dataidxs)
        data_local_num_dict[client_idx] = local_data_num
        # logging.info("client_idx = %d, local_sample_number = %d" % (client_idx, local_data_num))

        train_data_local = get_dataloader_CIFAR100(cifar100_train_ds, args.batch_size, True, dataidxs)
        test_data_local = get_dataloader_CIFAR100(cifar100_test_ds, args.batch_size, False, test_data_idx)

        train_data_local_dict[client_idx] = train_data_local
        test_data_local_dict[client_idx] = test_data_local

    return train_data_num, test_data_num, train_data_global, test_data_global, data_local_num_dict, \
           train_data_local_dict, test_data_local_dict, class_num, traindata_cls_counts

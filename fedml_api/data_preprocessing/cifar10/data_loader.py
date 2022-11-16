import copy
import logging

import numpy as np
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import tkinter
from .datasets import CIFAR10_truncated
from .cifar_lt import IMBALANCECIFAR10
from collections import Counter

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)

CLASS_NUM = 10 
# generate the non-IID distribution for all methods
def read_data_distribution(filename='./data_preprocessing/non-iid-distribution/CIFAR10/distribution.txt'):
    distribution = {}
    with open(filename, 'r') as data:
        for x in data.readlines():
            if '{' != x[0] and '}' != x[0]:
                tmp = x.split(':')
                if '{' == tmp[1].strip():
                    first_level_key = int(tmp[0])
                    distribution[first_level_key] = {}
                else:
                    second_level_key = int(tmp[0])
                    distribution[first_level_key][second_level_key] = int(tmp[1].strip().replace(',', ''))
    return distribution


def read_net_dataidx_map(filename='./data_preprocessing/non-iid-distribution/CIFAR10/net_dataidx_map.txt'):
    net_dataidx_map = {}
    with open(filename, 'r') as data:
        for x in data.readlines():
            if '{' != x[0] and '}' != x[0] and ']' != x[0]:
                tmp = x.split(':')
                if '[' == tmp[-1].strip():
                    key = int(tmp[0])
                    net_dataidx_map[key] = []
                else:
                    tmp_array = x.split(',')
                    net_dataidx_map[key] = [int(i.strip()) for i in tmp_array]
    return net_dataidx_map


def record_net_data_stats(y_train, net_dataidx_map):
    net_cls_counts = {}

    for net_i, dataidx in net_dataidx_map.items():
        unq, unq_cnt = np.unique(y_train[dataidx], return_counts=True)
        tmp = {unq[i]: unq_cnt[i] for i in range(len(unq))}
        net_cls_counts[net_i] = tmp
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


def _data_transforms_cifar10():
    CIFAR_MEAN = [0.49139968, 0.48215827, 0.44653124]
    CIFAR_STD = [0.24703233, 0.24348505, 0.26158768]

    train_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])

    # train_transform.transforms.append(Cutout(16))

    valid_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])

    return train_transform, valid_transform


def load_cifar10_data(args):
    train_transform, test_transform = _data_transforms_cifar10()
    cifar10_train_ds = CIFAR10_truncated(args.data_dir, train=True, download=True, transform=train_transform)
    cifar10_test_ds = CIFAR10_truncated(args.data_dir, train=False, download=True, transform=test_transform)

    return (cifar10_train_ds, cifar10_test_ds)

def load_cifar10_lt_data(args):
    train_transform, test_transform = _data_transforms_cifar10()

    cifar10_train_ds = IMBALANCECIFAR10(imbalance_ratio=args.imb_factor, root=args.data_dir,  train=True, download=True, transform=train_transform)
    cifar10_test_ds = IMBALANCECIFAR10(imbalance_ratio=args.imb_factor, root=args.data_dir, train=False, download=True, transform=test_transform)

    return (cifar10_train_ds, cifar10_test_ds)


def get_img_num_per_cls(client_data_num, cls_num, imb_type='exp', imb_factor=0.1):
    img_max = client_data_num / cls_num
    img_num_per_cls = []
    if imb_type == 'exp':
        for cls_idx in range(cls_num):
            num = img_max * (imb_factor**(cls_idx / (cls_num - 1.0)))
            img_num_per_cls.append(int(num))
    elif imb_type == 'step':
        for cls_idx in range(cls_num // 2):
            img_num_per_cls.append(int(img_max))
        for cls_idx in range(cls_num // 2):
            img_num_per_cls.append(int(img_max * imb_factor))
    else:
        img_num_per_cls.extend([int(img_max)] * cls_num)
    return img_num_per_cls


def partition_data(args, n_nets=0):
    logging.info("*********partition data***************")
    partition = args.partition_method
    if args.dataset == "cifar10_lt":
        cifar10_train_ds, cifar10_test_ds = load_cifar10_lt_data(args)
    else:
        cifar10_train_ds, cifar10_test_ds = load_cifar10_data(args)

    X_train, y_train = cifar10_train_ds.data, cifar10_train_ds.target

    n_train = X_train.shape[0]
    # n_test = X_test.shape[0]

    if partition == "homo":
        total_num = n_train
        idxs = np.random.permutation(total_num)
        batch_idxs = np.array_split(idxs, n_nets)
        net_dataidx_map = {i: batch_idxs[i] for i in range(n_nets)}

    elif partition == "hetero":
        min_size = 0
        min_num_per_class = 0
        K = CLASS_NUM
        N = n_train
        logging.info("N = " + str(N))
        net_dataidx_map = {}
        #
        # share_cls_data = []
        # for k in range(K):
        #     share_cls_data.append(np.where(y_train == k)[0][0])

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
                # min_num_per_class = min([len(i) for i in np.split(idx_k, proportions)])

        for j in range(n_nets):
            np.random.shuffle(idx_batch[j])
            net_dataidx_map[j] = idx_batch[j]

    if partition == "hetero-fix":
        distribution_file_path = './data_preprocessing/non-iid-distribution/CIFAR10/distribution.txt'
        traindata_cls_counts = read_data_distribution(distribution_file_path)
    else:
        traindata_cls_counts = record_net_data_stats(y_train, net_dataidx_map)

    return cifar10_train_ds, cifar10_test_ds, net_dataidx_map, traindata_cls_counts


def get_dataloader_CIFAR10(dataset, bs, shuffle, dataidxs=None):

    drop_last = False

    ds = copy.deepcopy(dataset)
    if dataidxs is not None:
        ds.data = ds.data[dataidxs]
        ds.target = ds.target[dataidxs]

    train_data = ds.data

    redundacy = train_data.shape[0] % bs
    if redundacy == 1:
        logging.info("Delete one sample to avoid bug in batch_norm for client")
        ds.data = np.delete(train_data, 1, axis=0)
        ds.target = np.delete(ds.target, 1, axis=0)

    dl = data.DataLoader(dataset=ds, batch_size=bs, shuffle=shuffle, drop_last=drop_last)

    return dl


def load_partition_data_cifar10(args):

    client_number = args.client_num_in_total
    cifar10_train_ds, cifar10_test_ds, net_dataidx_map, traindata_cls_counts = partition_data(args, n_nets=client_number)

    X_train, y_train = cifar10_train_ds.data, cifar10_train_ds.target
    X_test, y_test = cifar10_test_ds.data, cifar10_test_ds.target

    class_num = len(np.unique(y_train))
    num_per_cls = np.array(list(Counter(y_train).values()))
    clt_cls_list = []
    for i in traindata_cls_counts.values():
        cls_list = np.array([i[j] if j in i.keys() else 0 for j in range(class_num)])
        clt_cls_list.append(cls_list/np.sum(cls_list))
    clt_cls_list = np.array(clt_cls_list)
    print("avg client class ratoio:", np.sum(clt_cls_list, 0)/len(clt_cls_list))
    print("total class ratoio:", num_per_cls/np.sum(num_per_cls))

    logging.info("total sample number per class: " + str(num_per_cls))
    logging.info("traindata_cls_counts = " + str(traindata_cls_counts))
    train_data_num = sum([len(net_dataidx_map[r]) for r in range(client_number)])

    train_data_global = get_dataloader_CIFAR10(cifar10_train_ds, args.batch_size, True)
    test_data_global = get_dataloader_CIFAR10(cifar10_test_ds, args.batch_size, False)

    logging.info("train_dl_global number = " + str(len(train_data_global)))
    logging.info("test_dl_global number = " + str(len(test_data_global)))
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

        train_data_local = get_dataloader_CIFAR10(cifar10_train_ds, args.batch_size, True, dataidxs)
        test_data_local = get_dataloader_CIFAR10(cifar10_test_ds, args.batch_size, False, test_data_idx)

        train_data_local_dict[client_idx] = train_data_local
        test_data_local_dict[client_idx] = test_data_local

    return train_data_num, test_data_num, train_data_global, test_data_global, data_local_num_dict, \
           train_data_local_dict, test_data_local_dict, class_num, traindata_cls_counts

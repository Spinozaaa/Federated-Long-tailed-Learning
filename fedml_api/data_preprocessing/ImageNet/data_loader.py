import logging

import numpy as np
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import copy

from .datasets import ImageNet
from .datasets import ImageNet_truncated

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)

CLASS_NUM = 1000

def record_net_data_stats(y_train, net_dataidx_map):
    net_cls_counts = {}

    for net_i, dataidx in net_dataidx_map.items():
        unq, unq_cnt = np.unique(y_train[dataidx], return_counts=True)
        tmp = {unq[i]: unq_cnt[i] for i in range(len(unq))}
        net_cls_counts[net_i] = tmp
    # logging.debug('Data statistics: %s' % str(net_cls_counts))
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


def _data_transforms_ImageNet():
    # IMAGENET_MEAN = [0.5071, 0.4865, 0.4409]
    # IMAGENET_STD = [0.2673, 0.2564, 0.2762]
    logging.getLogger('PIL').setLevel(logging.WARNING)
    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]

    image_size = 32
    train_transform = transforms.Compose([
        # transforms.ToPILImage(),
        transforms.RandomResizedCrop(image_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    train_transform.transforms.append(Cutout(16))

    valid_transform = transforms.Compose([
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    return train_transform, valid_transform


# for centralized training
def get_dataloader(dataset, datadir, train_bs, test_bs, dataidxs=None):
    return get_dataloader_ImageNet(datadir, train_bs, test_bs, dataidxs)


# for local devices
def get_dataloader_test(dataset, datadir, train_bs, test_bs, dataidxs_train, dataidxs_test):
    return get_dataloader_test_ImageNet(datadir, train_bs, test_bs, dataidxs_train, dataidxs_test)


def get_dataloader_ImageNet_truncated(imagenet_dataset_train: ImageNet, imagenet_dataset_test: ImageNet, train_bs,
                                      test_bs, dataidxs=None, net_dataidx_map=None, args=None):

    dl_obj = ImageNet_truncated
    if args is not None and "paco" in args.method:
        transform_train, transform_test = paco_aug(dataset=args.dataset)
    else:
        transform_train, transform_test = _data_transforms_ImageNet()

    train_ds = dl_obj(imagenet_dataset_train, dataidxs, net_dataidx_map, train=True, transform=transform_train,
                      download=False)
    test_ds = dl_obj(imagenet_dataset_test, dataidxs=None, net_dataidx_map=None, train=False, transform=transform_test,
                     download=False)

    train_dl = data.DataLoader(dataset=train_ds, batch_size=train_bs, shuffle=True, drop_last=False)
    test_dl = data.DataLoader(dataset=test_ds, batch_size=test_bs, shuffle=False, drop_last=False)

    return train_dl, test_dl


def get_dataloader_ImageNet(datadir, train_bs, test_bs, dataidxs=None):
    dl_obj = ImageNet

    transform_train, transform_test = _data_transforms_ImageNet()

    train_ds = dl_obj(datadir, dataidxs=dataidxs, train=True, transform=transform_train, download=False)
    test_ds = dl_obj(datadir, dataidxs=None, train=False, transform=transform_test, download=False)

    train_dl = data.DataLoader(dataset=train_ds, batch_size=train_bs, shuffle=True, drop_last=False)
    test_dl = data.DataLoader(dataset=test_ds, batch_size=test_bs, shuffle=False, drop_last=False)

    return train_dl, test_dl


def get_dataloader_test_ImageNet(datadir, train_bs, test_bs, dataidxs_train=None, dataidxs_test=None):
    dl_obj = ImageNet

    transform_train, transform_test = _data_transforms_ImageNet()

    train_ds = dl_obj(datadir, dataidxs=dataidxs_train, train=True, transform=transform_train, download=True)
    test_ds = dl_obj(datadir, dataidxs=dataidxs_test, train=False, transform=transform_test, download=True)

    train_dl = data.DataLoader(dataset=train_ds, batch_size=train_bs, shuffle=True, drop_last=False)
    test_dl = data.DataLoader(dataset=test_ds, batch_size=test_bs, shuffle=False, drop_last=False)

    return train_dl, test_dl


def load_partition_data_ImageNet(args, get_bala_dataloader=False):

    train_dataset = ImageNet(data_dir=args.data_dir, dataidxs=None, train=True, args=args)
    lable_dict = train_dataset.label_dict
    test_dataset = ImageNet(data_dir=args.data_dir, dataidxs=None, train=False, args=args, label_dict=lable_dict)

    net_dataidx_map = train_dataset.get_net_dataidx_map()

    # logging.info("traindata_cls_counts = " + str(traindata_cls_counts))
    # train_data_num = sum([len(net_dataidx_map[r]) for r in range(client_number)])
    train_data_num = len(train_dataset)
    test_data_num = len(test_dataset)
    class_num_dict = train_dataset.get_data_local_num_dict()

    # train_data_global, test_data_global = get_dataloader(dataset, data_dir, batch_size, batch_size)

    train_data_global, test_data_global = get_dataloader_ImageNet_truncated(train_dataset, test_dataset,
                                                                            train_bs=args.batch_size, test_bs=args.batch_size,
                                                                            dataidxs=None, net_dataidx_map=None, )

    logging.info("train_dl_global number = " + str(len(train_data_global)))
    logging.info("test_dl_global number = " + str(len(test_data_global)))

    # get local dataset
    data_local_num_dict = train_dataset.get_data_local_num_dict()
    train_data_local_dict = dict()
    test_data_local_dict = dict()

    for client_idx in range(args.client_num_in_total):

        local_data_num = data_local_num_dict[client_idx]

        train_data_local, test_data_local = get_dataloader_ImageNet_truncated(train_dataset, test_dataset,
                                                                              train_bs=args.batch_size, test_bs=args.batch_size,
                                                                              dataidxs=client_idx,
                                                                              net_dataidx_map=net_dataidx_map, args=args)

        train_data_local_dict[client_idx] = train_data_local
        test_data_local_dict[client_idx] = test_data_local

    traindata_cls_counts = train_dataset.get_traindata_cls_counts()

    logging.info("data_local_num_dict: %s" % data_local_num_dict)
    return train_data_num, test_data_num, train_data_global, test_data_global, \
           data_local_num_dict, train_data_local_dict, test_data_local_dict, CLASS_NUM, traindata_cls_counts


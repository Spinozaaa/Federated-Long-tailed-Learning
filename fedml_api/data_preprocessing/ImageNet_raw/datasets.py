import copy
import os
import os.path
import numpy as np
import torch
import torch.utils.data as data
from PIL import Image
import logging
import collections
import pickle

def has_file_allowed_extension(filename, extensions):
    """Checks if a file is an allowed extension.

    Args:
        filename (string): path to a file

    Returns:
        bool: True if the filename ends with a known image extension
    """
    filename_lower = filename.lower()
    return any(filename_lower.endswith(ext) for ext in extensions)


# def find_classes(dir):
#     classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
#     classes.sort()
#     class_to_idx = {classes[i]: i for i in range(len(classes))}
#     return classes, class_to_idx

## Set lable according to sample number ##
def find_classes(dir):
    classes2filenum = {d: len(os.listdir(dir+"/"+d)) for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))}
    sorted_classes = sorted(classes2filenum.items(), key=lambda item:item[1], reverse=True)
    class_to_idx = {sorted_classes[i][0]: i for i in range(len(sorted_classes))}
    return classes2filenum, class_to_idx

def make_dataset(dir, class_to_idx, extensions):
    images = []

    data_local_num_dict = dict()
    net_dataidx_map = dict()
    sum_temp = 0
    dir = os.path.expanduser(dir)
    for target in sorted(os.listdir(dir)):
        d = os.path.join(dir, target)
        if not os.path.isdir(d):
            continue

        target_num = 0
        for root, _, fnames in sorted(os.walk(d)):
            for fname in sorted(fnames):
                if has_file_allowed_extension(fname, extensions):
                    path = os.path.join(root, fname)
                    item = (path, class_to_idx[target])
                    images.append(item)
                    target_num += 1

        net_dataidx_map[class_to_idx[target]] = (sum_temp, sum_temp + target_num)
        data_local_num_dict[class_to_idx[target]] = target_num
        sum_temp += target_num

    assert len(images) == sum_temp
    return images, data_local_num_dict, net_dataidx_map

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo)
    return dict

def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


def accimage_loader(path):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        pass


def default_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)


class ImageNet(data.Dataset):

    def __init__(self, data_dir, dataidxs=None, train=True, transform=None, target_transform=None, download=False, args=None, label_dict=None):
        """
            Generating this class too many times will be time-consuming.
            So it will be better calling this once and put it into ImageNet_truncated.
        """
        self.dataidxs = dataidxs
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.download = download
        self.loader = default_loader
        self.args = args
        self.traindata_cls_counts = None
        self.label_dict = label_dict

        if self.train:
            self.data_dir = os.path.join(data_dir, 'train')
        else:
            self.data_dir = os.path.join(data_dir, 'val')

        self.all_data, self.data_local_num_dict, self.net_dataidx_map = self.__getdatasets__()

        if dataidxs is None:
            self.local_data = self.all_data
        elif type(dataidxs) == int:
            (begin, end) = self.net_dataidx_map[dataidxs]
            self.local_data = self.all_data[begin: end]
        else:
            self.local_data = []
            for idxs in dataidxs:
                (begin, end) = self.net_dataidx_map[idxs]
                self.local_data += self.all_data[begin: end]

    def get_local_data(self):
        if "raw" in self.args.dataset:
            return self.local_data
        else:
            return None

    def get_data_and_label(self):
        return self.all_image, self.all_label

    def get_net_dataidx_map(self):
        return self.net_dataidx_map

    def get_data_local_num_dict(self):
        return self.data_local_num_dict

    def get_traindata_cls_counts(self):
        return self.traindata_cls_counts

    def __getdatasets__(self):
        # all_data = datasets.ImageFolder(data_dir, self.transform, self.target_transform)

        classes, class_to_idx = find_classes(self.data_dir)
        IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif']
        if "raw" in self.args.dataset:
            all_data, data_local_num_dict, net_dataidx_map = make_dataset(self.data_dir, class_to_idx, IMG_EXTENSIONS)
        else:
            self.all_image = None
            self.all_label = None
            if "32" in self.args.dataset:
                self.load_databatch(self.data_dir, 32)
            else:
                self.load_databatch(self.data_dir, 224)

            all_data = [(self.all_image[i], self.all_label[i]) for i in range(len(self.all_label))]

            data_local_num_dict = {}
            net_dataidx_map = {}

        if self.train:
            data_local_num_dict, net_dataidx_map = self.partition(all_data)

        if len(all_data) == 0:
            raise (RuntimeError("Found 0 files in subfolders of: " + self.data_dir + "\n"
                    "Supported extensions are: " + ",".join(IMG_EXTENSIONS)))

        return all_data, data_local_num_dict, net_dataidx_map

    def load_databatch(self, data_folder, img_size=224):
        data_file = os.path.join(data_folder, 'train_data_batch_')
        for root, ds, fs in os.walk(data_folder):
            files = fs
        for data_file in files:
            d = unpickle(os.path.join(data_folder, data_file))
            x = d['data']
            y = d['labels']

            # x = x / np.float32(255)
            # if self.train:
            #     mean_image = d['mean']
            #     mean_image = mean_image / np.float32(255)
            #     x -= mean_image

            y = [i for i in y]
            data_size = x.shape[0]

            img_size2 = img_size * img_size

            x = np.dstack((x[:, :img_size2], x[:, img_size2:2 * img_size2], x[:, 2 * img_size2:]))
            # x = x.reshape((x.shape[0], img_size, img_size, 3)).transpose(0, 3, 1, 2)
            x = x.reshape((x.shape[0], img_size, img_size, 3))

            # create mirrored images
            # X_train = x[0:data_size, :, :, :]
            # Y_train = y[0:data_size]
            if self.all_image is None:
                self.all_image = x
                self.all_label = y
            else:
                # ###for debug###
                # a = int(len(x)/10)
                # x = x[0:a]
                # y = y[0:a]
                # ###for debug###

                self.all_image = np.concatenate((self.all_image, x), 0)
                self.all_label.extend(y)
        self.all_label = np.array(self.all_label)

        ###Let the labels be sorted in descending order by the sample number
        if self.train:
            self.label_dict = {}
            self.change_label()
        else:
            if self.label_dict is not None:
                self.change_label()
        ###

        # X_train_flip = X_train[:, :, :, ::-1]
        # Y_train_flip = Y_train
        # X_train = np.concatenate((X_train, X_train_flip), axis=0)
        # Y_train = np.concatenate((Y_train, Y_train_flip), axis=0)

    def change_label(self):
        if self.train:
            all_label_old = copy.deepcopy(self.all_label)
            label_counter = collections.Counter(all_label_old).most_common()
            for idx, (i, _) in enumerate(label_counter):
                label_idx = np.where(all_label_old == i)
                self.all_label[label_idx] = idx
                self.label_dict[idx] = i
        else:
            all_label_old = copy.deepcopy(self.all_label)
            for idx, value in self.label_dict.items():
                lable_idx = np.where(all_label_old == value)
                self.all_label[lable_idx] = idx

    def record_net_data_stats(self, y_train, net_dataidx_map):
        net_cls_counts = {}

        for net_i, dataidx in net_dataidx_map.items():
            unq, unq_cnt = np.unique(y_train[dataidx], return_counts=True)
            tmp = {unq[i]: unq_cnt[i] for i in range(len(unq))}
            net_cls_counts[net_i] = tmp
        # logging.debug('Data statistics: %s' % str(net_cls_counts))
        return net_cls_counts

    def partition(self, all_data=None):

        if all_data is not None:
            target = np.array([label for path, label in all_data])
        else:
            target = np.array(self.all_label)

        n_train = target.shape[0]
        n_nets = self.args.client_num_in_total
        counter_num_per_class = collections.Counter(target)
        class_num = len(counter_num_per_class)
        num_per_class = {i: counter_num_per_class[i] for i in range(class_num)}
        print(num_per_class)
        if self.args.partition_method == "homo":
            total_num = n_train
            idxs = np.random.permutation(total_num)
            batch_idxs = np.array_split(idxs, n_nets)
            net_dataidx_map = {i: batch_idxs[i] for i in range(n_nets)}

        elif self.args.partition_method == "hetero":
            min_size = 0
            N = n_train
            net_dataidx_map = {}

            while min_size < 100:
                idx_batch = [[] for _ in range(n_nets)]
                # for each class in the dataset
                for class_num in range(class_num):
                    idx_k = np.where(target == class_num)[0]
                    np.random.shuffle(idx_k)
                    proportions = np.random.dirichlet(np.repeat(self.args.partition_alpha, n_nets))
                    ## Balance
                    proportions = np.array([p * (len(idx_j) < N / n_nets) for p, idx_j in zip(proportions, idx_batch)])
                    proportions = proportions / proportions.sum()
                    proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
                    idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]
                    min_size = min([len(idx_j) for idx_j in idx_batch])

            for j in range(n_nets):
                np.random.shuffle(idx_batch[j])
                net_dataidx_map[j] = idx_batch[j]
        else:
            logging.error("No partition method!")
            net_dataidx_map = None

        data_local_num_dict = [len(i) for i in net_dataidx_map.values()]
        self.traindata_cls_counts = self.record_net_data_stats(target, net_dataidx_map)

        return data_local_num_dict, net_dataidx_map

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        # img, target = self.data[index], self.target[index]

        img, target = self.local_data[index]
        if isinstance(img, str):
            img = self.loader(img)
        elif type(img) is np.ndarray:
            img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.local_data)


class ImageNet_truncated(data.Dataset):

    def __init__(self, imagenet_dataset: ImageNet, dataidxs, net_dataidx_map, train=True, transform=None,
                 target_transform=None, download=False):

        self.dataidxs = dataidxs
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.download = download
        self.net_dataidx_map = net_dataidx_map
        self.loader = default_loader
        self.all_data = imagenet_dataset.get_local_data()

        if self.all_data is not None:
            if dataidxs is None:
                self.local_data = self.all_data
            else:
                self.local_data = [self.all_data[i] for i in self.net_dataidx_map[dataidxs]]
        else:
            self.data, self.target = imagenet_dataset.get_data_and_label()
            if dataidxs is not None:
                self.data = self.data[self.net_dataidx_map[dataidxs]]
                self.target = self.target[self.net_dataidx_map[dataidxs]]
            self.local_data = self.data

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        # img, target = self.data[index], self.target[index]

        if self.all_data is not None:
            img, target = self.local_data[index]
        else:
            img = self.data[index]
            target = self.target[index]

        if isinstance(img, str):
            img = self.loader(img)
        elif type(img) is np.ndarray:
            img = Image.fromarray(img)

        if self.transform is not None:
            if isinstance(self.transform, list):
                sample1 = self.transform[0](img)
                sample2 = self.transform[1](img)
                sample3 = self.transform[2](img)
                if self.target_transform is not None:
                    target = self.target_transform(target)
                return [sample1, sample2, sample3], target
            else:
                img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.local_data)

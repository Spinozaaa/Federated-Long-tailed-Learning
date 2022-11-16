import logging

import numpy as np


class Client:

    def __init__(self, client_idx, local_training_data, local_test_data, local_sample_number, args, device,
                 model_trainer):
        self.client_idx = client_idx
        self.local_training_data = local_training_data
        self.local_test_data = local_test_data
        self.local_sample_number = local_sample_number
        # logging.info("self.local_sample_number = " + str(self.local_sample_number))

        self.args = args
        self.device = device
        self.model_trainer = model_trainer
        self.cls_num_list = None

        if "cifar100" in self.args.dataset:
            self.class_num = 100
        elif "cifar10" in self.args.dataset:
            self.class_num = 10
        elif "imagenet" in self.args.dataset:
            self.class_num = 1000

        if "cifar100_lt" in args.dataset:
            model_trainer.set_ltinfo(self.class_num, class_range=[35, 70])
        elif "cifar10_lt" in args.dataset:
            model_trainer.set_ltinfo(self.class_num, class_range=[4, 7])
        elif "imagenet" in self.args.dataset:
            model_trainer.set_ltinfo(self.class_num, class_range=[385, 864])
        else:
            model_trainer.set_ltinfo(self.class_num)

    def update_local_dataset(self, client_idx, local_training_data, local_test_data, local_sample_number):
        self.client_idx = client_idx
        self.local_training_data = local_training_data
        self.local_test_data = local_test_data
        self.local_sample_number = local_sample_number

    def get_sample_number(self):
        return self.local_sample_number

    def train(self, w_global, alpha=None, round=0):
        self.model_trainer.set_model_params(w_global)
        self.model_trainer.train(self.local_training_data, self.device, self.args, alpha, self.cls_num_list, round=round)

        weights = self.model_trainer.get_model_params()
        return weights

    def local_test(self, b_use_test_dataset):
        if b_use_test_dataset in self.args.method:
            test_data = self.local_test_data
        else:
            test_data = self.local_training_data

        metrics = self.model_trainer.test_for_all_labels(test_data, self.device)

        return metrics

    def set_cls_num_list(self, cls_num_list):
        if not isinstance(cls_num_list, list) and type(cls_num_list) is not np.ndarray:
            # cls_num_list = [cls_num_list[i] if i in cls_num_list.keys() else 0 for i in range(self.class_num)]
            cls_num_list = [cls_num_list[i] if i in cls_num_list.keys() else 0.1 for i in range(self.class_num)]
        self.cls_num_list = cls_num_list

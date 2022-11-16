import copy
import logging
from collections import Counter
import numpy as np
import torch
import wandb
import torchvision.transforms as trans
from fedml_api.clsimb_fedavg.client import Client

class FedAvgAPI(object):
    def __init__(self, dataset, device, args, model_trainer):
        self.device = device
        self.args = args

        [train_data_num, test_data_num, train_data_global, test_data_global,
         train_data_local_num_dict, train_data_local_dict, test_data_local_dict, class_num,
         traindata_cls_counts] = dataset

        self.all_data = dataset
        self.train_global = train_data_global
        self.test_global = test_data_global
        self.val_global = None
        self.train_data_num_in_total = train_data_num
        self.test_data_num_in_total = test_data_num
        self.traindata_cls_counts = traindata_cls_counts

        self.client_list = []
        self.train_data_local_num_dict = train_data_local_num_dict
        self.train_data_local_dict = train_data_local_dict
        self.test_data_local_dict = test_data_local_dict

        self.model_trainer = model_trainer
        self._setup_clients(train_data_local_num_dict, train_data_local_dict, test_data_local_dict, model_trainer)
        self.class_num = class_num

        if "esti_global" in self.args.method:
            self.avg_similarity = 0
            self.total_esti_cls_num = None
            self.count = 0
            self.esti_cls_num_list = []

        if "ldae" in self.args.method or "ride" in self.args.method:
            self.experts_acc_rounds_list = []

    def _setup_clients(self, train_data_local_num_dict, train_data_local_dict, test_data_local_dict, model_trainer):
        logging.info("############setup_clients (START)#############")
        for client_idx in range(self.args.client_num_per_round):
            c = Client(client_idx, train_data_local_dict[client_idx], test_data_local_dict[client_idx],
                       train_data_local_num_dict[client_idx], self.args, self.device, model_trainer)
            self.client_list.append(c)
        logging.info("############setup_clients (END)#############")

    def train(self):
        total_clt_num = self.args.client_num_in_total
        w_global = self.model_trainer.get_model_params()

        if "global" in self.args.method:
            total_cls_count = Counter(self.train_global.dataset.target)

        if "esti_global" in self.args.method:
            self.estimate_local_distribution(w_global)

        oringin_lr = self.args.lr

        round_range = range(self.args.comm_round)

        for round_idx in round_range:

            logging.info("################Communication round : {}".format(round_idx))

            ###lr schedular ###
            if self.args.lr_decay == 0:
                if round_idx >= int(self.args.comm_round * 1/10):
                    self.args.lr -= (oringin_lr / self.args.comm_round)
            elif self.args.lr_decay > 0:
                if round_idx == int(self.args.comm_round * 4/5):
                    self.args.lr = oringin_lr * self.args.lr_decay
                elif "imagenet224" in self.args.dataset and round_idx == int(self.args.comm_round * 9/10) \
                        and self.args.save_load != 3:
                    self.args.lr = self.args.lr * self.args.lr_decay
                    self.args.frequency_of_the_test = 50

            if "lade" in self.args.method:
                if "blsm" in self.args.method:
                    self.args.lade_weight = 0
                elif "imagenet224" in self.args.dataset:
                    self.args.lade_weight = 0.1
                elif "cifar100" in self.args.dataset:
                    self.args.lade_weight = 0.1
                else:
                    self.args.lade_weight = 0.01

            if "train_exp" in self.args.method and round_idx >= int(self.args.comm_round * 3/5):
                self.specifical_exp(round_idx=round_idx, expert_num=self.args.num_experts)
                if round_idx % self.args.frequency_of_the_test == 0 or round_idx == self.args.comm_round - 1:
                    self._global_test(round_idx)

                continue

            w_locals = []

            client_indexes = self._client_sampling(round_idx, total_clt_num,
                                                   self.args.client_num_per_round)
            logging.info("client_indexes = " + str(client_indexes))

            for idx, client in enumerate(self.client_list):
                # update dataset
                client_idx = client_indexes[idx]
                client.update_local_dataset(client_idx, self.train_data_local_dict[client_idx],
                                            self.test_data_local_dict[client_idx],
                                            self.train_data_local_num_dict[client_idx])

                if "real_global" in self.args.method:
                    client.set_cls_num_list(total_cls_count)
                elif "esti_global" in self.args.method:
                        client.set_cls_num_list(list(self.total_esti_cls_num))
                else:
                    client.set_cls_num_list(self.traindata_cls_counts[client_idx])

                if "ride" in self.args.method:
                    client_cls = self.traindata_cls_counts[client_idx]
                    max_cls_num = max(client_cls.values())
                    class_dist = {i: max_cls_num / client_cls[i] if i in client_cls.keys() else 0 for i in
                                  range(self.class_num)}

                    self.model_trainer.set_ltinfo(class_dist=torch.tensor(list(class_dist.values())))

                w = client.train(w_global, round=round_idx)

                w_locals.append((client.get_sample_number(), copy.deepcopy(w)))

            self.model_trainer.set_model_params(w_global)

            if round_idx == self.args.comm_round - 1:
                self._global_test(round_idx)
            elif round_idx % self.args.frequency_of_the_test == 0 and round_idx != 0:
                self._global_test(round_idx)

    def _aggregate(self, w_locals, client_ratio=None, global_model=None, client_cls_list=None, round_idx=0):
        training_num = 0
        ratio_training_num = 0

        if client_ratio is None or len(client_ratio) < len(w_locals):
            client_ratio = [1] * len(w_locals)
        else:
            logging.info("client_ratio", client_ratio)

        for idx in range(len(w_locals)):
            (sample_num, local_params) = w_locals[idx]

            if "esti_global" in self.args.method and self.count < self.args.client_num_in_total:
                self.server_estimate_global_distribution(local_params, global_model, sample_num, client_cls_list, idx)

            ratio_sample_num = sample_num * client_ratio[idx]
            ratio_training_num += ratio_sample_num

            training_num += sample_num

        (sample_num, averaged_params) = copy.deepcopy(w_locals[0])
        for k in averaged_params.keys():
            for i in range(0, len(w_locals)):
                local_sample_number, local_model_params = w_locals[i]
                w = local_sample_number / training_num
                if i == 0:
                    averaged_params[k] = local_model_params[k] * w
                else:
                    averaged_params[k] += local_model_params[k] * w

        return averaged_params

    def server_estimate_global_distribution(self, averaged_params, global_model, sample_num, client_cls_list, idx):
        classifier_key = list(averaged_params.keys())[-1]
        if "fc" in classifier_key:
            class_dist = self._get_weight_value(averaged_params["fc.weight"] - global_model["fc.weight"],
                                                "fc.weight", method="sum")
        elif "backbone.linear" in classifier_key:
            if self.args.num_experts > 1:
                local_linear = averaged_params["backbone.linears.0.weight"]
                global_linear = global_model["backbone.linears.0.weight"]

                for i in range(1, self.args.num_experts):
                    local_linear += averaged_params["backbone.linears." + str(i) + ".weight"]
                    global_linear += global_model["backbone.linears." + str(i) + ".weight"]

                local_linear /= self.args.num_experts
                global_linear /= self.args.num_experts

                class_dist = self._get_weight_value(local_linear - global_linear, "fc.weight", method="sum")
            else:
                class_dist = self._get_weight_value(averaged_params["backbone.linear.weight"] - global_model["backbone.linear.weight"],
                                                    "backbone.linear.weight", method="sum")

        real_dist = np.array(client_cls_list[idx])

        class_dist = class_dist.numpy()
        class_dist = np.where(class_dist <= 0, 1e-5, class_dist)
        class_dist = class_dist / sum(class_dist) if sum(class_dist) != 0 else [0] * self.args.client_num_per_round
        esti_cls_num = np.around(class_dist * sample_num).astype(np.int)

        if len(self.esti_cls_num_list) < self.args.client_num_in_total:
            self.esti_cls_num_list.append(list(esti_cls_num.astype(int)))

        similarity = cosine_similarity(esti_cls_num, real_dist)

        if self.total_esti_cls_num is None:
            self.total_esti_cls_num = copy.deepcopy(esti_cls_num)
            self.avg_similarity = similarity
            self.count = 1

        elif self.count < self.args.client_num_in_total:
            self.total_esti_cls_num += esti_cls_num
            # logging.info(str(esti_cls_num) + str(self.count) + "--Estimate\n" + str(real_dist) + "--Real; similarity:" + str(similarity))
            self.count += 1
            self.avg_similarity += similarity
            if self.count == self.args.client_num_in_total:
                self.total_esti_cls_num = [i if i >= 5 else 5 for i in self.total_esti_cls_num.astype(np.int)]

                # logging.info("Total_esti_cls_num: " + str(self.total_esti_cls_num))
                if "imagenet" in self.args.dataset:
                    real_total_cls_num = []
                    for label in range(self.class_num):
                        real_total_cls_num.append(0)
                        for clt in self.traindata_cls_counts.values():
                            if label in clt.keys():
                                real_total_cls_num[label] += clt[label]
                    real_total_cls_num = np.array(real_total_cls_num)
                else:
                    real_total_cls_num = np.array(list(Counter(self.train_global.dataset.target).values()))

                self.avg_similarity /= self.count
                logging.info("Avg similarity:" + str(self.avg_similarity))

                total_similarity = cosine_similarity(self.total_esti_cls_num, real_total_cls_num)
                logging.info("Total num similarity:" + str(total_similarity))

    def estimate_local_distribution(self, init_model):

        real_ep = self.args.epochs
        real_lr = self.args.lr
        real_bs = self.args.batch_size
        real_method = self.args.method

        self.args.epochs = 1

        self.args.method = "esti_global"

        client_cls_list = []
        w_locals = []

        client = self.client_list[0]

        esti_round = 1
        w_global = init_model

        client_indexes = self._client_sampling(0, self.args.client_num_in_total, self.args.client_num_in_total)

        for round in range(esti_round):
            for client_idx in client_indexes:
                client.update_local_dataset(client_idx, self.train_data_local_dict[client_idx],
                                            self.test_data_local_dict[client_idx],
                                            self.train_data_local_num_dict[client_idx])

                client.set_cls_num_list(self.traindata_cls_counts[client_idx])

                client_cls_dic = self.traindata_cls_counts[client_idx]
                # for calculate similarity
                client_cls_list.append(
                    [client_cls_dic[idx] if idx in client_cls_dic.keys() else 0 for idx in range(self.class_num)])

                w = client.train(w_global)

                w_locals.append((client.get_sample_number(), copy.deepcopy(w)))
            w_global = self._aggregate(w_locals, global_model=w_global, client_cls_list=client_cls_list)
            w_old_global = w_global

        for client_idx in client_indexes:
            client.update_local_dataset(client_idx, self.train_data_local_dict[client_idx],
                                        self.test_data_local_dict[client_idx],
                                        self.train_data_local_num_dict[client_idx])

            client.set_cls_num_list(self.traindata_cls_counts[client_idx])

            client_cls_dic = self.traindata_cls_counts[client_idx]
            # for calculate similarity
            client_cls_list.append(
                [client_cls_dic[idx] if idx in client_cls_dic.keys() else 0 for idx in range(self.class_num)])

            w = client.train(w_old_global)

            w_locals.append((client.get_sample_number(), copy.deepcopy(w)))
        self._aggregate(w_locals, global_model=w_old_global, client_cls_list=client_cls_list)

        self.args.epochs = real_ep
        self.args.lr = real_lr

        self.args.batch_size = real_bs
        self.args.method = real_method
        # self.args.client_num_per_round = real_client_num_per_round

    def specifical_exp(self, round_idx, expert_num):

        client_ratio = [self.args.beta] * expert_num

        clt_ratio_list = []
        if "real_global" in self.args.method:
            total_cls_count = np.array(list(Counter(self.train_global.dataset.target).values()))

            for (i, clt_cls_num) in self.traindata_cls_counts.items():
                clt_cls_num = np.array([clt_cls_num[i] if i in clt_cls_num.keys() else 0 for i in range(self.class_num)])
                clt_cls_num = np.array(clt_cls_num/sum(clt_cls_num))
                clt_ratio_list.append((sum(total_cls_count * clt_cls_num), i))

        elif "esti_global" in self.args.method:
            for (i, clt_cls_num) in enumerate(self.esti_cls_num_list):
                clt_cls_num = np.array(clt_cls_num/sum(clt_cls_num))
                clt_ratio_list.append((sum(self.total_esti_cls_num * clt_cls_num), i))

        clt_ratio_list.sort(reverse=True, key=lambda x: x[0])

        clts_sorted_by_cls = [i[1] for i in clt_ratio_list]
        # logging.info("###########clts_sorted_by_cls########\n" + str(clts_sorted_by_cls))
        expert_clt_idxs = []
        for i in range(expert_num):
            expert_clt_idxs.append(clts_sorted_by_cls[int(self.args.client_num_in_total * i/expert_num):int(self.args.client_num_in_total* (i+1)/expert_num)])

        for i in range(expert_num):
            if self.args.client_num_per_round * self.args.beta >= len(expert_clt_idxs[i]):
                clt_indexes = expert_clt_idxs[i]
                rand_expert_num = self.args.client_num_per_round * self.args.beta - len(clt_indexes)
                random_clt = np.random.choice(range(self.args.client_num_in_total), rand_expert_num, replace=False)
                # clt_indexes = expert_clt_idxs[2]
            else:
                np.random.seed(round_idx + 1)
                clt_indexes = np.random.choice(expert_clt_idxs[i], int(self.args.client_num_per_round * client_ratio[i]), replace=False)
                # clt_indexes = np.random.choice(expert_clt_idxs[2], int(self.args.client_num_per_round * client_ratio[i]), replace=False) ##rare client train
                random_clt = np.random.choice(range(self.args.client_num_in_total), int(self.args.client_num_per_round * (1-client_ratio[i])), replace=False)

            clt_indexes = np.concatenate((clt_indexes, random_clt))
            logging.info("expert :{0} client_indexes = :{1}".format(i, str(clt_indexes)))

            self.experts_train(round_idx=round_idx, train_experts=str(i), clt_indexes=clt_indexes)

    def experts_train(self, round_idx, train_experts=None, clt_indexes=None):
        w_global = self.model_trainer.get_model_params()
        w_locals = []

        for idx in range(len(clt_indexes)):
            # update dataset
            client = self.client_list[idx]
            client_idx = clt_indexes[idx]
            client.update_local_dataset(client_idx, self.train_data_local_dict[client_idx],
                                        self.test_data_local_dict[client_idx],
                                        self.train_data_local_num_dict[client_idx])

            # client.set_cls_num_list(self.traindata_cls_counts[client_idx]) #local
            if "real_global" in self.args.method:
                client.set_cls_num_list(Counter(self.train_global.dataset.target))
            elif "esti_global" in self.args.method:
                client.set_cls_num_list(list(self.total_esti_cls_num))
            else:
                client.set_cls_num_list(self.traindata_cls_counts[client_idx])

            self.freeze_layer(train_experts)

            w = client.train(w_global, round=round_idx)

            self.freeze_layer()

            w_locals.append((client.get_sample_number(), copy.deepcopy(w)))

        w_global = self._aggregate(w_locals, global_model=w_global)
        self.model_trainer.set_model_params(w_global)

    def freeze_layer(self, train_experts=None):
        self.model_trainer.training_exp = train_experts

        model = self.model_trainer.model
        if train_experts is not None:
            for name, para in model.named_parameters():
                if "s." in name and ("s." + train_experts) not in name:
                    para.requires_grad = False
                else:
                    para.requires_grad = True
        else:
            for name, para in model.named_parameters():
                para.requires_grad = True


    def _get_weight_value(self, param, round_idx=0, method=None):
        if param.shape[0] == self.class_num:
            dim = 1
        else:
            dim = 0

        if "abs_sum" in method:
            if len(param.shape) == 1:
                para_norm = abs(param)
            else:
                para_norm = abs(param)
                para_norm = para_norm.sum(0)
        elif "norm" in method:
            norm = torch.norm(param, 2, 1)
            if len(norm.shape) == 1:
                para_norm = norm
            else:
                para_norm = norm.sum(1)
        elif "min" in method:
            param = torch.min(param, dim=-1)
            para_norm = abs(param.values)
        elif "sum" in method:
            para_norm = torch.sum(param, dim=dim)
        else:
            logging.warning("No such Weight Value")
            return

        return para_norm

    def _client_sampling(self, round_idx, client_num_in_total, client_num_per_round):
        if client_num_in_total == client_num_per_round:
            client_indexes = [client_index for client_index in range(client_num_in_total)]
        else:
            num_clients = min(client_num_per_round, client_num_in_total)
            np.random.seed(round_idx + 1)  # make sure for each comparison, we are selecting the same clients each round
            client_indexes = np.random.choice(range(client_num_in_total), num_clients, replace=False)
        return client_indexes

    def _local_test_on_all_clients(self, round_idx):

        logging.info("################local_test_on_all_clients : {}".format(round_idx))

        train_metrics = {
            'num_samples': [],
            'num_correct': [],
            'losses': []
        }

        test_metrics = {
            'num_samples': [],
            'num_correct': [],
            'losses': []
        }

        client = self.client_list[0]

        for client_idx in range(self.args.client_num_in_total):

            """
            Note: for datasets like "fed_CIFAR100" and "fed_shakespheare",
            the training client number is larger than the testing client number
            """
            if self.test_data_local_dict[client_idx] is None:
                continue
            client.update_local_dataset(0, self.train_data_local_dict[client_idx],
                                        self.test_data_local_dict[client_idx],
                                        self.train_data_local_num_dict[client_idx])

            # train data
            train_local_metrics = client.local_test(False)
            train_metrics['num_samples'].append(copy.deepcopy(train_local_metrics['test_total']))
            train_metrics['num_correct'].append(copy.deepcopy(train_local_metrics['test_correct']))
            train_metrics['losses'].append(copy.deepcopy(train_local_metrics['test_loss']))

            # test data
            test_local_metrics = client.local_test(True)
            test_metrics['num_samples'].append(copy.deepcopy(test_local_metrics['test_total']))
            test_metrics['num_correct'].append(copy.deepcopy(test_local_metrics['test_correct']))
            test_metrics['losses'].append(copy.deepcopy(test_local_metrics['test_loss']))

            if 0 in train_local_metrics.keys():
                for i in range(self.class_num):
                    if i not in train_metrics.keys():
                        train_metrics[i] = []
                        test_metrics[i] = []
                    train_metrics[i].append(copy.deepcopy(train_local_metrics[i]))
                    test_metrics[i].append(copy.deepcopy(test_local_metrics[i]))

            """
            Note: CI environment is CPU-based computing. 
            The training speed for RNN training is to slow in this setting, so we only test a client to make sure there is no programming error.
            """
            if self.args.ci == 1:
                break

        # test on training dataset
        train_acc = sum(train_metrics['num_correct']) / sum(train_metrics['num_samples'])
        train_loss = sum(train_metrics['losses']) / sum(train_metrics['num_samples'])

        # test on test dataset
        test_acc = sum(test_metrics['num_correct']) / sum(test_metrics['num_samples'])
        test_loss = sum(test_metrics['losses']) / sum(test_metrics['num_samples'])

        stats = {'training_acc': train_acc, 'training_loss': train_loss}
        wandb.log({"Train/Acc": train_acc, "round": round_idx})
        wandb.log({"Train/Loss": train_loss, "round": round_idx})
        logging.info(stats)

        stats = {'test_acc': test_acc, 'test_loss': test_loss}
        wandb.log({"Test/Acc": test_acc, "round": round_idx})
        wandb.log({"Test/Loss": test_loss, "round": round_idx})
        logging.info(stats)

        if 0 in train_metrics.keys():
            lable_acc = {}
            for i in range(self.class_num):
                lable_acc[i] = sum(train_metrics[i]) / self.args.client_num_in_total
            # logging.info("train_acc_per_label:" + str(lable_acc))

            for i in range(self.class_num):
                lable_acc[i] = sum(test_metrics[i]) / self.args.client_num_in_total
            logging.info("test_acc_per_label:" + str(lable_acc))

    def _global_test(self, round_idx):

        logging.info("################global_test###################")


        train_metrics = {
            'test_correct': 1,
            'test_total': 1,
            'test_loss': 1
        }

        test_metrics = {
            'num_samples': 1,
            'num_correct': 1,
            'losses': 1
        }

        # train data
        if "imagenet" in self.args.dataset:
            train_local_metrics = train_metrics
        else:
            train_local_metrics = self.model_trainer.test(self.train_global, self.device, self.args)

        train_acc = copy.deepcopy(train_local_metrics['test_correct'])
        train_num = copy.deepcopy(train_local_metrics['test_total'])
        train_loss = copy.deepcopy(train_local_metrics['test_loss'])

        train_acc = train_acc / train_num
        train_loss = train_loss / train_num

        test_local_metrics = self.model_trainer.test(self.test_global, self.device, self.args)

        test_acc = copy.deepcopy(test_local_metrics['test_correct'])
        test_num = copy.deepcopy(test_local_metrics['test_total'])
        test_loss = copy.deepcopy(test_local_metrics['test_loss'])

        test_acc = test_acc / test_num
        test_loss = test_loss / test_num

        stats = {'training_acc': train_acc, 'training_loss': train_loss}
        wandb.log({"Train/Acc": train_acc, "round": round_idx})
        wandb.log({"Train/Loss": train_loss, "round": round_idx})
        logging.info(stats)

        stats = {'test_acc': test_acc, 'test_loss': test_loss}
        wandb.log({"Test/Acc": test_acc, "round": round_idx})
        wandb.log({"Test/Loss": test_loss, "round": round_idx})
        logging.info(stats)

        train_lable_matric = self.model_trainer.test_for_all_labels(self.train_global, self.device)
        test_label_matric = self.model_trainer.test_for_all_labels(self.test_global, self.device)

        train_label_acc = {}
        test_label_acc = {}
        for i in range(self.class_num):
            train_label_acc[i] = train_lable_matric[i]
            test_label_acc[i] = test_label_matric[i]
        #     wandb.log({"F1score class " + str(i): f1[i], "round": round_idx})
        wandb.log({"Many-shot acc ": test_label_matric["Many acc"], "round": round_idx})
        wandb.log({"Medium-shot acc ": test_label_matric["Medium acc"], "round": round_idx})
        wandb.log({"Few-shot acc ": test_label_matric["Few acc"], "round": round_idx})

        logging.info("train_acc_per_label:" + str(train_label_acc))
        logging.info("Many-shot acc:" + str(test_label_matric["Many acc"]))
        logging.info("Medium-shot acc:" + str(test_label_matric["Medium acc"]))
        logging.info("Few-shot acc:" + str(test_label_matric["Few acc"]))

    #### delete one of samples if there is one redundacy sample, due to the drop_last=True
    def fix_redundacy_sample(self, batch_size, train_data_local_dict):
        for idx in train_data_local_dict:
            data = train_data_local_dict[idx].dataset.data
            target = train_data_local_dict[idx].dataset.target
            redundacy = data.shape[0] % batch_size
            if redundacy == 1:
                logging.info("delete one sample to avoid bug in batch_norm for client " + str(idx))
                train_data_local_dict[idx] = torch.utils.data.DataLoader(dataset=train_data_local_dict[idx].dataset,
                                                                             batch_size=batch_size, shuffle=True,
                                                                             drop_last=True)


def cosine_similarity(x, y):
    similarity = float(np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y)))
    return similarity




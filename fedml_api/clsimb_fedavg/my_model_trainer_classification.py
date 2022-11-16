import matplotlib.pyplot as plt
import torch
from torchvision.utils import make_grid
import wandb

from .model_trainer import ModelTrainer
from .lossfns import *
from .multi_experts.trainer import Trainer as MultiExTrainer
from fedml_api.model.multiexp_model.ldam_drw_resnets.expert_resnet_cifar import NormedLinear


class MyModelTrainer(ModelTrainer):
    def __init__(self, model, args=None):
        super().__init__(model, args)
        self.extractor_model = None
        self.class_num = None
        self.class_dist = None
        self.islt = False
        self.mixtrain_flag = True
        self.total_cls_num = None
        self.class_range = None
        self.training_exp = None


    # for long-tail dataset
    def set_ltinfo(self, class_num=None, mixtrain_flag=None, class_dist=None, class_range=None):
        if class_num is not None:
            self.class_num = class_num

        if self.args is None or "lt" in self.args.dataset:
            self.islt = True
            if mixtrain_flag is not None:
                self.mixtrain_flag = mixtrain_flag
            if class_dist is not None:
                self.class_dist = class_dist
            if class_range is not None:
                self.class_range = class_range

    def wandb_watch_model(self):
        wandb.watch(self.model)

    def get_model_params(self):
        return self.model.cpu().state_dict()

    def set_model_params(self, model_parameters):
        self.model.load_state_dict(model_parameters)

    def get_model(self):
        return self.model

    def set_acc_in_weight(self, cls_acc_metrics, label_smaple_num, device):

        for label in range(self.class_num):
            if label_smaple_num[label] != 0:
                cls_acc_metrics[label] = cls_acc_metrics[label] / label_smaple_num[label]

        # logging.info("cls_acc_metrics in client" + str(cls_acc_metrics))
        model_para = self.get_model_params()
        fc_weight = model_para['fc.weight']

        for i in range(self.class_num):
            fc_weight[i][0] = cls_acc_metrics[i] * 0.1

        self.set_model_params(model_para)
        self.model.to(device)

    def train(self, train_data, device, args, alpha=None, cls_num_list=None, round=0):
        model = self.model
        model.to(device)
        model.train()

        criterion, optimizer = self.train_init(device, cls_num_list, alpha)

        epoch_loss = []

        if "ride" in args.method or "ldae" in args.method:
            multiext_trainer = MultiExTrainer(model=model, optimizer=optimizer, data_loader=train_data, args=args,
                                              cls_num_list=cls_num_list, training_exp=self.training_exp)
            multiext_trainer.train()

            return

        train_data.dataset.target = train_data.dataset.target.astype(np.int64)

        cls_acc_metrics = dict.fromkeys(range(self.class_num), 0)
        label_smaple_num = dict.fromkeys(range(self.class_num), 0)

        self.train_count = 0

        for epoch in range(args.epochs):

            batch_loss = []
            for batch_idx, (x, labels) in enumerate(train_data):
                model.zero_grad()

                if isinstance(x, list):
                    x[0], x[1], labels = x[0].to(device), x[1].to(device), labels.to(device)
                    features, labels, log_probs = model(im_q=x[0], im_k=x[1], labels=labels)
                    loss = criterion(features, labels, log_probs)
                else:
                    x, labels = x.to(device), labels.to(device)
                    log_probs = model(x)

                    if "lade" in args.method:
                        perform_loss = criterion["perform"](log_probs, labels)
                        routeweight_loss = criterion["routeweight"](log_probs, labels)
                        loss = perform_loss + args.lade_weight * routeweight_loss
                    elif "ldam" in args.method:
                        loss = criterion(log_probs, labels, round, device)
                    else:
                        loss = criterion(log_probs, labels)

                loss.backward()
                optimizer.step()

                batch_loss.append(loss.item())

                self.train_count += 1

                if self.islt and epoch == args.epochs - 1:
                    _, predicted = torch.max(log_probs, -1)
                    correct = predicted.eq(labels)

                    for (idx, label) in enumerate(labels):
                        if correct[idx]:
                            cls_acc_metrics[int(label.item())] += 1
                        label_smaple_num[int(label.item())] += 1

            epoch_loss.append(sum(batch_loss) / len(batch_loss))

    def train_init(self, device, cls_num_list=None, alpha=None):
        if self.args.client_optimizer == "sgd":
            optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, self.model.parameters()), lr=self.args.lr, momentum=0.9, weight_decay=self.args.wd)
        else:
            optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=self.args.lr,
                                         weight_decay=self.args.wd, amsgrad=True)

        if "focal" in self.args.method:
            # alpha = torch.Tensor([i/sum(cls_num_list) if i != 0 else 1/sum(cls_num_list) for i in cls_num_list])
            criterion = FocalLoss(gamma=0.5)
        elif "cbloss" in self.args.method:
            if "cifar10_lt" in self.args.dataset:
                beta = 0.999999
                gama = 1.0
            elif "cifar100_lt" in self.args.dataset:
                beta = 0.99
                gama = 0.8
            criterion = CB_loss(cls_num_list, self.class_num, device, beta=beta, gamma=gama)
        elif "lade" in self.args.method:
            criterion_perform = PriorCELoss(num_classes=self.class_num, prior_txt=cls_num_list).to(device)
            criterion_routeweight = LADELoss(num_classes=self.class_num, prior_txt=cls_num_list, remine_lambda=0.01).to(device)
            criterion = {"perform": criterion_perform, "routeweight": criterion_routeweight}
        elif "ldam" in self.args.method:
            criterion = LDAMLoss(cls_num_list=cls_num_list, reweight_ep=self.args.comm_round * 2/3)
        else:
            criterion = nn.CrossEntropyLoss().to(device)

        return criterion, optimizer

    def test(self, test_data, device, args):
        model = self.model

        model.to(device)
        model.eval()

        metrics = {
            'test_correct': 0,
            'test_loss': 0,
            'test_total': 0,
        }
        test_data.dataset.target = test_data.dataset.target.astype(np.int64)
        criterion = nn.CrossEntropyLoss().to(device)

        with torch.no_grad():
            for batch_idx, (x, target) in enumerate(test_data):
                x = x.to(device)

                target = target.to(device)
                pred = model(x)

                if "lade" in self.args.method:
                    pred += torch.log(torch.ones(self.class_num)/self.class_num).to(device)
                loss = criterion(pred, target)

                _, predicted = torch.max(pred, -1)
                correct = predicted.eq(target).sum()

                metrics['test_correct'] += correct.item()
                metrics['test_loss'] += loss.item() * target.size(0)
                metrics['test_total'] += target.size(0)

        return metrics

    def test_for_all_labels(self, test_data, device):
        model = self.model

        model.to(device)
        model.eval()

        metrics = {
            'test_correct': 0,
            'test_loss': 0,
            'test_total': 0,
            'all_preds': 0,
            'Many acc': 0,
            'Medium acc': 0,
            'Few acc': 0,
        }

        label_smaple_num = {}
        for i in range(self.class_num):
            metrics[i] = 0
            label_smaple_num[i] = 0

        test_data.dataset.target = test_data.dataset.target.astype(np.int64)
        criterion = nn.CrossEntropyLoss().to(device)

        all_preds = torch.tensor([])
        with torch.no_grad():
            for batch_idx, (x, target) in enumerate(test_data):
                x = x.to(device)
                target = target.to(device)
                pred = model(x)
                if "lade" in self.args.method:
                    pred += torch.log(torch.ones(self.class_num)/self.class_num).to(device)
                loss = criterion(pred, target)
                _, predicted = torch.max(pred, -1)
                correct = predicted.eq(target).sum()

                metrics['test_correct'] += correct.item()
                metrics['test_loss'] += loss.item() * target.size(0)
                metrics['test_total'] += target.size(0)

                all_preds = torch.cat((all_preds, predicted.cpu()), dim=0)

                for (idx, label) in enumerate(target):
                    if predicted[idx].eq(target[idx]):
                        metrics[label.item()] += 1
                    label_smaple_num[label.item()] += 1

            for label in range(self.class_num):
                if label_smaple_num[label] != 0:
                    metrics[label] = metrics[label] / label_smaple_num[label]

            if self.class_range is not None:
                for i in range(self.class_num):
                    if i < self.class_range[0]:
                        metrics['Many acc'] += metrics[i]
                    elif i < self.class_range[1]:
                        metrics['Medium acc'] += metrics[i]
                    else:
                        metrics['Few acc'] += metrics[i]

                metrics['Many acc'] /= self.class_range[0]
                metrics['Medium acc'] /= self.class_range[1] - self.class_range[0]
                if metrics['Medium acc'] < 0:
                    metrics['Medium acc'] = 0
                metrics['Few acc'] /= self.class_num - self.class_range[1]
                if metrics['Few acc'] < 0:
                    metrics['Few acc'] = 0

        metrics['all_preds'] = all_preds

        return metrics


def imshow(torch_batch_images, title = None):
    npimages = make_grid(torch_batch_images.detach().cpu())
    fig = plt.figure(figsize=(15, 15))
    plt.imshow(np.transpose(npimages,(1, 2, 0)))
    plt.title(torch_batch_images.__str__ if title is None else title)
    plt.show()





# From https://github.com/kaidic/LDAM-DRW/blob/master/models/resnet_cifar.py
'''
Properly implemented ResNet for CIFAR10 as described in paper [1].
The implementation and structure of this file is hugely influenced by [2]
which is implemented for ImageNet and doesn't have option A for identity.
Moreover, most of the implementations on the web is copy-paste from
torchvision's resnet and has wrong number of params.
Proper ResNet-s for CIFAR10 (for fair comparision and etc.) has following
number of layers and parameters:
name      | layers | params
ResNet20  |    20  | 0.27M
ResNet32  |    32  | 0.46M
ResNet44  |    44  | 0.66M
ResNet56  |    56  | 0.85M
ResNet110 |   110  |  1.7M
ResNet1202|  1202  | 19.4m
which this implementation indeed has.
Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
[2] https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
If you use this implementation in you work, please don't forget to mention the
author, Yerlan Idelbayev.
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.nn import Parameter

from fedml_api.model.basic.group_normalization import GroupNorm2d

import random

__all__ = ['ResNet_s', 'resnet20', 'resnet32', 'resnet44', 'resnet56', 'resnet110', 'resnet1202']

def _weights_init(m):
    classname = m.__class__.__name__
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)

class NormedLinear(nn.Module):

    def __init__(self, in_features, out_features):
        super(NormedLinear, self).__init__()
        self.weight = Parameter(torch.Tensor(in_features, out_features))
        self.weight.data.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul_(1e5)

    def forward(self, x):
        out = F.normalize(x, dim=1).mm(F.normalize(self.weight, dim=0))
        return out

class LambdaLayer(nn.Module):

    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, option='A'):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            if option == 'A':
                """
                For CIFAR10 ResNet paper uses option A.
                """
                self.planes = planes
                self.in_planes = in_planes
                # self.shortcut = LambdaLayer(lambda x: F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes // 4, planes // 4), "constant", 0))
                self.shortcut = LambdaLayer(lambda x:
                                            F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, (planes - in_planes) // 2, (planes - in_planes) // 2), "constant", 0))
                
            elif option == 'B':
                self.shortcut = nn.Sequential(
                     nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                     nn.BatchNorm2d(self.expansion * planes)
                )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet_s(nn.Module):

    def __init__(self, block, num_blocks, num_experts, num_classes=10, reduce_dimension=False, layer2_output_dim=None,
                 layer3_output_dim=None, use_norm=False, returns_feat=False, use_experts=None, s=30):
        super(ResNet_s, self).__init__()
        
        self.in_planes = 16
        self.num_experts = num_experts

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.in_planes = self.next_in_planes

        if layer2_output_dim is None:
            if reduce_dimension:
                layer2_output_dim = 24
            else:
                layer2_output_dim = 32

        if layer3_output_dim is None:
            if reduce_dimension:
                layer3_output_dim = 48
            else:
                layer3_output_dim = 64

        self.layer2s = nn.ModuleList([self._make_layer(block, layer2_output_dim, num_blocks[1], stride=2) for _ in range(num_experts)])
        self.in_planes = self.next_in_planes
        self.layer3s = nn.ModuleList([self._make_layer(block, layer3_output_dim, num_blocks[2], stride=2) for _ in range(num_experts)])
        self.in_planes = self.next_in_planes
        
        if use_norm:
            self.linears = nn.ModuleList([NormedLinear(layer3_output_dim, num_classes) for _ in range(num_experts)])
        else:
            self.linears = nn.ModuleList([nn.Linear(layer3_output_dim, num_classes) for _ in range(num_experts)])
            s = 1

        if use_experts is None:
            self.use_experts = list(range(num_experts))
        elif use_experts == "rand":
            self.use_experts = None
        else:
            self.use_experts = [int(item) for item in use_experts.split(",")]

        self.s = s

        self.apply(_weights_init)
        # self.moe_pool = nn.AdaptiveAvgPool2d((8, 8))
        # self.moe_nn = nn.Linear(16*8*8, self.num_experts, bias=False)
        # nn.init.constant_(self.moe_nn.weight, 1)
        self.softmax = nn.Softmax(dim=1)

        self.moe = False

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        self.next_in_planes = self.in_planes
        for stride in strides:
            layers.append(block(self.next_in_planes, planes, stride))
            self.next_in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def _hook_before_iter(self):
        assert self.training, "_hook_before_iter should be called at training time only, after train() is called"
        count = 0
        for module in self.modules():
            if isinstance(module, nn.BatchNorm2d):
                if module.weight.requires_grad == False:
                    module.eval()
                    count += 1

        if count > 0:
            print("Warning: detected at least one frozen BN, set them to eval state. Count:", count)

    def _separate_part(self, x, ind):
        out = x
        out = (self.layer2s[ind])(out)
        out = (self.layer3s[ind])(out)
        # self.feat_before_GAP.append(out)
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        # self.feat.append(out)
        out = (self.linears[ind])(out)
        out = out * self.s
        return out

    def forward(self, x, output_all=False):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        
        outs = []
        self.feat = []
        self.logits = outs
        self.feat_before_GAP = []
        
        if self.use_experts is None:
            use_experts = random.sample(range(self.num_experts), self.num_experts - 1)
        else:
            use_experts = self.use_experts
        
        for ind in use_experts:
            o = self._separate_part(out, ind)
            # out_min = torch.min(o)
            # out_max = torch.max(o)
            # o = (o - out_min)/(out_max - out_min)
            # o = o * 2 - 1
            outs.append(o)

        # self.feat = torch.stack(self.feat, dim=1)
        # self.feat_before_GAP = torch.stack(self.feat_before_GAP, dim=1)

        # if self.moe:
        #     moe_weight = self.moe_pool(out)
        #     moe_weight = moe_weight.view(moe_weight.size(0), -1)
        #     moe_weight = self.moe_nn(moe_weight)
        #     moe_weight = self.softmax(moe_weight).unsqueeze(-1)
        #
        #     self.moe_weight = moe_weight
        #     final_out = torch.stack(outs, dim=-1) @ moe_weight
        #     final_out = final_out.squeeze(-1)
        # else:
        final_out = torch.stack(outs, dim=1).mean(dim=1)

        if output_all:
            return final_out, outs
        else:
            return final_out

def resnet20():
    return ResNet_s(BasicBlock, [3, 3, 3])


def resnet32(num_classes=10, use_norm=False):
    return ResNet_s(BasicBlock, [5, 5, 5], num_classes=num_classes, use_norm=use_norm)


def resnet44():
    return ResNet_s(BasicBlock, [7, 7, 7])


def resnet56():
    return ResNet_s(BasicBlock, [9, 9, 9])


def resnet110():
    return ResNet_s(BasicBlock, [18, 18, 18])


def resnet1202():
    return ResNet_s(BasicBlock, [200, 200, 200])


def test(net):
    import numpy as np
    total_params = 0

    for x in filter(lambda p: p.requires_grad, net.parameters()):
        total_params += np.prod(x.data.numpy().shape)
    print("Total number of params", total_params)
    print("Total layers", len(list(filter(lambda p: p.requires_grad and len(p.data.size())>1, net.parameters()))))

def norm2d(planes, num_channels_per_group=32):

    if num_channels_per_group > 0:
        return GroupNorm2d(planes, num_channels_per_group, affine=True,
                           track_running_stats=False)
    else:
        return nn.BatchNorm2d(planes)


class BasicBlock_GN(BasicBlock):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, downsample=None, group_norm=0, option='A'):
        super(BasicBlock_GN, self).__init__(in_planes, planes, stride=stride, option=option)
        self.bn1 = norm2d(planes, group_norm)
        self.relu = nn.ReLU(inplace=True)
        self.bn2 = norm2d(planes, group_norm)
        self.downsample = downsample

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            if option == 'A':
                """
                For CIFAR10 ResNet paper uses option A.
                """
                self.planes = planes
                self.in_planes = in_planes
                # self.shortcut = LambdaLayer(lambda x: F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes // 4, planes // 4), "constant", 0))
                self.shortcut = LambdaLayer(lambda x: F.pad(x[:, :, ::2, ::2],
                                                  (0, 0, 0, 0, (planes - in_planes) // 2, (planes - in_planes) // 2),
                                                  "constant", 0))

            elif option == 'B':
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                    norm2d(self.expansion * planes)
                )

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class ResNet_GN(ResNet_s):

    def __init__(self, block, num_blocks, num_experts, num_classes=10, reduce_dimension=False, layer2_output_dim=None, layer3_output_dim=None, use_norm=False, use_experts=None, returns_feat=False, s=30, group_norm=1):

        super(ResNet_GN, self).__init__(block, num_blocks, num_experts, num_classes=num_classes, reduce_dimension=reduce_dimension, layer2_output_dim=layer2_output_dim,
                                        layer3_output_dim=layer3_output_dim, use_norm=use_norm, use_experts=use_experts, returns_feat=returns_feat, s=s)
        self.in_planes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm2d(64, group_norm)

        if layer2_output_dim is None:
            if reduce_dimension:
                layer2_output_dim = 128
            else:
                layer2_output_dim = 128

        if layer3_output_dim is None:
            if reduce_dimension:
                layer3_output_dim = 256
            else:
                layer3_output_dim = 256

        if reduce_dimension:
            layer4_output_dim = 512
        else:
            layer4_output_dim = 512

        self.layer1 = self._make_layer(block, self.in_planes, num_blocks[0], stride=1)
        self.in_planes = self.next_in_planes
        self.layer2s = nn.ModuleList([self._make_layer(block, layer2_output_dim, num_blocks[1], stride=2, group_norm=group_norm) for _ in range(num_experts)])
        self.in_planes = self.next_in_planes
        self.layer3s = nn.ModuleList([self._make_layer(block, layer3_output_dim, num_blocks[2], stride=2, group_norm=group_norm) for _ in range(num_experts)])
        self.in_planes = self.next_in_planes
        self.layer4s = nn.ModuleList([self._make_layer(block, layer4_output_dim, num_blocks[3], stride=2, group_norm=group_norm) for _ in range(num_experts)])
        self.in_planes = self.next_in_planes

        if use_norm:
            self.linears = nn.ModuleList([NormedLinear(layer4_output_dim, num_classes) for _ in range(num_experts)])
        else:
            self.linears = nn.ModuleList([nn.Linear(layer4_output_dim, num_classes) for _ in range(num_experts)])
            s = 1

        if use_experts is None:
            self.use_experts = list(range(num_experts))
        elif use_experts == "rand":
            self.use_experts = None
        else:
            self.use_experts = [int(item) for item in use_experts.split(",")]

        self.s = s
        self.apply(_weights_init)


    def _make_layer(self, block, planes, num_blocks, stride=1, group_norm=0):
        downsample = None
        self.next_in_planes = self.in_planes

        if stride != 1 or self.next_in_planes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.next_in_planes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                norm2d(planes * block.expansion, group_norm),
            )

        layers = []
        layers.append(block(self.next_in_planes, planes, stride, downsample, group_norm))
        for i in range(1, num_blocks):
            self.next_in_planes = planes * block.expansion
            layers.append(block(self.next_in_planes, planes, group_norm=group_norm))

        return nn.Sequential(*layers)

    def forward(self, x, output_all=False):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)

        outs = []
        self.feat = []
        self.logits = outs
        self.feat_before_GAP = []

        if self.use_experts is None:
            use_experts = random.sample(range(self.num_experts), self.num_experts - 1)
        else:
            use_experts = self.use_experts

        for ind in use_experts:
            outs.append(self._separate_part(out, ind))
        # self.feat = torch.stack(self.feat, dim=1)
        # self.feat_before_GAP = torch.stack(self.feat_before_GAP, dim=1)
        final_out = torch.stack(outs, dim=1).mean(dim=1)
        if output_all:
            return final_out, outs
        else:
            return final_out

    def _separate_part(self, x, ind):
        out = x
        out = (self.layer2s[ind])(out)
        out = (self.layer3s[ind])(out)
        out = (self.layer4s[ind])(out)
        # self.feat_before_GAP.append(out)
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        # self.feat.append(out)
        out = (self.linears[ind])(out)
        out = out * self.s
        return out

def ride_resnet18_gn(**kwargs):

    model = ResNet_GN(BasicBlock, [2, 2, 2, 2], **kwargs)

    return model

# if __name__ == "__main__":
#     for net_name in __all__:
#         if net_name.startswith('resnet'):
#             print(net_name)
#             test(globals()[net_name]())
#             print()
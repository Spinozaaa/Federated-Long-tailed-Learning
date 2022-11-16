from fedml_api.clsimb_fedavg.multi_experts.base import BaseModel
from .ldam_drw_resnets import resnet_cifar
from .ldam_drw_resnets import expert_resnet_cifar
from .ldam_drw_resnets import ea_resnet_cifar


class Model(BaseModel):
    requires_target = False

    def __init__(self, num_classes, backbone_class=None):
        super().__init__()
        if backbone_class is not None: # Do not init backbone here if None
            self.backbone = backbone_class(num_classes)

    def _hook_before_iter(self):
        self.backbone._hook_before_iter()

    def forward(self, x, mode=None):
        x = self.backbone(x)

        assert mode is None
        return x

class EAModel(BaseModel):
    requires_target = True
    confidence_model = True

    def __init__(self, num_classes, backbone_class=None):
        super().__init__()
        if backbone_class is not None: # Do not init backbone here if None
            self.backbone = backbone_class(num_classes)

    def _hook_before_iter(self):
        self.backbone._hook_before_iter()

    def forward(self, x, mode=None, target=None):
        x = self.backbone(x, target=target)

        assert isinstance(x, tuple) # logits, extra_info
        assert mode is None
        
        return x


class ResNet18GNModel(Model): # From LDAM_DRW
    def __init__(self, num_classes, reduce_dimension=False, layer2_output_dim=None, layer3_output_dim=None, use_norm=False, num_experts=1, **kwargs):
        super().__init__(num_classes, None)
        if num_experts == 1:
            self.backbone = resnet_cifar.ResNet_s(resnet_cifar.BasicBlock, [5, 5, 5], num_classes=num_classes, reduce_dimension=reduce_dimension, layer2_output_dim=layer2_output_dim, layer3_output_dim=layer3_output_dim, use_norm=use_norm, **kwargs)
        else:
            self.backbone = expert_resnet_cifar.ResNet_GN(expert_resnet_cifar.BasicBlock_GN, [2, 2, 2, 2], num_classes=num_classes, reduce_dimension=reduce_dimension, layer2_output_dim=layer2_output_dim, layer3_output_dim=layer3_output_dim, use_norm=use_norm, num_experts=num_experts, **kwargs)

class ResNet32Model(Model): # From LDAM_DRW
    def __init__(self, num_classes, reduce_dimension=False, layer2_output_dim=None, layer3_output_dim=None, use_norm=False, num_experts=1, **kwargs):
        super().__init__(num_classes, None)
        if num_experts == 1:
            self.backbone = resnet_cifar.ResNet_s(resnet_cifar.BasicBlock, [5, 5, 5], num_classes=num_classes, reduce_dimension=reduce_dimension, layer2_output_dim=layer2_output_dim, layer3_output_dim=layer3_output_dim, use_norm=use_norm, **kwargs)
        else:
            self.backbone = expert_resnet_cifar.ResNet_s(expert_resnet_cifar.BasicBlock, [5, 5, 5], num_classes=num_classes, reduce_dimension=reduce_dimension, layer2_output_dim=layer2_output_dim, layer3_output_dim=layer3_output_dim, use_norm=use_norm, num_experts=num_experts, **kwargs)

class ResNet32EAModel(EAModel): # From LDAM_DRW
    def __init__(self, num_classes, reduce_dimension=False, layer2_output_dim=None, layer3_output_dim=None, num_experts=2, **kwargs):
        super().__init__(num_classes, None)
        self.backbone = ea_resnet_cifar.ResNet_s(ea_resnet_cifar.BasicBlock, [5, 5, 5], num_classes=num_classes, reduce_dimension=reduce_dimension, layer2_output_dim=layer2_output_dim, layer3_output_dim=layer3_output_dim, num_experts=num_experts, **kwargs)


from torch import Tensor
from torchvision.models.resnet import ResNet
from torchvision.models.resnet import BasicBlock
import torch.nn as nn
from typing import Callable, List, Optional
### Custom implementation of ResNet without inplace ops that is 
# compatible with the shap package, otherwise follows the torchvision.models.resnet
class CustomResNet(ResNet):
    
    def __init__(self,
        block,
        layers: List[int],
        num_classes: int = 1000,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__(block, layers, num_classes, zero_init_residual, groups, width_per_group,
                         replace_stride_with_dilation, norm_layer)
        # need to change this because we can't do inplace operations
        # to calculate SHAP values
        self.relu = nn.ReLU(inplace=False)

class CustomBasicBlock(BasicBlock):
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__(inplanes, planes, stride, downsample, groups, base_width, dilation, norm_layer)
        self.relu = nn.ReLU(inplace=False)
        
        
    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = out + identity
        out = self.relu(out)

        return out

import os
import sys
import torch
import torch.nn as nn
from torchvision.models import resnet18

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from core.layers import OrganicSynapseConv

def get_organic_resnet18(device_profile, num_classes=5, pretrained=False):
    model = resnet18(weights='DEFAULT' if pretrained else None)
    
    def replace_layers(module):
        for name, child in module.named_children():
            if isinstance(child, nn.Conv2d):
                new_conv = OrganicSynapseConv(
                    in_channels=child.in_channels,
                    out_channels=child.out_channels,
                    kernel_size=child.kernel_size,
                    device_profile=device_profile,
                    stride=child.stride,
                    padding=child.padding,
                    dilation=child.dilation,
                    groups=child.groups,
                    bias=(child.bias is not None),
                    padding_mode=child.padding_mode
                )
                new_conv.weight.data = child.weight.data.clone()
                if child.bias is not None:
                    new_conv.bias.data = child.bias.data.clone()
                setattr(module, name, new_conv)
            else:
                replace_layers(child)
                
    replace_layers(model)
    
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    return model

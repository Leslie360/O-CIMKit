import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from core.layers import OrganicSynapseConv, QATMLPLayer

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, device_profile=None):
        super(BasicBlock, self).__init__()
        self.conv1 = OrganicSynapseConv(
            in_planes, planes, kernel_size=3, device_profile=device_profile, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        
        self.conv2 = OrganicSynapseConv(
            planes, planes, kernel_size=3, device_profile=device_profile, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                OrganicSynapseConv(
                    in_planes, self.expansion * planes, kernel_size=1, device_profile=device_profile, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.dropout(out)
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        out = self.dropout(out)
        return out

class ResNet18Grayscale(nn.Module):
    def __init__(self, num_classes=15, device_profile=None, pretrained=True):
        super(ResNet18Grayscale, self).__init__()
        self.in_planes = 64
        self.profile = device_profile

        # 1-channel grayscale input for 64x64 Yale Faces
        self.conv1 = OrganicSynapseConv(1, 64, kernel_size=3, device_profile=device_profile, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        
        self.layer1 = self._make_layer(BasicBlock, 64, 2, stride=1)
        self.layer2 = self._make_layer(BasicBlock, 128, 2, stride=2)
        self.layer3 = self._make_layer(BasicBlock, 256, 2, stride=2)
        self.layer4 = self._make_layer(BasicBlock, 512, 2, stride=2)
        
        # QAT Readout layer
        self.dropout_fc = nn.Dropout(p=0.4)
        self.linear = QATMLPLayer(512 * BasicBlock.expansion, num_classes, device_profile=device_profile, mode="minmax")

        if pretrained:
            self.load_pretrained_weights()

    def load_pretrained_weights(self):
        from torchvision.models import resnet18
        ref = resnet18(weights='DEFAULT')
        
        # Interpolate conv1 from 7x7 (3 channels) to 3x3 (1 channel)
        w_gray = ref.conv1.weight.data.mean(dim=1, keepdim=True) # [64, 1, 7, 7]
        w_interp = torch.nn.functional.interpolate(w_gray, size=(3, 3), mode='bilinear', align_corners=True) # [64, 1, 3, 3]
        self.conv1.weight.data = w_interp
        self.bn1.load_state_dict(ref.bn1.state_dict())
        
        # Layer 1
        for i in range(2):
            self.layer1[i].conv1.weight.data = ref.layer1[i].conv1.weight.data.clone()
            self.layer1[i].bn1.load_state_dict(ref.layer1[i].bn1.state_dict())
            self.layer1[i].conv2.weight.data = ref.layer1[i].conv2.weight.data.clone()
            self.layer1[i].bn2.load_state_dict(ref.layer1[i].bn2.state_dict())
            
        # Layer 2
        for i in range(2):
            self.layer2[i].conv1.weight.data = ref.layer2[i].conv1.weight.data.clone()
            self.layer2[i].bn1.load_state_dict(ref.layer2[i].bn1.state_dict())
            self.layer2[i].conv2.weight.data = ref.layer2[i].conv2.weight.data.clone()
            self.layer2[i].bn2.load_state_dict(ref.layer2[i].bn2.state_dict())
            if len(self.layer2[i].shortcut) > 0:
                self.layer2[i].shortcut[0].weight.data = ref.layer2[i].downsample[0].weight.data.clone()
                self.layer2[i].shortcut[1].load_state_dict(ref.layer2[i].downsample[1].state_dict())
                
        # Layer 3
        for i in range(2):
            self.layer3[i].conv1.weight.data = ref.layer3[i].conv1.weight.data.clone()
            self.layer3[i].bn1.load_state_dict(ref.layer3[i].bn1.state_dict())
            self.layer3[i].conv2.weight.data = ref.layer3[i].conv2.weight.data.clone()
            self.layer3[i].bn2.load_state_dict(ref.layer3[i].bn2.state_dict())
            if len(self.layer3[i].shortcut) > 0:
                self.layer3[i].shortcut[0].weight.data = ref.layer3[i].downsample[0].weight.data.clone()
                self.layer3[i].shortcut[1].load_state_dict(ref.layer3[i].downsample[1].state_dict())
                
        # Layer 4
        for i in range(2):
            self.layer4[i].conv1.weight.data = ref.layer4[i].conv1.weight.data.clone()
            self.layer4[i].bn1.load_state_dict(ref.layer4[i].bn1.state_dict())
            self.layer4[i].conv2.weight.data = ref.layer4[i].conv2.weight.data.clone()
            self.layer4[i].bn2.load_state_dict(ref.layer4[i].bn2.state_dict())
            if len(self.layer4[i].shortcut) > 0:
                self.layer4[i].shortcut[0].weight.data = ref.layer4[i].downsample[0].weight.data.clone()
                self.layer4[i].shortcut[1].load_state_dict(ref.layer4[i].downsample[1].state_dict())

        # Reset running stats of all batch norm layers to prevent ImageNet bias
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.reset_running_stats()

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(block(self.in_planes, planes, s, device_profile=self.profile))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        # Convert flat inputs [B, 4096] to image format [B, 1, 64, 64]
        if x.dim() == 2:
            x = x.view(-1, 1, 64, 64)
            
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        
        # Output spatial dimensions: 64x64 -> 64x64 -> 32x32 -> 16x16 -> 8x8
        out = F.avg_pool2d(out, 8)
        out = out.view(out.size(0), -1)
        out = self.dropout_fc(out)
        out = self.linear(out)
        return out

class FaceClassifier(ResNet18Grayscale):
    """Alias for backward compatibility with train.py"""
    def __init__(self, input_dim=4096, hidden_dim=256, num_classes=15, device_profile=None):
        super().__init__(num_classes=num_classes, device_profile=device_profile, pretrained=True)

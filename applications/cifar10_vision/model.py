import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from core.layers import OrganicSynapseConv

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

class ResNet18(nn.Module):
    def __init__(self, num_classes=10, device_profile=None, pretrained=True):
        super(ResNet18, self).__init__()
        self.in_planes = 64
        self.profile = device_profile

        self.conv1 = OrganicSynapseConv(3, 64, kernel_size=3, device_profile=device_profile, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        
        self.layer1 = self._make_layer(BasicBlock, 64, 2, stride=1)
        self.layer2 = self._make_layer(BasicBlock, 128, 2, stride=2)
        self.layer3 = self._make_layer(BasicBlock, 256, 2, stride=2)
        self.layer4 = self._make_layer(BasicBlock, 512, 2, stride=2)
        
        self.linear = nn.Linear(512 * BasicBlock.expansion, num_classes)

        if pretrained:
            self.load_pretrained_weights()

    def load_pretrained_weights(self):
        from torchvision.models import resnet18
        
        # Local model checkpoint path validation to avoid online download hangs
        cache_dir = os.path.expanduser("~/.cache/torch/hub/checkpoints")
        local_path1 = os.path.join(cache_dir, "resnet18-f37072fd.pth")
        local_path2 = os.path.join(cache_dir, "resnet18-f881a174.pth")
        
        ref = resnet18(weights=None)
        loaded = False
        for path in [local_path1, local_path2]:
            if os.path.exists(path):
                try:
                    ref.load_state_dict(torch.load(path, map_location='cpu'))
                    print(f"  Successfully loaded pre-trained ResNet-18 weights locally from {os.path.basename(path)}")
                    loaded = True
                    break
                except Exception as e:
                    print(f"  ⚠️ Error loading local weights {path}: {e}")
                    
        if not loaded:
            print("  ⚠️ No local ResNet-18 cache found! Attempting online download...")
            ref = resnet18(weights='DEFAULT')
        
        # Interpolate conv1 from 7x7 to 3x3 (3 channels)
        w_ref = ref.conv1.weight.data # [64, 3, 7, 7]
        w_interp = torch.nn.functional.interpolate(w_ref, size=(3, 3), mode='bilinear', align_corners=True) # [64, 3, 3, 3]
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
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

def get_model(device_profile=None, num_classes=10):
    return ResNet18(num_classes=num_classes, device_profile=device_profile, pretrained=True)

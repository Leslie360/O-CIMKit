import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from core.layers import OrganicSynapseConv, QATMLPLayer
from applications.optoelectronic_vision.dataset import simulate_physical_optic

class OptoelectronicPerception(nn.Module):
    """
    Optoelectronic Perception Layer: Simulates physical photo-electric capture
    followed by normalization to match standard neural network input expectations.
    """
    def __init__(self, max_photons=10000.0):
        super().__init__()
        self.max_photons = max_photons
        self.register_buffer('mean', torch.tensor([0.4914, 0.4822, 0.4465]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.2023, 0.1994, 0.2010]).view(1, 3, 1, 1))
        self.is_float_baseline = False
        
    def forward(self, x):
        if self.is_float_baseline:
            return (x - self.mean) / self.std
            
        # 1. Inverse Gamma correction (gamma = 2.2)
        x_linear = torch.pow(x, 2.2)
        # 2. Simulate physical photoelectric capture with Poisson shot noise
        x_optic = simulate_physical_optic(x_linear, self.max_photons)
        # 3. Apply standard normalization
        return (x_optic - self.mean) / self.std

class StandardBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(StandardBasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
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

class OptoelectronicVisionNet(nn.Module):
    """
    Optoelectronic Synapse Sensor-CIM Integrated Vision Network.
    - Front perception layer: OptoelectronicPerception (Low-light Poisson shot noise)
    - Physical convolution layer (conv1): OrganicSynapseConv mapped to OECT_Vision.json (30 states, C2C noise, LTP/LTD gradient Hook)
    - Feature extractor: Standard ResNet-18 layers (representing standard digital/precision-preserving back-end processing)
    - QAT readout layer: QATMLPLayer quantized to discrete states of OECT_Vision.json
    """
    def __init__(self, num_classes=10, device_profile=None, max_photons=10000.0, pretrained=True):
        super(OptoelectronicVisionNet, self).__init__()
        self.in_planes = 64
        self.profile = device_profile
        
        self.perception = OptoelectronicPerception(max_photons=max_photons)
        if device_profile is None:
            self.perception.is_float_baseline = True
        
        # Physical sensory convolution layer
        self.conv1 = OrganicSynapseConv(3, 64, kernel_size=3, device_profile=device_profile, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        
        # High-precision feature extraction
        self.layer1 = self._make_layer(StandardBasicBlock, 64, 2, stride=1)
        self.layer2 = self._make_layer(StandardBasicBlock, 128, 2, stride=2)
        self.layer3 = self._make_layer(StandardBasicBlock, 256, 2, stride=2)
        self.layer4 = self._make_layer(StandardBasicBlock, 512, 2, stride=2)
        
        # QAT fully connected readout layer
        self.linear = QATMLPLayer(512 * StandardBasicBlock.expansion, num_classes, device_profile=device_profile, mode="minmax")

        if pretrained:
            self.load_pretrained_weights()

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(block(self.in_planes, planes, s))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def load_pretrained_weights(self):
        from torchvision.models import resnet18
        
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
            
        # 1. Map conv1 (interpolate from 7x7 to 3x3)
        w_ref = ref.conv1.weight.data  # [64, 3, 7, 7]
        w_interp = torch.nn.functional.interpolate(w_ref, size=(3, 3), mode='bilinear', align_corners=True)  # [64, 3, 3, 3]
        self.conv1.weight.data = w_interp
        self.bn1.load_state_dict(ref.bn1.state_dict())
        
        # 2. Map standard basic blocks
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
                
        # 3. Readout classification head (initialize with reference fc weights if shape matches, otherwise default)
        if hasattr(ref, 'fc') and self.linear.weight.shape == ref.fc.weight.shape:
            self.linear.weight.data = ref.fc.weight.data.clone()
            self.linear.bias.data = ref.fc.bias.data.clone()

    def forward(self, x):
        # 1. Front-end perception
        out = self.perception(x)
        # 2. Hardware-aware conv1
        out = F.relu(self.bn1(self.conv1(out)))
        # 3. Backbone extraction
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        # 4. Pooling & Readout
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

def get_model(device_profile=None, num_classes=10, max_photons=10000.0, pretrained=True):
    return OptoelectronicVisionNet(
        num_classes=num_classes, 
        device_profile=device_profile, 
        max_photons=max_photons, 
        pretrained=pretrained
    )

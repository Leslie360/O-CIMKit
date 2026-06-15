import os
import sys
import torch
import torch.nn as nn

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from profiles.device_profile import DeviceProfile
from core.layers import QATMLPLayer

# 1. Load Device Profile
# You can load any parsed profile: FingerMemristor, OECT_Vision, PFO_AlOx_NonVolatile, etc.
profile_path = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 
    "profiles", "repository", "FingerMemristor.json"
)

if os.path.exists(profile_path):
    device_profile = DeviceProfile.from_json(profile_path)
    print(f"✅ Loaded device profile: {device_profile.device_name}")
    print(f"  - Conductance Range: {device_profile.conductance_min:.2e} ~ {device_profile.conductance_max:.2e} S")
    print(f"  - Discrete Quantization: {device_profile.discrete_states_count} states")
else:
    device_profile = None
    print("⚠️ Profile not found, running ideal simulation.")

# 2. Build model with hardware-aware layer
class FaceClassifier(nn.Module):
    def __init__(self, input_dim=4096, hidden_dim=256, num_classes=15, profile=None):
        super().__init__()
        # First layer incorporates LSQ QAT based on device states
        self.fc1 = QATMLPLayer(input_dim, hidden_dim, device_profile=profile)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        return self.fc2(x)

# Instantiate Model
model = FaceClassifier(profile=device_profile)
print(model)

# 3. Simulate training pass
dummy_input = torch.randn(4, 4096)
output = model(dummy_input)
print(f"Forward pass successful. Output shape: {output.shape}")

# Simulate backward pass
loss = output.sum()
loss.backward()
print("Backward pass with STE/LSQ updates successful.")

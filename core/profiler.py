import torch
import torch.nn as nn
from core.layers import QATMLPLayer, OrganicSynapseConv

class SystemProfiler:
    """
    Architectural Profiler for Computing-in-Memory (CIM) Hardware.
    Evaluates system-level metrics: Energy Efficiency (TOPS/W), Area, Latency.
    Compliant with benchmarking methodologies from ISCA, DAC, and NeuroSim.
    """
    def __init__(self, model: nn.Module, device_profile=None):
        self.model = model
        self.profile = device_profile
        self.mac_count = 0
        self.weight_params = 0
        
        # Default Hardware Metrics (can be overridden by device_profile)
        self.energy_per_mac_pj = getattr(device_profile, 'energy_per_mac_pj', 0.05) # 50 fJ per MAC (Array level)
        self.adc_dac_energy_pj = getattr(device_profile, 'adc_dac_energy_pj', 0.20) # 200 fJ per MAC (Peripheral overhead)
        
        self.cell_area_um2 = getattr(device_profile, 'cell_area_um2', 0.04) # 200x200 nm^2
        self.peripheral_area_ratio = getattr(device_profile, 'peripheral_area_ratio', 0.3) # 30% area overhead
        
        self.read_latency_ns = getattr(device_profile, 'read_latency_ns', 10.0) # 10 ns read
        self.crossbar_size = getattr(device_profile, 'crossbar_size', 256) # 256x256 tiles

    def _hook_fn(self, module, input, output):
        # Calculate MACs for QATMLPLayer
        if isinstance(module, QATMLPLayer):
            in_features = module.weight.shape[1]
            out_features = module.weight.shape[0]
            batch_size = input[0].shape[0]
            if len(input[0].shape) == 3: # Sequence data like Nano-GPT
                batch_size *= input[0].shape[1]
            macs = batch_size * in_features * out_features
            self.mac_count += macs
            self.weight_params += in_features * out_features
            
        # Calculate MACs for OrganicSynapseConv
        elif isinstance(module, OrganicSynapseConv):
            out_c, in_c, k_h, k_w = module.weight.shape
            batch_size = input[0].shape[0]
            out_h, out_w = output.shape[2], output.shape[3]
            macs = batch_size * out_c * out_h * out_w * in_c * k_h * k_w
            self.mac_count += macs
            self.weight_params += out_c * in_c * k_h * k_w

    def profile_model(self, dummy_input):
        """Run a forward pass to trace the MAC operations."""
        self.mac_count = 0
        self.weight_params = 0
        
        hooks = []
        for name, module in self.model.named_modules():
            if isinstance(module, (QATMLPLayer, OrganicSynapseConv)):
                hooks.append(module.register_forward_hook(self._hook_fn))
                
        # Forward pass
        with torch.no_grad():
            self.model(dummy_input)
            
        for hook in hooks:
            hook.remove()

    def get_report(self):
        """Calculate and return DAC/ISCA compliant architectural metrics."""
        # 1. Energy Calculation (in Joules)
        total_energy_pj = self.mac_count * (self.energy_per_mac_pj + self.adc_dac_energy_pj)
        total_energy_j = total_energy_pj * 1e-12
        
        # 2. Area Calculation (in mm^2)
        array_area_um2 = self.weight_params * self.cell_area_um2
        total_area_um2 = array_area_um2 * (1.0 + self.peripheral_area_ratio)
        total_area_mm2 = total_area_um2 * 1e-6
        
        # 3. Latency Calculation (in seconds)
        # Assuming parallel execution across crossbar tiles
        tiles_needed = max(1, self.weight_params / (self.crossbar_size ** 2))
        bottleneck_macs = self.mac_count / tiles_needed
        latency_ns = (bottleneck_macs / self.crossbar_size) * self.read_latency_ns
        latency_s = latency_ns * 1e-9
        
        # 4. Compute TOPS and TOPS/W
        if latency_s > 0:
            throughput_tops = (self.mac_count * 2) / latency_s / 1e12 # 2 ops per MAC (mult + add)
            power_w = total_energy_j / latency_s
            energy_efficiency_tops_w = throughput_tops / power_w if power_w > 0 else 0
        else:
            throughput_tops = 0
            energy_efficiency_tops_w = 0
            
        return {
            "Total MACs (M)": self.mac_count / 1e6,
            "Total Energy (uJ)": total_energy_pj / 1e6,
            "Total Area (mm^2)": total_area_mm2,
            "Latency (ms)": latency_ns / 1e6,
            "Energy Efficiency (TOPS/W)": energy_efficiency_tops_w,
            "Throughput (TOPS)": throughput_tops
        }

import os
import json
import numpy as np

class DeviceProfile:
    """
    Unified representation of an Organic Device (memristor, transistor, OFET, OECT, etc.)
    defining its physical properties, non-linearities, quantization states, and noise limits.
    """
    def __init__(self, name, device_type="memristor", **kwargs):
        self.device_name = name
        self.device_type = device_type
        
        # 1. Dynamic behavior / Memory properties
        self.is_volatile = kwargs.get("is_volatile", False)
        self.tau_volatile = kwargs.get("tau_volatile", None)  # volatile decay constant (seconds)
        self.leaking_rate_volatile = kwargs.get("leaking_rate_volatile", None)
        
        self.is_nonvolatile = kwargs.get("is_nonvolatile", True)
        self.discrete_states_count = kwargs.get("discrete_states_count", None) # None means analog/continuous
        
        # 2. Electrical / Conductance properties
        self.conductance_min = kwargs.get("conductance_min", 1.0)
        self.conductance_max = kwargs.get("conductance_max", 10.0)
        self.noise_std_ratio = kwargs.get("noise_std_ratio", 0.0)  # noise relative to conductance range
        
        # 3. LTP / LTD Non-linearity Coefficients (3rd-order polynomials)
        self.ltp_poly_coefficients = kwargs.get("ltp_poly_coefficients", [0.0, 0.0, 0.0, 0.0])
        self.ltd_poly_coefficients = kwargs.get("ltd_poly_coefficients", [0.0, 0.0, 0.0, 0.0])
        
        # Calculate leaking rate if tau is provided
        if self.is_volatile and self.tau_volatile is not None and self.leaking_rate_volatile is None:
            # Alpha leaking rate matching ESN discrete updates: alpha = 1 - exp(-1/tau)
            self.leaking_rate_volatile = float(1.0 - np.exp(-1.0 / self.tau_volatile))

    @classmethod
    def from_json(cls, json_path):
        """Load a device profile from a JSON configuration file."""
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        kwargs = {}
        if "dynamic_parameters" in data:
            kwargs.update(data["dynamic_parameters"])
        if "fitting_parameters" in data:
            kwargs.update(data["fitting_parameters"])
            
        return cls(
            name=data["device_name"],
            device_type=data.get("device_type", "memristor"),
            **kwargs
        )

    def to_json(self, json_path):
        """Save device profile configuration to JSON file."""
        data = {
            "device_name": self.device_name,
            "device_type": self.device_type,
            "dynamic_parameters": {
                "is_volatile": self.is_volatile,
                "tau_volatile": self.tau_volatile,
                "leaking_rate_volatile": self.leaking_rate_volatile,
                "is_nonvolatile": self.is_nonvolatile,
                "discrete_states_count": self.discrete_states_count
            },
            "fitting_parameters": {
                "conductance_min": self.conductance_min,
                "conductance_max": self.conductance_max,
                "noise_std_ratio": self.noise_std_ratio,
                "ltp_poly_coefficients": list(self.ltp_poly_coefficients),
                "ltd_poly_coefficients": list(self.ltd_poly_coefficients)
            }
        }
        
        os.makedirs(os.path.dirname(json_path), exist_ok=True)
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4, ensure_ascii=False)

    def map_to_physical(self, w_math):
        """Map mathematical weights [-1, 1] to physical conductance [G_min, G_max]."""
        w_norm = (w_math + 1.0) / 2.0
        w_phys = w_norm * (self.conductance_max - self.conductance_min) + self.conductance_min
        return w_phys

    def map_to_math(self, w_phys):
        """Map physical conductance [G_min, G_max] back to mathematical weights [-1, 1]."""
        w_norm = (w_phys - self.conductance_min) / (self.conductance_max - self.conductance_min)
        w_math = w_norm * 2.0 - 1.0
        return w_math

    def get_noise_std(self):
        """Calculate the absolute noise standard deviation in Siemens/Amperes."""
        return (self.conductance_max - self.conductance_min) * self.noise_std_ratio

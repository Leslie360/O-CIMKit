import torch
import torch.nn as nn
from typing import Any, Tuple

class STEFunction(torch.autograd.Function):
    """
    Straight-Through Estimator (STE) Function.
    Forward pass: round to nearest integer.
    Backward pass: identity gradient flow.
    """
    @staticmethod
    def forward(ctx: Any, input: torch.Tensor) -> torch.Tensor:
        """Forward pass rounds the input to the nearest integer."""
        return torch.round(input)

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> torch.Tensor:
        """Backward pass propagates the gradient unchanged."""
        return grad_output

class STEClamp(torch.autograd.Function):
    """
    Straight-Through Estimator (STE) Clamp.
    Clamps inputs during forward, passes gradients through during backward.
    """
    @staticmethod
    def forward(ctx: Any, input: torch.Tensor, min_val: float, max_val: float) -> torch.Tensor:
        """Forward pass clamps the input to [min_val, max_val]."""
        return torch.clamp(input, min_val, max_val)

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> Tuple[torch.Tensor, None, None]:
        """Backward pass propagates the gradient unchanged."""
        return grad_output, None, None

class LSQQuantizer(nn.Module):
    """
    Learned Step Size Quantization (LSQ) Module.
    Supports dynamic learnable step size 's' for quantized state optimization.
    """
    def __init__(self, num_states: int) -> None:
        """
        Initializes the LSQQuantizer.
        
        Args:
            num_states (int): Number of discrete states.
        """
        super().__init__()
        self.num_states = num_states
        
        # Calculate quantization boundaries
        # For N-states (e.g. 64-states), standard integer grid is [-N/2, N/2 - 1]
        self.q_min = -(num_states // 2)
        self.q_max = (num_states // 2) - 1
        
        # Step size parameter (initialized as a floating scale)
        self.s = nn.Parameter(torch.tensor(1.0))
        self.initialized = False

    def initialize_step_size(self, weights: torch.Tensor) -> None:
        """
        Initialize step size 's' based on the weight magnitude.
        
        Args:
            weights (torch.Tensor): Weight tensor to compute initial step size.
        """
        with torch.no_grad():
            max_val = torch.max(torch.abs(weights))
            self.s.data.fill_(max_val / (self.q_max - self.q_min))
            self.initialized = True

    def forward(self, weights: torch.Tensor) -> torch.Tensor:
        """Forward pass applying LSQ quantization to the weights."""
        if not self.initialized:
            self.initialize_step_size(weights)
            
        # 1. Scale weights
        scaled_weights = weights / self.s
        
        # 2. Clip values to boundaries
        clipped_weights = torch.clamp(scaled_weights, self.q_min, self.q_max)
        
        # 3. Apply STE rounding
        rounded_weights = STEFunction.apply(clipped_weights)
        
        # 4. De-scale back to math domain
        quantized_weights = rounded_weights * self.s
        return quantized_weights

class MinMaxQuantizer(nn.Module):
    """
    Dynamic Min-Max Quantization Module.
    Quantizes weights dynamically to a fixed number of levels based on min/max of weights in each pass.
    """
    def __init__(self, num_states: int) -> None:
        """
        Initializes the MinMaxQuantizer.
        
        Args:
            num_states (int): Number of discrete states.
        """
        super().__init__()
        self.num_states = num_states

    def forward(self, weights: torch.Tensor) -> torch.Tensor:
        """Forward pass applying MinMax quantization to the weights."""
        w_min = weights.min()
        w_max = weights.max()
        if w_min == w_max:
            return weights.clone()
        
        # Scale to [0, num_states - 1]
        scaled = (weights - w_min) / (w_max - w_min + 1e-8) * (self.num_states - 1)
        # Apply STE rounding
        rounded = STEFunction.apply(scaled)
        # Scale back to original domain
        quantized = rounded / (self.num_states - 1) * (w_max - w_min) + w_min
        return quantized


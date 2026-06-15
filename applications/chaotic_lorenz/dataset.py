import numpy as np

def generate_lorenz_1d(steps=5000, dt=0.01):
    """Generates Lorenz chaos 1D trajectory (X axis) normalized to [-1, 1]."""
    sigma = 10.0
    rho = 28.0
    beta = 8.0 / 3.0
    
    x, y, z = 1.0, 1.0, 1.0
    data = []
    
    for _ in range(steps):
        dx = sigma * (y - x) * dt
        dy = (x * (rho - z) - y) * dt
        dz = (x * y - beta * z) * dt
        x += dx
        y += dy
        z += dz
        data.append(x)
        
    data = np.array(data)
    # Normalize to [-1, 1]
    return (data - data.min()) / (data.max() - data.min()) * 2 - 1

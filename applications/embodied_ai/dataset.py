import numpy as np

def generate_tactile_dataset_with_temporal_dependency(n_samples=2000, n_timesteps=100, tau_volatile=3.64):
    """
    Generates tactile dataset with temporal dependency for sensor fusion evaluation.
    """
    np.random.seed(42)
    materials = ['glass', 'sandpaper', 'rubber', 'metal']
    X = []
    y = []
    material_baselines = []

    for i in range(n_samples):
        material = materials[i % len(materials)]
        t = np.linspace(0, tau_volatile, n_timesteps)

        # Baseline friction coefficient
        if material == 'glass':
            mu_base = 0.2
        elif material == 'sandpaper':
            mu_base = 0.8
        elif material == 'rubber':
            mu_base = 0.5
        elif material == 'metal':
            mu_base = 0.3

        # Temporal dependency (friction coefficient changes over time)
        mu_temporal = mu_base * (1.0 + 0.1 * np.sin(2.0 * np.pi * 0.5 * t))
        # Sliding speed variation
        v_slide = 0.1 + 0.05 * np.sin(2.0 * np.pi * 0.3 * t)
        # Pressure variation
        pressure = 1.0 + 0.2 * np.sin(2.0 * np.pi * 0.2 * t)
        
        signal = mu_temporal * v_slide * pressure

        # Add material-specific noise
        if material == 'glass':
            noise = 0.02 * np.random.randn(n_timesteps)
        elif material == 'sandpaper':
            noise = 0.1 * np.random.randn(n_timesteps)
        elif material == 'rubber':
            noise = 0.05 * np.random.randn(n_timesteps)
        elif material == 'metal':
            noise = 0.03 * np.random.randn(n_timesteps)

        signal = signal + noise
        baseline = mu_base

        X.append(signal)
        y.append(materials.index(material))
        material_baselines.append(baseline)

    return np.array(X), np.array(y), materials, np.array(material_baselines)

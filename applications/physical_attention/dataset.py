import numpy as np

def generate_long_dependency_task(n_samples=2000, n_timesteps=200, dependency_length=60):
    """
    Generates long dependency time-series prediction task.
    Predicts a rare event of 3 classes (no event, small event, large event)
    that occurred dependency_length seconds ago.
    """
    np.random.seed(42)
    X = []
    y = []

    for i in range(n_samples):
        t = np.linspace(0, dependency_length, n_timesteps)
        base = np.sin(2.0 * np.pi * 0.1 * t) + 0.5 * np.sin(2.0 * np.pi * 0.3 * t)
        
        rare_event = np.zeros(n_timesteps)
        event_type = np.random.choice([0, 1, 2])

        if event_type == 1:
            rare_event[0:10] = np.random.randn(10) * 1.0
        elif event_type == 2:
            rare_event[0:10] = np.random.randn(10) * 3.0

        signal = base + rare_event + 0.1 * np.random.randn(n_timesteps)
        X.append(signal)
        y.append(event_type)

    return np.array(X), np.array(y)

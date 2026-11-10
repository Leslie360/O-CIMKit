import os
import sys
import numpy as np
import wfdb

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

def load_ecg_data(dataset_name="mitdb", data_dir=None, return_groups=False):
    """
    Load 180-sample heartbeat windows with optional MIT-BIH record identifiers.

    Record identifiers support group-isolated validation. Synthetic PTB-DB
    fallback data have no patient identifiers and do not support grouped folds.
    """
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    if dataset_name == "ptbdb":
        if return_groups:
            raise ValueError("Synthetic PTB-DB data have no patient identifiers for grouped validation.")
        if data_dir is None:
            data_dir = os.path.join(project_root, "data", "datasets", "ptbdb")
        if os.path.exists(data_dir) and any(os.listdir(data_dir)):
            raise NotImplementedError("Real PTB-DB parsing is not implemented; no synthetic data will be substituted silently.")

        print("Generating explicitly labelled synthetic PTB-DB example waveforms...")
        np.random.seed(42)
        t = np.linspace(0, 1.8, 180)
        normal_pulse = np.sin(2 * np.pi * t) + 0.3 * np.sin(10 * np.pi * t) * np.exp(-100 * (t - 0.5)**2)
        mi_pulse = np.sin(2 * np.pi * t) + 0.1 * np.sin(10 * np.pi * t) * np.exp(-100 * (t - 0.5)**2) - 0.2 * np.exp(-50 * (t - 0.9)**2)

        all_X, all_y = [], []
        for _ in range(1000):
            all_X.append(normal_pulse + np.random.normal(0, 0.05, 180))
            all_y.append(0)
            all_X.append(mi_pulse + np.random.normal(0, 0.05, 180))
            all_y.append(1)

        X, y = np.array(all_X), np.array(all_y)
        idx = np.arange(len(X))
        np.random.shuffle(idx)
        return X[idx], y[idx]

    if dataset_name != "mitdb":
        raise ValueError(f"Unsupported ECG dataset: {dataset_name}")
    if data_dir is None:
        data_dir = os.path.join(project_root, "data", "datasets", "mitdb")
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"MIT-BIH dataset directory not found: {data_dir}")

    print(f"Loading MIT-BIH dataset from: {data_dir}")
    all_X, all_y, record_ids = [], [], []
    label_map = {'N': 0, 'L': 0, 'R': 0, 'A': 1, 'a': 1, 'V': 2, 'E': 2}

    for filename in sorted(os.listdir(data_dir)):
        if not filename.endswith('.hea'):
            continue
        record = filename[:-4]
        try:
            sig = wfdb.rdrecord(os.path.join(data_dir, record)).p_signal[:, 0]
            ann = wfdb.rdann(os.path.join(data_dir, record), 'atr')
        except Exception as exc:
            print(f"Warning: skipping unreadable record {record}: {exc}")
            continue
        for sample, symbol in zip(ann.sample, ann.symbol):
            if symbol not in label_map:
                continue
            start, end = sample - 90, sample + 90
            if start < 0 or end >= len(sig):
                continue
            heartbeat = sig[start:end]
            heartbeat = (heartbeat - heartbeat.mean()) / (heartbeat.std() + 1e-8)
            all_X.append(heartbeat)
            all_y.append(int(label_map[symbol] > 0))
            record_ids.append(record)

    if not all_X:
        raise ValueError(f"No supported ECG annotations were loaded from {data_dir}")
    X, y = np.array(all_X), np.array(all_y)
    normal_idx = np.where(y == 0)[0]
    abnormal_idx = np.where(y == 1)[0]
    np.random.seed(42)
    normal_sampled = np.random.choice(normal_idx, min(len(abnormal_idx) * 3, 2832), replace=False)
    abnormal_sampled = np.random.choice(abnormal_idx, min(len(abnormal_idx), 944), replace=False)
    selected = np.concatenate([normal_sampled, abnormal_sampled])
    np.random.shuffle(selected)

    print(f"Sampled {len(selected)} beats from {len(set(record_ids))} MIT-BIH records "
          f"(Normal: {len(normal_sampled)}, Abnormal: {len(abnormal_sampled)})")
    if return_groups:
        return X[selected], y[selected], np.array(record_ids)[selected]
    return X[selected], y[selected]


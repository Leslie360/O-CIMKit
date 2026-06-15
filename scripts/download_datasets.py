import os
import urllib.request
import numpy as np
import torch
import torchvision

def download_all_datasets():
    print("🚀 [O-CIMKit] Starting automated dataset acquisition and generation...")
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    datasets_dir = os.path.join(project_root, "data", "datasets")
    os.makedirs(datasets_dir, exist_ok=True)
    
    # 1. Download Torchvision standard datasets
    print("📦 Pre-downloading MNIST...")
    torchvision.datasets.MNIST(root=datasets_dir, train=True, download=True)
    print("📦 Pre-downloading CIFAR-10...")
    torchvision.datasets.CIFAR10(root=datasets_dir, train=True, download=True)
    
    # 2. Generate lightweight Mock Datasets for proprietary/medical/physical domains
    # This ensures that researchers cloning the repo can instantly run out-of-the-box evaluations
    # without needing massive proprietary datasets (like MIT-BIH, Sleep-EDF, CWRU Bearing Fault)
    print("🔬 Generating lightweight mock datasets for medical/physical domains...")
    
    # Mock ECG (MIT-BIH)
    ecg_dir = os.path.join(datasets_dir, "mit_bih")
    os.makedirs(ecg_dir, exist_ok=True)
    np.save(os.path.join(ecg_dir, "train_data.npy"), np.random.randn(100, 1, 300))
    np.save(os.path.join(ecg_dir, "train_labels.npy"), np.random.randint(0, 5, 100))
    
    # Mock EEG (Sleep-EDF)
    eeg_dir = os.path.join(datasets_dir, "sleep_edf")
    os.makedirs(eeg_dir, exist_ok=True)
    np.save(os.path.join(eeg_dir, "eeg_mock.npy"), np.random.randn(50, 3000))
    
    # Mock CWRU Bearing Fault
    cwru_dir = os.path.join(datasets_dir, "cwru")
    os.makedirs(cwru_dir, exist_ok=True)
    np.save(os.path.join(cwru_dir, "vibration_data.npy"), np.random.randn(200, 1024))
    
    # Mock Face Rec (Yale)
    yale_dir = os.path.join(datasets_dir, "yale_faces")
    os.makedirs(yale_dir, exist_ok=True)
    np.save(os.path.join(yale_dir, "faces.npy"), np.random.randn(50, 1, 64, 64))
    
    # Mock Multimodal E-skin/Embodied
    eskin_dir = os.path.join(datasets_dir, "e_skin")
    os.makedirs(eskin_dir, exist_ok=True)
    np.save(os.path.join(eskin_dir, "tactile.npy"), np.random.randn(500, 16, 16))

    print("✅ All datasets prepared successfully! You can now run all O-CIMKit applications.")

if __name__ == "__main__":
    download_all_datasets()

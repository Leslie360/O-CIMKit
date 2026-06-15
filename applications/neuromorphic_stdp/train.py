import os
import sys
import time
import torch
import torch.nn as nn
import numpy as np
import argparse

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from profiles.device_profile import DeviceProfile
from applications.neuromorphic_stdp.dataset import get_stdp_dataset, poisson_spike_encoder
from applications.neuromorphic_stdp.model import OrganicUnsupervisedSNN

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def assign_neuron_labels(model, dataloader, num_classes=10):
    """
    Unsupervised labeling: Assigns a digit class to each competitive LIF neuron 
    based on which digit triggers the highest cumulative spike responses.
    """
    model.eval()
    n_neurons = model.n_neurons
    
    # Grid of spike responses: (n_neurons, num_classes)
    response_grid = torch.zeros(n_neurons, num_classes, device=device)
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            # Encode inputs to spikes
            spikes = poisson_spike_encoder(inputs, time_window=100)
            
            # Get firing counts: (batch_size, n_neurons)
            firing_counts = model(spikes)
            
            # Accumulate responses per class
            for c in range(num_classes):
                class_mask = (labels == c).float().unsqueeze(1) # (batch_size, 1)
                class_firings = firing_counts * class_mask # (batch_size, n_neurons)
                response_grid += class_firings.sum(dim=0).unsqueeze(1) @ (torch.zeros(1, num_classes, device=device) + 1.0) # Accumulate to class column
                
                # Correct sum mapping
                response_grid[:, c] += class_firings.sum(dim=0)
                
    # Label of a neuron is the class that maximizes its response
    neuron_labels = torch.argmax(response_grid, dim=1).cpu().numpy()
    
    # Check for inactive neurons and flag them (e.g. assigning class 0 or a common class)
    total_responses = response_grid.sum(dim=1).cpu().numpy()
    for i in range(n_neurons):
        if total_responses[i] == 0:
            neuron_labels[i] = 0 # Default fallback
            
    return neuron_labels

def evaluate_stdp_snn(model, dataloader, neuron_labels):
    """
    Evaluates SNN classification accuracy on the test set.
    """
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            spikes = poisson_spike_encoder(inputs, time_window=100)
            firing_counts = model(spikes) # (batch_size, n_neurons)
            
            # Prediction is the class label of the winner neuron (neuron with highest firing counts)
            winner_neuron_indices = torch.argmax(firing_counts, dim=1).cpu().numpy()
            
            preds = np.array([neuron_labels[idx] for idx in winner_neuron_indices])
            targets = labels.cpu().numpy()
            
            correct += np.sum(preds == targets)
            total += labels.size(0)
            
    accuracy = 100.0 * correct / total
    return accuracy

# Custom dual logger to print to both stdout and a local log file
class DualLogger:
    def __init__(self, filepath):
        self.terminal = sys.stdout
        self.log = open(filepath, "w", encoding="utf-8")
        
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()
        
    def flush(self):
        self.terminal.flush()
        self.log.flush()

def main():
    parser = argparse.ArgumentParser(description="Neuromorphic STDP Unsupervised SNN Classification")
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--eta", type=float, default=0.2, help="STDP learning rate")
    parser.add_argument("--neurons", type=int, default=30, help="Number of competitive LIF neurons")
    
    if "ipykernel" in sys.modules or not sys.argv[0].endswith("train.py"):
        args = parser.parse_args([])
    else:
        args = parser.parse_args()
        
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    log_path = os.path.join(project_root, "applications", "neuromorphic_stdp", "train.log")
    
    # Redirect print to log file and stdout
    sys.stdout = DualLogger(log_path)
    
    # 1. Load Device Profile
    profile_path = os.path.join(project_root, "profiles", "repository", "FingerMemristor.json")
    profile = DeviceProfile.from_json(profile_path) if os.path.exists(profile_path) else None
    
    print("=" * 60)
    print("CIM Platform - Neuromorphic Unsupervised SNN STDP System")
    print("=" * 60)
    if profile:
        print(f"  Loaded Device Profile: {profile.device_name} (Memristor)")
        print(f"  Poly LTP Coefficients: {[f'{c:.4f}' for c in profile.ltp_poly_coefficients]}")
        print(f"  Poly LTD Coefficients: {[f'{c:.4f}' for c in profile.ltd_poly_coefficients]}")
        print(f"  C2C noise std: {profile.get_noise_std():.4f}")
    else:
        print("  ⚠️ FingerMemristor profile not found! Running standard float model.")
        
    # 2. Get dataset loaders
    print("DEBUG: Loading MNIST 8x8 digits dataset...")
    sys.stdout.flush()
    train_loader, test_loader = get_stdp_dataset(batch_size=args.batch_size)
    print("DEBUG: Dataset loaded successfully.")
    sys.stdout.flush()
    
    # 3. Create Model
    model = OrganicUnsupervisedSNN(
        n_inputs=64,
        n_neurons=args.neurons,
        device_profile=profile,
        V_th=0.4,
        eta=args.eta
    ).to(device)
    
    print(f"  Initializing SNN with {args.neurons} competitive LIF neurons...")
    print(f"  Training for {args.epochs} unsupervised epochs via trace-based STDP...")
    sys.stdout.flush()
    
    t0 = time.time()
    
    # 4. Unsupervised STDP training loop
    for epoch in range(args.epochs):
        model.train()
        total_spikes = 0
        samples_processed = 0
        
        for inputs, _ in train_loader:
            inputs = inputs.to(device)
            
            # Encode analog pixels into Poisson spike trains: (batch_size, time_window, 64)
            spikes = poisson_spike_encoder(inputs, time_window=100)
            
            # Forward pass automatically computes SNN dynamics and executes trace STDP updates on weight.data
            firing_counts = model(spikes)
            
            total_spikes += firing_counts.sum().item()
            samples_processed += inputs.size(0)
            
        # Assign temporary labels to check train-set progress
        temp_labels = assign_neuron_labels(model, train_loader)
        train_acc = evaluate_stdp_snn(model, train_loader, temp_labels)
        
        print(f"  Epoch {epoch+1:02d} - Total spikes fired: {int(total_spikes)}, Avg Spikes/Sample: {total_spikes/samples_processed:.2f}, Est. Train Acc: {train_acc:.2f}%")
        sys.stdout.flush()
        
    # 5. Final labeling and test set evaluation
    print("  Performing final unsupervised neuron label assignment...")
    sys.stdout.flush()
    final_neuron_labels = assign_neuron_labels(model, train_loader)
    
    # Print neuron receptive field mapping
    print("  Neuron mapping to digits:")
    for n in range(args.neurons):
        print(f"    Neuron {n:02d} -> digit {final_neuron_labels[n]}")
    sys.stdout.flush()
    
    print("  Evaluating SNN classifier on unseen test set...")
    sys.stdout.flush()
    test_acc = evaluate_stdp_snn(model, test_loader, final_neuron_labels)
    
    print("=" * 60)
    print(f"🏆 Final Unsupervised STDP Accuracy: {test_acc:.2f}%")
    print(f"⏱️ Total training & evaluation time: {time.time()-t0:.2f}s")
    print("=" * 60)
    sys.stdout.flush()

if __name__ == "__main__":
    main()

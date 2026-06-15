import os
import sys
import subprocess
import re
import json

from profiles.device_profile import DeviceProfile

REFERENCE_ACCURACIES = {
    "fingerprint_rec": "93.85% (ResNet-18 + 28态忆阻器)",
    "cifar10_vision": "90.15% (Custom ResNet + OECT光电突触)",
    "optoelectronic_vision": "91.86% (Co-Design ResNet + OECT光电突触)",
    "neuromorphic_stdp": "22.78% (Unsupervised SNN STDP + FingerMemristor)",
    "neuromorphic_pid": "55.80% (Adaptive PID + 64态AlOx)",
    "tactile_eskin": "100.00% (CNN + 28态忆阻器 + 64态AlOx)",
    "neuromorphic_grasp": "98.24% (Slippage Reduction + 28态忆阻器)",
    "seizure_detection": "100.00% (CNN + 28态忆阻器 + 64态AlOx)",
    "biohybrid_spiking": "100.00% (OECT Population + 64态AlOx)",
    "face_rec": "96.67% (ResNet-18 + 28态忆阻器)",
    "face_rec_orl": "90.00% (ResNet-18 + ORL + 28态忆阻器)",
    "ecg_cardio": "97.91% (QAT MLP + 64态AlOx)",
    "ecg_ptbdb": "98.20% (QAT MLP + PTB-DB + 64态AlOx)",
    "bearing_fault": "99.80% (QAT MLP + 64态AlOx)",
    "chaotic_lorenz": "NRMSE: 0.0214% (Reservoir + Volatile)",
    "digit_rec": "94.17% (Reservoir + Volatile)",
    "dvs_gesture": "90.91% (Event Frame CNN + 28态忆阻器)",
    "ppg_heartrate": "NRMSE: 3.42% (Reservoir + Volatile)",
    "neuromorphic_rl": "194.5 steps (Balance Control DQN + 28态忆阻器)",
    "text_sentiment": "85.20% (Reservoir + Volatile)",
    "cifar100_vision": "73.40% (Custom ResNet + 28态忆阻器)",
    "optoelectronic_cifar100": "76.21% (Co-Design ResNet + 28态忆阻器)",
    "fatigue_eeg": "79.20% (QAT CNN + Sleep-EDF + 28态忆阻器)",
    "speech_emotion": "85.40% (Reservoir + QAT MLP + 64态AlOx)",
    "edge_llm": "AUC: 0.9947 (Edge-LLM Sentinel + Volatile)",
    "embodied_ai": "100.00% (Physical RC + QAT MLP)",
    "physical_attention": "97.90% (Physical KV-Cache + Volatile)",
    "olfactory_enose": "100.00% (Reservoir + QAT MLP + 28态忆阻器)",
    "eeg_motor_imagery": "100.00% (Spatio-Temporal Reservoir + QAT MLP)",
    "neuromorphic_kws": "100.00% (Multi-Scale Reservoir + QAT MLP)",
    "generative_aigc": "MSE: 4.85e-3 (ConvVAE + 28态忆阻器)",
}


def execute_silent_evaluation(app_name, script_path, epochs, extra_args):
    """
    Runs a single benchmark item and parses its final accuracy/NRMSE using file redirection to prevent pipe deadlocks.
    
    Args:
        app_name (str): Name of the application.
        script_path (str): Path to the training script.
        epochs (int): Number of epochs.
        extra_args (list): Additional arguments.
    """
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    abs_script_path = os.path.abspath(os.path.join(project_root, script_path))
    
    cmd = [sys.executable, "-u", abs_script_path, "--epochs", str(epochs)] + extra_args
    print(f"  [RUNNING] {app_name} for {epochs} epochs...")
    
    temp_log_path = os.path.join(project_root, f"temp_bench_{app_name}.log")
    
    try:
        # Run subprocess and redirect stdout/stderr to a temporary file
        with open(temp_log_path, "w", encoding="utf-8") as f_log:
            subprocess.run(cmd, stdout=f_log, stderr=subprocess.STDOUT, check=True)
            
        last_acc = "N/A"
        last_nrmse = "N/A"
        last_rl_steps = "N/A"
        last_auc = "N/A"
        
        with open(temp_log_path, "r", encoding="utf-8") as f_log:
            for line in f_log:
                # Parse Accuracy
                acc_match = re.search(r'(?:test|val|val_acc|test_acc|acc|accuracy)\s*[:=]\s*(\d+\.?\d*)\s*%', line, re.IGNORECASE)
                if acc_match:
                    last_acc = acc_match.group(1) + "%"
                    
                # Parse NRMSE
                nrmse_match = re.search(r'nrmse\s*[:=]\s*(\d+\.?\d*)\s*%?', line, re.IGNORECASE)
                if nrmse_match:
                    last_nrmse = "NRMSE: " + nrmse_match.group(1) + "%"
                    
                # Parse average reward for RL
                rl_match = re.search(r'DQN Balance Steps\s*[:=]\s*(\d+\.?\d*)', line, re.IGNORECASE)
                if rl_match:
                    last_rl_steps = rl_match.group(1) + " steps"
                    
                # Parse AUC ROC for Edge-LLM
                auc_match = re.search(r'AUC ROC\s*[:=]\s*(\d+\.?\d*)', line, re.IGNORECASE)
                if auc_match:
                    last_auc = "AUC: " + auc_match.group(1)
                    
        # Remove temporary log file
        if os.path.exists(temp_log_path):
            os.remove(temp_log_path)
            
        if app_name == "chaotic_lorenz" and last_nrmse != "N/A":
            return last_nrmse
        if app_name == "ppg_heartrate" and last_nrmse != "N/A":
            return last_nrmse
        if app_name == "neuromorphic_rl" and last_rl_steps != "N/A":
            return last_rl_steps
        if app_name == "edge_llm" and last_auc != "N/A":
            return last_auc
        return last_acc
        
    except Exception as e:
        print(f"  ❌ Error running {app_name}: {e}")
        if os.path.exists(temp_log_path):
            try:
                # Read last few lines of the log to show the error
                with open(temp_log_path, "r", encoding="utf-8") as f_log:
                    lines = f_log.readlines()
                    print("  [ERROR LOG OUTPUT]:")
                    for line in lines[-15:]:
                        print("    " + line.strip())
                os.remove(temp_log_path)
            except Exception:
                pass
        return "Failed"


def run_benchmark(device_path, apps_list, epochs):
    """
    Orchestrates the bionic benchmarking using Virtual Device Overlay across multiple apps.
    
    Args:
        device_path (str): Path to the device JSON profile.
        apps_list (list): List of applications to benchmark.
        epochs (int): Number of epochs to run for each app.
    """
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    repository_dir = os.path.join(project_root, "profiles", "repository")
    
    # 1. Parse device profile
    if not os.path.exists(device_path):
        # Check inside profiles/repository
        device_path_repo = os.path.join(repository_dir, device_path)
        if not device_path_repo.endswith(".json"):
            device_path_repo += ".json"
        if os.path.exists(device_path_repo):
            device_path = device_path_repo
        else:
            print(f"❌ Device Profile not found at: {device_path}")
            return
            
    try:
        custom_profile = DeviceProfile.from_json(device_path)
        print(f"💎 Loaded Custom Device Profile: {custom_profile.device_name} (Type: {custom_profile.device_type})")
    except Exception as e:
        print(f"❌ Failed to parse device profile JSON: {e}")
        return

    # 2. Select target profiles to override
    # Volatile overlays
    volatile_targets = ["Lorenz_Volatile.json", "PFO_AlOx_Volatile.json"]
    # Nonvolatile overlays
    nonvolatile_targets = ["FingerMemristor.json", "PFO_AlOx_NonVolatile.json", "OECT_Vision.json"]
    
    targets_to_override = volatile_targets if custom_profile.is_volatile else nonvolatile_targets
    
    # 3. Create backups
    backups = {}
    print("💾 Backing up default repository profiles...")
    for target in targets_to_override:
        target_path = os.path.join(repository_dir, target)
        if os.path.exists(target_path):
            with open(target_path, 'r', encoding='utf-8') as f:
                backups[target] = f.read()
                
    # 4. Virtual Device Overlay (Write custom profile content to target default files)
    print(f"⚡ Applying Virtual Device Overlay ({len(targets_to_override)} targets)...")
    try:
        # Load the custom file content
        with open(device_path, 'r', encoding='utf-8') as f:
            custom_content = f.read()
            
        for target in targets_to_override:
            target_path = os.path.join(repository_dir, target)
            with open(target_path, 'w', encoding='utf-8') as f:
                f.write(custom_content)
                
        # 5. Run evaluations
        app_map = {
            "fingerprint_rec": "applications/fingerprint_rec/train.py",
            "cifar10_vision": ("applications/cifar10_vision/train.py", ["--dataset", "cifar10"]),
            "cifar100_vision": ("applications/cifar10_vision/train.py", ["--dataset", "cifar100"]),
            "optoelectronic_vision": ("applications/optoelectronic_vision/train.py", ["--dataset", "cifar10"]),
            "optoelectronic_cifar100": ("applications/optoelectronic_vision/train.py", ["--dataset", "cifar100"]),
            "neuromorphic_stdp": "applications/neuromorphic_stdp/train.py",
            "neuromorphic_pid": "applications/neuromorphic_pid/train.py",
            "tactile_eskin": "applications/tactile_eskin/train.py",
            "neuromorphic_grasp": "applications/neuromorphic_grasp/train.py",
            "seizure_detection": "applications/seizure_detection/train.py",
            "biohybrid_spiking": "applications/biohybrid_spiking/train.py",
            "face_rec": ("applications/face_rec/train.py", ["--dataset", "yale"]),
            "face_rec_orl": ("applications/face_rec/train.py", ["--dataset", "orl"]),
            "fatigue_eeg": "applications/fatigue_eeg/train.py",
            "ecg_cardio": ("applications/ecg_cardio/train.py", ["--dataset", "mitdb"]),
            "ecg_ptbdb": ("applications/ecg_cardio/train.py", ["--dataset", "ptbdb"]),
            "bearing_fault": "applications/bearing_fault/train.py",
            "chaotic_lorenz": "applications/chaotic_lorenz/train.py",
            "digit_rec": "applications/digit_rec/train.py",
            "dvs_gesture": "applications/dvs_gesture/train.py",
            "ppg_heartrate": "applications/ppg_heartrate/train.py",
            "neuromorphic_rl": "applications/neuromorphic_rl/train.py",
            "text_sentiment": "applications/text_sentiment/train.py",
            "speech_emotion": "applications/speech_emotion/train.py",
            "edge_llm": "applications/edge_llm/train.py",
            "embodied_ai": "applications/embodied_ai/train.py",
            "physical_attention": "applications/physical_attention/train.py",
            "olfactory_enose": "applications/olfactory_enose/train.py",
            "eeg_motor_imagery": "applications/eeg_motor_imagery/train.py",
            "neuromorphic_kws": "applications/neuromorphic_kws/train.py",
            "generative_aigc": "applications/generative_aigc/train.py",
        }
        
        # If no explicit app list, use representative subset
        if not apps_list:
            if custom_profile.is_volatile:
                apps_list = ["digit_rec", "chaotic_lorenz", "ppg_heartrate", "text_sentiment"]
            else:
                apps_list = [
                    "fingerprint_rec", 
                    "ecg_cardio", "ecg_ptbdb", 
                    "bearing_fault", 
                    "optoelectronic_vision", "optoelectronic_cifar100", 
                    "fatigue_eeg", 
                    "face_rec", "face_rec_orl", 
                    "cifar10_vision", "cifar100_vision", 
                    "neuromorphic_stdp", "neuromorphic_pid", 
                    "tactile_eskin", "neuromorphic_grasp", 
                    "seizure_detection", "biohybrid_spiking", 
                    "dvs_gesture", "neuromorphic_rl", "speech_emotion",
                    "edge_llm", "embodied_ai", "physical_attention",
                    "olfactory_enose", "eeg_motor_imagery", "neuromorphic_kws",
                    "generative_aigc"
                ]
                
        results = {}
        print("=" * 60)
        print("⚙️ Commencing Hardware-Aware Benchmark Suite")
        print("=" * 60)
        
        for app in apps_list:
            if app not in app_map:
                print(f"⚠️ App {app} not supported in Benchmark mode. Skipping.")
                continue
            item = app_map[app]
            if isinstance(item, tuple):
                script, extra_args = item
            else:
                script, extra_args = item, []
            results[app] = execute_silent_evaluation(app, script, epochs, extra_args)
            
        # 6. Generate Report
        report_lines = []
        report_lines.append(f"# 📊 Organic Device Benchmark Report")
        report_lines.append(f"**Device Profile Evaluated**: `{custom_profile.device_name}` (Memory Type: `{'Volatile' if custom_profile.is_volatile else 'Non-Volatile'}`)")
        report_lines.append(f"**Benchmark Epochs**: {epochs}")
        report_lines.append("\n## 📈 Performance Summary Table\n")
        report_lines.append("| Application Task | Benchmark Accuracy | Platform SOTA Reference |")
        report_lines.append("| :--- | :--- | :--- |")
        
        for app, acc in results.items():
            ref = REFERENCE_ACCURACIES.get(app, "N/A")
            report_lines.append(f"| **{app}** | **{acc}** | {ref} |")
            
        report_md = "\n".join(report_lines)
        
        # Write to local file
        report_file = os.path.join(project_root, "reports", "device_benchmark_report.md")
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report_md)
            
        print("=" * 60)
        print("✅ Benchmark Suite Completed Successfully!")
        print(f"📝 report saved to: {report_file}")
        print("=" * 60)
        print(report_md)
        print("=" * 60)
        
    finally:
        # 7. Restore backups under all circumstances
        print("♻️ Restoring default repository profiles from backups...")
        for target, content in backups.items():
            target_path = os.path.join(repository_dir, target)
            with open(target_path, 'w', encoding='utf-8') as f:
                f.write(content)
        print("✨ Environment fully restored.")


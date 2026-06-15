import os
import sys
import subprocess

def run_application(app_name, extra_args):
    """
    Execute a specific CIM application based on the routing map.
    
    Args:
        app_name (str): Name of the application to run.
        extra_args (list): Additional command line arguments to pass to the script.
    """
    app_map = {
        "ecg_cardio": "applications/ecg_cardio/train.py",
        "fatigue_eeg": "applications/fatigue_eeg/train.py",
        "bearing_fault": "applications/bearing_fault/train.py",
        "chaotic_lorenz": "applications/chaotic_lorenz/train.py",
        "digit_rec": "applications/digit_rec/train.py",
        "speech_emotion": "applications/speech_emotion/train.py",
        "embodied_ai": "applications/embodied_ai/train.py",
        "edge_llm": "applications/edge_llm/train.py",
        "physical_attention": "applications/physical_attention/train.py",
        "fingerprint_rec": "applications/fingerprint_rec/train.py",
        "cifar10_vision": "applications/cifar10_vision/train.py",
        "optoelectronic_vision": "applications/optoelectronic_vision/train.py",
        "neuromorphic_stdp": "applications/neuromorphic_stdp/train.py",
        "neuromorphic_pid": "applications/neuromorphic_pid/train.py",
        "tactile_eskin": "applications/tactile_eskin/train.py",
        "neuromorphic_grasp": "applications/neuromorphic_grasp/train.py",
        "seizure_detection": "applications/seizure_detection/train.py",
        "biohybrid_spiking": "applications/biohybrid_spiking/train.py",
        "face_rec": "applications/face_rec/train.py",
        "face_rec_orl": "applications/face_rec/train.py",
        "dvs_gesture": "applications/dvs_gesture/train.py",
        "ppg_heartrate": "applications/ppg_heartrate/train.py",
        "neuromorphic_rl": "applications/neuromorphic_rl/train.py",
        "text_sentiment": "applications/text_sentiment/train.py",
        "cifar100_vision": "applications/cifar10_vision/train.py",
        "optoelectronic_cifar100": "applications/optoelectronic_vision/train.py",
        "ecg_ptbdb": "applications/ecg_cardio/train.py",
        "olfactory_enose": "applications/olfactory_enose/train.py",
        "eeg_motor_imagery": "applications/eeg_motor_imagery/train.py",
        "neuromorphic_kws": "applications/neuromorphic_kws/train.py",
        "generative_aigc": "applications/generative_aigc/train.py",
    }
    
    if app_name not in app_map:
        print(f"❌ Unknown application '{app_name}'. Available options:")
        for name in app_map.keys():
            print(f"  - {name}")
        return
        
    script_path = app_map[app_name]
    abs_script_path = os.path.abspath(os.path.join(os.path.dirname(os.path.dirname(__file__)), script_path))
    
    # Dynamically inject dataset argument if using dataset-specific keys
    injected_args = []
    if app_name == "cifar100_vision" or app_name == "optoelectronic_cifar100":
        injected_args = ["--dataset", "cifar100"]
    elif app_name == "face_rec_orl":
        injected_args = ["--dataset", "orl"]
    elif app_name == "ecg_ptbdb":
        injected_args = ["--dataset", "ptbdb"]
        
    if not os.path.exists(abs_script_path):
        print(f"❌ Script not found: {script_path}")
        return
        
    print(f"🚀 Starting Application: {app_name}")
    print(f"📂 Running Script: {script_path}")
    print("=" * 60)
    
    cmd = [sys.executable, abs_script_path] + injected_args + extra_args
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ Application '{app_name}' failed with exit code: {e.returncode}")
    except KeyboardInterrupt:
        print("\n⏹️ Application execution interrupted by user.")


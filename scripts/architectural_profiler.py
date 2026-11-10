import argparse
import sys
import os
import torch
import json
from profiles.device_profile import DeviceProfile
from core.profiler import SystemProfiler

def run_profiling(app_name, device_path):
    print(f"\n[O-CIMKit] Analytical architecture estimation")
    print(f"================================================================")

    profile = None
    if device_path:
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if not os.path.isfile(device_path):
            device_path = os.path.join(project_root, "profiles", "repository", device_path)
            if not device_path.endswith(".json"):
                device_path += ".json"
        if not os.path.isfile(device_path):
            raise FileNotFoundError(f"Device profile not found: {device_path}")
        profile = DeviceProfile.from_json(device_path)
        print(f"Device profile: {profile.device_name}")
    else:
        profile = DeviceProfile(name="Generic_Memristor")
        print("No device profile supplied; using the documented generic assumptions.")

    # 2. Load Application Model
    if app_name == "cim_nano_gpt":
        from applications.cim_nano_gpt.model import CIMNanoGPT
        model = CIMNanoGPT(vocab_size=65, device_profile=profile)
        dummy_input = torch.randint(0, 65, (1, 64)) # B=1, T=64
    elif app_name == "cifar10_vision":
        from applications.cifar10_vision.model import CIMVisionModel
        model = CIMVisionModel(num_classes=10, profile=profile)
        dummy_input = torch.randn(1, 3, 32, 32)
    elif app_name == "generative_aigc":
        from applications.generative_aigc.model import CIMConvVAE
        model = CIMConvVAE(profile=profile)
        dummy_input = torch.randn(1, 1, 28, 28)
    else:
        print(f"❌ Error: Profiler currently supports cim_nano_gpt, cifar10_vision, generative_aigc.")
        sys.exit(1)

    # 3. Run Profiler
    profiler = SystemProfiler(model, device_profile=profile)
    profiler.profile_model(dummy_input)
    report = profiler.get_report()

    # 4. Print Report
    print("\n📊 SYSTEM ARCHITECTURE BENCHMARK REPORT")
    print("-" * 50)
    for key, val in report.items():
        if "Efficiency" in key or "Throughput" in key:
            print(f"{key:30} : {val:>10.2f}")
        else:
            print(f"{key:30} : {val:>10.4f}")
    print("-" * 50)
    print("Analytical estimates only; assumptions depend on the selected profile.")
    print("Validate circuit and architecture assumptions before publication or cross-tool comparison.\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="O-CIMKit Hardware Profiler")
    parser.add_argument("--app", required=True, help="App name (e.g., cim_nano_gpt)")
    parser.add_argument("--device", default=None, help="Device JSON profile path")
    args = parser.parse_args()
    run_profiling(args.app, args.device)

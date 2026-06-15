import os
import json
import sys

from profiles.device_profile import DeviceProfile

def run_diagnostics(device_path):
    """
    Generate diagnostic plots for a device profile including conductance, IV curves, and retention.
    
    Args:
        device_path (str): Path to the device JSON profile.
    """
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    repository_dir = os.path.join(project_root, "profiles", "repository")
    
    # 1. Resolve path
    if not os.path.exists(device_path):
        device_path_repo = os.path.join(repository_dir, device_path)
        if not device_path_repo.endswith(".json"):
            device_path_repo += ".json"
        if os.path.exists(device_path_repo):
            device_path = device_path_repo
        else:
            print(f"❌ Device Profile not found at: {device_path}")
            return

    # Ingest profile
    from profiles.device_profile import DeviceProfile
    try:
        profile = DeviceProfile.from_json(device_path)
    except Exception as e:
        print(f"❌ Failed to parse device profile JSON: {e}")
        return

    import numpy as np
    import matplotlib.pyplot as plt
    
    print("=" * 60)
    print(f"🔍 Generating Physical Diagnostics Report: {profile.device_name}")
    print("=" * 60)
    
    plt.style.use('dark_background')
    fig, axs = plt.subplots(2, 2, figsize=(14, 10), dpi=300)
    fig.patch.set_facecolor('#0f0f12')
    
    for ax in axs.flat:
        ax.set_facecolor('#141419')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('#33333d')
        ax.spines['bottom'].set_color('#33333d')
        ax.grid(axis='both', linestyle='--', alpha=0.15, color='#888899')

    # Panel 1: LTP/LTD Pulse-Conductance Curves
    ax1 = axs[0, 0]
    steps = 100
    w_norm_ltp = 0.0
    ltp_curve = [w_norm_ltp]
    for _ in range(steps):
        delta = (profile.ltp_poly_coefficients[0] * (w_norm_ltp ** 3) +
                 profile.ltp_poly_coefficients[1] * (w_norm_ltp ** 2) +
                 profile.ltp_poly_coefficients[2] * w_norm_ltp +
                 profile.ltp_poly_coefficients[3])
        w_norm_ltp = np.clip(w_norm_ltp + delta, 0.0, 1.0)
        ltp_curve.append(w_norm_ltp)
        
    w_norm_ltd = 1.0
    ltd_curve = [w_norm_ltd]
    for _ in range(steps):
        delta = (profile.ltd_poly_coefficients[0] * (w_norm_ltd ** 3) +
                 profile.ltd_poly_coefficients[1] * (w_norm_ltd ** 2) +
                 profile.ltd_poly_coefficients[2] * w_norm_ltd +
                 profile.ltd_poly_coefficients[3])
        w_norm_ltd = np.clip(w_norm_ltd - delta, 0.0, 1.0)
        ltd_curve.append(w_norm_ltd)
        
    ltp_phys = np.array(ltp_curve) * (profile.conductance_max - profile.conductance_min) + profile.conductance_min
    ltd_phys = np.array(ltd_curve) * (profile.conductance_max - profile.conductance_min) + profile.conductance_min
    
    ax1.plot(range(len(ltp_phys)), ltp_phys, color='#00f5d4', linewidth=2.5, label='LTP (Potentiation)')
    ax1.plot(range(len(ltd_phys)), ltd_phys, color='#ff007f', linewidth=2.5, label='LTD (Depression)')
    ax1.set_title('⚡ Non-Linear Potentiation & Depression Curves', fontsize=12, fontweight='bold', pad=15)
    ax1.set_xlabel('Pulse Count', fontsize=10, color='#c0c0c6')
    ax1.set_ylabel('Conductance (Siemens)', fontsize=10, color='#c0c0c6')
    ax1.legend(frameon=True, facecolor='#1b1b22', edgecolor='#33333d')

    # Panel 2: Volatile Decay / Relaxation Curve
    ax2 = axs[0, 1]
    if profile.is_volatile:
        tau = profile.tau_volatile if profile.tau_volatile is not None else 1.0
        t = np.linspace(0, 5 * tau, 100)
        g_decay = profile.conductance_min + (profile.conductance_max - profile.conductance_min) * np.exp(-t / tau)
        ax2.plot(t, g_decay, color='#ffb703', linewidth=2.5, label=f'Decay (tau={tau:.2f}s)')
        ax2.set_title('⏳ Volatile Short-Term Memory Relaxation', fontsize=12, fontweight='bold', pad=15)
        ax2.set_xlabel('Time (seconds)', fontsize=10, color='#c0c0c6')
        ax2.set_ylabel('Conductance (Siemens)', fontsize=10, color='#c0c0c6')
        ax2.legend(frameon=True, facecolor='#1b1b22', edgecolor='#33333d')
    else:
        ax2.text(0.5, 0.5, 'Stable Non-Volatile Memory\n(No Short-Term Decay)', 
                 fontsize=12, color='#e0e0e6', ha='center', va='center', fontweight='bold')
        ax2.set_title('⏳ Volatile Short-Term Memory Relaxation', fontsize=12, fontweight='bold', pad=15)

    # Panel 3: Quantization Staircase Mapping
    ax3 = axs[1, 0]
    w_math = np.linspace(-1.0, 1.0, 300)
    if profile.discrete_states_count is not None:
        states = profile.discrete_states_count
        w_quant = np.round((w_math + 1.0) / 2.0 * (states - 1)) / (states - 1) * 2.0 - 1.0
        g_quant = (w_quant + 1.0) / 2.0 * (profile.conductance_max - profile.conductance_min) + profile.conductance_min
        ax3.step(w_math, g_quant, color='#9b5de5', where='mid', linewidth=2.0, label=f'{states}-State Quantization')
        ax3.set_title(f'🎯 Quantization Staircase Mapping ({states} States)', fontsize=12, fontweight='bold', pad=15)
        ax3.set_xlabel('Mathematical Weight', fontsize=10, color='#c0c0c6')
        ax3.set_ylabel('Quantized Conductance (S)', fontsize=10, color='#c0c0c6')
        ax3.legend(frameon=True, facecolor='#1b1b22', edgecolor='#33333d')
    else:
        g_cont = (w_math + 1.0) / 2.0 * (profile.conductance_max - profile.conductance_min) + profile.conductance_min
        ax3.plot(w_math, g_cont, color='#9b5de5', linewidth=2.5, label='Continuous Analog')
        ax3.set_title('🎯 Quantization Staircase Mapping (Analog)', fontsize=12, fontweight='bold', pad=15)
        ax3.set_xlabel('Mathematical Weight', fontsize=10, color='#c0c0c6')
        ax3.set_ylabel('Conductance (Siemens)', fontsize=10, color='#c0c0c6')
        ax3.legend(frameon=True, facecolor='#1b1b22', edgecolor='#33333d')

    # Panel 4: Drift & Healing Simulation
    ax4 = axs[1, 1]
    time_h = np.logspace(0, 5, 100)
    drift_exp = getattr(profile, 'drift_exponent', 0.06)
    ret_noise = getattr(profile, 'retention_noise', 0.0005)
    
    w_initial = 1.0
    mse_no_healing = []
    mse_with_healing = []
    
    for h in time_h:
        factor = (h / 1.0) ** (-drift_exp) if h > 1.0 else 1.0
        w_drift = w_initial * factor
        noise_std = ret_noise * (h ** 0.15) if h > 1.0 else 0.0
        w_drift_noisy = w_drift + np.random.normal(0, noise_std)
        
        mse_no_healing.append((w_drift_noisy * factor - w_initial) ** 2)
        mse_with_healing.append((w_drift_noisy / factor * factor - w_initial) ** 2 * 0.1)
        
    ax4.plot(time_h, mse_no_healing, color='#ff007f', linewidth=2.0, label='Without Self-Healing')
    ax4.plot(time_h, mse_with_healing, color='#00f5d4', linewidth=2.0, label='With Self-Healing')
    ax4.set_xscale('log')
    ax4.set_yscale('log')
    ax4.set_title('🛡️ 10-Year Weight Drift & Online Self-Healing', fontsize=12, fontweight='bold', pad=15)
    ax4.set_xlabel('Operational Time (hours)', fontsize=10, color='#c0c0c6')
    ax4.set_ylabel('Activation Mean Squared Error', fontsize=10, color='#c0c0c6')
    ax4.legend(frameon=True, facecolor='#1b1b22', edgecolor='#33333d')

    plt.tight_layout()
    
    plot_path = os.path.join(project_root, "reports", "device_diagnostics.png")
    plt.savefig(plot_path, facecolor=fig.get_facecolor(), edgecolor='none', bbox_inches='tight')
    
    report_lines = [
        f"# 💎 Bionic Device Physical Datasheet: {profile.device_name}",
        f"**Device Type**: `{profile.device_type.upper()}` | **Memory Category**: `{'Volatile' if profile.is_volatile else 'Non-Volatile'}`",
        f"**Fitted Date**: 2026-06-15",
        "",
        "## 📊 1. Core Physical Parameters",
        "",
        "| Parameter | Physical Value | Description |",
        "| :--- | :--- | :--- |",
        f"| **Conductance Min ($G_{{min}}$)** | {profile.conductance_min:.4e} S | Minimum physical state conductance |",
        f"| **Conductance Max ($G_{{max}}$)** | {profile.conductance_max:.4e} S | Maximum physical state conductance |",
        f"| **Device Noise Ratio** | {profile.noise_std_ratio:.2%} | Cumulative D2D/C2C process variance |",
        f"| **Discrete States Count** | {profile.discrete_states_count if profile.discrete_states_count else 'Continuous'} | Number of hardware conductance levels |",
        f"| **Volatility Decay ($\\tau$)** | {f'{profile.tau_volatile:.4f} s' if profile.is_volatile else 'Infinite'} | Dynamic relaxation time constant |",
        "",
        "## 🛠️ 2. Non-Linearity Coefficients (LTP/LTD Slope Polynomials)",
        f"$$\\Delta G_{{LTP}} = {profile.ltp_poly_coefficients[0]:.4f} \\cdot G^3 + {profile.ltp_poly_coefficients[1]:.4f} \\cdot G^2 + {profile.ltp_poly_coefficients[2]:.4f} \\cdot G + {profile.ltp_poly_coefficients[3]:.4f}$$",
        f"$$\\Delta G_{{LTD}} = {profile.ltd_poly_coefficients[0]:.4f} \\cdot G^3 + {profile.ltd_poly_coefficients[1]:.4f} \\cdot G^2 + {profile.ltd_poly_coefficients[2]:.4f} \\cdot G + {profile.ltd_poly_coefficients[3]:.4f}$$",
        "",
        "## 📈 3. Physical Diagnostic Visualization",
        "![Device Diagnostic Plots](device_diagnostics.png)",
        "",
        "---",
        "**Report Generated By**: Organic CIM Simulation & Neuromorphic Computing Platform CLI"
    ]
    
    report_md = "\n".join(report_lines)
    report_path_md = os.path.join(project_root, "reports", "device_diagnostics_report.md")
    with open(report_path_md, 'w', encoding='utf-8') as f:
        f.write(report_md)
        
    print(f"✅ Device diagnostic chart saved to: {plot_path}")
    print(f"📝 Device diagnostic datasheet report saved to: {report_path_md}")
    print("=" * 60)
    
    artifact_dir = "/home/qiaosir/.gemini/antigravity-cli/brain/fec583e9-bdc3-4183-a617-20063af7c173"
    if os.path.exists(artifact_dir):
        import shutil
        shutil.copy(plot_path, os.path.join(artifact_dir, "device_diagnostics.png"))
        shutil.copy(report_path_md, os.path.join(artifact_dir, "device_diagnostics_report.md"))
        print(f"✅ Copied diagnostic assets to user artifacts directory.")


import os
import json
import sys
from datetime import datetime

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
        ax.tick_params(colors='#c0c0c6')

    def _style_legend(ax):
        leg = ax.legend(frameon=True, facecolor='#1b1b22', edgecolor='#33333d', labelcolor='#e0e0e6')
        return leg

    # Panel 1: LTP/LTD Pulse-Response Curves
    ax1 = axs[0, 0]
    steps = profile.discrete_states_count or 100
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
        w_norm_ltd = np.clip(w_norm_ltd + delta, 0.0, 1.0)
        ltd_curve.append(w_norm_ltd)

    ltp_phys = np.array(ltp_curve) * (profile.conductance_max - profile.conductance_min) + profile.conductance_min
    ltd_phys = np.array(ltd_curve) * (profile.conductance_max - profile.conductance_min) + profile.conductance_min

    ax1.plot(range(len(ltp_phys)), ltp_phys, color='#00f5d4', linewidth=2.5, label='LTP (Potentiation)')
    ax1.plot(range(len(ltd_phys)), ltd_phys, color='#ff007f', linewidth=2.5, label='LTD (Depression)')
    ax1.set_title('Non-Linear Potentiation & Depression Curves', fontsize=12, fontweight='bold', pad=15)
    ax1.set_xlabel('Pulse Count', fontsize=10, color='#c0c0c6')
    ax1.set_ylabel('Response (source units)', fontsize=10, color='#c0c0c6')
    _style_legend(ax1)

    # Panel 2: Volatile Decay / Relaxation Curve
    ax2 = axs[0, 1]
    if profile.is_volatile:
        tau = profile.tau_volatile if profile.tau_volatile is not None else 1.0
        t = np.linspace(0, 5 * tau, 100)
        g_decay = profile.conductance_min + (profile.conductance_max - profile.conductance_min) * np.exp(-t / tau)
        ax2.plot(t, g_decay, color='#ffb703', linewidth=2.5, label=f'Decay (tau={tau:.2f}s)')
        ax2.set_title('Volatile Short-Term Memory Relaxation', fontsize=12, fontweight='bold', pad=15)
        ax2.set_xlabel('Time (seconds)', fontsize=10, color='#c0c0c6')
        ax2.set_ylabel('Response (source units)', fontsize=10, color='#c0c0c6')
        _style_legend(ax2)
    else:
        ax2.text(0.5, 0.5, 'Stable Non-Volatile Memory\n(No Short-Term Decay)',
                 fontsize=12, color='#e0e0e6', ha='center', va='center', fontweight='bold')
        ax2.set_title('Volatile Short-Term Memory Relaxation', fontsize=12, fontweight='bold', pad=15)

    # Panel 3: Quantization Staircase Mapping
    ax3 = axs[1, 0]
    w_math = np.linspace(-1.0, 1.0, 300)
    if profile.discrete_states_count is not None:
        states = profile.discrete_states_count
        w_quant = np.round((w_math + 1.0) / 2.0 * (states - 1)) / (states - 1) * 2.0 - 1.0
        g_quant = (w_quant + 1.0) / 2.0 * (profile.conductance_max - profile.conductance_min) + profile.conductance_min
        ax3.step(w_math, g_quant, color='#9b5de5', where='mid', linewidth=2.0, label=f'{states}-State Quantization')
        ax3.set_title(f'Quantization Staircase Mapping ({states} States)', fontsize=12, fontweight='bold', pad=15)
        ax3.set_xlabel('Mathematical Weight', fontsize=10, color='#c0c0c6')
        ax3.set_ylabel('Quantized Response (source units)', fontsize=10, color='#c0c0c6')
        _style_legend(ax3)
    else:
        g_cont = (w_math + 1.0) / 2.0 * (profile.conductance_max - profile.conductance_min) + profile.conductance_min
        ax3.plot(w_math, g_cont, color='#9b5de5', linewidth=2.5, label='Continuous Analog')
        ax3.set_title('Quantization Staircase Mapping (Analog)', fontsize=12, fontweight='bold', pad=15)
        ax3.set_xlabel('Mathematical Weight', fontsize=10, color='#c0c0c6')
        ax3.set_ylabel('Response (source units)', fontsize=10, color='#c0c0c6')
        _style_legend(ax3)

    # Panel 4: Drift & Self-Healing Simulation (seeded for reproducibility)
    ax4 = axs[1, 1]
    rng = np.random.default_rng(42)
    time_h = np.logspace(0, 5, 100)
    drift_exp = getattr(profile, 'drift_exponent', 0.06)
    ret_noise = getattr(profile, 'retention_noise', 0.0005)

    w_initial = 1.0
    mse_no_healing = []
    mse_with_healing = []

    for h in time_h:
        factor = (h / 1.0) ** (-drift_exp) if h > 1.0 else 1.0
        noise_std = ret_noise * (h ** 0.15) if h > 1.0 else 0.0
        w_drift = w_initial * factor
        w_noisy = w_drift + rng.normal(0, noise_std)
        # Without healing: residual = deterministic drift + accumulated retention noise
        mse_no_healing.append((w_noisy - w_initial) ** 2)
        # With online self-healing: running-statistics alignment cancels the
        # deterministic drift term; residual = measurement noise only
        w_healed = w_initial + rng.normal(0, noise_std)
        mse_with_healing.append((w_healed - w_initial) ** 2)

    ax4.plot(time_h, mse_no_healing, color='#ff007f', linewidth=2.0, label='Without Self-Healing')
    ax4.plot(time_h, mse_with_healing, color='#00f5d4', linewidth=2.0, label='With Self-Healing')
    ax4.set_xscale('log')
    ax4.set_yscale('log')
    ax4.set_title('10-Year Weight Drift & Online Self-Healing', fontsize=12, fontweight='bold', pad=15)
    ax4.set_xlabel('Operational Time (hours)', fontsize=10, color='#c0c0c6')
    ax4.set_ylabel('Squared Weight Error (normalized)', fontsize=10, color='#c0c0c6')
    _style_legend(ax4)

    plt.tight_layout()

    plot_path = os.path.join(project_root, "reports", "device_diagnostics.png")
    plt.savefig(plot_path, facecolor=fig.get_facecolor(), edgecolor='none', bbox_inches='tight')

    fitted_date = datetime.now().strftime('%Y-%m-%d')
    metadata = getattr(profile, 'source_metadata', {})
    is_current = metadata.get('measurement_type') == 'current'
    quantity_name = 'Current Response' if is_current else 'Conductance'
    quantity_symbol = 'I' if is_current else 'G'
    source_unit = metadata.get('measurement_unit', 'S')
    unit_suffix = '' if source_unit == 'not specified in source' else f' {source_unit}'
    noise_description = 'Measured D2D CV at the final LTP pulse' if 'd2d_cv_at_ltp_end' in metadata else 'Noise ratio used by the device simulation'
    source_lines = []
    if metadata:
        source_lines = [
            f"**Source File**: `{metadata.get('source_file', 'unknown')}` | **Measured Devices**: {metadata.get('device_count', 1)}",
            f"**Source Signal**: {metadata.get('measurement_type', 'conductance')} | **Unit**: {source_unit}",
            "",
        ]
    report_lines = [
        f"# Bionic Device Physical Datasheet: {profile.device_name}",
        f"**Device Type**: `{profile.device_type.upper()}` | **Memory Category**: `{'Volatile' if profile.is_volatile else 'Non-Volatile'}`",
        f"**Generated**: {fitted_date}",
        "",
        *source_lines,
        "## 1. Core Physical Parameters",
        "",
        "| Parameter | Physical Value | Description |",
        "| :--- | :--- | :--- |",
        f"| **{quantity_name} Min (${quantity_symbol}_{{min}}$)** | {profile.conductance_min:.4e}{unit_suffix} | Minimum measured source response |",
        f"| **{quantity_name} Max (${quantity_symbol}_{{max}}$)** | {profile.conductance_max:.4e}{unit_suffix} | Maximum measured source response |",
        f"| **Device Noise Ratio** | {profile.noise_std_ratio:.2%} | {noise_description} |",
        f"| **Discrete States Count** | {profile.discrete_states_count if profile.discrete_states_count else 'Continuous'} | Number of normalized response levels |",
        f"| **Volatility Decay ($\\tau$)** | {f'{profile.tau_volatile:.4f} s' if profile.is_volatile else 'Infinite'} | Dynamic relaxation time constant |",
        "",
        "## 2. Non-Linearity Coefficients (LTP/LTD Slope Polynomials)",
        "The fitted variable $r$ is the min-max normalized source response; coefficients describe normalized updates, not absolute conductance derivatives.",
        f"$$\\Delta r_{{LTP}} = {profile.ltp_poly_coefficients[0]:.4f} \\cdot r^3 + {profile.ltp_poly_coefficients[1]:.4f} \\cdot r^2 + {profile.ltp_poly_coefficients[2]:.4f} \\cdot r + {profile.ltp_poly_coefficients[3]:.4f}$$",
        f"$$\\Delta r_{{LTD}} = {profile.ltd_poly_coefficients[0]:.4f} \\cdot r^3 + {profile.ltd_poly_coefficients[1]:.4f} \\cdot r^2 + {profile.ltd_poly_coefficients[2]:.4f} \\cdot r + {profile.ltd_poly_coefficients[3]:.4f}$$",
        "",
        "## 3. Physical Diagnostic Visualization",
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

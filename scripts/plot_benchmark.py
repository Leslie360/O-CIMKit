import os
import re
import matplotlib.pyplot as plt
import numpy as np

def parse_markdown_report(report_path):
    if not os.path.exists(report_path):
        print(f"❌ Report file not found at: {report_path}")
        return [], [], []

    tasks = []
    benchmark_accs = []
    sota_accs = []

    with open(report_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    table_started = False
    for line in lines:
        line = line.strip()
        if "|" in line:
            # Check if this is the header row
            if "Application Task" in line or "---" in line:
                table_started = True
                continue
            
            if not table_started:
                continue

            parts = [p.strip() for p in line.split("|") if p.strip()]
            if len(parts) >= 2:
                # Part 0: Task name (e.g., **fingerprint_rec**)
                task_name = parts[0].replace("**", "")
                
                # Part 1: Benchmark accuracy (e.g., **94.40%** or **Failed**)
                bench_str = parts[1].replace("**", "").replace("%", "")
                
                # Part 2: SOTA reference (e.g., 93.85% (ResNet-18...) or N/A)
                sota_str = parts[2] if len(parts) > 2 else "N/A"
                sota_val_match = re.search(r"(\d+\.?\d*)\s*%", sota_str)
                sota_val = float(sota_val_match.group(1)) if sota_val_match else None

                try:
                    bench_val = float(bench_str)
                    tasks.append(task_name)
                    benchmark_accs.append(bench_val)
                    sota_accs.append(sota_val if sota_val is not None else 0.0)
                except ValueError:
                    # Skip failed or non-numeric tasks
                    continue
                    
    return tasks, benchmark_accs, sota_accs

def main():
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    report_path = os.path.join(project_root, "reports", "device_benchmark_report_nonvolatile.md")
    
    tasks, bench_accs, sota_accs = parse_markdown_report(report_path)
    if not tasks:
        print("⚠️ No valid benchmark accuracy entries found to plot.")
        return

    print(f"📊 Found {len(tasks)} benchmark tasks for plotting.")

    # 1. Styling configuration (Premium Dark Theme)
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(12, 7), dpi=300)
    fig.patch.set_facecolor('#0f0f12')
    ax.set_facecolor('#141419')

    # Font setup
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'sans-serif']
    plt.rcParams['axes.unicode_minus'] = False

    x = np.arange(len(tasks))
    width = 0.35

    # Curated color palette
    color_bench = '#00f5d4' # Neon turquoise
    color_sota = '#7b2cbf'  # Electric purple / violet

    # 2. Draw Grouped Bars
    rects1 = ax.bar(x - width/2, bench_accs, width, label='Our Organic Device (FingerMemristor)', color=color_bench, edgecolor='#00f5d4', alpha=0.9, zorder=3)
    
    # Only draw SOTA bars if there's non-zero SOTA references
    rects2 = ax.bar(x + width/2, sota_accs, width, label='Platform SOTA Reference', color=color_sota, edgecolor='#7b2cbf', alpha=0.8, zorder=3)

    # 3. Custom Labels and Grid
    ax.set_title('📊 Organic CIM Device Hardware-Aware Benchmark Results', fontsize=16, fontweight='bold', pad=25, color='#ffffff')
    ax.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold', labelpad=15, color='#e0e0e6')
    ax.set_xticks(x)
    ax.set_xticklabels(tasks, rotation=30, ha='right', fontsize=10, fontweight='bold', color='#c0c0c6')
    ax.set_ylim(0, 115)
    
    # Hide top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#33333d')
    ax.spines['bottom'].set_color('#33333d')

    # Premium grid styling
    ax.grid(axis='y', linestyle='--', alpha=0.2, color='#888899', zorder=0)

    # 4. Add Legends and Value Annotations
    ax.legend(loc='upper right', frameon=True, facecolor='#1b1b22', edgecolor='#33333d', fontsize=10)

    # Label values on top of bars
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            if height > 0.0:
                ax.annotate(f'{height:.1f}%',
                            xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, 4),  # 3 points vertical offset
                            textcoords="offset points",
                            ha='center', va='bottom', fontsize=8, color='#e0e0e6', fontweight='bold')

    autolabel(rects1)
    autolabel(rects2)

    # Adjust layout
    plt.tight_layout()
    
    # 5. Save figure
    output_path = os.path.join(project_root, "reports", "benchmark_comparison.png")
    plt.savefig(output_path, facecolor=fig.get_facecolor(), edgecolor='none', bbox_inches='tight')
    print(f"✅ Benchmark comparison chart saved successfully to: {output_path}")

if __name__ == "__main__":
    main()

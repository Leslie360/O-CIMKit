# O-CIMKit

**A source-aware research data and intelligent-analysis platform for organic computing-in-memory devices.**

O-CIMKit connects experimental LTP/LTD measurements, reusable device profiles, hardware-aware PyTorch models, and reproducible application reports. It is intended for device and materials researchers who want to evaluate a measured device across AI tasks without rebuilding the modelling workflow for every task.

[English](#overview) · [中文](#项目简介) · [Review dashboard](applications/web_dashboard/index.html) · [Measured-data case](#measured-data-case) · [License](LICENSE)

## Overview

- **Experiment ingestion:** Excel/TXT input, multi-device aggregation, source metadata, LTP/LTD fitting, and device-to-device endpoint variation.
- **Device-aware models:** quantization-aware layers, volatile reservoir dynamics, nonlinear-update approximations, and output-statistics drift compensation.
- **Application catalog:** 28 directories containing `train.py` modules covering biomedical signals, sensing, vision, control, and generative modelling. The catalog size is not a claim that all modules have been rerun with the XJTU measured profile.
- **Research outputs:** per-fold JSON results, diagnostic plots, cross-application evaluation, and an offline SVG-based review dashboard.

### Evidence boundaries

| Capability | Current evidence |
| --- | --- |
| XJTU LTP/LTD measurements | 12 single-device sheets; 100 pulses per device; source unit not specified |
| XJTU device-profile simulation | Profile-driven quantized/noisy training and inference; not fabricated-chip measurements |
| ECG validation | Public MIT-BIH records; the released headline result uses beat-level five-fold validation, not patient-independent validation |
| Lifetime compensation | Device-aging simulation; not ten years of physical operation |
| TOPS/W, area, and latency | Analytical architecture estimates using explicit model assumptions; not cycle-accurate hardware measurements |
| Literature/knowledge agent | Planned future work, not an implemented platform feature |

### Measured-data case

The XJTU profile is aggregated from 12 measured devices, with 50 LTP and 50 LTD pulses per device. The source contains current responses without a stated unit, so the profile retains source metadata and does not claim calibrated absolute conductance.

For the MIT-BIH binary heartbeat task, the recorded five-fold result with seed 42 is:

| Track | Accuracy | Macro-F1 |
| --- | --- | --- |
| Floating-point baseline | **98.09% ± 0.51 pp** | 0.9744 |
| XJTU measured-profile constraints | **97.93% ± 0.51 pp** | 0.9722 |

The accuracy difference is **−0.16 percentage points**: the constrained result is comparable to the baseline, not an improvement over it. The experiment uses 3,776 sampled beats from the 15 locally available MIT-BIH records. Full fold-level results are in [`reports/xjtu_measured_ecg_results.json`](reports/xjtu_measured_ecg_results.json).

A separate five-fold **record-isolated** experiment holds out three MIT-BIH records per fold. It gives **80.04% ± 12.23 pp** for the float baseline and **77.10% ± 14.46 pp** for the measured-profile model (Macro-F1 0.6983 and 0.6651; difference −2.95 pp). This exposes substantial record-to-record variability and is not evidence of patient-independent clinical performance. The two validation protocols are not interchangeable. See [`reports/xjtu_measured_ecg_patient_group_results.json`](reports/xjtu_measured_ecg_patient_group_results.json).

The measured profile supplies the readout's discrete-state count and noise scale. The LTP/LTD polynomial fitting and volatile reservoir use separate modelling paths; this experiment does not establish a complete calibrated device-to-chip model.

## Installation

```bash
git clone https://github.com/Leslie360/O-CIMKit.git
cd O-CIMKit
python3 -m pip install -e .
```

Run the test suite:

```bash
python3 run_tests.py
```

`o-cimkit prepare-data` downloads MNIST and CIFAR-10 and creates small example arrays. Other applications require their respective public datasets or explicitly generated mock data. Always distinguish downloaded, measured, and mock data in a report.

## Measured-profile reproduction

With the authorized source workbook in `data/devices/` and the MIT-BIH records in `data/datasets/mitdb/`:

```bash
python3 profiles/parser.py \
  --file data/devices/LTP-LTD.xlsx \
  --name XJTU_Measured_Memristor \
  --type nonvolatile \
  --measurement-type current

python3 applications/ecg_cardio/train.py \
  --dataset mitdb \
  --epochs 150 \
  --device-profile XJTU_Measured_Memristor \
  --validation beat-kfold \
  --seed 42 \
  --output-json reports/xjtu_measured_ecg_results.json
```

To regenerate the dashboard's measured curves and result data after rerunning the experiment:

```bash
python3 scripts/export_dashboard_data.py
```

## Application and diagnostic commands

```bash
o-cimkit run digit_rec
o-cimkit run cifar10_vision --epochs 10
o-cimkit run cim_nano_gpt
o-cimkit benchmark --device FingerMemristor --apps cifar10_vision,ecg_cardio --epochs 3
o-cimkit diagnose --device FingerMemristor
o-cimkit codesign --device FingerMemristor
o-cimkit profile --app cim_nano_gpt
```

The application runner accepts application-specific options after the application name. Profile selection and data-preparation support vary by application; consult the module and its CLI help before a comparative run.

## Project structure

```text
core/                 Device-aware layers, quantizers, compiler, and analytical profiler
profiles/             DeviceProfile schema, experiment parser, and profile repository
applications/         28 training modules and the offline review dashboard
scripts/              Data preparation, evaluation, and dashboard-data export
cli/                  Command-line entry points
reports/              Reproduction results and model-based comparison reports
tests/                Unit and parser tests
```

## Review dashboard

Open [`applications/web_dashboard/index.html`](applications/web_dashboard/index.html) directly in a browser. No web server, external chart package, or font service is required. The page separates team measurements, device simulation, and planned work; measured curves and ECG results are generated from local evidence files.

## Related tools

O-CIMKit complements rather than replaces full analog-array or circuit simulators:

- [IBM AIHWKit](https://aihwkit.readthedocs.io/en/latest/): analog training/inference and hardware-calibrated device presets.
- [CrossSim](https://cross-sim.sandia.gov/): experiment-derived update models and analog crossbar accuracy simulation.
- [NeuroSim](https://shimeng.ece.gatech.edu/downloads/): circuit- and architecture-level performance evaluation.

The distinguishing scope of O-CIMKit is an accessible organic-device experiment-to-application workflow, with mixed volatile/nonvolatile models and reviewable research outputs. Quantitative comparisons with these tools require matched devices, models, and validation protocols.

---

## 项目简介

**O-CIMKit：面向有机存算一体器件的垂域科研数据底座与智能分析平台。**

平台将原始 LTP/LTD 实验表格转化为带来源信息的统一器件画像，接入量化感知训练、储备池计算和可靠性仿真，并输出可复现的应用评测报告。当前仓库包含 **28 个训练模块**，其中已完成的团队实测画像闭环以心电分类为代表；不能将训练入口数量表述为28项实测器件验证成果。

### 当前核心证据

- 12颗单器件工作表，每颗100个脉冲，自动聚合为50态画像。
- 源表未标注单位，保留“原始电流响应”语义，不将其宣称为绝对电导。
- MIT-BIH 心搏二分类五折结果：浮点基线 **98.09% ± 0.51 pp**，实测画像约束 **97.93% ± 0.51 pp**，差值 **−0.16个百分点**。
- 该结果来自本地15条公开记录中的3,776个抽样心搏，采用心搏级交叉验证；不是患者独立测试、临床诊断结论或芯片实测。
- 记录隔离补充五折：浮点基线 **80.04% ± 12.23 pp**，画像约束 **77.10% ± 14.46 pp**，差值 **−2.95个百分点**，宏平均F1分别为0.6983和0.6651；逐折测试记录可核查。该结果揭示跨记录泛化仍需优化，不能与心搏级结果混用。
- 可靠性为老化模型仿真；能耗、面积和时延为带假设的体系结构估算。
- 文献检索、参数溯源和科研智能体属于规划能力，不计入当前已实现成果。

### 安装、测试与复现

```bash
git clone https://github.com/Leslie360/O-CIMKit.git
cd O-CIMKit
python3 -m pip install -e .
python3 run_tests.py
```

放入经授权的实测工作表和 MIT-BIH 数据后，运行：

```bash
python3 profiles/parser.py --file data/devices/LTP-LTD.xlsx --name XJTU_Measured_Memristor --type nonvolatile --measurement-type current
python3 applications/ecg_cardio/train.py --dataset mitdb --epochs 150 --device-profile XJTU_Measured_Memristor --validation beat-kfold --seed 42 --output-json reports/xjtu_measured_ecg_results.json
```

详细评测边界见 [`applications/ecg_cardio/README.md`](applications/ecg_cardio/README.md)。完整逐折结果见 [`reports/xjtu_measured_ecg_results.json`](reports/xjtu_measured_ecg_results.json)。

### 离线展示

直接用浏览器打开 `applications/web_dashboard/index.html`。页面不依赖外部图表或字体服务，支持单器件曲线核查、逐折结果展开和 JSON 下载；`poster.html` 提供独立学术海报版式。

### 工程与数据许可

代码许可见 [MIT License](LICENSE)，贡献流程见 [CONTRIBUTING.md](CONTRIBUTING.md)。数据集、实验原始表格和文献内容仍需分别遵守其来源许可与团队公开授权；代码开源不等于这些数据自动获得公开许可。

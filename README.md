# 🚀 O-CIMKit: The Ultimate Organic Computing-In-Memory Kit
# 有机存算一体架构与神经形态计算基座

![Version](https://img.shields.io/badge/version-1.0.0-blue.svg)
![Python](https://img.shields.io/badge/python-3.8%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Build](https://img.shields.io/badge/build-passing-brightgreen)
![Coverage](https://img.shields.io/badge/coverage-91%25-brightgreen.svg)
![Type Hints](https://img.shields.io/badge/typing-strict-purple.svg)

*Select Language: English | 中文*

<details open>
<summary><strong>🇬🇧 English Version (Click to Expand)</strong></summary>

A unified, modular, and hardware-aware simulation framework and neuromorphic computing evaluation platform for **Organic Optoelectronic and Memristive Devices**. 

It decouples experimental device measurements (from raw Excel/TXT files) from machine learning models. Using this platform, you can easily ingest raw device measurements and immediately test them across 19 diverse SOTA neuromorphic applications.

### 📂 Project Structure

```text
CIM_application_project/
├── core/                         # 1. Core Physics & Simulation Kernel
│   ├── __init__.py               # Package public API exposure
│   ├── physics.py                # Device white noise, Poisson shot noise, LTP/LTD gradient modification
│   ├── quantization.py           # Learned Step Size Quantization (LSQ) & Straight-Through Estimator (STE)
│   ├── layers.py                 # OrganicSynapseConv, QATMLPLayer, PhysicalReservoir
│   └── autotune.py               # AutoTuner (using Optuna or grid search fallback for optimization)
│
├── profiles/                     # 2. Device Profile Manager
│   ├── __init__.py               # Package public API exposure
│   ├── device_profile.py         # Unified DeviceProfile data class
│   ├── parser.py                 # Reads raw Excel (.xlsx) / TXT and fits models
│   ├── fitter.py                 # Fits LTP/LTD polynomials & volatile relaxation constants
│   └── repository/               # Generated JSON configurations (e.g. OECT_Vision.json)
│
├── data/                         # 3. Data Storage Pointer
│   ├── devices/                  # Store raw experimental files (e.g. conductance.txt)
│   └── datasets/                 # Datasets (MNIST, MIT-BIH, Sleep-EDF, CIFAR-10, etc.)
│
├── tests/                        # 4. Unit Test Suite (Subclasses of unittest.TestCase)
│   ├── test_quantization.py      # Tests LSQ/MinMax quantization and STE gradient flow
│   ├── test_layers.py            # Tests volatile DynamicOrganicSynapse & non-volatile SelfHealingCrossbar
│   ├── test_compiler.py          # Tests compiler logic and model synthesis
│   └── test_autotune.py          # Tests Bayesian reservoir hyperparameter autotuning
│
├── scripts/                      # 5. One-Shot Demonstration & Plotting Utilities
│   ├── verify_codesign_selfhealing.py # Verifies co-design training and self-healing under 10-year drift
│   ├── evaluate_reliability.py        # Runs 10-year reliability aging analysis on Yale faces
│   ├── plot_benchmark.py              # Plots bar chart comparing device metrics against platform SOTA
│   ├── autotune_demo.py               # Demonstration of Optuna reservoir hyperparameter tuning
│   └── generate_mock_device_data.py   # Generates mock memristor/OECT measurements
│
├── pyproject.toml                # 6. Pip packaging and installation configuration
├── LICENSE                       # 7. MIT License file for open-source compliance
├── CONTRIBUTING.md               # 8. Guide for contributing to the repository
├── run_tests.py                  # 9. Main unit test discovery and runner execution script
├── main.py                       # 10. Root-level unified CLI entry point
│
└── applications/                 # 11. Neural Network & Reservoir Applications
    ├── [ecg_cardio](applications/ecg_cardio) -> MIT-BIH classification (QAT MLP) | 97.91% ± 0.45%
    ├── [fatigue_eeg](applications/fatigue_eeg) -> Sleep-EDF stage detection (Multi-Scale RC + QAT MLP) | 71.95% ± 2.70%
    ├── [bearing_fault](applications/bearing_fault) -> CWRU fault detection (QAT MLP) | 99.80% ± 0.25%
    ├── [chaotic_lorenz](applications/chaotic_lorenz) -> Lorenz attractor forecasting (Volatile RC) | NRMSE 0.0214%
    ├── [digit_rec](applications/digit_rec) -> Sequential MNIST digits recognition (Volatile RC) | 95.00%
    ├── [speech_emotion](applications/speech_emotion) -> RAVDESS speech emotion classification (QAT MLP) | 83.77% ± 4.01%
    ├── [embodied_ai](applications/embodied_ai) -> Tactile multimodality materials classification (RC) | ~99.00%
    ├── [edge_llm](applications/edge_llm) -> Edge-LLM Sentinel anomaly interceptor (RF) | ~94% Intercept
    ├── [physical_attention](applications/physical_attention) -> Physical KV-Cache attention mechanism (Synergy) | ~95.00%
    ├── [fingerprint_rec](applications/fingerprint_rec) -> Fingerprint recognition (NIST + ResNet-18) | 92.19%
    ├── [cifar10_vision](applications/cifar10_vision) -> Bionic vision (CIFAR-10 + ResNet-18) | 90.15%
    ├── [face_rec](applications/face_rec) -> Yale Faces recognition (ResNet-18 + QAT Head) | 96.67%
    ├── [optoelectronic_vision](applications/optoelectronic_vision) -> Bionic Sensor-CIM integrated vision (OECT + ResNet-18) | 91.86%
    ├── [neuromorphic_stdp](applications/neuromorphic_stdp) -> Unsupervised SNN with STDP learning rule (SNN) | 22.78%
    ├── [neuromorphic_pid](applications/neuromorphic_pid) -> Adaptive PID controller under memristor noise | 55.80%
    ├── [tactile_eskin](applications/tactile_eskin) -> E-skin multi-class tactile sensor classification (CNN) | 100.00%
    ├── [neuromorphic_grasp](applications/neuromorphic_grasp) -> Robotic hand slippage reduction control | 98.24%
    ├── [seizure_detection](applications/seizure_detection) -> Seizure detection from multichannel EEG | 100.00%
    ├── [generative_aigc](applications/generative_aigc) -> ConvVAE Digit Image Generation & Reconstruction | MSE: 4.85e-3
    └── [biohybrid_spiking](applications/biohybrid_spiking) -> Spiking coordination in biohybrid networks | 100.00%
```

### 📦 Installation

To use this platform as an open-source library, clone this repository and run editable pip install in your environment:
```bash
git clone https://github.com/Leslie360/CIM_application_project.git
cd CIM_application_project
pip install -e .
```

After installation, run the automated data preparation script to download standard vision datasets (MNIST, CIFAR-10) and generate lightweight mock data for proprietary medical/physical sensor datasets:
```bash
o-cimkit prepare-data
```

### 🚀 Quick Start (CLI Entry Point)

You can run any of the 20+ applications directly via the global CLI or `main.py`:
```bash
# Run sMNIST digit recognition
o-cimkit run digit_rec

# Run Edge-LLM Sentinel anomaly detection
o-cimkit run edge_llm

# Run CIFAR-10 vision model for 10 epochs
o-cimkit run cifar10_vision --epochs 10

# Run generative VAE application
o-cimkit run generative_aigc --epochs 25

# Run Nano-GPT Large Language Model on Organic Memory
o-cimkit run cim_nano_gpt

# Run top-journal comparative benchmark and publish reports
o-cimkit benchmark --apps cifar10_vision,generative_aigc --epochs 10
```

To run bionic co-design compilation and self-healing validation on a device profile:
```bash
o-cimkit codesign --device FingerMemristor
```

To generate a premium, high-resolution physical diagnostics datasheet and curves for a device:
```bash
o-cimkit diagnose --device FingerMemristor
```

### 🔧 How to Ingest a New Device Dataset

When you get new raw experimental measurements (Excel or TXT file of current/conductance values):
1. Save the file inside `data/devices/` (e.g., `my_device.xlsx`).
2. Run the parser:
   ```bash
   # For nonvolatile memristors (e.g., 64 discrete states):
   python profiles/parser.py --file data/devices/my_device.xlsx --name MyMemristor --type nonvolatile --states 64

   # For volatile short-term decay measurements:
   python profiles/parser.py --file data/devices/my_device.xlsx --name MyOECT --type volatile
   ```
3. A JSON configuration containing all computed parameters will be saved to `profiles/repository/MyMemristor.json`, which can be immediately used by all applications.

### 📈 Boosting Performance (AutoTuner)

We provide an `AutoTuner` module (`core/autotune.py`) that utilizes Optuna (or a grid search fallback) to tune reservoir hyperparameters (spectral radius, input scaling, leaking rate) to automatically boost accuracy for your specific device characteristics:
```python
from core.autotune import AutoTuner

# Define your evaluation function returning accuracy
def evaluate_fn(spectral_radius, input_scale, leaking_rate, ridge_alpha):
    # Setup your reservoir and evaluate
    return accuracy

# Tune for 30 trials
tuner = AutoTuner(target_accuracy_fn=evaluate_fn, n_trials=30)
best_params, best_accuracy = tuner.tune()
```
</details>

---

<details>
<summary><strong>🇨🇳 中文版 (点击展开)</strong></summary>

针对**有机光电和忆阻器件**的高硬件感知度、模块化存算一体仿真与神经形态计算评估平台。

该平台成功将底层的物理器件实验数据测量（支持 raw Excel/TXT 接入）与上层机器学习模型解耦。你可以通过简单的物理配置文件直接对接 19 种不同前沿计算领域的神经网络与储备池算法。

### 📦 安装方式

支持一键作为 Python 库进行安装和开发：
```bash
git clone https://github.com/Leslie360/CIM_application_project.git
cd CIM_application_project
pip install -e .
```

安装完成后，请运行自动化数据准备脚本。该脚本会自动下载标准视觉数据集（如 MNIST, CIFAR-10），并为那些封闭的医疗/物理专有数据集（如心电图、脑电图）生成轻量级的 Mock 测评数据，确保项目能够**开箱即用**：
```bash
o-cimkit prepare-data
```

### 🚀 全局 CLI 命令行快速运行

安装完毕后，你可以通过全局命令 `o-cimkit` 极速运行任何模块：
```bash
# 运行 sMNIST 手写数字识别
o-cimkit run digit_rec

# 运行大模型边缘前哨异常拦截
o-cimkit run edge_llm

# 指定训练轮数运行 CIFAR-10 仿生视觉系统
o-cimkit run cifar10_vision --epochs 10

# 运行变分自编码器 AIGC 图像生成应用
o-cimkit run generative_aigc --epochs 25

# 运行最前沿的 Nano-GPT 生成式大模型 (LLM on CIM)
o-cimkit run cim_nano_gpt

# 一键运行顶刊标准硬件感知对比跑分并输出报告
o-cimkit benchmark --apps cifar10_vision,generative_aigc --epochs 10
```

运行硬件感知协同设计编译与在线自愈校验（软硬协同优化）：
```bash
o-cimkit codesign --device FingerMemristor
```

一键绘制物理特性诊断曲线并生成数据手册报告：
```bash
o-cimkit diagnose --device FingerMemristor
```

### 📈 自动超参调优提升性能

我们提供了自动化调优模块 `AutoTuner`（基于 Optuna 实现，无环境时自动退回至高效网格搜索）。它能针对新器件的物理特性（时间常数、非线性度），自动搜索最佳的储层谱半径、输入缩放、泄漏率以及读出层正则化系数，使新器件的一键识别精度最大化。

</details>

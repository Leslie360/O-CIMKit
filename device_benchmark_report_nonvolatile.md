# 📊 Organic Device Benchmark Report
**Device Profile Evaluated**: `FingerMemristor` (Memory Type: `Non-Volatile`)
**Benchmark Epochs**: 3

## 📈 Performance Summary Table

| Application Task | Benchmark Accuracy | Platform SOTA Reference |
| :--- | :--- | :--- |
| **fingerprint_rec** | **92.47%** | 93.85% (ResNet-18 + 28态忆阻器) |
| **ecg_cardio** | **97.40%** | 97.91% (QAT MLP + 64态AlOx) |
| **ecg_ptbdb** | **100.00%** | 98.20% (QAT MLP + PTB-DB + 64态AlOx) |
| **bearing_fault** | **99.69%** | 99.80% (QAT MLP + 64态AlOx) |
| **optoelectronic_vision** | **94.88%** | 91.86% (Co-Design ResNet + OECT光电突触) |
| **optoelectronic_cifar100** | **77.75%** | 76.21% (Co-Design ResNet + 28态忆阻器) |
| **fatigue_eeg** | **78.05%** | 79.20% (QAT CNN + Sleep-EDF + 28态忆阻器) |
| **face_rec** | **90.00%** | 96.67% (ResNet-18 + 28态忆阻器) |
| **face_rec_orl** | **100.00%** | 90.00% (ResNet-18 + ORL + 28态忆阻器) |
| **cifar10_vision** | **93.58%** | 90.15% (Custom ResNet + OECT光电突触) |
| **cifar100_vision** | **75.06%** | 73.40% (Custom ResNet + 28态忆阻器) |
| **neuromorphic_stdp** | **26.11%** | 22.78% (Unsupervised SNN STDP + FingerMemristor) |
| **neuromorphic_pid** | **57.21%** | 55.80% (Adaptive PID + 64态AlOx) |
| **tactile_eskin** | **100.00%** | 100.00% (CNN + 28态忆阻器 + 64态AlOx) |
| **neuromorphic_grasp** | **97.95%** | 98.24% (Slippage Reduction + 28态忆阻器) |
| **seizure_detection** | **100.00%** | 100.00% (CNN + 28态忆阻器 + 64态AlOx) |
| **biohybrid_spiking** | **93.00%** | 100.00% (OECT Population + 64态AlOx) |
| **dvs_gesture** | **100.00%** | 90.91% (Event Frame CNN + 28态忆阻器) |
| **neuromorphic_rl** | **87.1 steps** | 194.5 steps (Balance Control DQN + 28态忆阻器) |
| **speech_emotion** | **84.96%** | 85.40% (Reservoir + QAT MLP + 64态AlOx) |
| **edge_llm** | **AUC: 0.9947** | AUC: 0.9947 (Edge-LLM Sentinel + Volatile) |
| **embodied_ai** | **99.95%** | 100.00% (Physical RC + QAT MLP) |
| **physical_attention** | **97.90%** | 97.90% (Physical KV-Cache + Volatile) |
| **olfactory_enose** | **100.00%** | 100.00% (Reservoir + QAT MLP + 28态忆阻器) |
| **eeg_motor_imagery** | **100.00%** | 100.00% (Spatio-Temporal Reservoir + QAT MLP) |
| **neuromorphic_kws** | **100.00%** | 100.00% (Multi-Scale Reservoir + QAT MLP) |
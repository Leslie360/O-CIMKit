# MIT-BIH 心电分类：器件画像约束评测

本模块演示如何将非易失器件画像接入心电二分类流程。它回答的是“实测曲线提取的量化状态与噪声尺度能否进入模型训练与评估”，不是临床诊断性能验证。

## 模型与数据

1. 从 MIT-BIH 记录中读取带 `N/L/R/A/a/V/E` 标注的心搏，截取标注点前后各90个采样点，得到180点序列。
2. 将 `N/L/R` 归为正常，将 `A/a/V/E` 归为异常。当前本地数据含15条记录，抽样后包含2,832个正常和944个异常心搏，共3,776个样本。
3. 使用2,000节点的挥发储备池提取均值、标准差、最大值、最小值和末态特征，形成10,000维输入。
4. 读出网络为 `10,000 → 256 → 2`，包含ReLU与dropout。浮点基线与器件约束模型使用相同特征、数据划分、训练轮数和逐折随机种子。
5. `QATMLPLayer` 对读出权重执行画像指定的50态MinMax量化，并使用画像响应范围进行映射。

当前画像中的 `noise_std_ratio` 由多颗器件在LTP末端的差异估计；该值作为画像元数据保存，但不注入本模块的读出层，不能据此宣称已验证写入噪声的影响。LTP/LTD多项式供支持非线性梯度近似的层使用，本模块的 `QATMLPLayer` 不调用该梯度钩子。

## 已复跑结果

固定 `seed=42`、每折150轮、心搏级分层五折交叉验证：

| 评测方式 | 准确率 | 宏平均 F1 |
| --- | --- | --- |
| 浮点基线 | 98.09% ± 0.51个百分点 | 0.9744 |
| XJTU实测画像约束 | 97.93% ± 0.51个百分点 | 0.9722 |

差值为−0.16个百分点，结论是二者在该实验流程中表现相当，不是“零损失”或“超越基线”。逐折结果见 `reports/xjtu_measured_ecg_results.json`。

### 记录隔离补充验证

固定 `seed=42`、每折150轮，使用 `StratifiedGroupKFold` 按15条MIT-BIH记录分组，每折训练12条、测试3条：

| 评测方式 | 准确率（均值±折间标准差） | 宏平均 F1 |
| --- | --- | --- |
| 浮点基线 | 80.04% ± 12.23个百分点 | 0.6983 |
| XJTU实测画像约束 | 77.10% ± 14.46个百分点 | 0.6651 |

差值为−2.95个百分点。每折测试记录均有正常与异常心搏，训练与测试记录零重叠；逐折记录号、类别分布和完整结果见 `reports/xjtu_measured_ecg_patient_group_results.json`。结果反映明显的记录间差异，说明跨记录泛化仍需改进；两种划分协议不能混用，也不能据此宣称临床或患者独立性能。

增加 `--export-predictions` 后，还会输出全部留出预测、混淆矩阵及正常/异常心搏实例。合并3,776个留出心搏时，浮点与画像约束的样本加权准确率分别为 **78.42% / 75.53%**，正常召回率分别为78.53% / 76.52%，异常召回率分别为78.07% / 72.56%。各折等权均值与合并预测采用不同统计口径，不应当作矛盾。

## 复现命令

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

记录隔离的补充验证使用独立结果文件，不覆盖上述心搏级基线：

```bash
python3 applications/ecg_cardio/train.py \
  --dataset mitdb \
  --epochs 150 \
  --device-profile XJTU_Measured_Memristor \
  --validation patient-group-kfold \
  --seed 42 \
  --output-json reports/xjtu_measured_ecg_patient_group_results.json \
  --export-predictions
```

参数：

- `--epochs`：每折训练轮数，默认150。
- `--device-profile`：非易失画像名称或JSON路径，默认 `PFO_AlOx_NonVolatile`。
- `--validation`：数据划分方式；`beat-kfold` 用于复现上述历史基线，`patient-group-kfold` 按MIT-BIH记录隔离训练与测试。
- `--seed`：模型优化和数据加载随机种子，默认42；心搏级划分固定为42以保持历史结果可复现。
- `--output-json`：保存每折准确率、宏平均F1、样本数、验证协议和画像来源。
- `--export-predictions`：额外保存全部留出预测CSV、混淆矩阵及真实心搏实例，供网页和申报图共用。

## 结果解释边界

- `beat-kfold` 在心搏层面随机划分，同一记录的心搏可能同时出现在训练集和测试集，不证明跨患者泛化能力。
- `patient-group-kfold` 使用记录号作为组标识，避免同一记录跨越训练集与测试集；小记录集合上的分组结果应单独报告，不与心搏级结果直接比较。
- MIT-BIH记录号作为记录级隔离标识，用于降低相邻心搏泄漏风险，不等同于临床多中心外部验证。
- 结果来自公开心电数据与实测曲线驱动画像的器件仿真，不是芯片实测或医疗器械性能证明。
- 数据预处理、抽样策略、模型规模及记录范围均会影响结果，跨论文比较必须保持协议一致。

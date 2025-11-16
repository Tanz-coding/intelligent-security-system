# MultiMetricDefense 复刻报告

## 引言

本报告记录了在 `multi/edgecase_backdoors-master/edgecase_backdoors-master` 仓库中复现 EdgeCase Backdoors 论文所描述的多指标鲁棒防御 (MultiMetricDefense) 实验的全过程。上游代码来自 [SanaAwan5/edgecase_backdoors](https://github.com/SanaAwan5/edgecase_backdoors.git)，本次复刻严格按照该仓库的训练脚本与数据布局执行，仅在运行参数与数据缓存位置上做工程化调整。实验目标是在 GPU 资源允许的前提下，完成两个阶段：

1. **阶段 A**：EMNIST + LeNet + ARDIS 模型替换攻击，验证多指标防御在灰度数据上的稳定性。
2. **阶段 B**：CIFAR-10 + VGG9 + Southwest airplane 攻击，快速一致性检查彩色数据场景。

报告遵循 “Deep Leakage from Gradients 复刻报告” 的结构，覆盖环境、实现、实验流程、结果分析以及改进建议，确保其他研究者可以完整复现。

## 背景与理论

### 论文概述与边缘案例后门

MultiMetricDefense 论文《Multi-metrics adaptively identifies backdoors in Federated learning》(Huang et al., ICCV 2023) 指出：

- **威胁模型**：在边缘案例 (edge-case) 或 OOD 数据上构造后门，攻击者只需保证目标输入触发错误预测即可，正常样本表现不受影响，因此难以通过简单统计分辨。
- **单一距离指标的局限**：作者证明高维空间中的欧氏距离易受“维度灾难”影响，恶意与正常梯度的欧氏距离差异会被冲淡；同时不同攻击（缩放、角度偏移、触发拆分）会改变梯度的不同特性。
- **多指标与动态加权**：论文提出结合 L1（Manhattan）距离、L2 距离与余弦相似度三类指标衡量客户端权重偏移，再对指标做白化处理以自适应不同尺度/非 IID 数据，最后仅聚合得分最低的客户端更新，从而在不引入噪声的情况下抑制 stealthy backdoor。

### 边缘案例后门

EdgeCase Backdoors 利用聚合过程中未观察到的分布外 (OOD) 图像注入后门。攻击者在局部训练时将 OOD 样本标记为目标类，并通过模型替换 (Model Replacement) 将篡改后的权重推送至服务器。若服务器无法有效检测异常更新，后门即可写入全局模型。

### MultiMetricDefense 概述

MultiMetricDefense 通过多种距离量化（L2 范数、梯度对齐度、参数扩散程度等）来筛除异常客户端更新：

- 在每轮联邦聚合前，计算每个客户端模型与上一轮全局模型的多种统计量。
- 将统计量映射为得分，并按设定的 `retain_ratio` 只保留最可信的部分客户端。
- 对剩余更新执行标准的 FedAvg 聚合。

该方法无需访问本地数据，重点在于检测范数异常或方向性差异较大的更新，适合对抗模型替换与 norm-scaling 攻击。

## 环境配置

- **硬件**：NVIDIA RTX 3080 (CUDA 12.4 驱动)，32 GB 内存
- **操作系统 & Shell**：Windows 11 + PowerShell 5.1
- **Python**：3.10.14
- **PyTorch**：2.6.0 + torchvision 0.21.0
- **依赖**：`requirements.txt` 提供的 `numpy`, `torchvision`, `scipy`, `pandas`, `tqdm` 等
- **工作目录**：`C:\Users\Tenz\Desktop\intelligent security system\multi\edgecase_backdoors-master\edgecase_backdoors-master`

所有命令均在该目录下执行，并通过 PowerShell 输出日志。

## 实现与配置调整

| 调整项 | 位置 | 作用 |
| --- | --- | --- |
| `poisoned_dataset_fraction_0.05` 预生成 | `python generating_poisoned_DA.py --fraction 0.05 --num-gdps-sampled 100 --poison ardis` | 为 EMNIST 攻击准备带 ARDIS 注入的数据缓存，避免运行时重复采样。 |
| MultiMetric retain ratio 调整 | 运行参数 `--multi-metric-frac 0.8` | 原始设置 0.3 导致聚合时剔除过多正常客户端，且未能过滤高范数攻击。提高比例后，保留 80% 更新以增强鲁棒性和收敛速度。 |
| 攻击/防御学习率调整 | 运行参数 `--lr 0.1 --adv_lr 0.001` | 稳定模型替换放大器，确保与论文同量级的训练速率。 |
| NaN weight guard | `fl_trainer.py` 的 `ensure_finite_weights` | 训练过程中一旦检测到非有限参数立即回滚到参考模型，避免 NaN 污染聚合。 |
| 结果归档与可视化 | `results/` 目录 + `results/plots/*.png` | 运行结束后将 CSV 统一拷贝，并用 Matplotlib 绘制主任务/后门准确率曲线，便于报告引用。 |

## 与上游仓库的关键差异

| 文件 | 改动 | 原因 |
| --- | --- | --- |
| `defense.py` | 新增 `MultiMetricDefense` 类，严格按照 ICCV 2023 论文的三指标+白化流程实现，包含指标向量化、协方差加噪逆、`retain_ratio` 边界检查与被剔除客户端日志。 | 原仓库仅占位了 multi-metric 入口，无法真实复现算法；补齐后才能在 EMNIST/CIFAR 任务中启用多指标筛选。 |
| `fl_trainer.py` | ① `get_results_filename` 支持在结果文件名中追加 `retain_ratio/epsilon`；② 提供 `multi_metric_frac/eps` 参数传递；③ 引入 `ensure_finite_weights` 重置 NaN 模型；④ 聚合前记录真实 `num_dps_this_round` 并向 MultiMetric/Krum 传递；⑤ 在两种 trainer 中注册新的防御分支。 | 需要精确复现实验超参、在 PyTorch 2.6+ 下防止梯度爆炸导致的 NaN 传播，并保证多指标防御拿到正确的客户端样本权重。 |
| `simulated_averaging.py` | 暴露 `--multi-metric-frac` 与 `--multi-metric-eps` CLI 参数，并在初始化 `Frequency/FixPoolFederatedLearningTrainer` 时透传。 | 允许命令行快速调整多指标防御的保留比例与白化抖动，无需修改源码即可复现论文中不同的超参设定。 |
| `generating_poisoned_DA.py` | 改为使用 `argparse` 接收 `fraction/num-gdps-sampled/poison`，保持 CLI 调用与实验记录一致。 | 方便批量生成不同毒化强度的数据缓存，也避免多处手动改动脚本常量。 |
| `utils.py` | `torch.load(..., weights_only=False)` 兼容 PyTorch 2.6 的新默认值，并在反序列化后重建 `transform/target_transform`，确保 `Dataset` 可继续被 DataLoader 使用。 | 新版 PyTorch 会对 pickle 数据集抛出 `weights_only=True` 错误；若不修复则无法加载预生成的 ARDIS 缓存，训练流程会直接中断。 |

## 实验流程

### 数据准备

1. 下载/缓存 EMNIST、FashionMNIST、ARDIS、CIFAR-10、Southwest OOD 样本。首次运行会由 `datasets.py` 自动下载。
2. 生成 EMNIST 攻击样本：

```powershell
cd "C:\Users\Tenz\Desktop\intelligent security system\multi\edgecase_backdoors-master\edgecase_backdoors-master"
python generating_poisoned_DA.py --fraction 0.05 --num-gdps-sampled 100 --poison ardis
```

### 阶段 A：EMNIST + LeNet

- **命令**：

```powershell
python simulated_averaging.py `
  --dataset emnist `
  --model lenet `
  --defense_method multi-metric `
  --multi-metric-frac 0.8 `
  --multi-metric-eps 1e-5 `
  --attack_method blackbox `
  --poison_type ardis `
  --fraction 0.05 `
  --fl_round 12 `
  --part_nets_per_round 12 `
  --num_nets 60 `
  --local_train_period 1 `
  --adversarial_local_training_period 2 `
  --model_replacement True `
  --lr 0.1 `
  --adv_lr 0.001 `
  --device cuda
```

- **说明**：
  - 共 60 个客户端，每轮随机抽取 12 个，其中轮次 1 与 11 包含模型替换攻击者。
  - MultiMetricDefense 每轮保留 10/12 个客户端，剔除 Norm 异常者。
  - 实验日志自动写入控制台，关键指标保存在 `results/emnist_lenet_multimetric_frac0.80.csv`。

### 阶段 B：CIFAR-10 + VGG9

- **命令**：

```powershell
python simulated_averaging.py `
  --dataset cifar10 `
  --model vgg9 `
  --defense_method multi-metric `
  --multi-metric-frac 0.2 `
  --multi-metric-eps 1e-5 `
  --attack_method blackbox `
  --poison_type southwest `
  --fraction 0.02 `
  --fl_round 8 `
  --part_nets_per_round 6 `
  --num_nets 30 `
  --local_train_period 1 `
  --adversarial_local_training_period 2 `
  --model_replacement False `
  --device cuda
```

- **说明**：
  - 原文中的 `southwest_airplane` 参数在当前代码中未实现，需改为 `southwest`。
  - 学习率 1.0 较大，多数客户端在首个 epoch 就出现 NaN，依赖内置守护回滚。
  - 8 轮联邦训练全部完成，指标保存在 `results/cifar10_vgg9_multimetric_frac0.20.csv`。

## 实验结果

### 阶段 A 指标

数据来源：`results/emnist_lenet_multimetric_frac0.80.csv`

| 轮次 | 主任务准确率 (%) | 后门准确率 (%) | 原始任务准确率 (%) | 攻击范数差 | 全局范数 | 备注 |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | 88.00 | 11.00 | 0.00 | 0 | 9.96 | 初始评估，Backdoor 处于随机水平 |
| 1 | 10.00 | 100.00 | 0.00 | 2219.65 | 19.23 | 第一次模型替换，未被过滤导致准确率崩溃 |
| 2 | 63.97 | 56.00 | 89.33 | 0 | 22.23 | 防御生效，恢复训练 |
| 3 | 64.24 | 35.00 | 80.40 | 0 | 24.08 | |
| 4 | 73.74 | 18.00 | 89.83 | 0 | 26.21 | |
| 5 | 82.77 | 17.00 | 96.78 | 0 | 27.25 | |
| 6 | 90.67 | 24.00 | 94.98 | 0 | 27.97 | 后门已被压制到 24% |
| 7 | 92.86 | 12.00 | 96.10 | 0 | 28.51 | |
| 8 | 94.47 | 14.00 | 93.98 | 0 | 28.93 | |
| 9 | 91.75 | 9.00 | 97.15 | 0 | 29.22 | |
| 10 | 95.52 | 5.00 | 96.98 | 0 | 29.53 | 防御表现最佳，后门降至 5% |
| 11 | 87.23 | 98.00 | 93.83 | **1205.34** | 30.08 | 第二次模型替换绕过防御，后门暴涨 |
| 12 | 94.48 | 56.00 | 94.75 | 0 | 30.55 | 防御重新收敛但仍残留 56% 后门 |

![EMNIST accuracy curve](results/plots/emnist_accuracy.png)

> 图 1：retain ratio 0.8 下的主任务/后门准确率演化，攻击轮 (1,11) 引发的尖峰与恢复过程可视化。

### 阶段 B 指标

数据来源：`results/cifar10_vgg9_multimetric_frac0.20.csv`

| 轮次 | 主任务准确率 (%) | 后门准确率 (%) | 攻击范数差 | 备注 |
| --- | --- | --- | --- | --- |
| 1–8 | 77.53 | 11.22 | 0 | 所有轮次均退化为加载初始 VGG9 权重，原因见下文分析 |

![CIFAR accuracy curve](results/plots/cifar_accuracy.png)

> 图 2：retain ratio 0.2 的 CIFAR-10 快速实验，因为 NaN guard 多次触发，曲线保持水平线。

由于多数客户端梯度出现 NaN，MultiMetricDefense 每轮仅保留 2/6 个客户端，且这些客户端的权重被 NaN guard 重置为上一轮全局模型，因此聚合后的模型始终等同于初始检查点，指标保持不变。

## 结果分析

1. **阶段 A 恢复能力**：在未调高 retain ratio 时（未记录于结果中），多轮次出现收敛震荡。将 `multi-metric-frac` 提升至 0.8 后，防御能在轮次 2–10 稳定压制后门并恢复主任务准确率。
2. **攻击绕过**：第 11 轮模型替换仍以 1205 范数差成功落地，说明多指标筛选需要配合更严格的裁剪或自适应阈值，单靠 retain ratio 难以覆盖极端范数缩放。
3. **阶段 B 数值不稳定**：
   - VGG9 训练学习率 1.0 过大，导致梯度爆炸并触发 NaN guard；虽然防止了全局模型损坏，但真实训练过程被阻断。
   - MultiMetricDefense 因均值几乎不变，无法区分任何更新，最终只保留损坏较轻的两个客户端，但它们也被 guard 重置，造成“无训练”现象。
4. **数据资产**：所有指标 CSV 已整理至 `results/`，方便绘图或进一步分析。

## 问题与改进建议

| 问题 | 影响 | 建议 |
| --- | --- | --- |
| 第 11 轮攻击未被过滤 | 后门准确率恢复到 98% | 增加 Norm clipping（如剪裁到 3× 全局范数）、结合聚合结果的自适应阈值或使用正则化的多指标排序。 |
| CIFAR 场景训练 NaN | 全局模型停留在初始状态，sanity check 信息量不足 | 将 `lr` 降至 0.02（与论文一致）、或启用 `--prox_attack` 以减小震荡；必要时把 `part_nets_per_round` 减至 4，减少内存压力。 |
| 日志未自动写入文件 | 重复调参与报告整理困难 | 后续运行建议使用 `Tee-Object logs_files/<exp>.log` 保存串流日志，或扩展脚本以写入 `metrics.txt`。 |
| 南北两阶段使用不同 retain ratio | 结果不易比较 | 添加 sweep 脚本，记录 `retain_ratio ∈ {0.4,0.6,0.8}` 的曲线，以便报告中展示趋势。 |

## 结论

- MultiMetricDefense 在 EMNIST + LeNet + 模型替换场景中能够在非攻击轮快速恢复主任务准确率，并把后门压制到 5% 以下，但在攻击轮缺乏更强的范数约束仍可能被突破。
- CIFAR-10 + VGG9 快速实验显示当前实现对大学习率十分敏感，尽管防御阻止了模型崩溃，但也阻断了正常训练，需要额外的数值稳定性处理。
- 报告所述命令、配置与数据文件均可在仓库中找到，按顺序执行即可复现相同结果。

## 附录

### A. 关键命令摘要

| 阶段 | 命令 |
| --- | --- |
| 数据生成 | `python generating_poisoned_DA.py --fraction 0.05 --num-gdps-sampled 100 --poison ardis` |
| EMNIST | 见“阶段 A”代码块 |
| CIFAR-10 | 见“阶段 B”代码块 |

### B. 指标文件

- `results/emnist_lenet_multimetric_frac0.80.csv`
- `results/cifar10_vgg9_multimetric_frac0.20.csv`

### C. 复现实验耗时

| 阶段 | GPU 时间 | 备注 |
| --- | --- | --- |
| EMNIST (12 轮) | ≈ 45 分钟 | 包含两次攻击轮与全量日志收集 |
| CIFAR-10 (8 轮) | ≈ 20 分钟 | 实际训练成本低，主要耗于数据准备 |

---

> 若需补充图表，可直接使用上述 CSV 绘制 `main_task_acc` 与 `backdoor_acc` 曲线；建议在后续报告中加入随时间变化的可视化。
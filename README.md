
# Deep Leakage from Gradients (DLG) 复现实验与代码修改说明

## 0.更多说明
如需查看multi信息，请点击：[README_multi](multi/README_multi.md)

本仓库基于 mit-han-lab/dlg（NeurIPS 2019）进行复现与工程化改造，面向 Windows/PowerShell 环境提供一键运行与更丰富的实验配置（数据集选择、批量样本、TV 正则、指标导出与结果保存）。

- 上游项目主页: https://github.com/mit-han-lab/dlg
- 论文: https://arxiv.org/abs/1906.08935

本文档包含：
- 复现实验步骤（MNIST / CIFAR-100 / LFW）
- 关键命令（Windows PowerShell）
- 代码相对于上游的修改列表与动机
- 结果产出（history.png / recon.png / metrics.txt）

---

## 1. 环境准备

建议使用 Python 3.8+；在仓库根目录安装依赖：

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

requirements.txt：
- torch, torchvision（请根据本机 CUDA 选择合适版本）
- pillow, matplotlib

---

## 2. 快速体验（单样本，CIFAR-100）

```powershell
# 运行 300 次迭代，保存历史图
python main.py --dataset cifar100 --index 25 --iters 300 --save-history --save-final
```

产物：
- history.png：每 10 次迭代的重建快照
- recon.png / target.png：最终重建和原图
- metrics.txt：MSE 与 PSNR 指标

可选参数：
- `--tv-weight 0.005`：开启 TV 正则，提升视觉质量
- `--indices 10 25 70`：批量恢复多个样本

---

## 3. 复现实验配置

### 3.1 MNIST 实验（类似截图中的 5.1.1）

```powershell
# MNIST 单样本，1200 次迭代
python main.py --dataset mnist --model lenetmnist --index 5 --iters 1200 --save-history --save-final --tv-weight 0.005
```

说明：
- 模型：`lenetmnist`（1x28x28 输入），与上游 LeNet 结构一致风格但针对 MNIST 自动计算 FC 维度。
- 指标：执行结束后在 metrics.txt 输出 MSE 和 PSNR。

### 3.2 CIFAR-100 实验（类似截图中的 5.1.2）

```powershell
# CIFAR-100 单样本，1200 次迭代
python main.py --dataset cifar100 --model lenet --index 25 --iters 1200 --tv-weight 0.005 --save-history --save-final
```

- 模型：`lenet`（3x32x32）；也可用 `resnet18` 试验不同结构。
- 建议开启 TV 正则，重建更平滑。

### 3.3 LFW 人脸实验（类似截图中的 5.1.3）

```powershell
# LFWPeople 数据集（自动下载），图像统一到 64x64
python main.py --dataset lfw --model lenet64 --index 10 --iters 1500 --tv-weight 0.005 --save-history --save-final
```

- 模型：`lenet64`（3x64x64），自动推断 FC 输入维度。
- LFW 类别数较多，优化更慢；可尝试更高迭代数或更强 TV。

### 3.4 批量（Batch）实验

```powershell
# 同时恢复多个索引样本
python main.py --dataset cifar100 --indices 10 25 70 123 --iters 800 --tv-weight 0.003 --save-final
```

- 将根据 batch 的平均梯度进行匹配，虚拟变量 `dummy_data` / `dummy_label` 将是同 batch 维度。
- 输出会保存 recon_*.png / target_*.png。

### 3.5 自定义图片

```powershell
# 用外部图片替换数据集样本，指定标签与类别数
python main.py --dataset cifar100 --image .\path\to\your.jpg --label 0 --num-classes 100 --iters 600 --save-final
```

- 程序会根据所选数据集自动 resize 到合适尺寸；MNIST 会转为灰度。
- 若不指定 `--label`，默认使用当前数据集索引对应的标签（但当以自定义图片替换时建议显式传入 `--label` 和 `--num-classes`）。

---

## 4. 重要参数汇总

- `--dataset {cifar100,mnist,lfw}`：选择数据集
- `--index / --indices`：单样本或多样本索引
- `--model {lenet,lenetmnist,lenet64,resnet18}`：模型结构
- `--iters`：迭代次数（300~2000 视任务而定）
- `--tv-weight`：TV 正则权重（0~0.01 常用）
- `--save-history / --save-final`：保存历史/最终图像
- `--device`：指定 cuda/cpu（默认自动检测）
- `--seed`：随机种子

---

## 5. 相对于上游仓库的代码修改

文件：`main.py`
- 新增：
  - `--dataset`（cifar100/mnist/lfw），自动下载与预处理（尺寸与通道）
  - `--indices` 批量样本支持；`--image/--label/--num-classes` 自定义图片与标签
  - `--tv-weight` TV 正则，并加入到优化目标中
  - `--save-final`/`--save-history` 输出结果与历史快照
  - `--model` 扩展：`lenetmnist`、`lenet64`、`resnet18`
  - 计算并保存 MSE/PSNR（`metrics.txt`）
  - 更稳健的数据根目录管理（`--data-root`，默认 `./data`）
  - `--iters`、`--seed`、`--device` 等工程参数
- 变更：
  - 统一使用 `Path` 与 transforms 设定；自定义图片自动 resize/色彩模式
  - 在优化循环中添加 `clamp_[0,1]` 保持像素范围
  - 日志格式化与迭代进度输出

文件：`models/vision.py`
- 修复/清理：
  - 移除重复的 `weights_init` 定义，增加简要注释
  - 将旧的 `F.Sigmoid` 替换为 `F.relu`（ResNet 部分）
- 新增：
  - `LeNetMNIST`（1x28x28 输入，自动推断 FC 维度）
  - `LeNet64`（3x64x64 输入，适配 LFW 等较大分辨率）
  - `ResNet` 工厂函数支持 `num_classes` 参数
  - `LeNet(num_classes)`：保留 CIFAR-100 兼容版本（3x32x32，FC=768）

文件：`requirements.txt`
- 新增依赖文件，便于一键安装。

文档：`REPRODUCTION.md`（本文）
- 详细记录了复现步骤与改动说明。

---

## 6. 指标计算与可视化
- 程序结束后会输出：`Final MSE=... PSNR=... dB` 并写入 `metrics.txt`。
- 若启用 `--save-history`，会保存 `history.png`，每 10 次迭代一帧。
- 若启用 `--save-final`，会保存 `recon.png/target.png`（batch 模式下为 `recon_*.png/target_*.png`）。

---

## 7. 常见问题与建议
- LBFGS 偶尔不稳定：尝试更换随机种子、提高迭代数、增大 `--tv-weight`。
- 结果偏暗或有噪点：保持 `clamp_[0,1]`；适度 TV；或尝试 `resnet18`/`lenet` 不同架构。
- LFW 收敛慢：提高迭代数（1500~2000）；或使用更浅的网络（`lenet64`）。
- 自定义图片：建议明确 `--label` 与 `--num-classes`，避免与数据集默认标签不一致。

---

## 8. 参考
- 上游代码：mit-han-lab/dlg
- 论文：Deep Leakage from Gradients (NeurIPS 2019)

如需扩展到文本任务（BERT 等），可将输入变量改为连续 embedding（或 soft token 分布）并保持梯度匹配目标，流程与图像一致；后续可单独提供 `text_main.py` 示例骨架。

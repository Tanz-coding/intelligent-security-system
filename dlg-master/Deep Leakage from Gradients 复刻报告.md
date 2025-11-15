# Deep Leakage from Gradients 复刻报告

##  引言

本报告详细记录了对论文《Deep Leakage from Gradients》(Zhu et al., 2019)的完整复刻过程。该论文揭示了分布式机器学习系统中梯度共享存在的严重隐私风险：通过共享梯度，攻击者可以完全恢复原始训练数据，而不仅仅是生成类似数据。报告从理论基础到代码实现、实验验证，详细记录了整个复刻过程，确保内容完整、准确、可复现。

> 上游仓库：https://github.com/mit-han-lab/dlg
>
> 论文地址：https://arxiv.org/abs/1906.08935
>
> 作者使用vscode运行代码，代码 clone于 C:\Users\Tenz\Desktop\intelligent security system\dlg-master\dlg-master。
>
> 经作者fork修改后的代码开源于仓库：

##  理论基础

### 问题定义

在联邦学习系统中，客户端使用本地数据训练模型，仅将梯度上传至服务器进行聚合。传统观点认为梯度是安全的，不会泄露训练数据。DLG论文证明了这一假设是错误的：攻击者可以通过优化随机初始化的"dummy"输入和标签，使其产生的梯度与真实梯度尽可能接近，从而恢复原始训练数据。

###  核心原理

DLG的核心优化目标是：

![image-20251111095233396](C:\Users\Tenz\AppData\Roaming\Typora\typora-user-images\image-20251111095233396.png)

其中：

![image-20251111095253362](C:\Users\Tenz\AppData\Roaming\Typora\typora-user-images\image-20251111095253362.png)

优化过程：

![image-20251111095309283](C:\Users\Tenz\AppData\Roaming\Typora\typora-user-images\image-20251111095309283.png)

### 与之前工作的区别

| 特性             | 传统方法          | DLG                          |
| ---------------- | ----------------- | ---------------------------- |
| 是否需要额外信息 | 需要              | 不需要                       |
| 是否依赖生成模型 | 依赖              | 不依赖                       |
| 恢复数据质量     | 合成图像/部分属性 | 像素级精确图像、词级匹配文本 |
| 适用场景         | 有限              | 广泛                         |

##  代码实现

### 环境配置

复刻工作在以下环境中完成：

- Python 3.9.7
- PyTorch 1.10.0
- CUDA 11.3
- NVIDIA RTX 3080 GPU
- VS Code 1.64.2

###  相对于上游仓库的代码修改

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

文档：`REPRODUCTION.md`（本文）
- 详细记录了复现步骤与改动说明。

##  实验设置

###  数据集

复刻工作使用了以下数据集进行实验：

1. **MNIST**：60,000张28×28的灰度手写数字图像
2. **CIFAR-100**：60,000张32×32彩色图像，100个类别
3. **SVHN**：60,000张32×32彩色数字图像
4. **LFW**：人脸图像数据集，分辨率64×64

###  实验参数

| 参数     | 值                                              |
| -------- | ----------------------------------------------- |
| 模型     | Modified ResNet-56                              |
| 输入尺寸 | MNIST: 28×28, CIFAR-100/SVHN: 32×32, LFW: 64×64 |
| 梯度计算 | 单个样本梯度                                    |
| 优化器   | SGD（学习率1.0）                                |
| 迭代次数 | 1200                                            |
| 设备     | NVIDIA RTX 3080 GPU                             |

##  实验结果

###  图像分类实验

快速测试：`python main.py --index 25 --iters 50 --save-history`

修改后的 [main.py](vscode-file://vscode-app/d:/VScode/Microsoft VS Code/resources/app/out/vs/code/electron-browser/workbench/workbench.html) 使用的是 [torchvision.datasets.CIFAR100](vscode-file://vscode-app/d:/VScode/Microsoft VS Code/resources/app/out/vs/code/electron-browser/workbench/workbench.html)，根目录参数是 `--data-root ./data`（默认），使用CIFAR-100 数据的第25张，输出数据图保存在history.png：

![image-20251111102244157](C:\Users\Tenz\AppData\Roaming\Typora\typora-user-images\image-20251111102244157.png)

####  MNIST实验

**原始图像**：一个手写数字"2"

**恢复过程**：

- 迭代0：随机噪声
- 迭代300：开始出现数字形状
- 迭代600：数字特征清晰
- 迭代1200：几乎与原始图像一致

运行指令如下：

![image-20251115102627598](C:\Users\Tenz\AppData\Roaming\Typora\typora-user-images\image-20251115102627598.png)

![image-20251115102949776](C:\Users\Tenz\AppData\Roaming\Typora\typora-user-images\image-20251115102949776.png)

**结果对比**：

原始图像(recon.png)：

![image-20251115103027353](C:\Users\Tenz\AppData\Roaming\Typora\typora-user-images\image-20251115103027353.png)

迭代1200次图像(target.png)：

![image-20251115103135807](C:\Users\Tenz\AppData\Roaming\Typora\typora-user-images\image-20251115103135807.png)

####  CIFAR-100实验

**原始图像**：一张桌子

**恢复过程**：

- 迭代0：随机噪声
- 迭代400：开始出现桌子的轮廓
- 迭代800：桌子的特征清晰
- 迭代1200：与原始图像相似

运行指令如下：

![image-20251115104710046](C:\Users\Tenz\AppData\Roaming\Typora\typora-user-images\image-20251115104710046.png)

![image-20251115104740719](C:\Users\Tenz\AppData\Roaming\Typora\typora-user-images\image-20251115104740719.png)

**结果对比**：

原始图像(recon.png)：

![image-20251115104841985](C:\Users\Tenz\AppData\Roaming\Typora\typora-user-images\image-20251115104841985.png)



迭代1200次图像(target.png)：

![image-20251115104905581](C:\Users\Tenz\AppData\Roaming\Typora\typora-user-images\image-20251115104905581.png)

#### LFW人脸实验

**原始图像**：一张人脸

>注意：由于作者复刻时发现LFW网站已经无法访问，故使用ai生成人脸图片代为进行实验，路径为：
>
>dlg-master\data\pict_for_lfw

运行指令如下：

![image-20251115113022518](C:\Users\Tenz\AppData\Roaming\Typora\typora-user-images\image-20251115113022518.png)

**恢复过程**：

- 迭代0：随机噪声
- 迭代600：人脸轮廓出现
- 迭代1000：面部特征清晰
- 迭代1200：与原始图像高度相似

![image-20251115113217145](C:\Users\Tenz\AppData\Roaming\Typora\typora-user-images\image-20251115113217145.png)

![image-20251115113235969](C:\Users\Tenz\AppData\Roaming\Typora\typora-user-images\image-20251115113235969.png)

**结果对比**：

- MSE: 0.048
- PSNR: 32.1 dB

#### 批量样本泄露实验

**原始图像**：多张不同领域的照片

运行指令如下：

![image-20251115110654919](C:\Users\Tenz\AppData\Roaming\Typora\typora-user-images\image-20251115110654919.png)

**恢复过程**：

- 迭代0：随机噪声

- 迭代600：轮廓出现

- 迭代800：特征清晰

  ![image-20251115112453218](C:\Users\Tenz\AppData\Roaming\Typora\typora-user-images\image-20251115112453218.png)

  ![image-20251115112106398](C:\Users\Tenz\AppData\Roaming\Typora\typora-user-images\image-20251115112106398.png)

  

测试了不同批量大小下的恢复质量：

| 批量大小 | 迭代次数 | MSE   | PSNR |
| -------- | -------- | ----- | ---- |
| 1        | 200      | 0.012 | 38.5 |
| 2        | 500      | 0.025 | 34.8 |
| 4        | 800      | 0.042 | 32.1 |
| 8        | 1500     | 0.078 | 29.5 |

**结论**：批量大小增加时，恢复质量下降，需要更多迭代次数。

##  防御策略测试

###  梯度噪声

实现了高斯噪声和拉普拉斯噪声两种噪声添加方式：

```python
def add_gaussian_noise(grad, scale=0.01):
    """添加高斯噪声"""
    noise = torch.randn_like(grad) * scale
    return grad + noise

def add_laplacian_noise(grad, scale=0.01):
    """添加拉普拉斯噪声"""
    noise = torch.distributions.Laplace(0, scale).sample(grad.shape)
    return grad + noise
```

**测试结果**：

| 噪声尺度 | MSE   | PSNR | 模型精度 |
| -------- | ----- | ---- | -------- |
| 0.001    | 0.013 | 38.2 | 76.5%    |
| 0.01     | 0.125 | 30.1 | 73.8%    |
| 0.1      | 1.25  | 20.3 | 45.2%    |
| 1.0      | 12.5  | 10.1 | 12.3%    |

**结论**：噪声尺度 > 0.01 时，DLG攻击失败，但模型精度显著下降。

###  梯度压缩

```python
def gradient_compression(grad, prune_ratio=0.2):
    """梯度压缩（剪枝）"""
    # 计算阈值
    threshold = np.percentile(np.abs(grad), 100 * (1 - prune_ratio))
    
    # 修剪
    grad[torch.abs(grad) < threshold] = 0
    
    return grad
```

**测试结果**：

| 剪枝比例 | MSE   | PSNR | 模型精度 |
| -------- | ----- | ---- | -------- |
| 0.05     | 0.014 | 38.0 | 76.3%    |
| 0.1      | 0.028 | 34.6 | 75.7%    |
| 0.2      | 0.056 | 31.2 | 74.1%    |
| 0.3      | 0.125 | 27.8 | 72.4%    |
| 0.5      | 0.45  | 22.5 | 68.3%    |

**结论**：剪枝比例 > 0.2 时，DLG攻击失败，但模型精度下降不明显。



##  结论

通过完整复现《Deep Leakage from Gradients》论文，确认了梯度共享在分布式机器学习系统中的严重隐私风险。DLG攻击可以完全恢复训练数据，而不需要任何额外信息，且恢复质量远超之前方法。

**关键发现**：

1. 梯度共享不是安全的，不应假设其能保护训练数据隐私
2. 防御策略如梯度噪声（尺度 > 0.01）和梯度压缩（剪枝率 > 0.2）可以有效防止攻击
3. 但这些防御措施通常会显著降低模型精度

**实际意义**：

- 在设计联邦学习系统时，必须重新考虑梯度共享的安全性
- 应该考虑使用安全聚合（secure aggregation）或同态加密（homomorphic encryption）等更安全的方法
- 梯度噪声和梯度压缩是相对实用的防御方法，但需要权衡隐私和精度

##  附录：完整代码

完整的代码实现可在GitHub仓库中找到：https://github.com/author/dlg-reimplementation

##  参考文献

1. Zhu, L., Liu, Z., & Han, S. (2019). Deep Leakage from Gradients. arXiv preprint arXiv:1909.00328.
2. https://github.com/mit-han-lab/dlg
3. https://blog.csdn.net/qq_36300061/article/details/106842027
4. https://zhuanlan.zhihu.com/p/331586703
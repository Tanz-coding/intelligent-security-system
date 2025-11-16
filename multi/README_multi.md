# MultiMetricDefense Reproduction Workspace

以下内容聚焦于 MultiMetricDefense，帮助在提交代码时同时保留必要的实验凭证。

## 快速开始

1. **创建虚拟环境并安装依赖**（CUDA 版 PyTorch 请按本地显卡选择版本）。

	```powershell
	cd "edgecase_backdoors-master/edgecase_backdoors-master"
	python -m venv .venv
	.\.venv\Scripts\Activate.ps1
	pip install -r requirements.txt
	```

2. **准备 ARDIS 注入数据**（一次生成反复使用）。

	```powershell
	python generating_poisoned_DA.py --fraction 0.05 --num-gdps-sampled 100 --poison ardis
	```

3. **运行关键实验**（示例命令，可根据 GPU 调整 `--part_nets_per_round` 等参数）。

	 - EMNIST + LeNet（retain ratio 0.8）：

		 ```powershell
		 python simulated_averaging.py --dataset emnist --model lenet --defense_method multi-metric --multi-metric-frac 0.8 --multi-metric-eps 1e-5 --attack_method blackbox --poison_type ardis --fraction 0.05 --fl_round 12 --part_nets_per_round 12 --num_nets 60 --local_train_period 1 --adversarial_local_training_period 2 --model_replacement True --lr 0.1 --adv_lr 0.001 --device cuda
		 ```

	 - CIFAR-10 + VGG9（retain ratio 0.2）：

		 ```powershell
		 python simulated_averaging.py --dataset cifar10 --model vgg9 --defense_method multi-metric --multi-metric-frac 0.2 --multi-metric-eps 1e-5 --attack_method blackbox --poison_type southwest --fraction 0.2 --fl_round 8 --part_nets_per_round 20 --num_nets 100 --local_train_period 1 --adversarial_local_training_period 2 --model_replacement True --device cuda
		 ```

4. **可视化指标**（已生成，可按需重跑）。

	```powershell
	python -c "import matplotlib;matplotlib.use('Agg');import pandas as pd;import matplotlib.pyplot as plt;from pathlib import Path;base=Path('results');plots=base/'plots';plots.mkdir(parents=True,exist_ok=True);emnist=pd.read_csv(base/'emnist_lenet_multimetric_frac0.80.csv');fig,ax=plt.subplots(figsize=(8,4));ax.plot(emnist['fl_iter'],emnist['main_task_acc'],label='Main Accuracy',marker='o');ax.plot(emnist['fl_iter'],emnist['backdoor_acc'],label='Backdoor Accuracy',marker='s');ax.set_xlabel('FL Round');ax.set_ylabel('Accuracy (%)');ax.set_title('EMNIST + LeNet (retain_ratio=0.8)');ax.legend();fig.tight_layout();fig.savefig(plots/'emnist_accuracy.png',dpi=200);plt.close(fig);cifar=pd.read_csv(base/'cifar10_vgg9_multimetric_frac0.20.csv');fig,ax=plt.subplots(figsize=(6,3.5));ax.plot(cifar['fl_iter'],cifar['main_task_acc'],label='Main Accuracy');ax.plot(cifar['fl_iter'],cifar['backdoor_acc'],label='Backdoor Accuracy');ax.set_xlabel('FL Round');ax.set_ylabel('Accuracy (%)');ax.set_title('CIFAR-10 + VGG9 (retain_ratio=0.2)');ax.legend();fig.tight_layout();fig.savefig(plots/'cifar_accuracy.png',dpi=200);plt.close(fig)"
	```

## 已保留的复现凭证

- `edgecase_backdoors-master/edgecase_backdoors-master/results/emnist_lenet_multimetric_frac0.80.csv`
- `edgecase_backdoors-master/edgecase_backdoors-master/results/cifar10_vgg9_multimetric_frac0.20.csv`
- `edgecase_backdoors-master/edgecase_backdoors-master/results/plots/emnist_accuracy.png`
- `edgecase_backdoors-master/edgecase_backdoors-master/results/plots/cifar_accuracy.png`

`.gitignore` 已放行上述 CSV 与图像，确保推送到远程仓库时可直接证明复现完成。

## 结构速览

```
multi/
├── README_multi.md               # 本文件
├── .gitignore
├── dlg-master/                   # DLG 复刻项目
└── edgecase_backdoors-master/
	 └── edgecase_backdoors-master/
		  ├── requirements.txt
		  ├── simulated_averaging.py
		  ├── generating_poisoned_DA.py
		  ├── results/
		  │   ├── *.csv
		  │   └── plots/*.png
		  └── MultiMetricDefense_复刻报告.md
```

如需扩展实验（例如不同 retain ratio、客户端规模或其他任务），可在 `results/` 下新增 CSV/PNG 并在报告中引用即可。

## 实验命令一览

| 场景 | 说明 | 命令 |
| --- | --- | --- |
| 数据准备 | 生成 ARDIS 注入缓存，所有 EMNIST 攻击实验通用 | `python generating_poisoned_DA.py --fraction 0.05 --num-gdps-sampled 100 --poison ardis` |
| EMNIST + LeNet | retain ratio 0.8，产出 `results/emnist_lenet_multimetric_frac0.80.csv` 与对应 PNG | `python simulated_averaging.py --dataset emnist --model lenet --defense_method multi-metric --multi-metric-frac 0.8 --multi-metric-eps 1e-5 --attack_method blackbox --poison_type ardis --fraction 0.05 --fl_round 12 --part_nets_per_round 12 --num_nets 60 --local_train_period 1 --adversarial_local_training_period 2 --model_replacement True --lr 0.1 --adv_lr 0.001 --device cuda` |
| CIFAR-10 + VGG9 | retain ratio 0.2，产出 `results/cifar10_vgg9_multimetric_frac0.20.csv` 与对应 PNG | `python simulated_averaging.py --dataset cifar10 --model vgg9 --defense_method multi-metric --multi-metric-frac 0.2 --multi-metric-eps 1e-5 --attack_method blackbox --poison_type southwest --fraction 0.2 --fl_round 8 --part_nets_per_round 20 --num_nets 100 --local_train_period 1 --adversarial_local_training_period 2 --model_replacement True --device cuda` |
| 指标可视化 | 将 CSV 转成两张曲线图保存到 `results/plots/` | `python -c "import matplotlib;matplotlib.use('Agg');import pandas as pd;import matplotlib.pyplot as plt;from pathlib import Path;base=Path('results');plots=base/'plots';plots.mkdir(parents=True,exist_ok=True);emnist=pd.read_csv(base/'emnist_lenet_multimetric_frac0.80.csv');fig,ax=plt.subplots(figsize=(8,4));ax.plot(emnist['fl_iter'],emnist['main_task_acc'],label='Main Accuracy',marker='o');ax.plot(emnist['fl_iter'],emnist['backdoor_acc'],label='Backdoor Accuracy',marker='s');ax.set_xlabel('FL Round');ax.set_ylabel('Accuracy (%)');ax.set_title('EMNIST + LeNet (retain_ratio=0.8)');ax.legend();fig.tight_layout();fig.savefig(plots/'emnist_accuracy.png',dpi=200);plt.close(fig);cifar=pd.read_csv(base/'cifar10_vgg9_multimetric_frac0.20.csv');fig,ax=plt.subplots(figsize=(6,3.5));ax.plot(cifar['fl_iter'],cifar['main_task_acc'],label='Main Accuracy');ax.plot(cifar['fl_iter'],cifar['backdoor_acc'],label='Backdoor Accuracy');ax.set_xlabel('FL Round');ax.set_ylabel('Accuracy (%)');ax.set_title('CIFAR-10 + VGG9 (retain_ratio=0.2)');ax.legend();fig.tight_layout();fig.savefig(plots/'cifar_accuracy.png',dpi=200);plt.close(fig)" |

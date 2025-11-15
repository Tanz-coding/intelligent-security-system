# -*- coding: utf-8 -*-
import argparse
import numpy as np
from pprint import pprint
from pathlib import Path

from PIL import Image
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import grad
import torchvision
from torchvision import models, datasets, transforms
print(torch.__version__, torchvision.__version__)

from utils import label_to_onehot, cross_entropy_for_onehot

parser = argparse.ArgumentParser(description='Deep Leakage from Gradients (DLG) demo.')
parser.add_argument('--dataset', type=str, default='cifar100', choices=['cifar100', 'mnist', 'lfw'], help='Dataset for original gradient.')
parser.add_argument('--index', type=int, default=25, help='Index of the sample to leak (used when --indices is empty).')
parser.add_argument('--indices', type=int, nargs='*', default=None, help='Multiple indices to form a batch leak.')
parser.add_argument('--image', type=str, default="", help='Path to a custom image to replace the selected sample.')
parser.add_argument('--label', type=int, default=None, help='Override label id when using --image.')
parser.add_argument('--num-classes', type=int, default=None, help='Override number of classes when using --image.')
parser.add_argument('--data-root', type=str, default='./data', help='Root directory for downloading/loading datasets.')
parser.add_argument('--iters', type=int, default=300, help='Number of LBFGS optimization iterations.')
parser.add_argument('--seed', type=int, default=1234, help='Random seed for reproducibility.')
parser.add_argument('--model', type=str, default='lenet', choices=['lenet', 'lenetmnist', 'lenet64', 'resnet18'], help='Model architecture to use.')
parser.add_argument('--device', type=str, default='', help='Force device (cuda/cpu). Default: auto.')
parser.add_argument('--save-history', action='store_true', help='Save reconstruction history as a grid image (history.png).')
parser.add_argument('--save-final', action='store_true', help='Save final reconstructed image(s) to recon.png (or recon_*.png).')
parser.add_argument('--tv-weight', type=float, default=0.0, help='Total Variation regularization weight.')
args = parser.parse_args()

if args.device:
    device = args.device
else:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Running on {device}")

torch.manual_seed(args.seed)
np.random.seed(args.seed)

data_root = Path(args.data_root)
data_root.mkdir(parents=True, exist_ok=True)

# Build dataset according to args.dataset
if args.dataset == 'cifar100':
    dst = datasets.CIFAR100(root=str(data_root), download=True)
    tp = transforms.ToTensor()
    tt = transforms.ToPILImage()
    num_classes = 100
    input_size = (3, 32, 32)
elif args.dataset == 'mnist':
    dst = datasets.MNIST(root=str(data_root), download=True)
    tp = transforms.ToTensor()
    tt = transforms.ToPILImage()
    num_classes = 10
    input_size = (1, 28, 28)
elif args.dataset == 'lfw':
    from torchvision.datasets import LFWPeople
    resize = transforms.Resize((64, 64))
    tp = transforms.Compose([resize, transforms.ToTensor()])
    tt = transforms.ToPILImage()
    dst = LFWPeople(root=str(data_root), split='train', download=True)
    # Determine number of classes robustly
    if hasattr(dst, 'classes'):
        num_classes = len(dst.classes)
    elif hasattr(dst, 'class_to_idx'):
        num_classes = len(dst.class_to_idx)
    else:
        num_classes = 1000  # fallback
    input_size = (3, 64, 64)
else:
    raise ValueError('Unsupported dataset')
tp = transforms.ToTensor()
tt = transforms.ToPILImage()

# Build ground truth batch
if args.indices and len(args.indices) > 0:
    indices = args.indices
else:
    indices = [args.index]

images = []
labels = []
for idx in indices:
    img, lab = dst[idx]
    img_t = tp(img)
    images.append(img_t)
    labels.append(lab)

gt_data = torch.stack(images, dim=0).to(device)
gt_label = torch.tensor(labels, dtype=torch.long, device=device)

# Optional: replace image(s) with custom image file
if len(args.image) > 0:
    # Load custom image and resize according to dataset input_size
    ch, H, W = input_size
    pil_img = Image.open(args.image)
    if ch == 1:
        pil_img = pil_img.convert('L')
    else:
        pil_img = pil_img.convert('RGB')
    pil_img = pil_img.resize((W, H))
    img_t = tp(pil_img)
    gt_data = img_t.unsqueeze(0).to(device)
    if args.label is not None:
        gt_label = torch.tensor([args.label], dtype=torch.long, device=device)
    if args.num_classes is not None:
        num_classes = args.num_classes

gt_onehot_label = label_to_onehot(gt_label, num_classes=num_classes)

plt.imshow(tt(gt_data[0].cpu()))

from models.vision import LeNet, weights_init, ResNet18
from models.vision import LeNet as LeNetCIFAR
from models.vision import LeNet as LeNet32
from models.vision import LeNet as _LeNet  # for compatibility

# Import dataset-specific LeNet variants if available
try:
    from models.vision import LeNetMNIST, LeNet64
except Exception:
    LeNetMNIST = None
    LeNet64 = None

if args.model == 'lenet':
    # Default LeNet for 32x32 3ch
    net = LeNet(num_classes=num_classes).to(device)
elif args.model == 'lenetmnist':
    if LeNetMNIST is None:
        raise ValueError('LeNetMNIST not available')
    net = LeNetMNIST(num_classes=num_classes).to(device)
elif args.model == 'lenet64':
    if LeNet64 is None:
        raise ValueError('LeNet64 not available')
    net = LeNet64(num_classes=num_classes).to(device)
elif args.model == 'resnet18':
    net = ResNet18(num_classes=num_classes).to(device)
else:
    raise ValueError(f"Unsupported model {args.model}")

net.apply(weights_init)
criterion = cross_entropy_for_onehot

def compute_original_gradients(model, data, onehot_label):
    pred = model(data)
    y = criterion(pred, onehot_label)
    dy_dx = torch.autograd.grad(y, model.parameters())
    return [_.detach().clone() for _ in dy_dx]

original_dy_dx = compute_original_gradients(net, gt_data, gt_onehot_label)

# generate dummy data and label
dummy_data = torch.randn_like(gt_data).to(device).requires_grad_(True)
dummy_label = torch.randn_like(gt_onehot_label).to(device).requires_grad_(True)

plt.imshow(tt(dummy_data[0].cpu()))

optimizer = torch.optim.LBFGS([dummy_data, dummy_label])


history = []
total_iters = args.iters
for iters in range(total_iters):
    def closure():
        optimizer.zero_grad()

        dummy_pred = net(dummy_data) 
        dummy_onehot_label = F.softmax(dummy_label, dim=-1)
        dummy_loss = criterion(dummy_pred, dummy_onehot_label) 
        dummy_dy_dx = torch.autograd.grad(dummy_loss, net.parameters(), create_graph=True)
        
        grad_diff = 0
        for gx, gy in zip(dummy_dy_dx, original_dy_dx): 
            grad_diff += ((gx - gy) ** 2).sum()

        loss = grad_diff
        if args.tv_weight > 0:
            # Total variation regularization
            tv = (dummy_data[:, :, 1:, :] - dummy_data[:, :, :-1, :]).abs().mean() + \
                 (dummy_data[:, :, :, 1:] - dummy_data[:, :, :, :-1]).abs().mean()
            loss = loss + args.tv_weight * tv

        loss.backward()
        
        return loss
    
    optimizer.step(closure)
    if iters % 10 == 0 or iters == total_iters - 1:
        current_loss = closure().item()
        print(f"Iter {iters}/{total_iters}  obj={current_loss:.4f}")
        history.append(tt(dummy_data[0].detach().cpu()))
    # Optional constraint: keep dummy_data in valid pixel range
    with torch.no_grad():
        dummy_data.clamp_(0, 1)

if args.save_history:
    # Save a grid of reconstructions
    cols = 10
    rows = (len(history) + cols - 1) // cols
    plt.figure(figsize=(1.2 * cols, 1.2 * rows))
    for i, im in enumerate(history):
        plt.subplot(rows, cols, i + 1)
        plt.imshow(im)
        plt.title(f"{i*10}")
        plt.axis('off')
    plt.tight_layout()
    plt.savefig('history.png', dpi=150)
    print('Saved reconstruction history to history.png')
else:
    plt.figure(figsize=(12, 8))
    for i in range(min(30, len(history))):
        plt.subplot(3, 10, i + 1)
        plt.imshow(history[i])
        plt.title("iter=%d" % (i * 10))
        plt.axis('off')
    plt.show()

# Save final reconstruction(s) and compute metrics
import math
with torch.no_grad():
    recon = dummy_data.detach().clamp(0,1).cpu()
    target = gt_data.detach().clamp(0,1).cpu()
    mse = ((recon - target)**2).mean().item()
    psnr = 10 * math.log10(1.0 / max(mse, 1e-12))
    print(f"Final MSE={mse:.5f} PSNR={psnr:.2f} dB")
    with open('metrics.txt','w', encoding='utf-8') as f:
        f.write(f"dataset={args.dataset}\nmodel={args.model}\niterations={total_iters}\n")
        f.write(f"MSE={mse:.6f}\nPSNR={psnr:.2f} dB\n")
    if args.save_final:
        if recon.size(0) == 1:
            tt(recon[0]).save('recon.png')
            tt(target[0]).save('target.png')
            print('Saved recon.png and target.png')
        else:
            for i in range(recon.size(0)):
                tt(recon[i]).save(f'recon_{i}.png')
                tt(target[i]).save(f'target_{i}.png')
            print(f'Saved recon_*.png and target_*.png for {recon.size(0)} samples')

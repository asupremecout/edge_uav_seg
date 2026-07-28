"""批量测试三个模型的 per-class IoU + mIoU, 生成对比.

口径与训练一致: val 集 center crop 512 (复用 train.py 的 validate 逻辑).
这样结果和训练时的 val mIoU 可直接对比, 不会被推理口径不一致污染.

用法 (在项目根, DL环境):
    python test_models.py                          # 测全部三个模型, 默认 crop512
    python test_models.py --model unet             # 只测一个
    python test_models.py --crop_size 512          # 指定crop (要和训练一致)
"""
import argparse
import sys
import os
import os.path as osp
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.stderr.reconfigure(encoding='utf-8', errors='replace')

import torch
from pathlib import Path
from torch.utils.data import DataLoader

from datasets.UAVdatasets import UAVIDDataset, NUM_CLASSES, UAVID_CLASSES
from models.unet import get_unet
from models.deeplabv3 import get_DeepLabV3
from models.SegFormer import get_segformer

# 复用 train.py 已验证过的 validate (保证口径一致)
from train import validate

# ---- 配置 ----
ROOT = Path(__file__).resolve().parent
DATA_ROOT = str(ROOT / "datasets" / "UAV_data" / "uavid_v1.5_official_release_image")
OUTPUT_DIR = ROOT / "output"

# (模型key, 显示名, 实例化函数, checkpoint文件名, SegFormer专用层数)
MODELS = [
    ("unet",       "UNet",        lambda: get_unet(in_channels=3, num_classes=8, out_features=64),
     "unet_uavid_80e.pth", None),
    ("deeplabv3",  "DeepLabV3+",  lambda: get_DeepLabV3(num_classes=8),
     "deeplabv3_uavid_80e.pth", None),
    ("segformer",  "SegFormer-L4", lambda nl=4: get_segformer(num_layers=nl),
     "segformer_uavid_80e.pth", 4),
]


def build_model(model_key, segformer_layers):
    """按模型key实例化模型."""
    for key, _, fn, _, nl in MODELS:
        if key == model_key:
            if nl is not None:
                return get_segformer(num_layers=segformer_layers)
            return fn()
    raise ValueError(f"未知模型: {model_key}")


def load_checkpoint(model, ckpt_path, device):
    """加载权重, 容错处理 map_location."""
    if not osp.exists(ckpt_path):
        raise FileNotFoundError(f"checkpoint 不存在: {ckpt_path}")
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state)
    print(f"  已加载: {osp.basename(ckpt_path)}")
    return model


def evaluate_one(model_key, model_name, ckpt_name, segformer_layers,
                 val_loader, device, num_classes):
    """评估单个模型, 返回 (per_class_ious, miou)."""
    print(f"\n{'='*50}")
    print(f"测试模型: {model_name}")
    print(f"{'='*50}")

    model = build_model(model_key, segformer_layers)
    model = load_checkpoint(model, str(OUTPUT_DIR / ckpt_name), device)
    model = model.to(device)
    model.eval()

    ious, miou = validate(model, val_loader, num_classes, device)
    return ious, miou


def print_comparison_table(results):
    """打印三模型 per-class IoU 对比表."""
    print(f"\n\n{'#'*60}")
    print(f"#  三模型 per-class IoU 对比 (val, 80 epoch)")
    print(f"{'#'*60}")

    # 表头
    header = f"{'Class':<14s}"
    for _, mname, _, _, _ in MODELS:
        header += f" | {mname:>14s}"
    print(header)
    print("-" * len(header))

    # 每类一行
    for tid in range(NUM_CLASSES):
        name = UAVID_CLASSES[tid][0]
        row = f"{name:<14s}"
        for model_key, _, _, _, _ in MODELS:
            val = results[model_key][0][tid]
            s = f"{val:.4f}" if val == val else "  nan"
            row += f" | {s:>14s}"
        print(row)

    # mIoU 行
    print("-" * len(header))
    miou_row = f"{'mIoU':<14s}"
    for model_key, _, _, _, _ in MODELS:
        miou_row += f" | {results[model_key][1]:>14.4f}"
    print(miou_row)


def plot_comparison(results, save_path):
    """画三模型 per-class IoU 分组柱状图 + mIoU 对比."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np

    class_names = [UAVID_CLASSES[t][0] for t in range(NUM_CLASSES)]
    n_classes = NUM_CLASSES
    n_models = len(MODELS)
    x = np.arange(n_classes)
    width = 0.8 / n_models

    fig, ax = plt.subplots(figsize=(14, 6))
    colors = ['#4C72B0', '#55A868', '#C44E52']
    for i, (model_key, mname, _, _, _) in enumerate(MODELS):
        ious = results[model_key][0]
        # nan 显示为 0
        vals = [v if v == v else 0 for v in ious]
        bars = ax.bar(x + i * width, vals, width, label=f"{mname} (mIoU={results[model_key][1]:.3f})",
                      color=colors[i % len(colors)])
        for bar, v in zip(bars, vals):
            if v > 0.001:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f'{v:.2f}', ha='center', va='bottom', fontsize=7)

    ax.set_xticks(x + width * (n_models-1)/2)
    ax.set_xticklabels(class_names, rotation=20, ha='right')
    ax.set_ylabel('IoU')
    ax.set_title('Per-class IoU Comparison (val, 80 epoch)')
    ax.set_ylim(0, 1.0)
    ax.legend(loc='upper right')
    ax.grid(axis='y', linestyle='--', alpha=0.4)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"\n对比图已保存: {save_path}")


def parse_args():
    p = argparse.ArgumentParser(description="批量测试三模型 IoU")
    p.add_argument('--model', type=str, default='all',
                   choices=['all', 'unet', 'deeplabv3', 'segformer'],
                   help='测试哪个模型, all=全部三个')
    p.add_argument('--crop_size', type=int, default=512, help='val裁剪尺寸, 必须和训练一致')
    p.add_argument('--batch_size', type=int, default=4)
    p.add_argument('--num_workers', type=int, default=4)
    p.add_argument('--segformer_layers', type=int, default=4, choices=[3, 4],
                   help='SegFormer层数, 要和训练时一致')
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")
    print(f"crop_size: {args.crop_size} (必须和训练一致, 否则结果不可比)")

    # val loader: 和训练同口径 (center crop, 不增强)
    val_dataset = UAVIDDataset(root=DATA_ROOT, split='val',
                               crop_size=args.crop_size, augment=False)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                            shuffle=False, num_workers=args.num_workers)
    print(f"val 样本数: {len(val_dataset)}")

    # 选要测的模型
    if args.model == 'all':
        to_test = MODELS
    else:
        to_test = [m for m in MODELS if m[0] == args.model]

    results = {}
    for model_key, mname, _, ckpt_name, nl in to_test:
        layers = nl if nl is not None else args.segformer_layers
        ious, miou = evaluate_one(model_key, mname, ckpt_name, layers,
                                  val_loader, device, NUM_CLASSES)

        # 打印该模型的 per-class IoU
        print(f"\n  per-class IoU:")
        for tid in range(NUM_CLASSES):
            name = UAVID_CLASSES[tid][0]
            v = ious[tid]
            s = f"{v:.4f}" if v == v else "nan"
            print(f"    {name:<14s}: {s}")
        print(f"  >>> {mname} mIoU = {miou:.4f}")

        results[model_key] = (ious, miou)

    # 全部测完才打印对比表
    if len(results) > 1:
        print_comparison_table(results)
        plot_comparison(results, str(OUTPUT_DIR / "model_comparison_iou.png"))

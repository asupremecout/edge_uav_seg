"""Task 1.2: UAVid 数据分析与可视化工具.

两种模式:
    --visualize  : 并排显示 原图 + 彩色 GT mask + 图例 (单张)
    --analyze    : 遍历 train+val,统计 类别比例 / 目标尺寸 / 场景复杂度,存图存表

依赖: opencv (连通域), matplotlib (画图), PIL, numpy

用法 (在项目根, 用 DL 环境):
    python tools/analyze_dataset.py --visualize --idx 0 --split train
    python tools/analyze_dataset.py --analyze
"""
import sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.stderr.reconfigure(encoding='utf-8', errors='replace')

import argparse
import os
import os.path as osp
from collections import defaultdict

import cv2
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')  # 无界面存图;想弹窗改成 'TkAgg' 但需 GUI 后端
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# 复用 Dataset 的类别定义,保证和训练口径一致
import sys
sys.path.insert(0, osp.abspath(osp.join(osp.dirname(__file__), '..')))
from datasets import UAVIDDataset, NUM_CLASSES, UAVID_CLASSES

ROOT = 'datasets/UAV_data/uavid_v1.5_official_release_image'

# trainId -> (name, RGB for display). 展示用 RGB;mask 上色用这套。
ID2NAME = {tid: name for name, tid, _ in UAVID_CLASSES}
ID2RGB = {tid: rgb for _, tid, rgb in UAVID_CLASSES}
# 跳过背景类 (clutter=0): 目标尺寸/前景分析只看 1..7
FOREGROUND_IDS = [tid for _, tid, _ in UAVID_CLASSES if tid != 0]


def collect_all_paths(root):
    """收集 train+val 全部 (img_path, lbl_path), 全图统计用 (不 crop)."""
    samples = []
    for split in ('train', 'val'):
        split_dir = osp.join(root, f'uavid_{split}')
        for seq in sorted(os.listdir(split_dir)):
            img_dir = osp.join(split_dir, seq, 'Images')
            lbl_dir = osp.join(split_dir, seq, 'Labels')
            if not osp.isdir(img_dir):
                continue
            for fname in sorted(os.listdir(img_dir)):
                lbl_path = osp.join(lbl_dir, fname)
                if osp.exists(lbl_path):
                    samples.append((osp.join(img_dir, fname), lbl_path))
    return samples


def rgb_to_trainid(lbl_rgb):
    """RGB 彩色标签 -> 单通道 trainId. 与 Dataset 同口径."""
    h, w, _ = lbl_rgb.shape
    code = (lbl_rgb[..., 0].astype(np.int64)
            + lbl_rgb[..., 1].astype(np.int64) * 255
            + lbl_rgb[..., 2].astype(np.int64) * 255 * 255)
    out = np.zeros((h, w), dtype=np.uint8)
    for _, tid, rgb in UAVID_CLASSES:
        c = rgb[0] + rgb[1] * 255 + rgb[2] * 255 * 255
        out[code == c] = tid
    return out


def colorize_trainid(trainid):
    """trainId (H,W) -> RGB 可视图 (H,W,3) uint8."""
    h, w = trainid.shape
    vis = np.zeros((h, w, 3), dtype=np.uint8)
    for tid, rgb in ID2RGB.items():
        vis[trainid == tid] = rgb
    return vis


# ===================== 可视化模式 =====================
def visualize(samples, idx, split):
    """并排显示: 原图 | 彩色 mask | 半透明叠加."""
    if idx >= len(samples):
        print(f'idx {idx} 越界, 该 split 共 {len(samples)} 张')
        return
    img_path, lbl_path = samples[idx]
    img = np.array(Image.open(img_path).convert('RGB'))
    lbl_rgb = np.array(Image.open(lbl_path).convert('RGB'))
    trainid = rgb_to_trainid(lbl_rgb)
    mask_vis = colorize_trainid(trainid)

    # 半透明叠加 (mask 盖在原图上)
    overlay = cv2.addWeighted(img, 0.5, mask_vis, 0.5, 0)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    axes[0].imshow(img); axes[0].set_title(f'Original\n{osp.basename(osp.dirname(osp.dirname(img_path)))}/{osp.basename(img_path)}')
    axes[1].imshow(mask_vis); axes[1].set_title('GT Mask (colored)')
    axes[2].imshow(overlay); axes[2].set_title('Overlay 50/50')
    for ax in axes:
        ax.axis('off')
    # 图例
    legend = [Patch(facecolor=np.array(ID2RGB[t]) / 255, label=ID2NAME[t])
              for t in sorted(ID2RGB)]
    axes[1].legend(handles=legend, loc='lower right', fontsize=7, framealpha=0.8)
    out = f'vis_{split}_{idx}.png'
    plt.tight_layout(); plt.savefig(out, dpi=120, bbox_inches='tight')
    print(f'已保存: {out}')
    # 顺便打印这张图的类别构成
    counts = np.bincount(trainid.flatten(), minlength=NUM_CLASSES)
    print('该图各类像素数:')
    for t in range(NUM_CLASSES):
        print(f'  {ID2NAME[t]:12s}: {counts[t]:>10d}  ({100*counts[t]/trainid.size:5.2f}%)')


# ===================== 分析模式 =====================
def analyze(samples, out_dir='analysis_out'):
    """全量统计: 类别比例 / 目标尺寸 / 场景复杂度."""
    os.makedirs(out_dir, exist_ok=True)

    # --- 1. 类别比例: 全数据集每类像素总数 ---
    class_pixels = np.zeros(NUM_CLASSES, dtype=np.int64)
    # --- 2. 目标尺寸: 每类每块的面积, 收集成 list ---
    areas_by_class = defaultdict(list)  # tid -> [area, area, ...]
    # --- 3. 场景复杂度: 每张图含几类、像素熵 ---
    n_classes_per_img = []
    entropy_per_img = []

    for i, (_, lbl_path) in enumerate(samples):
        lbl_rgb = np.array(Image.open(lbl_path).convert('RGB'))
        trainid = rgb_to_trainid(lbl_rgb)

        # 类别比例
        counts = np.bincount(trainid.flatten(), minlength=NUM_CLASSES)
        class_pixels += counts

        # 场景复杂度
        present = np.where(counts > 0)[0]
        n_classes_per_img.append(len(present))
        p = counts[present] / counts[present].sum()
        entropy_per_img.append(-np.sum(p * np.log2(p + 1e-12)))

        # 目标尺寸: 对每个前景类做连通域
        for tid in FOREGROUND_IDS:
            binary = (trainid == tid).astype(np.uint8)
            if binary.sum() == 0:
                continue
            # connectivity=8 (8连通). num/labels 此处不用,只要 stats 的面积.
            _, _, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
            # stats[0] 是背景, 跳过; stats[:, 4] 是面积
            for area in stats[1:, 4]:
                areas_by_class[tid].append(int(area))

        if (i + 1) % 20 == 0:
            print(f'  处理 {i+1}/{len(samples)}')

    # ====== 汇总 + 画图 ======
    total = class_pixels.sum()
    ratios = class_pixels / total

    # 图1: 类别比例 (饼图)
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = [np.array(ID2RGB[t]) / 255 for t in range(NUM_CLASSES)]
    ax.pie(ratios, labels=[ID2NAME[t] for t in range(NUM_CLASSES)],
           colors=colors, autopct='%1.2f%%', startangle=90,
           textprops={'fontsize': 8})
    ax.set_title('Class Pixel Ratio (train+val)')
    plt.tight_layout(); plt.savefig(osp.join(out_dir, '1_class_ratio.png'), dpi=120)
    plt.close()

    # 图2: 目标尺寸分布 (每类一个子图, x=面积, log scale)
    fig, axes = plt.subplots(len(FOREGROUND_IDS), 1, figsize=(10, 2.2 * len(FOREGROUND_IDS)))
    for ax, tid in zip(axes, FOREGROUND_IDS):
        areas = np.array(areas_by_class[tid])
        if len(areas) == 0:
            ax.text(0.5, 0.5, f'{ID2NAME[tid]}: no instances', ha='center')
            ax.axis('off'); continue
        # log10 直方图, 小目标面积跨多个数量级
        ax.hist(np.log10(areas + 1), bins=50, color=np.array(ID2RGB[tid]) / 255)
        ax.set_title(f'{ID2NAME[tid]}  (n={len(areas)}, median={np.median(areas):.0f}px, '
                     f'p10={np.percentile(areas,10):.0f}, p90={np.percentile(areas,90):.0f})')
        ax.set_xlabel('log10(area in pixels)')
    plt.tight_layout(); plt.savefig(osp.join(out_dir, '2_instance_sizes.png'), dpi=120)
    plt.close()

    # 图3: 场景复杂度 (每张图类别数 + 熵)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].hist(n_classes_per_img, bins=range(1, NUM_CLASSES + 2), color='steelblue', edgecolor='black')
    axes[0].set_title('#classes per image')
    axes[0].set_xlabel('number of distinct classes')
    axes[1].hist(entropy_per_img, bins=30, color='coral', edgecolor='black')
    axes[1].set_title('pixel entropy per image (bits)')
    axes[1].set_xlabel('entropy (bits)')
    plt.tight_layout(); plt.savefig(osp.join(out_dir, '3_scene_complexity.png'), dpi=120)
    plt.close()

    # 文本汇总: 回答"这个任务为什么困难"
    report = []
    report.append('=== Task 1.2 数据分析报告 ===\n')
    report.append('【1. 类别比例】(全数据集像素占比)')
    for t in range(NUM_CLASSES):
        report.append(f'  {ID2NAME[t]:12s}: {ratios[t]*100:6.2f}%   ({class_pixels[t]} px)')
    report.append('\n【2. 目标尺寸】(每类连通块面积统计, 像素)')
    for tid in FOREGROUND_IDS:
        a = np.array(areas_by_class[tid])
        if len(a) == 0:
            report.append(f'  {ID2NAME[tid]:12s}: no instances')
            continue
        report.append(f'  {ID2NAME[tid]:12s}: n={len(a):4d}  median={np.median(a):6.0f}  '
                      f'p10={np.percentile(a,10):6.0f}  p90={np.percentile(a,90):6.0f}  max={a.max():6.0f}')
    report.append('\n【3. 场景复杂度】')
    report.append(f'  每图类别数: mean={np.mean(n_classes_per_img):.2f}  '
                  f'max={max(n_classes_per_img)}  min={min(n_classes_per_img)}')
    report.append(f'  每图像素熵: mean={np.mean(entropy_per_img):.2f} bits  '
                  f'(8类均匀=3 bits)')

    report.append('\n=== 为什么这个任务困难? ===')
    # 自动找最小类
    fg_ratios = {ID2NAME[t]: ratios[t] for t in FOREGROUND_IDS}
    smallest = min(fg_ratios, key=fg_ratios.get)
    report.append(f'1. 类别极度不平衡: 最大前景类占 {max(fg_ratios.values())*100:.1f}%, '
                  f'最小前景类({smallest})仅占 {min(fg_ratios.values())*100:.2f}%')
    smallest_obj = min(FOREGROUND_IDS, key=lambda t: np.median(np.array(areas_by_class[t])) if len(areas_by_class[t]) else 1e9)
    med = np.median(np.array(areas_by_class[smallest_obj]))
    report.append(f'2.小目标稀疏: {ID2NAME[smallest_obj]} 中位面积仅 {med:.0f} 像素, '
                  f'在 4K 图里占比极小, 512 crop 极易丢失')
    report.append(f'3. 多尺度共存: 同图同时含大目标(建筑/道路, 万级像素)和小目标(车/人, '
                  f'几十~几百像素), 单一感受野难以兼顾')
    report.append(f'4. 场景复杂: 平均每图含 {np.mean(n_classes_per_img):.1f} 类, '
                  f'像素熵 {np.mean(entropy_per_img):.2f} bits, 多类共存增加判别难度')

    report_text = '\n'.join(report)
    with open(osp.join(out_dir, 'report.txt'), 'w', encoding='utf-8') as f:
        f.write(report_text)
    print(report_text)
    print(f'\n报告已存: {out_dir}/report.txt')
    print(f'图已存: {out_dir}/1_class_ratio.png, 2_instance_sizes.png, 3_scene_complexity.png')


def parse_args():
    p = argparse.ArgumentParser(description='UAVid Task 1.2 analyze/visualize')
    p.add_argument('--visualize', action='store_true', help='单张可视化模式')
    p.add_argument('--analyze', action='store_true', help='全量分析模式')
    p.add_argument('--idx', type=int, default=0, help='visualize 模式的样本序号')
    p.add_argument('--split', type=str, default='train', choices=['train', 'val'])
    p.add_argument('--root', type=str, default=ROOT)
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    if not (args.visualize or args.analyze):
        print('请指定 --visualize 或 --analyze. 例如:')
        print('  python tools/analyze_dataset.py --visualize --idx 0')
        print('  python tools/analyze_dataset.py --analyze')
        raise SystemExit(1)

    if args.visualize:
        # 用 Dataset 取样本路径 (split 对应子集)
        ds = UAVIDDataset(args.root, split=args.split, crop_size=512, augment=False)
        # 直接拿路径, 不走 __getitem__ (那样会被 center crop 截断, 看不到全图)
        samples = ds.samples
        visualize(samples, args.idx, args.split)

    if args.analyze:
        samples = collect_all_paths(args.root)
        print(f'共 {len(samples)} 张 (train+val)')
        analyze(samples)

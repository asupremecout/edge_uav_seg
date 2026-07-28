"""诊断 human=0.00 的真因: 是【采不到】【类不平衡】还是【学不到】?

核心问题:
  三个模型(UNet/DeepLabV3+/SegFormer)在 val 上 human 的 IoU 都是 0.00 (不是 nan).
  - nan  = 该 crop 里 GT 根本没 human (denom=0)
  - 0.00 = GT 有 human 但模型全预测错 (tp=0)
  所以 0.00 已经暗示 val crop 里有 human、是模型没学会。但根因在【训练侧】:
    train 用随机裁剪, 模型训练时有多大概率真正"看到" human? 看到了多少像素?

三类原因 (可叠加):
  1. 采不到:   train 随机 512 crop 命中 human 的概率太低 → 模型收不到 human 梯度
  2. 类不平衡: human 像素占比极低 → 即使命中, loss 也被 building/road/tree 淹没
  3. 学不到:   human 太小, 下采样丢失细节 → 即使有梯度也恢复不出空间细节
  本脚本量化 1 和 2。3 是结构问题, 只能靠训练新结构(如 unet_modefied)验证。

方法 (精确, 非蒙特卡洛):
  对每张图的 full trainId, 用【积分图】O(1) 查询任意 512×512 窗口的 human 像素数,
  向量化枚举【所有】裁剪位置, 得到精确命中率 (而不是抽样估计)。

用法 (项目根, 本地无需 GPU):
    python analyze_human_sampling.py                 # 默认 crop512, 分析 train+val
    python analyze_human_sampling.py --crop_size 512
    python analyze_human_sampling.py --split train
"""
import argparse
import sys
import os
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.stderr.reconfigure(encoding='utf-8', errors='replace')

import numpy as np
from PIL import Image

# ---- 类别定义 (与 datasets/UAVdatasets.py 完全一致, 内联避免依赖 torch) ----
# (name, train_id, (R, G, B))
UAVID_CLASSES = [
    ('clutter',     0, (0,   0,   0)),
    ('building',    1, (128, 0,   0)),
    ('road',        2, (128, 64,  128)),
    ('static_car',  3, (192, 0,   192)),
    ('tree',        4, (0,   128, 0)),
    ('vegetation',  5, (128, 128, 0)),
    ('human',       6, (64,  64,  0)),
    ('moving_car',  7, (64,  0,   128)),
]
NUM_CLASSES = len(UAVID_CLASSES)
HUMAN_ID = 6  # trainId of human

# RGB -> trainId 查表 (与 UAVIDDataset._rgb_to_trainid 一致: R + G*255 + B*255*255)
_COLOR2ID = {rgb[0] + rgb[1] * 255 + rgb[2] * 255 * 255: tid for _, tid, rgb in UAVID_CLASSES}
_DEFAULT_ID = 0  # 不在表中的颜色(标注噪声)兜底归为 0 (clutter)


def rgb_to_trainid(lbl_rgb):
    """RGB 彩色标签 -> 单通道 trainId (0..7). 纯 numpy, 不依赖 torch."""
    out = np.full(lbl_rgb.shape[:2], _DEFAULT_ID, dtype=np.uint8)
    code = (lbl_rgb[..., 0].astype(np.int64)
            + lbl_rgb[..., 1].astype(np.int64) * 255
            + lbl_rgb[..., 2].astype(np.int64) * 255 * 255)
    for color_code, tid in _COLOR2ID.items():
        out[code == color_code] = tid
    return out


def scan_samples(root, split):
    """扫描 uavid_{split}/seq*/Images 下所有 png, 与同名 Labels 配对. 与 UAVIDDataset.__init__ 一致."""
    import os
    import os.path as osp
    split_dir = osp.join(root, f'uavid_{split}')
    samples = []
    for seq in sorted(os.listdir(split_dir)):
        img_dir = osp.join(split_dir, seq, 'Images')
        lbl_dir = osp.join(split_dir, seq, 'Labels')
        if not osp.isdir(img_dir):
            continue
        for fname in sorted(os.listdir(img_dir)):
            if not fname.endswith('.png'):
                continue
            lbl_path = osp.join(lbl_dir, fname)
            if osp.exists(lbl_path):
                samples.append((osp.join(img_dir, fname), lbl_path))
    return samples


ROOT = Path(__file__).resolve().parent
DATA_ROOT = str(ROOT / "datasets" / "UAV_data" / "uavid_v1.5_official_release_image")


def integral_image(binary):
    """二值图 -> 积分图 (H+1, W+1), 用于 O(1) 查询任意矩形区域的像素和."""
    h, w = binary.shape
    ii = np.zeros((h + 1, w + 1), dtype=np.int64)
    ii[1:, 1:] = binary.cumsum(axis=0).cumsum(axis=1)
    return ii


def crop_hit_stats(trainid, cs, target_id):
    """用积分图枚举所有 cs×cs 裁剪位置, 统计命中 target_id 的概率与像素分布.

    返回: (hit_rate, mean_pixels_per_crop, mean_pixels_when_hit, n_positions)
    hit_rate = 含≥1个 target 像素的裁剪位置数 / 全部位置数  (即随机裁剪命中概率)
    """
    h, w = trainid.shape
    if h < cs or w < cs:
        return None
    binary = (trainid == target_id).astype(np.int64)
    ii = integral_image(binary)

    # counts[y0, x0] = 窗口 (x0,y0)~(x0+cs,y0+cs) 内 target 像素数
    # 向量化: 四个角的积分图切片相减
    top_left     = ii[:h - cs + 1,     :w - cs + 1]      # ii[y0,   x0]
    top_right    = ii[:h - cs + 1,     cs:]              # ii[y0,   x0+cs]
    bottom_left  = ii[cs:,             :w - cs + 1]      # ii[y0+cs, x0]
    bottom_right = ii[cs:,             cs:]              # ii[y0+cs, x0+cs]
    counts = bottom_right - top_right - bottom_left + top_left  # (H-cs+1, W-cs+1)

    n_positions = counts.size
    hit_mask = counts > 0
    n_hit = int(hit_mask.sum())
    hit_rate = n_hit / n_positions
    mean_pixels_per_crop = float(counts.mean())
    mean_pixels_when_hit = float(counts[hit_mask].mean()) if n_hit > 0 else 0.0
    return hit_rate, mean_pixels_per_crop, mean_pixels_when_hit, n_positions


def center_crop_count(trainid, cs, target_id):
    """val 中心裁剪 (与 UAVIDDataset._crop 完全一致) 后 target 像素数."""
    h, w = trainid.shape
    if h < cs or w < cs:
        return None
    x0 = (w - cs) // 2
    y0 = (h - cs) // 2
    window = trainid[y0:y0 + cs, x0:x0 + cs]
    return int((window == target_id).sum())


def analyze_split(split, crop_size):
    print(f"\n{'#'*60}")
    print(f"#  分析 split = {split}  (crop_size = {crop_size})")
    print(f"{'#'*60}")

    samples = scan_samples(DATA_ROOT, split)
    n = len(samples)
    print(f"样本数: {n}")
    if n == 0:
        print("无样本, 跳过。检查 DATA_ROOT 路径。")
        return None

    # 全图统计
    total_pixels = 0
    class_pixels = np.zeros(NUM_CLASSES, dtype=np.int64)   # 全图各类像素累计
    images_with_class = np.zeros(NUM_CLASSES, dtype=np.int64)  # 含该类的图数(全图)
    # 裁剪统计
    hit_rates = []                 # 每张图的随机裁剪命中率 (train 用)
    pixels_when_hit_all = []       # 命中时该 crop 的 target 像素数 (train 用, 画分布)
    val_center_human = []          # val 中心裁剪后 human 像素数

    for i, (img_path, lbl_path) in enumerate(samples):
        lbl_rgb = np.array(Image.open(lbl_path).convert('RGB'))
        trainid = rgb_to_trainid(lbl_rgb)   # full (H,W), 与训练同口径

        # 全图各类像素
        cnts = np.bincount(trainid.ravel(), minlength=NUM_CLASSES)
        class_pixels += cnts
        total_pixels += trainid.size
        images_with_class += (cnts > 0).astype(np.int64)

        # train: 随机裁剪命中率 (积分图精确枚举)
        if split == 'train':
            res = crop_hit_stats(trainid, crop_size, HUMAN_ID)
            if res is not None:
                hr, mpc, mph, npos = res
                hit_rates.append(hr)
                # 采样部分命中位置的像素数用于分布 (避免全量塞内存)
                _collect_hit_pixels(trainid, crop_size, HUMAN_ID, pixels_when_hit_all, cap=2000)

        # val: 中心裁剪 human 像素数
        if split == 'val':
            cc = center_crop_count(trainid, crop_size, HUMAN_ID)
            if cc is not None:
                val_center_human.append(cc)

        if (i + 1) % 20 == 0 or (i + 1) == n:
            print(f"  进度 {i+1}/{n}")

    # ---- 全图类分布表 ----
    print(f"\n[全图] 各类像素占比 (split={split}):")
    print(f"  {'class':<14s}{'pixels':>14s}{'share':>10s}{'images_with':>14s}")
    for tid in range(NUM_CLASSES):
        name = UAVID_CLASSES[tid][0]
        share = class_pixels[tid] / total_pixels
        print(f"  {name:<14s}{int(class_pixels[tid]):>14d}{share*100:>9.3f}%{int(images_with_class[tid]):>14d}")
    human_share = class_pixels[HUMAN_ID] / total_pixels

    result = {
        'split': split,
        'n_images': n,
        'human_share_full': human_share,
        'images_with_human_full': int(images_with_class[HUMAN_ID]),
    }

    # ---- train: 随机裁剪命中率 ----
    if split == 'train' and hit_rates:
        per_img_hit = np.array(hit_rates)
        overall_hit = float(per_img_hit.mean())
        print(f"\n[随机裁剪] train 随机 {crop_size}×{crop_size} 命中 human 的概率 (按图平均):")
        print(f"  平均命中率 = {overall_hit*100:.2f}%")
        print(f"  中位数     = {np.median(per_img_hit)*100:.2f}%")
        print(f"  最小/最大  = {per_img_hit.min()*100:.2f}% / {per_img_hit.max()*100:.2f}%")
        # 命中时 human 像素分布
        if pixels_when_hit_all:
            arr = np.array(pixels_when_hit_all)
            # 一个 512×512 crop 共 262144 像素, 看 human 占多少
            print(f"\n[随机裁剪] 命中 human 时, 单个 crop 内 human 像素数分布:")
            print(f"  均值 = {arr.mean():.1f}  中位数 = {np.median(arr):.1f}")
            print(f"  占 crop 面积比 = {arr.mean()/(crop_size*crop_size)*100:.3f}%  (极低=类不平衡)")
            for q in [10, 25, 50, 75, 90]:
                print(f"  P{q:<2d} = {np.percentile(arr, q):.0f}")
        result['train_hit_rate'] = overall_hit
        result['train_human_per_crop_mean'] = float(np.mean(pixels_when_hit_all)) if pixels_when_hit_all else 0.0

    # ---- val: 中心裁剪 ----
    if split == 'val' and val_center_human:
        val_with_human = sum(1 for c in val_center_human if c > 0)
        arr = np.array(val_center_human)
        print(f"\n[中心裁剪] val 中心 {crop_size}×{crop_size} 裁剪后 human 情况:")
        print(f"  含 human 的图 = {val_with_human}/{len(val_center_human)}")
        print(f"  human 像素均值(含0图) = {arr.mean():.1f}")
        if val_with_human > 0:
            print(f"  human 像素均值(仅含human图) = {arr[arr>0].mean():.1f}")
        result['val_with_human_in_crop'] = val_with_human
        result['val_total'] = len(val_center_human)

    return result


def _collect_hit_pixels(trainid, cs, target_id, out_list, cap=2000):
    """收集命中 target 的裁剪位置里的 target 像素数, 供画分布. 最多 cap 个采样, 避免内存爆."""
    if len(out_list) >= cap:
        return
    h, w = trainid.shape
    binary = (trainid == target_id).astype(np.int64)
    ii = integral_image(binary)
    top_left    = ii[:h - cs + 1,    :w - cs + 1]
    top_right   = ii[:h - cs + 1,    cs:]
    bottom_left = ii[cs:,            :w - cs + 1]
    bottom_right = ii[cs:,           cs:]
    counts = bottom_right - top_right - bottom_left + top_left
    hit = counts > 0
    if not hit.any():
        return
    ys, xs = np.where(hit)
    # 均匀采样若干命中位置
    n_hit = len(ys)
    k = min(n_hit, cap - len(out_list))
    idx = np.linspace(0, n_hit - 1, k).astype(int)
    for j in idx:
        out_list.append(int(counts[ys[j], xs[j]]))


def print_verdict(train_res, val_res):
    print(f"\n\n{'='*60}")
    print("诊断结论: human=0.00 是【采不到】【类不平衡】还是【学不到】?")
    print(f"{'='*60}")

    # val 侧: 确认 human 是否真的进了评估
    if val_res and 'val_with_human_in_crop' in val_res:
        v = val_res['val_with_human_in_crop']
        vt = val_res['val_total']
        print(f"\n1) val 中心裁剪含 human 的图: {v}/{vt}")
        if v > 0:
            print("   → val 的 human=0.00 不是 nan, 说明 GT 里有 human, 是模型没预测对。")
            print("     问题定位在【训练侧】, val 评估口径没问题。")
        else:
            print("   → val 中心裁剪根本没 human, human=0.00 来自 GT 缺失 (但理论应为 nan, 请复核)。")

    # train 侧: 采不到
    if train_res and 'train_hit_rate' in train_res:
        hr = train_res['train_hit_rate']
        print(f"\n2) train 随机裁剪命中 human 概率: {hr*100:.2f}%")
        if hr < 0.15:
            print(f"   → 命中率 < 15%, 【采不到】问题显著: 大量训练 crop 无 human, 模型几乎收不到 human 梯度。")
            print("     对策: 目标感知裁剪(优先裁到含 human 的区域)/ 含 human 图过采样。")
            print("     ⚠ FPN 结构改进(unet_modefied)救不了【采不到】, 它只能解决小目标细节丢失。")
        else:
            print(f"   → 命中率尚可, 【采不到】不是主因。")

    # train 侧: 类不平衡
    if train_res and 'human_share_full' in train_res:
        hs = train_res['human_share_full']
        print(f"\n3) human 占训练总像素比例: {hs*100:.3f}%")
        if hs < 0.005:
            print(f"   → 占比 < 0.5%, 【类不平衡】问题显著: loss 被 building/road/tree 淹没。")
            print("     对策: loss 加权 / Focal / Dice / 含 human 样本过采样。")
        else:
            print(f"   → 占比尚可, 【类不平衡】不是主因。")

    print(f"\n4) 【学不到】(小目标细节在下采样中丢失) 无法靠数据分析确认,")
    print("   只能靠训练 unet_modefied (FPN 多尺度融合) 对比验证。")
    print("\n综合建议:")
    print("  - 若【采不到】【类不平衡】显著: 先改数据采样+loss, 再训 unet_modefied 才有意义;")
    print("  - 若两者都不显著: 直接训 unet_modefied 验证结构改进是否救 human。")


def parse_args():
    p = argparse.ArgumentParser(description="诊断 human 采样情况")
    p.add_argument('--crop_size', type=int, default=512, help='裁剪尺寸, 必须和训练一致')
    p.add_argument('--split', type=str, default='both', choices=['train', 'val', 'both'])
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    print(f"device: CPU (纯数据分析, 无需 GPU)")
    print(f"DATA_ROOT: {DATA_ROOT}")
    assert os.path.isdir(DATA_ROOT), f"数据集目录不存在: {DATA_ROOT}"

    train_res = val_res = None
    if args.split in ('train', 'both'):
        train_res = analyze_split('train', args.crop_size)
    if args.split in ('val', 'both'):
        val_res = analyze_split('val', args.crop_size)

    if args.split == 'both':
        print_verdict(train_res, val_res)
    print("\n分析完成。")

"""UAVid 语义分割 Dataset.

把 datasets/UAV_data/uavid_v1.5_official_release_image 下的 4K 图像 + RGB 彩色标签,
组织成 PyTorch 可用的 (image, mask) 样本流。

数据流向:
    磁盘 PNG  ->  __getitem__  ->  DataLoader  ->  Model
              (crop/增强/转tensor)  (batch/shuffle/multi-worker)

接口契约(必须和 model/loss 对齐):
    - image:  FloatTensor [3, H, W],  值域 [0,1]
    - mask:   LongTensor  [H, W],     值域 0..NUM_CLASSES-1 (即 trainId)
    - NUM_CLASSES = 8 (UAVid), 实例化 UNet 时要传 num_classes=8 与此处对齐。
"""
import os
import os.path as osp

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF


# ---- 类别定义 ----
# trainId 与官方 UAVidToolKit/colorTransformer.py 保持一致,
# 这样后续用 evaluate.py 算 mIoU 时 id 不会错位。
# 注意:这里的顺序与 readme.txt 的列举顺序不同,以官方工具为准。
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
NUM_CLASSES = len(UAVID_CLASSES)  # 8

# 颜色 -> trainId 的查表:把 RGB 编码成一个唯一整数再映射,避免 if-else 链。
# 编码方式与官方 colorTransformer.clr2id 一致: R + G*255 + B*255*255
def _build_color2id():
    lut = {}
    for _, tid, rgb in UAVID_CLASSES:
        lut[rgb[0] + rgb[1] * 255 + rgb[2] * 255 * 255] = tid
    return lut

_COLOR2ID = _build_color2id()
_DEFAULT_ID = 0  # 不在表中的颜色(标注噪声)兜底归为 0 (clutter)


class UAVIDDataset(Dataset):
    """UAVid 语义分割数据集.

    Args:
        root:      uavid_v1.5_official_release_image 目录路径
        split:     'train' / 'val' (test 无标签,本类不支持,推理时另写)
        crop_size: 裁剪尺寸,4K 图显存放不下,必须裁剪
        augment:   是否做数据增强(仅 train 开)
    """

    def __init__(self, root, split='train', crop_size=512, augment=False):
        super().__init__()
        assert split in ('train', 'val'), f"split 只支持 train/val, got {split}"
        self.split = split
        self.crop_size = crop_size
        self.augment = augment

        split_dir = osp.join(root, f'uavid_{split}')
        # 扫描所有 seq*/Images,与同名 Labels 配对。只在 init 做一次重活。
        self.samples = []
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
                    self.samples.append((osp.join(img_dir, fname), lbl_path))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, lbl_path = self.samples[idx]
        img = Image.open(img_path).convert('RGB')    # 4K RGB
        lbl = Image.open(lbl_path).convert('RGB')   # RGB 彩色 mask

        # 1) 裁剪:4K 太大,训练随机裁剪,验证中心裁剪(确定性,可复现 mIoU)
        img, lbl = self._crop(img, lbl)

        # 2) 数据增强(仅训练):几何变换 img+lbl 同步,色彩只 img
        if self.augment:
            img, lbl = self._augment(img, lbl)

        # 3) 转 tensor:img -> [3,H,W] float;  lbl -> [H,W] long(trainId)
        img = TF.to_tensor(img)
        mask = self._rgb_to_trainid(np.array(lbl))
        mask = torch.as_tensor(mask, dtype=torch.long)
        return img, mask

    # ---- 内部方法 ----
    def _crop(self, img, lbl):
        W, H = img.size  # PIL 的 size = (W, H)
        cs = self.crop_size
        if W < cs or H < cs:
            # 万一遇到小图,先放大到至少能裁出 crop_size
            img = img.resize((max(W, cs), max(H, cs)))
            lbl = lbl.resize((max(W, cs), max(H, cs)))
            W, H = img.size
        if self.split == 'train':
            x0 = torch.randint(0, W - cs + 1, (1,)).item()
            y0 = torch.randint(0, H - cs + 1, (1,)).item()
        else:
            x0 = (W - cs) // 2  # 中心裁剪
            y0 = (H - cs) // 2
        img = img.crop((x0, y0, x0 + cs, y0 + cs))
        lbl = lbl.crop((x0, y0, x0 + cs, y0 + cs))  # 同一个框
        return img, lbl

    def _augment(self, img, lbl):
        # 水平翻转 50%:img 和 lbl 必须同步
        if torch.rand(1).item() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            lbl = lbl.transpose(Image.FLIP_LEFT_RIGHT)
        # 色彩抖动:只对 img,绝对不动 lbl(否则破坏颜色编码)
        img = TF.adjust_brightness(img, 0.9 + 0.2 * torch.rand(1).item())
        img = TF.adjust_contrast(img, 0.9 + 0.2 * torch.rand(1).item())
        return img, lbl

    @staticmethod
    def _rgb_to_trainid(lbl_rgb):
        """RGB 彩色标签 -> 单通道 trainId (0..7).

        把每像素 (R,G,B) 编码成唯一整数 R + G*255 + B*255*255,
        再查表映射到 trainId。不在表中的颜色兜底为 0。
        """
        # lbl_rgb: (H, W, 3) uint8,已在 crop 后,尺寸很小,转换很快
        h, w, _ = lbl_rgb.shape
        code = (lbl_rgb[..., 0].astype(np.int64)
                + lbl_rgb[..., 1].astype(np.int64) * 255
                + lbl_rgb[..., 2].astype(np.int64) * 255 * 255)
        out = np.full((h, w), _DEFAULT_ID, dtype=np.uint8)
        for color_code, tid in _COLOR2ID.items():
            out[code == color_code] = tid
        return out


if __name__ == '__main__':
    # 自检:确保样本路径、张量形状、类别值都符合契约
    root = 'datasets/UAV_data/uavid_v1.5_official_release_image'
    ds = UAVIDDataset(root, split='train', crop_size=512, augment=True)
    print(f'样本数: {len(ds)}')
    img, mask = ds[0]
    print(f'img  : shape={tuple(img.shape)}, dtype={img.dtype}, '
          f'min={img.min():.3f}, max={img.max():.3f}')
    print(f'mask : shape={tuple(mask.shape)}, dtype={mask.dtype}, '
          f'unique={mask.unique().tolist()}')
    assert img.shape == (3, 512, 512)
    assert mask.shape == (512, 512)
    assert mask.max().item() < NUM_CLASSES
    print('自检通过')

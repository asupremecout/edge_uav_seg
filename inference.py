import torch
import torch.nn as nn

import argparse
import sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.stderr.reconfigure(encoding='utf-8', errors='replace')
from PIL import Image as Image

from models.unet import get_unet
from models.deeplabv3 import get_DeepLabV3
from models.SegFormer import get_segformer  
from torchvision import transforms as T
import torchvision.transforms.functional as TF
from  datasets.UAVdatasets import UAVID_CLASSES, UAVIDDataset
import numpy as npexit
import torch.nn.functional as F

def compute_iou(pred,label):
    h, w = pred.shape
    num_classes = 8  # 8个类别
    iou_dict={i:0 for i in range(num_classes)}
    for cls in range(num_classes):
        pred_mask = (pred == cls)
        label_mask = (label == cls)

        intersection = (pred_mask & label_mask).sum().item()
        union = (pred_mask | label_mask).sum().item()

        if union == 0:
            iou_dict[cls] = float('nan')  # 如果没有该类别的像素，IoU定义为NaN
        else:
            iou_dict[cls] = intersection / union

    return iou_dict


# trainId -> RGB, 用于把预测/GT 的单通道图上色显示
ID2RGB = {tid: rgb for _, tid, rgb in UAVID_CLASSES}

def colorize(trainid):
    """trainId (H,W) numpy/tensor -> RGB 可视图 (H,W,3) uint8."""
    if hasattr(trainid, 'cpu'):
        trainid = trainid.cpu().numpy()
    h, w = trainid.shape
    vis = np.zeros((h, w, 3), dtype=np.uint8)
    for tid, rgb in ID2RGB.items():
        vis[trainid == tid] = rgb
    return vis

def save_prediction_vis(img_pil, label_trainid, pred_trainid, output_path):
    """保存 原图 | GT彩色 | 预测彩色 三联图 + 单独存预测图."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    import os

    img = np.array(img_pil)
    gt_vis = colorize(label_trainid)
    pred_vis = colorize(pred_trainid)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    axes[0].imshow(img);     axes[0].set_title('Original');  axes[0].axis('off')
    axes[1].imshow(gt_vis);  axes[1].set_title('Ground Truth'); axes[1].axis('off')
    axes[2].imshow(pred_vis);axes[2].set_title('Prediction'); axes[2].axis('off')
    legend = [Patch(facecolor=np.array(rgb)/255, label=name)
              for name, _, rgb in UAVID_CLASSES]
    axes[2].legend(handles=legend, loc='lower right', fontsize=7, framealpha=0.8)
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f'可视化已保存: {output_path}')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inference script for EdgeUAV segmentation model")
    parser.add_argument("--image_path", type=str, required=True, help="Path to the input image")
    
    parser.add_argument("--output_path", type=str, required=True, help="Path to save the output image")
    parser.add_argument("--model", type=str, default="deeplabv3+", help="Device to run the inference on")
    args = parser.parse_args()

    if args.model == "unet":
        model = get_unet()
        model.load_state_dict(torch.load("D:\\pycharm\\EdgeUAV_seg\\output\\unet_uavid_10e.pth"))
    elif args.model == "deeplabv3+":
        model = get_DeepLabV3()
        model.load_state_dict(torch.load("D:\\pycharm\\EdgeUAV_seg\\output\\deeplabv3_uavid_10e.pth"))
    elif args.model == "segformer":
        model = get_segformer()
        model.load_state_dict(torch.load("D:\\pycharm\\EdgeUAV_seg\\output\\segformer_uavid_10e.pth"))
    else:
        raise ValueError(f"Unsupported model type: {args.model}")

    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)



    img_path=args.image_path
    label_path=img_path.replace("Images","Labels") # 将图片路径中的"Images"替换为"Labels"，得到对应的标签路径
    img=Image.open(args.image_path).convert("RGB")
    # 注意: 训练时 UAVIDDataset 只做了 to_tensor, 没有 normalize.
    # 推理必须与训练同口径, 否则模型见到没见过的输入分布, 预测全错. 所以这里不归一化.
    img_tensor = TF.to_tensor(img).to(device)           # (3,H,W)
    input_tensor = img_tensor.unsqueeze(0)               # (1,3,H,W) 加 batch 维

    # label 是 RGB 彩色 mask, 必须转成单通道 trainId (0..7) 才能和 pred 比.
    label_rgb = np.array(Image.open(label_path).convert("RGB"))
    label = UAVIDDataset._rgb_to_trainid(label_rgb)     # (H,W) uint8, 0..7
    label_tensor = torch.as_tensor(label, device=device) # 放到同 device

    with torch.no_grad():
        logits = model(input_tensor)                     # (1,8,H,W)
        pred = logits.argmax(dim=1).squeeze(0)           # (H,W) 预测的 trainId

    iou = compute_iou(pred, label_tensor)
    print('per-class IoU:')
    for tid, val in iou.items():
        name = UAVID_CLASSES[tid][0]
        print(f'  {name:12s}: {val:.4f}' if not (isinstance(val, float) and val != val)
              else f'  {name:12s}: nan (该类在GT中不存在)')
    valid = [v for v in iou.values() if not (isinstance(v, float) and v != v)]
    print(f'mIoU = {sum(valid)/len(valid):.4f}  (over {len(valid)} present classes)')

    # 保存预测可视化图: 原图 | GT彩色 | 预测彩色 三联
    save_prediction_vis(img, label, pred, args.output_path)


    




    
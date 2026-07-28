import torch
import torch.nn as nn
import torch.optim as optim
from datasets.UAVdatasets import UAVIDDataset
from torch.utils.data import DataLoader
from models.unet import get_unet
from models.deeplabv3 import get_DeepLabV3
import argparse
import sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
from pathlib import Path
from losses.combined_loss import conbined_loss
from models.SegFormer import get_segformer
from datasets.UAVdatasets import NUM_CLASSES, UAVID_CLASSES

from models.unet_modefied import get_unet_modefied
device=torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
amp_enabled = device.type == "cuda"

# Windows下cuDNN在多worker/动态形状下偶发 "Unable to find a valid cuDNN algorithm" 报错.
# 关掉benchmark(不试多种卷积算法)+ deterministic(用确定性算法), 牺牲少量速度换稳定性.
if device.type == "cuda":
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


@torch.no_grad()
def validate(model, val_loader, num_classes, device):
    """在val集上算per-class IoU和mIoU, 用混淆矩阵批量累加(高效)."""
    model.eval()
    cm = torch.zeros(num_classes, num_classes, dtype=torch.int64, device=device)
    for img, mask in val_loader:
        img, mask = img.to(device), mask.to(device)
        logits = model(img)
        pred = logits.argmax(dim=1)  # (B,H,W)
        # 把 (真实, 预测) 二维坐标编码成一维下标, bincount 一次统计整个batch
        flat = mask.flatten() * num_classes + pred.flatten()
        cm += torch.bincount(flat, minlength=num_classes*num_classes).reshape(num_classes, num_classes)
    # 从混淆矩阵导出每类 IoU
    ious = []
    for c in range(num_classes):
        tp = cm[c, c].item()
        fp = cm[:, c].sum().item() - tp
        fn = cm[c, :].sum().item() - tp
        denom = tp + fp + fn
        ious.append(tp / denom if denom > 0 else float('nan'))
    # mIoU 只对存在的类(非nan)求平均
    valid = [v for v in ious if v == v]  # nan != nan, 用来过滤
    miou = sum(valid) / len(valid) if valid else 0.0
    model.train()
    return ious, miou


if amp_enabled:
    from torch.cuda.amp import autocast, GradScaler


else:
    autocast = None
    GradScaler = None


def plot_loss_curve(loss_values, save_path=None, show=False):
    """Plot and optionally save the training loss curve."""
    if not loss_values:
        print("No loss values to plot.")
        return

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is not installed, skip loss visualization.")
        return

    epochs = list(range(1, len(loss_values) + 1))
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, loss_values, marker='o', linewidth=2, label='Train Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Curve')
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.legend()
    plt.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=200)
        print(f'Loss curve saved to: {save_path}')

    if show:
        plt.show()

    plt.close()


def store_model(model,output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    pth_path = output_dir+"unet_uavid_10e.pth"
    torch.save(model.state_dict(), pth_path)
    print(f"Model weights saved to: {pth_path}")



if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument('--lr',type=float,default=1e-3,help="learing_rate")

    parser.add_argument('--epoch',default=5,type=int)
    parser.add_argument('--batch_size',default=4,type=int,help='物理 batch size, 4060 8GB + AMP 下 crop512 时 4 够用')
    parser.add_argument('--accum_steps',default=1,type=int,help='梯度累加步数, 等效batch = batch_size * accum_steps')
    parser.add_argument('--optim',default="Adam",type=str)
    parser.add_argument('--mode',default="train",choices=("train", "val"),type=str)
    parser.add_argument('--loss_plot', default='loss_curve.png', type=str, help='path to save loss curve image')
    parser.add_argument('--show_loss_plot', action='store_true', help='show loss curve after training')
    parser.add_argument('--crop_size', default=512, type=int, help='random/center crop size, 4K图必须裁剪')
    parser.add_argument('--model',default="unet",choices=("unet","deeplabv3","segformer","unet_modefied"),type=str)
    parser.add_argument('--resume', type=str, default=None, help='resume from checkpoint path')
    parser.add_argument('--val_interval', default=1, type=int, help='每几个epoch验证一次mIoU')
    parser.add_argument('--num_workers', default=4, type=int, help='DataLoader多进程读图, 0=主进程串行(慢)')
    parser.add_argument('--segformer_layers', default=4, type=int, choices=[3,4], help='SegFormer层数: 3=轻量(31M), 4=标准(59M)')

    args=parser.parse_args()

    argment=True if  args.mode=="train" else False


    data_root = Path(__file__).resolve().parent / "datasets" / "UAV_data" / "uavid_v1.5_official_release_image"
    train_dataset=UAVIDDataset(root=str(data_root), split='train', crop_size=args.crop_size, augment=True)
    dataloader=DataLoader(train_dataset,batch_size=args.batch_size,shuffle=True,num_workers=args.num_workers)
    # val 集: 不增强, 中心裁剪, 用于算 mIoU 监控收敛
    val_dataset=UAVIDDataset(root=str(data_root), split='val', crop_size=args.crop_size, augment=False)
    val_loader=DataLoader(val_dataset,batch_size=args.batch_size,shuffle=False,num_workers=args.num_workers)


    if args.resume:
        if args.model == "unet":
            model=get_unet(in_channels=3,num_classes=8,out_features=64).to(device)
        elif args.model == "deeplabv3":
            model=get_DeepLabV3(num_classes=8).to(device)
        elif args.model == "segformer":
            model=get_segformer(num_layers=args.segformer_layers).to(device)
        elif args.model == "unet_modefied":
            model=get_unet_modefied(in_channels=3,num_classes=8,out_features=64).to(device)
        output_dir = Path(__file__).resolve().parent / "output"
        pth_path = output_dir / "unet_uavid_10e.pth"
        if pth_path.exists():
            model.load_state_dict(torch.load(pth_path, map_location=device))
            print(f"Resumed training from checkpoint: {pth_path}")
        else:
            print(f"No checkpoint found at: {pth_path}. Starting training from scratch.")
    else:
        if args.model == "unet":
                model=get_unet(in_channels=3,num_classes=8,out_features=64).to(device)
        elif args.model == "deeplabv3":
                model=get_DeepLabV3(num_classes=8).to(device)
        elif args.model == "segformer":
                model=get_segformer(num_layers=args.segformer_layers).to(device)
        elif args.model == "unet_modefied":
                model=get_unet_modefied(in_channels=3,num_classes=8,out_features=64).to(device)


    total_loss=[]
    loss=conbined_loss
    scaler = GradScaler(enabled=amp_enabled) if amp_enabled else None
    if args.optim=="Adam":
        optimizer=optim.Adam(model.parameters(),lr=args.lr)
    elif args.optim=="SGD":
        optimizer=optim.SGD(model.parameters(),lr=args.lr,momentum=0.9)
    else:
        raise ValueError("optimizer must be Adam or SGD")

    for i in range(args.epoch):
        model.train()
        epoch_loss=0
        if len(dataloader) == 0:
            raise RuntimeError("DataLoader is empty. Check dataset path and split.")
        optimizer.zero_grad()  # 梯度累加: 每个epoch开头清零
        for batch_idex,(img,mask) in enumerate(dataloader):
            img=img.to(device)
            mask=mask.to(device)
            with autocast(enabled=amp_enabled):
                pred=model(img)
                l=loss(pred,mask)/args.accum_steps  # 梯度累加: loss 除以累加步数
            epoch_loss+=l.item()*args.accum_steps  # 记录原始loss(还原)

            if amp_enabled:
                scaler.scale(l).backward()
                # 每 accum_steps 步才真正更新一次参数
                if (batch_idex+1) % args.accum_steps == 0:
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
            else:
                l.backward()
                if (batch_idex+1) % args.accum_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad()

        total_loss.append(epoch_loss / len(dataloader))
        print(f'epoch {i+1}/{args.epoch}  train_loss={total_loss[-1]:.4f}')

        # 每 val_interval 个epoch验证一次, 监控 mIoU 收敛趋势
        if (i+1) % args.val_interval == 0:
            ious, miou = validate(model, val_loader, NUM_CLASSES, device)
            print(f'  val mIoU={miou:.4f}  per-class:')
            for tid, iou in enumerate(ious):
                name = UAVID_CLASSES[tid][0]
                s = f'{iou:.4f}' if iou==iou else 'nan'
                print(f'    {name:12s}: {s}')

    plot_loss_curve(total_loss, save_path=args.loss_plot, show=args.show_loss_plot)

    # ---- 保存模型 ----
    output_dir = Path(__file__).resolve().parent / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.model == "unet":
        pth_path = output_dir / "unet_uavid_10e.pth"
    elif args.model == "deeplabv3":
        pth_path = output_dir / "deeplabv3_uavid_10e.pth"
    elif args.model == "segformer":
        pth_path = output_dir / "segformer_uavid_10e.pth"
    elif args.model == "unet_modefied":
        pth_path = output_dir / "unet_modefied_uavid_10e.pth"
    torch.save(model.state_dict(), pth_path)
    print(f"Model weights saved to: {pth_path}")







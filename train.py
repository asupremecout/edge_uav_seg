import torch
import torch.nn as nn
import torch.optim as optim
from datasets.UAVdatasets import UAVIDDataset
from torch.utils.data import DataLoader
from models.unet import get_unet
from models.deeplabv3 import get_DeepLabV3
import argparse
from pathlib import Path
from losses.combined_loss import conbined_loss

device=torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu") 
amp_enabled = device.type == "cuda"

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
    parser.add_argument('--batch_size',default=1,type=int)
    parser.add_argument('--optim',default="Adam",type=str)
    parser.add_argument('--mode',default="train",choices=("train", "val"),type=str)
    parser.add_argument('--loss_plot', default='loss_curve.png', type=str, help='path to save loss curve image')
    parser.add_argument('--show_loss_plot', action='store_true', help='show loss curve after training')
    parser.add_argument('--crop_size', default=256, type=int, help='random/center crop size')
    parser.add_argument('--model',default="unet",choices=("unet","deeplabv3"),type=str)
    parser.add_argument('--resume', type=str, default=None, help='resume from checkpoint path')
    
    args=parser.parse_args()
    
    argment=True if  args.mode=="train" else False


    data_root = Path(__file__).resolve().parent / "datasets" / "UAV_data" / "uavid_v1.5_official_release_image"
    MyDataset=UAVIDDataset(root=str(data_root), split=args.mode, crop_size=args.crop_size, augment=argment)
    dataloader=DataLoader(MyDataset,batch_size=args.batch_size,shuffle=(args.mode=="train"),num_workers=0)


    if args.resume:
        if args.model == "unet":
            model=get_unet(in_channels=3,num_classes=8,out_features=64).to(device)
        elif args.model == "deeplabv3":
            model=get_DeepLabV3(num_classes=8).to(device)
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
        for batch_idex,(img,mask) in enumerate(dataloader):
            img=img.to(device)
            mask=mask.to(device)
            with autocast(enabled=amp_enabled):
                pred=model(img)
                l=loss(pred,mask)
            epoch_loss+=l.item()
            
            optimizer.zero_grad()
            if amp_enabled:
                scaler.scale(l).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                l.backward()
                optimizer.step()

        total_loss.append(epoch_loss / len(dataloader))

    plot_loss_curve(total_loss, save_path=args.loss_plot, show=args.show_loss_plot)

    # ---- 保存模型 ----
    output_dir = Path(__file__).resolve().parent / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.model == "unet":
        pth_path = output_dir / "unet_uavid_10e.pth"
    elif args.model == "deeplabv3":
        pth_path = output_dir / "deeplabv3_uavid_10e.pth"
    torch.save(model.state_dict(), pth_path)
    print(f"Model weights saved to: {pth_path}")







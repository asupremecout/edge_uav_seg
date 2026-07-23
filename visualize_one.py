import torch
import torch.nn as nn
from models.unet import get_unet
import PIL.Image as Image
from pathlib import Path
import numpy as np
import torchvision


device=torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model():
    model=get_unet()
    model.eval()
    pth = Path(__file__).resolve().parent / "output" / "unet_uavid_10e.pth"
    model.load_state_dict(torch.load(str(pth), map_location="cpu"))
    model.to(device)
    return model


img_path=r"datasets\UAV_data\uavid_v1.5_official_release_image\uavid_val\seq16\Images\000000.png"
label_path=r"datasets\UAV_data\uavid_v1.5_official_release_image\uavid_val\seq16\Labels\000000.png"


img=Image.open(img_path).convert("RGB")
img=torchvision.transforms.ToTensor()(img).to(device)


mask_pil=Image.open(label_path).convert("RGB")


model=load_model()


with torch.no_grad():
    pred=model(img.unsqueeze(0)).argmax(dim=1).squeeze(0)           # [H, W]


# ---- 可视化 ----
from datasets.UAVdatasets import UAVID_CLASSES

NUM_CLASSES=len(UAVID_CLASSES)
COLOR_MAP=np.zeros((NUM_CLASSES,3),dtype=np.uint8)
for _,tid,rgb in UAVID_CLASSES:
    COLOR_MAP[tid]=rgb

import matplotlib.pyplot as plt

pred_np=pred.cpu().numpy()
pred_color=COLOR_MAP[pred_np]

fig,axes=plt.subplots(1,3,figsize=(18,6))
axes[0].imshow(img.permute(1,2,0).cpu().numpy())
axes[0].set_title("Input Image")
axes[0].axis("off")

axes[1].imshow(pred_color)
axes[1].set_title("UNet Prediction")
axes[1].axis("off")

axes[2].imshow(mask_pil)
axes[2].set_title("Ground Truth")
axes[2].axis("off")

plt.tight_layout()
save_path=Path(__file__).resolve().parent/"vis_result.png"
plt.savefig(str(save_path),dpi=200,bbox_inches="tight")
print(f"Visualization saved to {save_path}")
plt.close()





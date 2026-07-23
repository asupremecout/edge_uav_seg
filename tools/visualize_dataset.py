import os
from datasets.UAVdatasets import UAVID_CLASSES
import cv2
import numpy as np
import PIL.Image as Image
myroot="D:\pycharm\EdgeUAV_seg\datasets\UAV_data\uavid_v1.5_official_release_image"
mode=["train","val"]

id_to_rgb={id:rgb for _,id,rgb in UAVID_CLASSES}

id_to_name={id:name for name,id,_ in UAVID_CLASSES}

id={id for _,id,_ in UAVID_CLASSES if id not in[0]}

def load_image(img_path): #加载索引版的图片
    img=Image.open(img_path)
    img=np.array(img,dtype=np.uint8)
    return img

def color_image(idx_img):
    h,w=idx_img.shape
    color_img=np.zeros((h,w,3),dtype=np.uint8) #建立
    for i in range(h):
        for j in range(w):
            color_img[i,j]=id_to_rgb[idx_img[i,j]]
    return color_img



def collect_all_path(root):
    collection=[]
    for pointer in mode:
        point_root=os.path.join(root,f'uavid_{pointer}')

        for seq in os.listdir(point_root):
            seq_image_path=os.path.join(point_root,seq,"Images")
            seq_label_path=os.path.join(point_root,seq,"Labels")

            for filename in os.listdir(seq_image_path):
                if filename.endswith(".png"):
                    filename=os.path.join(seq_image_path,filename)
                    collection.append(filename)
                else:
                    continue

            for filename in os.listdir(seq_label_path):
                if filename.endswith(".png"):
                    filename=os.path.join(seq_image_path,filename)
                    collection.append(filename)
                else:
                    continue

    return collection




    









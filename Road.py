# %%
import torch.nn as nn
import torch.nn.functional as F
import torch
import pickle
import numpy as np
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
import cv2 as cv
import os 
import sys
import pandas as pd
sys.path.append(os.path.dirname(os.path.abspath('.')))
from unet import UNet
from AttnUNet import AttU_Net
from ResUNet import Res_UNet
from PIL import Image
import matplotlib.pyplot as plt

from matplotlib.widgets import Slider


# %%
class Road_Lane_Segmentation(Dataset):
    ROAD_COLOR = np.array([128, 0, 0])
    
    def __init__(self, image_paths , label_paths):
        super().__init__()
        self.image_paths  = image_paths 
        self.label_paths  = label_paths 
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx])
        img = img.resize((512, 512), Image.BILINEAR)
        img = np.array(img)

        mask = Image.open(self.label_paths[idx])
        mask = mask.resize((512, 512), Image.NEAREST)   
        mask = np.array(mask)

        road_mask = (mask == 1).astype(np.float32)

        img = torch.tensor(img, dtype=torch.float).permute(2, 0, 1)       # (3, H, W)
        road_mask = torch.tensor(road_mask, dtype=torch.float).unsqueeze(0)  # (1, H, W)        


        return img, road_mask
    
train_csv = pd.read_csv("train_file_list.csv")
valid_csv = pd.read_csv("valid_file_list.csv")

root_path_train = r"data\train"
root_path_valid = r"data\valid"
train_images = [os.path.join(root_path_train, fname) for fname in train_csv["image"]]
train_masks = [os.path.join(root_path_train, fname) for fname in train_csv["mask"]]

valid_images = [os.path.join(root_path_valid, fname) for fname in valid_csv["image"]]
valid_masks = [os.path.join(root_path_valid, fname) for fname in valid_csv["mask"]]


train_dataset=Road_Lane_Segmentation(train_images, train_masks)
valid_dataset=Road_Lane_Segmentation(valid_images, valid_masks)

train_loader=DataLoader(train_dataset,batch_size=8,shuffle=True)
val_loader=DataLoader(valid_dataset,batch_size=8,shuffle=False)

device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# %%
AttU_model = AttU_Net(3, 1).to(device)
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(AttU_model.parameters(),lr=1e-4)

# %%
AttU_model.load_state_dict(torch.load('att_unet_road_100_1e4.pth',map_location=device))

# %%
AttU_model.eval() #AttU_model, UNet_model, ResUNet_model

all_predictions=[]
all_inputs=[]

for batch_x,batch_y in val_loader:
    batch_x, batch_y = batch_x.to(device),batch_y.to(device)
    with torch.no_grad():
        output=AttU_model(batch_x)
    for i in range(batch_x.size(0)):
        all_inputs.append(batch_x[i].cpu())     # (3,H,W)
        all_predictions.append(output[i].cpu())     # (1,H,W)


# %%
all_overlayer = []


for i in range(len(all_predictions)):
    img = all_inputs[i].cpu()                   
    pred = all_predictions[i].cpu()            


    img_norm = (img - img.min()) / (img.max() - img.min())
    img_norm = img_norm.permute(1,2,0).numpy()  


    logits = pred.squeeze().numpy()            
    logits_norm = (logits - logits.min()) / (logits.max() - logits.min())

    heat = np.zeros_like(img_norm)
    heat[..., 0] = logits_norm   

    alpha = 0.5
    overlay = (1 - alpha) * img_norm + alpha * heat

    all_overlayer.append(overlay)

# %%
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import numpy as np

num_frames = len(all_overlayer)

fig, ax = plt.subplots(figsize=(8, 5))
plt.subplots_adjust(bottom=0.25)

frame0 = all_overlayer[0]
img = ax.imshow(frame0)
ax.axis('off')

ax_slider = plt.axes([0.2, 0.1, 0.6, 0.03])
slider = Slider(
    ax=ax_slider, 
    label='Frame', 
    valmin=0, 
    valmax=num_frames - 1, 
    valinit=0, 
    valstep=1
)

def update(val):
    idx = int(slider.val)
    img.set_data(all_overlayer[idx])
    fig.canvas.draw_idle()

slider.on_changed(update)

plt.show()




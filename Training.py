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


class LaneDataset(Dataset):
    '''Expects x and y to be np arrays
    x.shape=(num_samples,80,160,3)
    y.shape=(num_samples,80,160,1)
    converts them to pytroch (3,80,160) and (1,80,160)'''
    def __init__(self,images,labels):
        super().__init__()
        self.images = images
        self.labels=labels

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img = torch.tensor(self.images[idx],dtype=torch.float).permute(2,0,1) #change to (3,80,160)
        label = torch.tensor(self.labels[idx],dtype=torch.float).permute(2,0,1) #change to (1,80,160)
        return img,label
    

train_pickle=pickle.load(open('data/full_CNN_train.p','rb'))
train_labels=pickle.load(open('data/full_CNN_labels.p','rb'))
train_features=np.array(train_pickle)
train_labels=np.array(train_labels)/255 #normalize
# train_features,train_labels=shuffle(train_features,train_labels)
X_train, X_val, y_train, y_val=train_test_split(train_features,train_labels)
train_dataset=LaneDataset(X_train, y_train)
val_dataset=LaneDataset(X_val,y_val)
train_loader=DataLoader(train_dataset,batch_size=128,shuffle=True)
val_loader=DataLoader(val_dataset,batch_size=128,shuffle=True)
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')


AttU_model = AttU_Net(3, 1).to(device)
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(AttU_model.parameters(),lr=1e-4)
loss_history = []

num_epochs=1000
for epoch in range(num_epochs):
    AttU_model.train()
    running_loss=0
    for batch_x,batch_y in train_loader:
        batch_x, batch_y = batch_x.to(device),batch_y.to(device)

        optimizer.zero_grad()
        output=AttU_model(batch_x)
        loss=criterion(output,batch_y)
        loss.backward()
        optimizer.step()
        running_loss+=loss.item()
    print(f'Epoch {epoch+1}, loss: {running_loss/len(train_loader)}')
    loss_history.append(running_loss/len(train_loader))

torch.save(AttU_model.state_dict(),'AttU_model.pth')
df = pd.DataFrame({'epoch': list(range(1, num_epochs+1)), 'loss': loss_history})
df.to_csv('AttU_model_training_loss.csv', index=False)

#==========================================================================================

UNet_model = UNet(3,1).to(device)
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(UNet_model.parameters(),lr=1e-4)
loss_history = []

num_epochs=1000
for epoch in range(num_epochs):
    UNet_model.train()
    running_loss=0
    for batch_x,batch_y in train_loader:
        batch_x, batch_y = batch_x.to(device),batch_y.to(device)

        optimizer.zero_grad()
        output=UNet_model(batch_x)
        loss=criterion(output,batch_y)
        loss.backward()
        optimizer.step()
        running_loss+=loss.item()

    print(f'Epoch {epoch+1}, loss: {running_loss/len(train_loader)}')
    loss_history.append(running_loss/len(train_loader))

torch.save(UNet_model.state_dict(),'UNet_model.pth')
df = pd.DataFrame({'epoch': list(range(1, num_epochs+1)), 'loss': loss_history})
df.to_csv('UNet_model_training_loss.csv', index=False)

#==========================================================================================

ResUNet_model = Res_UNet(3,1).to(device)
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(ResUNet_model.parameters(),lr=1e-4)
loss_history = []


num_epochs=1000
for epoch in range(num_epochs):
    ResUNet_model.train()
    running_loss=0
    for batch_x,batch_y in train_loader:
        batch_x, batch_y = batch_x.to(device),batch_y.to(device)

        optimizer.zero_grad()
        output=ResUNet_model(batch_x)
        loss=criterion(output,batch_y)
        loss.backward()
        optimizer.step()
        running_loss+=loss.item()
    print(f'Epoch {epoch+1}, loss: {running_loss/len(train_loader)}')
    loss_history.append(running_loss/len(train_loader))

torch.save(ResUNet_model.state_dict(),'ResUNet_model.pth')
df = pd.DataFrame({'epoch': list(range(1, num_epochs+1)), 'loss': loss_history})
df.to_csv('ResUNet_model_training_loss.csv', index=False)

#==========================================================================================

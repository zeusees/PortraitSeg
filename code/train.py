import os
import time
import numpy as np
import pandas as pd
import cv2 as cv
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
from pandas.core.frame import DataFrame
from sklearn.model_selection import train_test_split

import albumentations as A
import segmentation_models_pytorch as smp

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import models
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast
from torch.cuda.amp import GradScaler

import onnx
import onnxsim

if torch.cuda.is_available():
    print('GPU numbers:',torch.cuda.device_count())
else:
    print('CPU Mode')

torch.cuda.set_device(0)


# ------------------------------------ config ------------------------------------

train_image_width, train_image_height = 192, 288

# Please make sure that your data set directory is the same as below,
# or you can rewrite the code of the import data part according to your data set situation.
'''
--code
  --train.py
--dataset
  --AiSeg
    --matting_human_half
      --clip_img
        --1803151818
          --clip_00000000
            --1803151818-00000003.jpg
      --matting
        --1803151818
          --matting_00000000
            --1803151818-00000003.png
  --Automatic
    --training
      --00001.png
      --00001_matte.png
    --testing
      --00001.png
      --00001_matte.png
  --SuperviselyPersonDataset
    --img
      --ache-adult-depression-expression-41253.jpeg
    --seg
      --ache-adult-depression-expression-41253.jpeg
'''


# ------------------------------------ tool function ------------------------------------

class MyDataset(Dataset):
    def __init__(self, data, augmentation=None):
        self.augmentation = augmentation
        self.image_data = []
        self.label_data = []
        kernel = cv.getStructuringElement(cv.MORPH_RECT,(9,9))
        for image_name, label_name in tqdm(data):
            image = cv.resize(cv.imread(image_name,cv.IMREAD_UNCHANGED),(train_image_width,train_image_height),interpolation=cv.INTER_AREA)
            if len(image.shape) != 3:
                print('file: ',image_name,' shape is not in diment 3')
                continue
            label = cv.resize(cv.imread(label_name,cv.IMREAD_UNCHANGED),(train_image_width,train_image_height),interpolation=cv.INTER_AREA)
            if len(label.shape)==3 and label.shape[2] == 4: # AiSeg data
                label = label[:,:,3] # get alpha channel
            else: # Automatic and COCO data
                label = label # get origin
            self.image_data.append(image)
            self.label_data.append(label)
        self.len = len(self.image_data)

    def __getitem__(self, index):
        transformed  = self.augmentation(image=self.image_data[index],mask=self.label_data[index])
        img = transformed['image']
        seg = transformed['mask']
        img = (img-127.5)/127.5
        seg = seg/255.0
        return img.transpose([2,0,1]), seg.reshape(1,train_image_height,train_image_width)

    def __len__(self):
        return self.len


# ------------------------------------ import dataset ------------------------------------

# import AiSeg dataset
BigData_image_root = '../dataset/AiSeg/matting_human_half/clip_img'
BigData_label_root = '../dataset/AiSeg/matting_human_half/matting'
BigData_file = []
for _sub_floder_ in os.listdir(BigData_image_root):
    sub_floder = os.path.join(BigData_image_root,_sub_floder_)
    sub_sub_floder = os.listdir(sub_floder)
    sub_sub_floder_num = len(sub_sub_floder)
    for i in range(sub_sub_floder_num):
        image_floder_path = os.path.join(sub_floder,'clip_'+str(i).rjust(8,'0'))
        label_floder_path = os.path.join(os.path.join(BigData_label_root,_sub_floder_),'matting_'+str(i).rjust(8,'0'))
        if os.path.exists(image_floder_path) and os.path.exists(label_floder_path):
            floder_file = os.listdir(image_floder_path)
            for _file_ in floder_file:
                image_file = os.path.join(image_floder_path,_file_)
                label_file = os.path.join(label_floder_path,_file_.split('.')[0]+'.png')
                if os.path.exists(image_file) and os.path.exists(label_file):
                    BigData_file.append([image_file,label_file])
                else:
                    print('file no exist: {}, {}'.format(image_file,label_file))
        else:
            print('path no exist: {}, {}'.format(image_floder_path,label_floder_path))

print('AiSeg dataset number: ',len(BigData_file))


# import Automatic dataset
Automatic_train_data_root = '../dataset/Automatic/training'
Automatic_train_file_name = [str(i).rjust(5,'0') for i in range(1,1700+1)]
Automatic_train_file = [[os.path.join(Automatic_train_data_root,i+'.png'),os.path.join(Automatic_train_data_root,i+'_matte.png')] for i in Automatic_train_file_name]
Automatic_valid_data_root = '../dataset/Automatic/testing'
Automatic_valid_file_name = [str(i).rjust(5,'0') for i in range(1,300+1)]
Automatic_valid_file = [[os.path.join(Automatic_valid_data_root,i+'.png'),os.path.join(Automatic_valid_data_root,i+'_matte.png')] for i in Automatic_valid_file_name]
Automatic_file = Automatic_train_file + Automatic_valid_file

print('Automatic dataset number: ',len(Automatic_file))


# import Supervise.ly dataset
SuperviselyPersonDataset_train_image_data_root = '../dataset/SuperviselyPersonDataset/img'
SuperviselyPersonDataset_train_label_data_root = '../dataset/SuperviselyPersonDataset/seg'
SuperviselyPersonDataset_train_image_file = os.listdir(SuperviselyPersonDataset_train_image_data_root)
SuperviselyPersonDataset_file = [[os.path.join(SuperviselyPersonDataset_train_image_data_root,i),os.path.join(SuperviselyPersonDataset_train_label_data_root,i)] for i in SuperviselyPersonDataset_train_image_file]

print('SuperviselyPerson dataset number: ',len(SuperviselyPersonDataset_file))


# AiSeg & Supervise.ly for train
# Automatic for valid
train_file = BigData_file + SuperviselyPersonDataset_file
valid_file = Automatic_file
print('train:valid ---- {}:{}'.format(len(train_file),len(valid_file)))


# ------------------------------------ import data & model & train config ------------------------------------

train_A = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.RandomGamma(gamma_limit=(50, 150), p=0.9),
    A.ShiftScaleRotate(shift_limit=0.2,scale_limit=0.5,rotate_limit=45,border_mode=cv.BORDER_CONSTANT,value=0,mask_value=0,p=0.9),
])
valid_A = A.Compose([
])
train_dataset = MyDataset(train_file, augmentation=train_A)
valid_dataset = MyDataset(valid_file, augmentation=valid_A)
train_dataloader = DataLoader(dataset=train_dataset, batch_size=128, num_workers=8, drop_last=False, pin_memory=True, shuffle=True)
valid_dataloader = DataLoader(dataset=valid_dataset, batch_size=128, num_workers=8, drop_last=False, pin_memory=True, shuffle=False)


model = smp.DeepLabV3Plus(
    encoder_name="timm-mobilenetv3_small_minimal_100",
    encoder_weights=None,
    in_channels=3,
    classes=1,
    decoder_channels=48,
)
model = model.cuda()
print(model(torch.rand(2,3,train_image_height,train_image_width, device='cuda')).shape)

criterion = nn.L1Loss()
optimizer = optim.Adam(model.parameters(), lr=0.008)
scheduler = optim.lr_scheduler.MultiStepLR(optimizer,milestones=[5,30,60,90,120,150],gamma=0.5)


# ------------------------------------ start training ------------------------------------

print('start train')

epoch = 150

scaler = GradScaler()
for epo in range(epoch):
    train_loss, valid_loss = 0.0, 0.0
    t1 = time.time()
    model.train()
    for _, [img, seg] in enumerate(train_dataloader):
        inputs, labels = img.float().cuda(), seg.float().cuda()
        # mixed position
        optimizer.zero_grad()
        with autocast():
            output = model(inputs)
            loss = criterion(output,labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        train_loss += loss.item() * labels.size(0)
    t2 = time.time()

    # update lr
    scheduler.step()

    with torch.no_grad():
        model.eval()
        for _, [img, seg] in enumerate(valid_dataloader):
            inputs, labels = img.float().cuda(), seg.float().cuda()
            output = model(inputs)
            loss = criterion(output,labels)
            valid_loss += loss.item() * labels.size(0)
    t3 = time.time()

    print('Time:[{},{}], Epoch:[{}/{}] ---- train loss:{:.6f}, valid loss:{:.6f}'
        .format(int(t2-t1),int(t3-t2),epo+1,epoch,train_loss/train_dataset.__len__(),valid_loss/valid_dataset.__len__()))


# ------------------------------------ Save & Export Model ------------------------------------

model_name = 'model'

# save pytorch model
torch.save(model.state_dict(),'{}.pth'.format(model_name))
# save onnx model
torch.onnx.export(model, torch.rand(1,3,train_image_height,train_image_width, device='cuda'), '{}.onnx'.format(model_name),
                    export_params=True,
                    verbose=True,
                    input_names=['inputs_3_{}_{}'.format(train_image_height,train_image_width)],
                    output_names=["output_1_{}_{}".format(train_image_height,train_image_width)],
                    opset_version=11)
# simplify onnx model
onnx_model = onnx.load('{}.onnx'.format(model_name))
onnx_model_sim, check = onnxsim.simplify(onnx_model)
onnx.save(onnx_model_sim,'{}-sim.onnx'.format(model_name))


# ------------------------------------ Over ------------------------------------

torch.cuda.empty_cache()
print('Thanks')

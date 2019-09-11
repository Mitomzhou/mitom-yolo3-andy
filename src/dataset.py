'''
Created on Jul 14, 2019

@author: monky
'''

import numpy as np
import random
import os
from PIL import Image

import torch
import torchvision
from torch.utils.data import Dataset
import torch.utils.data as data


class YoloDataset(Dataset):
    
    def __init__(self, root, shape=None, shuffle=True, transform=None, train=True, seen=0, batch_size=64, num_workers=4):
        # read voc_train.txt
        with open(root, 'r') as file:
            self.lines = file.readlines()
            
        if shuffle:
            random.shuffle(self.lines)
        
        self.shape = shape
        self.nSamples = len(self.lines)
        self.transform = transform
        self.train = train
        self.seen = seen
        self.batch_size = batch_size
        self.num_workers = num_workers
    
    
    def __len__(self):
        return self.nSamples
    
    
    def __getitem__(self, index):
        imgpath = self.lines[index].rstrip()
        if self.train:
            img, label = load_data_detection(imgpath, self.shape)
            label = torch.Tensor(label)
        else:
            img, label = load_data_detection(imgpath, self.shape)
            label = torch.Tensor(label)    
         
        if self.transform is not None:
            img = self.transform(img)
        return (img, label)
            
            
            
def load_data_detection(imgpath, shape):
    labpath = imgpath.replace('images', 'labels').replace('JPEGImages', 'labels').replace('.jpg', '.txt').replace('.png','.txt')
    #load img
    img = Image.open(imgpath).convert('RGB')
    img = img.resize(shape)
    
    #load label -> list[0~100]  20x5
    max_boxes = 40
    label = np.zeros((max_boxes, 5))
    if os.path.getsize(labpath):
        bndboxes = np.loadtxt(labpath)
        if bndboxes is None:
            return label
        #bndboxes ndim=1
        if bndboxes.ndim == 1:
            bndboxes = np.expand_dims(bndboxes, axis=0)
        for i in range(bndboxes.shape[0]):
            label[i] = bndboxes[i]
            if i >= 40:
                break
            
    label = np.reshape(label, (-1))
    return img, label
    
    
if __name__ == '__main__':
    c = 0
    
    train_loader = torch.utils.data.DataLoader(
        YoloDataset('../cfg/voc_train.txt', 
                    shape=(448, 448), 
                    shuffle=True, 
                    transform=torchvision.transforms.ToTensor(), 
                    train=True, 
                    seen=0, 
                    batch_size=128, 
                    num_workers=4),
        batch_size=32, shuffle=True, num_workers=4)
    for batch_idx, (data, target) in enumerate(train_loader):
        print(target.size())
        c = c+1
        print("------------------: ", c)
        
        
#     test_loader = torch.utils.data.DataLoader(
#     YoloDataset('../cfg/2007_test.txt', 
#                 shape=(448, 448), 
#                 shuffle=False, 
#                 transform=torchvision.transforms.ToTensor(), 
#                 train=False, 
#                 seen=0, 
#                 batch_size=32, 
#                 num_workers=4),
#     batch_size=32, shuffle=False, num_workers=4)
#     for batch_idx, (data, target) in enumerate(test_loader):
#         print(target.size())
#         c = c+1
#         print("------------------: ", c)
    
    
    
    
    
    
    
    
            
            
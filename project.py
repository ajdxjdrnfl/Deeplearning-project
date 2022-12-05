#!/usr/bin/env python
# coding: utf-8



import torch
import torch.nn as nn
import torch.functional as F
from torchvision import transforms
from network import resnet
from src import dataset
from tqdm import tqdm
from torch.optim import adam
import numpy as np
import math
import matplotlib.pyplot as plt





def dft(img):
    array = torch.fft.fftshift(torch.fft.fft2(img, norm='ortho'))
    epsilon = 1e-12
    array = torch.abs(array)
    array += epsilon
    array = torch.log(array)
    
    return array

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, device="cpu", eps=1e-10):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.device = device
        self.eps = eps
    
    def forward(self, input, target):
        p = torch.sigmoid(input)
        pt = p * target.float() + (1.0-p)*(1-target).float()
        alpha_t = (1.0 - self.alpha) * target.float() + self.alpha * (1-target).float()
        loss = - 1.0 * torch.pow((1-pt), self.gamma)*torch.log(pt+self.eps)
        return loss.sum()

class sumNetwork(nn.Module):
    def __init__(self, dft, num_classes):
        super().__init__()
        self.src = resnet.resnetlayer34()
        self.tgt = resnet.resnetlayer34()
        self.dft = dft
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(1*1*512, num_classes)
        
    def forward(self, x):
        y = dft(x)
        
        x = self.src(x)
        y = self.tgt(y)
        x = self.avgpool(x)
        y = self.avgpool(y)
        z = torch.flatten(x+y, start_dim=1)
        z = self.fc(z)
        return z
        
class concatNetwork(nn.Module):
    def __init__(self, dft, num_classes):
        super().__init__()
        self.src = resnet.resnetlayer34()
        self.tgt = resnet.resnetlayer34()
        self.dft = dft
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(1*1*1024, num_classes)
        
    def forward(self, x):
        y = dft(x)
        
        x = self.src(x)
        y = self.tgt(y)
        x = self.avgpool(x)
        y = self.avgpool(y)
        z = torch.flatten(torch.cat((x,y), dim=1), start_dim=1)
        z = self.fc(z)
        return z

def SequenceMask(X, X_len,value=0):
    maxlen = X.size(1)
    mask = torch.arange((maxlen),dtype=torch.float)[None, :] < X_len[:, None]    
    X[~mask]=value
    return X

def masked_softmax(X, valid_length):
    # X: 3-D tensor, valid_length: 1-D or 2-D tensor
    softmax = nn.Softmax()
    if valid_length is None:
        return softmax(X)
    else:
        shape = X.shape
        if valid_length.dim() == 1:
            valid_length = torch.FloatTensor(valid_length.numpy().repeat(shape[1], axis=0))
        else:
            valid_length = valid_length.reshape((-1,))
        # fill masked elements with a large negative, whose exp is 0
        X = SequenceMask(X.reshape((-1, shape[-1])), valid_length)
        return softmax(X).reshape(shape)

class CrossAttention(nn.Module): 
    def __init__(self, dropout, **kwargs):
        super(CrossAttention, self).__init__(**kwargs)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, valid_length=None):
        d = query.shape[-1]
        
        scores = torch.bmm(query, key.transpose(1,2)) / math.sqrt(d)
        attention_weights = self.dropout(masked_softmax(scores, valid_length))
        return torch.bmm(attention_weights, value)

class customNetwork(nn.Module):
    def __init__(self, dft, attn, dimension, num_classes):
        super().__init__()
        self.src = resnet.resnetlayer34()
        self.tgt = resnet.resnetlayer34()
        self.Wk = nn.Sequential(nn.Linear(100, dimension), )
        self.Wq = nn.Sequential(nn.Linear(100, dimension), )
        self.Wv = nn.Sequential(nn.Linear(100, dimension), )
        self.dft = dft
        self.attn = attn
        self.classification = nn.Sequential(nn.LayerNorm(dimension), nn.Linear(dimension, num_classes))
        self.fc = nn.Linear(1*1*512, num_classes)
    def forward(self, x):
        y = dft(x)
        
        x = self.src(x)
        y = self.tgt(y)
    
        x = torch.flatten(x, start_dim=2)
        y = torch.flatten(y, start_dim=2)
       
        query = self.Wq(y)
        key = self.Wk(x)
        value = self.Wv(x)
        
        z = self.attn(query, key, value)
  
        z = torch.mean(z, dim=2)

        z = self.fc(z)
        return z





num_classes = 1
device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
focal_loss = FocalLoss()
bce_loss = nn.BCELoss().to(device)





data_dir = "/content/"
transform =transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
myDataSet = dataset.MyDataSet(data_dir, transform=transform)
total_ratio = 1.0
train_ratio = 0.6
val_ratio = 0.2
test_ratio = total_ratio - train_ratio - val_ratio

dataset_size = len(myDataSet)
train_size = int(dataset_size * train_ratio)
val_size = int(dataset_size * val_ratio)
test_size = int(dataset_size * test_ratio)
residual_size = dataset_size - train_size - val_size - test_size
train_dataset, val_dataset, test_dataset, _ = torch.utils.data.random_split(myDataSet, [train_size, val_size, test_size, residual_size])





batch_size = 32
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True, drop_last=True)




epoch = 10
def init_weights(m):
    if type(m) == nn.Linear or type(m) == nn.Conv2d:
        torch.nn.init.xavier_uniform_(m.weight)




baseNet = resnet.resnet50(num_classes).to(device)
base_optimizer = adam.Adam(baseNet.parameters(), lr=0.001, weight_decay=0.0001)
baseNet.apply(init_weights)



baseCost = []
baseValCost = []
baseAcc = []
for _ in range(epoch):
    train_loss = 0.0
    baseCorrect = 0
    baseTotal = 0
    for image, label in tqdm(train_dataloader):
        image = image.to(device)
        label = label.to(device).float()
        
        output = baseNet(image)
        output = torch.sigmoid(output)
        
        batch_loss = bce_loss(output, label)
        
        base_optimizer.zero_grad()
        batch_loss.backward()
        base_optimizer.step()
        
        train_loss += batch_loss.item()
    cost = train_loss/len(train_dataloader)
    
    with torch.no_grad():
        baseNet.eval()
        val_loss = 0.0
        for image, label in (val_dataloader):
            image = image.to(device)
            label = label.to(device).float()
            output = baseNet(image)
            output = torch.sigmoid(output)
            batch_val_loss = bce_loss(output, label)
            val_loss += batch_val_loss.item()
          
            predicted = (output > 0.5)  
            baseTotal += label.size(0)
            baseCorrect += (predicted == label).sum().item()
         
        baseAcc.append(100 * baseCorrect/baseTotal)
        total = val_loss/len(val_dataloader)
        baseValCost.append(total)
    
    baseCost.append(cost)
    print(cost)



base_test = 0.0
with torch.no_grad():
        baseTotal = 0
        baseCorrect = 0
        for image, label in (val_dataloader):
            image = image.to(device)
            label = label.to(device).float()
            output = baseNet(image)
            print(output)
            output = torch.sigmoid(output)
            print(output)
            predicted = (output > 0.5)  
            baseTotal += label.size(0)
            baseCorrect += (predicted == label).sum().item()
           
        base_test = (100 * baseCorrect/baseTotal)
        print(baseCost)
        print(baseValCost)
        print(baseAcc)
        print(base_test)



sumModel = sumNetwork(dft, num_classes)
sumModel = sumModel.to(device)
sumModel.apply(init_weights)
sum_optimizer = adam.Adam(sumModel.parameters(), lr=0.001, weight_decay=0.0001)




sumCost = []
sumValCost = []
sumAcc = []
for _ in range(epoch):
    train_loss = 0.0
    baseCorrect = 0
    baseTotal = 0
    for image, label in tqdm(train_dataloader):
        image = image.to(device)
        label = label.to(device).float()
        
        output = sumModel(image)
        output = torch.sigmoid(output)
        
        batch_loss = bce_loss(output, label)
        
        sum_optimizer.zero_grad()
        batch_loss.backward()
        sum_optimizer.step()
        
        train_loss += batch_loss.item()
    cost = train_loss/len(train_dataloader)
    
    with torch.no_grad():
        sumModel.eval()
        val_loss = 0.0
        for image, label in (val_dataloader):
            image = image.to(device)
            label = label.to(device).float()
            output = sumModel(image)
            output = torch.sigmoid(output)
            batch_val_loss = bce_loss(output, label)
            val_loss += batch_val_loss.item()

            predicted = (output > 0.5)  
            baseTotal += label.size(0)
            baseCorrect += (predicted == label).sum().item()

        sumAcc.append(100 * baseCorrect/baseTotal)
        total = val_loss/len(val_dataloader)
        sumValCost.append(total)
    
    sumCost.append(cost)
    print(cost)



sum_test = 0.0
with torch.no_grad():
        baseTotal = 0
        baseCorrect = 0
        for image, label in (val_dataloader):
            image = image.to(device)
            label = label.to(device).float()
            output = sumModel(image)
            output = torch.sigmoid(output)
            predicted = (output > 0.5)  
            baseTotal += label.size(0)
            baseCorrect += (predicted == label).sum().item()
           
        sum_test = (100 * baseCorrect/baseTotal)
        print(sumCost)
        print(sumValCost)
        print(sumAcc)
        print(sum_test)



concatModel = concatNetwork(dft, num_classes)
concatModel.to(device)
concatModel.apply(init_weights)
concat_optimizer = adam.Adam(concatModel.parameters(), lr=0.001, weight_decay=0.0001)



concatCost = []
concatValCost = []
concatAcc = []
epoch=10
for _ in range(epoch):
    train_loss = 0.0
    baseCorrect = 0
    baseTotal = 0
    for image, label in tqdm(train_dataloader):
        image = image.to(device)
        label = label.to(device).float()
        
        output = concatModel(image)
        output = torch.sigmoid(output)
        
        batch_loss = bce_loss(output, label)
        
        concat_optimizer.zero_grad()
        batch_loss.backward()
        concat_optimizer.step()
        
        train_loss += batch_loss.item()
    cost = train_loss/len(train_dataloader)
    
    with torch.no_grad():
        concatModel.eval()
        val_loss = 0.0
        for image, label in (val_dataloader):
            image = image.to(device)
            label = label.to(device).float()
            output = concatModel(image)
            output = torch.sigmoid(output)
            batch_val_loss = bce_loss(output, label)
            val_loss += batch_val_loss.item()

            predicted = (output > 0.5)  
            baseTotal += label.size(0)
            baseCorrect += (predicted == label).sum().item()

        concatAcc.append(100 * baseCorrect/baseTotal)
        total = val_loss/len(val_dataloader)
        concatValCost.append(total)
    
    concatCost.append(cost)
    print(cost)


concat_test = 0.0
with torch.no_grad():
        baseTotal = 0
        baseCorrect = 0
        for image, label in (val_dataloader):
            image = image.to(device)
            label = label.to(device).float()
            output = concatModel(image)
            
            output = torch.sigmoid(output)
            
            predicted = (output > 0.5)  
            baseTotal += label.size(0)
            baseCorrect += (predicted == label).sum().item()
            
        concat_test = (100 * baseCorrect/baseTotal)
        print(concatCost)
        print(concatValCost)
        print(concatAcc)
        print(concat_test)



crossModel = customNetwork(dft, CrossAttention(0.1),3, 1)
crossModel.to(device)
crossModel.apply(init_weights)
cross_optimizer = adam.Adam(crossModel.parameters(), lr=0.002, weight_decay=0.0001)


attent = CrossAttention(0.1)
crossModel = customNetwork(dft, attent, 10, 1)
crossModel.to(device)
crossModel.apply(init_weights)
cross_optimizer = adam.Adam(crossModel.parameters(), lr=0.001, weight_decay=0.0001)



crossCost = []
crossValCost = []
crossAcc = []
for _ in range(epoch):
    train_loss = 0.0
    baseCorrect = 0
    baseTotal = 0
    for image, label in tqdm(train_dataloader):
        image = image.to(device)
        label = label.to(device).float()
        
        output = crossModel(image)
        output = torch.sigmoid(output)
        
        batch_loss = bce_loss(output, label)
        
        cross_optimizer.zero_grad()
        batch_loss.backward()
        cross_optimizer.step()
        
        train_loss += batch_loss.item()
    cost = train_loss/len(train_dataloader)
    
    with torch.no_grad():
        crossModel.eval()
        val_loss = 0.0
        for image, label in (val_dataloader):
            image = image.to(device)
            label = label.to(device).float()
            output = crossModel(image)
            output = torch.sigmoid(output)
            batch_val_loss = bce_loss(output, label)
            val_loss += batch_val_loss.item()

            predicted = (output > 0.5)  
            baseTotal += label.size(0)
            baseCorrect += (predicted == label).sum().item()

        crossAcc.append(100 * baseCorrect/baseTotal)
        total = val_loss/len(val_dataloader)
        crossValCost.append(total)
    
    crossCost.append(cost)
    print(cost)


cross_test = 0.0
with torch.no_grad():
        baseTotal = 0
        baseCorrect = 0
        for image, label in (val_dataloader):
            image = image.to(device)
            label = label.to(device).float()
            output = crossModel(image)
            output = torch.sigmoid(output)

            predicted = (output > 0.5)  
            baseTotal += label.size(0)
            baseCorrect += (predicted == label).sum().item()
           
        cross_test = (100 * baseCorrect/baseTotal)
        print(crossCost)
        print(crossValCost)
        print(crossAcc)
        print(cross_test)


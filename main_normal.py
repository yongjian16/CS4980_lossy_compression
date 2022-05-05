#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 11 10:46:16 2022

@author: yz
"""

import torch
import torch.optim as optim
import torchvision
from torch.optim.optimizer import Optimizer
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torchvision import datasets
import numpy as np
import torch.nn as nn
from FI import FI
from models import FiAlexNet, FiResnet18, Resnet18, AlexNet_normal

device = 'cuda' if torch.cuda.is_available() else 'cpu'


transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ])

transform_val = transforms.Compose([
            transforms.ToTensor(),
    transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

epoch = 100
batch_size = 64
coefficient = 0.01
error_type = 'baseline'
model_name = 'resnet'

train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
#train_dataset, val_dataset = random_split(train_dataset,[len(train_dataset)-1000,1000])
test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_val)
train_loader = DataLoader(train_dataset, batch_size, shuffle=True, num_workers=4, pin_memory=True)
#val_loader = DataLoader(val_dataset, batch_size*2, num_workers=4, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size*2, num_workers=4, pin_memory=True)

if error_type == 'baseline':
    if model_name == 'resnet':
        model = Resnet18().to(device)
    else:
        model = AlexNet_normal().to(device)
else:
    if model_name == 'resnet':
        model = FiResnet18(coefficient=coefficient, error_type=error_type, batch_size=batch_size).to(device)
    else:
        model = FiAlexNet(coefficient=coefficient, error_type=error_type, batch_size=batch_size).to(device)
    
    
optimizer = optim.SGD(model.parameters(), lr=1e-3, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
criterion = nn.CrossEntropyLoss()

ACC_tr = []
ACC_te = []
LOS_tr = []
LOS_te = []




def run_train(epoch, T, error, Warmup):
    print('\nEpoch: %d' % epoch)
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets) 
        loss.backward()
        
        optimizer.step()
        T += 1

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    print('Training Loss: %.3f, Acc: %.3f'%(train_loss/(batch_idx+1), 100.*correct/total))
    ACC_tr.append(100.*correct/total)
    LOS_tr.append(train_loss/(batch_idx+1))
    
def run_test(epoch):
    global best_acc
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
        print('Testing Loss: %.3f, Acc: %.3f'%(test_loss/(batch_idx+1), 100.*correct/total))
    ACC_te.append(100.*correct/total)
    LOS_te.append(test_loss/(batch_idx+1))


T = 0
Warmup = True
error = 10
for i in range(epoch):
    run_train(i, T, error, Warmup)
    run_test(i)
    scheduler.step()

np.save('train_accuracy_'+error_type+'_'+model_name+'.npy',ACC_tr)
np.save('train_loss_'+error_type+'_'+model_name+'.npy',LOS_tr)
np.save('test_accuracy_'+error_type+'_'+model_name+'.npy',ACC_te)
np.save('test_loss_'+error_type+'_'+model_name+'.npy',LOS_te)


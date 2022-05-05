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
from models import FiNet, RealexNet, AlexNet

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def get_weights(model):
    weights = []
    grads = []
    params = list(model.parameters())
    for i in range(2,6):
        weights.append(params[i].detach())
        grads.append(params[i].grad)
    return weights,grads

def set_grad(model, grads):
    params = list(model.parameters())
    for i in range(4):
        params[i+2].grad = grads[i].detach()

def run_train(epoch,beta, T,coef,train_loader, model1, model2,model3,criterion, optimizer1,optimizer3, ACC_tr, LOS_tr):
    print('\nEpoch: %d' % epoch)
    model1.train()
    train_loss = 0
    correct = 0
    total = 0
    coefficient = 0
    
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer1.zero_grad()
        optimizer3.zero_grad()
        outputs, activations = model1(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        
        
        grad_out = model1.act.grad
        weights,grads = get_weights(model1)
        errors = model3()
        for i,j in zip(errors,activations):
            if i.shape!=j.shape:
                print('stop')
        errors = [i+j for i,j in zip(errors,activations) ]
        e_grad = model2(grad_out, errors, weights)
        sparsity = [torch.mean((i!=0).float()).cpu() for i in activations]
        a_mean = [i.mean() for i in activations]
        eb = [np.sqrt(i*64)*j.cpu() for i,j in zip(sparsity,a_mean)]
        loss2 = 0
        loss3 = 0
        for i in range(4):
            loss2 += 0.25*torch.square(errors[i]/eb[i]).mean()
            loss3 += 0.25*torch.square(grads[i]-e_grad[i]).mean()
        
        floss = loss3 - beta*loss2
        floss.backward()
        set_grad(model1, e_grad)
        
        optimizer1.step()
        optimizer3.step()
        T += 1

        coefficient += loss2.item()
        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    print('Training Loss: %.3f, Acc: %.3f, Coef: %5.3f'%(train_loss/(batch_idx+1), 100.*correct/total, coefficient))
    ACC_tr.append(100.*correct/total)
    LOS_tr.append(train_loss/(batch_idx+1))
    coef.append(coefficient/(batch_idx+1))
    
def run_test(epoch,test_loader, model, ACC_te, LOS_te):
    global best_acc
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs, activations = model(inputs)
            loss = criterion(outputs, targets)
            
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
        print('Testing Loss: %.3f, Acc: %.3f'%(test_loss/(batch_idx+1), 100.*correct/total))
    ACC_te.append(100.*correct/total)
    LOS_te.append(test_loss/(batch_idx+1))



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
error_type = 'generative'
model_name = 'alexnet'
beta = 1e-3

train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
#train_dataset, val_dataset = random_split(train_dataset,[len(train_dataset)-1000,1000])
test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_val)
train_loader = DataLoader(train_dataset, batch_size, shuffle=True, num_workers=2, drop_last=True, pin_memory=True)
#val_loader = DataLoader(val_dataset, batch_size*2, num_workers=4, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size*2, num_workers=2, pin_memory=True)
model1 = AlexNet().to(device)
model2 = RealexNet().to(device)
model3 = FiNet().to(device)
optimizer1 = optim.SGD(model1.parameters(), lr=1e-3, weight_decay=5e-4)
optimizer3 = optim.Adam(model3.parameters(), lr=1e-3, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer1, T_max=100)
criterion = nn.CrossEntropyLoss()


ACC_tr = []
ACC_te = []
LOS_tr = []
LOS_te = []
coef = []


T = 0
Warmup = True
error = 10
for i in range(epoch):
    run_train(i, beta,T,coef, train_loader, model1, model2,model3,criterion, optimizer1,optimizer3, ACC_tr, LOS_tr)
    run_test(i,test_loader, model1, ACC_te, LOS_te)
    scheduler.step()

np.save('train_accuracy_'+error_type+'_'+model_name+'.npy',ACC_tr)
np.save('train_loss_'+error_type+'_'+model_name+'.npy',LOS_tr)
np.save('test_accuracy_'+error_type+'_'+model_name+'.npy',ACC_te)
np.save('test_loss_'+error_type+'_'+model_name+'.npy',LOS_te)
np.save('training_error_'+error_type+'_'+model_name+'.npy',coef)
torch.save(model3.state_dict(), 'FiNet_1e3.pth')

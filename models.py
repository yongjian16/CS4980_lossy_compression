#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 22 01:27:18 2022

@author: yz
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from FI import FiConv2d


class FiResBlock(nn.Module):
    def __init__(self, inchannel, outchannel, stride=1, coefficient=0., error_type='uniform', batch_size=64):
        super(FiResBlock, self).__init__()

        self.left = nn.Sequential(
            FiConv2d(inchannel, outchannel, kernel_size=3, 
                     stride=stride, padding=1, bias=False,
                     coefficient=coefficient,
                     error_type=error_type,
                     batch_size=batch_size),
            nn.BatchNorm2d(outchannel),
            nn.LeakyReLU(0.1),
            FiConv2d(outchannel, outchannel, kernel_size=3,
                     stride=1, padding=1, bias=False,
                     coefficient=coefficient,
                     error_type=error_type,
                     batch_size=batch_size),
            nn.BatchNorm2d(outchannel)
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or inchannel != outchannel:

            self.shortcut = nn.Sequential(
                nn.Conv2d(inchannel, outchannel, kernel_size=1, 
                          stride=stride, bias=False),
                nn.BatchNorm2d(outchannel)
            )
            
    def forward(self, x):
        out = self.left(x)
        out = out + self.shortcut(x)
        out = nn.LeakyReLU(0.1)(out)
        
        return out
    
class ResBlock(nn.Module):
    def __init__(self, inchannel, outchannel, stride=1):
        super(ResBlock, self).__init__()

        self.left = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, kernel_size=3, 
                     stride=stride, padding=1, bias=False,),
            nn.BatchNorm2d(outchannel),
            nn.LeakyReLU(0.1),
            nn.Conv2d(outchannel, outchannel, kernel_size=3,
                     stride=1, padding=1, bias=False,),
            nn.BatchNorm2d(outchannel)
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or inchannel != outchannel:

            self.shortcut = nn.Sequential(
                nn.Conv2d(inchannel, outchannel, kernel_size=1, 
                          stride=stride, bias=False),
                nn.BatchNorm2d(outchannel)
            )
            
    def forward(self, x):
        out = self.left(x)
        out = out + self.shortcut(x)
        out = nn.LeakyReLU(0.1)(out)
        
        return out

class Resnet18(nn.Module):
    def __init__(self, num_classes=10):
        super(Resnet18, self).__init__()
        self.inchannel = 64
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1)
        )

        self.layer1 = self.make_layer(ResBlock, 64, 2, stride=1)
        self.layer2 = self.make_layer(ResBlock, 128, 2, stride=2)
        self.layer3 = self.make_layer(ResBlock, 256, 2, stride=2)        
        self.layer4 = self.make_layer(ResBlock, 512, 2, stride=2)        
        self.fc = nn.Linear(512, num_classes)

    def make_layer(self, block, channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.inchannel, channels, stride))
            self.inchannel = channels
        return nn.Sequential(*layers)
    
    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


class FiResnet18(nn.Module):
    def __init__(self, num_classes=10, coefficient=0., error_type='uniform', batch_size=64):
        super(FiResnet18, self).__init__()
        self.inchannel = 64
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1)
        )
        self.coefficient = coefficient
        self.error_type = error_type
        self.batch_size = batch_size
        self.layer1 = self.make_layer(FiResBlock, 64, 2, stride=1)
        self.layer2 = self.make_layer(FiResBlock, 128, 2, stride=2)
        self.layer3 = self.make_layer(FiResBlock, 256, 2, stride=2)        
        self.layer4 = self.make_layer(FiResBlock, 512, 2, stride=2)        
        self.fc = nn.Linear(512, num_classes)

    def make_layer(self, block, channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.inchannel, channels, stride,
                                self.coefficient,self.error_type,self.batch_size))
            self.inchannel = channels
        return nn.Sequential(*layers)
    
    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out
    
    
class FiAlexNet(nn.Module):
    def __init__(self, classes=10, coefficient=0., error_type='uniform', batch_size=64):
        super(FiAlexNet, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3,
                      out_channels=96,
                      kernel_size=2,
                      stride=1,
                      padding=1),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv2 = nn.Sequential(
            FiConv2d(in_channels=96,
                      out_channels=256,
                      kernel_size=2,
                      stride=1,
                      padding=1,
                      coefficient=coefficient,
                      error_type=error_type,
                      batch_size=batch_size
                      ),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(kernel_size=2)
        )

        self.conv3 = nn.Sequential(
            FiConv2d(in_channels=256,
                      out_channels=384,
                      kernel_size=3,
                      stride=1,
                      padding=1,
                      coefficient=coefficient,
                      error_type=error_type,
                      batch_size=batch_size
                      ),
            nn.LeakyReLU(0.1),
        )
        self.conv4 = nn.Sequential(
            FiConv2d(in_channels=384,
                      out_channels=384,
                      kernel_size=3,
                      stride=1,
                      padding=1,
                      coefficient=coefficient,
                      error_type=error_type,
                      batch_size=batch_size
                      ),
            nn.LeakyReLU(0.1),
        )
        self.conv5 = nn.Sequential(
            FiConv2d(in_channels=384,
                      out_channels=256,
                      kernel_size=2,
                      stride=1,
                      padding=1,
                      coefficient=coefficient,
                      error_type=error_type,
                      batch_size=batch_size,
                      ),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(kernel_size=2)
        )

        self.fc1 = nn.Sequential(
            nn.Linear(4 * 4 * 256, 4096),
            nn.BatchNorm1d(4096),
            nn.ReLU(),
        )

        self.fc2 = nn.Sequential(
            nn.Linear(4096, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
        )

        self.out = nn.Linear(1024, classes)

    def forward(self, inputs):
        network = self.conv1(inputs)
        network = self.conv2(network)
        network = self.conv3(network)
        network = self.conv4(network)
        network = self.conv5(network)
        network = network.view(network.size(0), -1)
        network = self.fc1(network)
        network = self.fc2(network)
        out = self.out(network)
        return out
    
class AlexNet_normal(nn.Module):
    def __init__(self, classes=10):
        super(AlexNet_normal, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3,
                      out_channels=96,
                      kernel_size=2,
                      stride=1,
                      padding=1),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=96,
                      out_channels=256,
                      kernel_size=2,
                      stride=1,
                      padding=1,bias=False),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(kernel_size=2)
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=256,
                      out_channels=384,
                      kernel_size=3,
                      stride=1,
                      padding=1,bias=False),
            nn.LeakyReLU(0.1),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=384,
                      out_channels=384,
                      kernel_size=3,
                      stride=1,
                      padding=1,bias=False),
            nn.LeakyReLU(0.1),
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels=384,
                      out_channels=256,
                      kernel_size=2,
                      stride=1,
                      padding=1,bias=False),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(kernel_size=2)
        )

        self.fc1 = nn.Sequential(
            nn.Linear(4 * 4 * 256, 4096),
            nn.BatchNorm1d(4096),
            nn.ReLU(),
        )

        self.fc2 = nn.Sequential(
            nn.Linear(4096, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
        )

        self.out = nn.Linear(1024, classes)

    def forward(self, inputs):
        network = self.conv1(inputs)
        network = self.conv2(network)
        network = self.conv3(network)
        network = self.conv4(network)
        network = self.conv5(network)
        network = network.view(network.size(0), -1)
        network = self.fc1(network)
        network = self.fc2(network)
        out = self.out(network)
        return out


class AlexNet(nn.Module):
    def __init__(self, classes=10):
        super(AlexNet, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3,
                      out_channels=96,
                      kernel_size=2,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=96,
                      out_channels=256,
                      kernel_size=2,
                      stride=1,
                      padding=1,bias=False),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=256,
                      out_channels=384,
                      kernel_size=3,
                      stride=1,
                      padding=1,bias=False),
            nn.ReLU(),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=384,
                      out_channels=384,
                      kernel_size=3,
                      stride=1,
                      padding=1,bias=False),
            nn.ReLU(),
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels=384,
                      out_channels=256,
                      kernel_size=2,
                      stride=1,
                      padding=1,bias=False),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )

        self.fc1 = nn.Sequential(
            nn.Linear(4 * 4 * 256, 4096),
            nn.BatchNorm1d(4096),
            nn.ReLU(),
        )

        self.fc2 = nn.Sequential(
            nn.Linear(4096, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
        )

        self.out = nn.Linear(1024, classes)
        self.act = 0
        
    def forward(self, inputs):
        activations = []
        network = self.conv1(inputs)
        activations.append(network.detach())
        network = self.conv2(network)
        activations.append(network.detach())
        network = self.conv3(network)
        activations.append(network.detach())
        network = self.conv4(network)
        activations.append(network.detach())
        network = self.conv5(network)
        if self.training:
            self.act = network
            self.act.retain_grad()
        network = network.view(network.size(0), -1)
        network = self.fc1(network)
        network = self.fc2(network)
        out = self.out(network)
        return out, activations
    
class RealexNet(nn.Module):
    def __init__(self, batch_size=64):
        super(RealexNet,self).__init__()
        self.batch_size = batch_size
    
    def forward(self, grad_out, activations, weights):
        a2, a3, a4, a5 = activations
        w2, w3, w4, w5 = weights
        # layer 5
        with torch.no_grad():
            b = F.conv2d(a5,w5,stride=1,padding=1)
            mask1 = (b>0).float()
            mask2, indices = F.max_pool2d(torch.relu(b), kernel_size=2, return_indices=True)
            grad_out = F.max_unpool2d(grad_out, indices, kernel_size=2, output_size=b.size() )
            grad_out = grad_out*mask1
            
        in_grad = torch.nn.grad.conv2d_input(a5.shape, w5, grad_out,stride=1,padding=1)
        w5_grad = torch.nn.grad.conv2d_weight(a5, w5.shape, grad_out,stride=1,padding=1)
        
        # layer 4
        with torch.no_grad():
            b = F.conv2d(a4,w4,stride=1,padding=1)
            mask1 = (b>0).float()
            in_grad = in_grad*mask1
            
        in_grad1 = torch.nn.grad.conv2d_input(a4.shape, w4, in_grad,stride=1,padding=1)
        w4_grad = torch.nn.grad.conv2d_weight(a4, w4.shape, in_grad,stride=1,padding=1)
        
        # layer 3
        with torch.no_grad():
            b = F.conv2d(a3,w3,stride=1,padding=1)
            mask1 = (b>0).float()
            in_grad = in_grad*mask1
            
        in_grad = torch.nn.grad.conv2d_input(a3.shape, w3, in_grad1,stride=1,padding=1)
        w3_grad = torch.nn.grad.conv2d_weight(a3, w3.shape, in_grad1,stride=1,padding=1)
        
        # layer 2
        with torch.no_grad():
            b = F.conv2d(a2,w2,stride=1,padding=1)
            mask1 = (b>0).float()
            mask2, indices = F.max_pool2d(torch.relu(b), kernel_size=2, return_indices=True)
            in_grad = F.max_unpool2d(in_grad, indices, kernel_size=2, output_size=b.size() )
            in_grad = in_grad*mask1
            
        in_grad1 = torch.nn.grad.conv2d_input(a2.shape, w2, in_grad,stride=1,padding=1)
        w2_grad = torch.nn.grad.conv2d_weight(a2, w2.shape, in_grad,stride=1,padding=1)
        return w2_grad,w3_grad,w4_grad,w5_grad

class FiNet(nn.Module):
    def __init__(self, batch_size=64):
        super(FiNet,self).__init__()
        self.FI = nn.Linear(50, 128)
        self.FI2 = nn.Linear(128, 24576)
        self.FI3 = nn.Linear(128, 16384)
        self.FI4 = nn.Linear(128, 24576)
        self.FI5 = nn.Linear(128, 24576)
        self.batch_size = batch_size
        self.bn = nn.BatchNorm1d(128)

    def forward(self):
        z = torch.randn(self.batch_size,50).cuda()
        z = self.FI(z)
        z = self.bn(z)
        z = nn.LeakyReLU(0.2, inplace=True)(z)
        z2 = self.FI2(z).view(self.batch_size, 96, 16, 16)
        z3 = self.FI3(z).view(self.batch_size, 256, 8, 8)
        z4 = self.FI4(z).view(self.batch_size, 384, 8, 8)
        z5 = self.FI5(z).view(self.batch_size, 384, 8, 8)
        return z2,z3,z4,z5

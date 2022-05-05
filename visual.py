#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  4 21:04:03 2022

@author: yz
"""

import seaborn as sns
from models import FiNet
import torch
import matplotlib.pyplot as plt
import numpy as np


sns.set_style("white")
model3 = FiNet().cuda()
model3.load_state_dict(torch.load('FiNet_1e3.pth'))
layers = model3.forward()

colors = ["dodgerblue", "orange", "deeppink","gold"]
for i in range(4):
    items = np.round(layers[i].detach().cpu().view(-1).numpy()*1e+7)
    plt.figure(figsize=(10,7), dpi= 80)
    sns.displot(items, color=colors[i],bins=15000,kde=True)
    print(layers[i].mean(),layers[i].std(),layers[i].var())
    plt.xlim(-10000,10000)
    plt.xticks([])
    plt.show()
    plt.savefig('layer'+str(i)+'.png')
    
# encoding: utf-8
# !/usr/bin/env python3

import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)


class ConvNet(nn.Module):
    def __init__(self, din=25, dout=100, C=[100, 100]):
        super(ConvNet, self).__init__()
        self.C = C
        self.conv1 = nn.Conv2d(1, C[0], 3, padding=0)
        self.conv2 = nn.Conv2d(C[0], C[1], 3, padding=0)
        self.conv3 = nn.Conv2d(C[1], dout, 3, padding=0)

        self.pool1 = nn.AvgPool2d(2, 2)
        self.pool2 = nn.AvgPool2d(2, 2)

        self.bn1 = nn.BatchNorm2d(C[0])
        self.bn2 = nn.BatchNorm2d(C[1])
        self.bn3 = nn.BatchNorm2d(dout)

    def forward(self, x):
        y = []
        x = x.unsqueeze(1)

        x = F.selu(self.conv1(x))
        x = self.pool1(x)
        y.append(x.view(x.shape[0], -1))

        x = F.selu(self.conv2(x))
        x = self.pool2(x)
        y.append(x.view(x.shape[0], -1))

        x = F.selu(self.conv3(x))
        x = self.pool2(x)
        y.append(x.view(x.shape[0], -1))

        return x, y


class MLP(nn.Module):
    def __init__(self, din=25, dout=2, C=[20, 20]):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(din, C[0])  # 6*6 from image dimension
        self.fc2 = nn.Linear(C[0], C[1])
        self.fc3 = nn.Linear(C[1], dout)

    def forward(self, x):
        x = F.selu(self.fc1(x))
        x = F.selu(self.fc2(x))
        x = self.fc3(x)
        return x


class DMLLinear(ConvNet):
    def __init__(self, din=25):
        super(DMLLinear, self).__init__()
        self.out = nn.Linear(din, 1)

    def forward(self, x):
        x = self.out(x.view(x.shape[0], -1))
        return x


class DMLConvNet(ConvNet):
    def __init__(self, din=25, dout=100, C=[100, 100]):
        super().__init__(din=din, dout=dout, C=C)
        self.out = nn.Linear(dout*3, 1)
        self.dp = nn.Dropout(0.1)
        # super(DMLConvNet, self).__init__()
        '''
        self.C = C
        self.conv1 = nn.Conv2d(1, C[0], 3, padding=0)
        self.conv2 = nn.Conv2d(C[0], C[1], 3, padding=0)
        self.conv3 = nn.Conv2d(C[1], dout, 3, padding=0)

        self.pool1 = nn.AvgPool2d(2, 2)
        self.pool2 = nn.AvgPool2d(2, 2)

        self.bn1 = nn.BatchNorm2d(C[0])
        self.bn2 = nn.BatchNorm2d(C[1])
        self.bn3 = nn.BatchNorm2d(dout)
        '''

    def forward(self, x):
        y = []
        x = x.unsqueeze(1)

        x = F.selu(self.conv1(x))
        x = self.pool1(x)
        y.append(x.view(x.shape[0], -1))

        x = F.selu(self.conv2(x))
        x = self.pool2(x)
        y.append(x.view(x.shape[0], -1))

        x = F.selu(self.conv3(x))
        x = self.pool2(x)
        y.append(x.view(x.shape[0], -1))

        x = self.out(x.view(x.shape[0], -1))
        y.append(x.view(x.shape[0], -1))

        return x, y

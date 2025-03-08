# -*- coding: utf-8 -*-
# @Time    : 2023/2/6 20:26
# @Author  : ht
# @File    : CNN_model.py
# @Software: PyCharm
import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNNet(nn.Module):
    def __init__(self, in_planes, output=128):
        super(CNNNet, self).__init__()
        self.in_planes = in_planes
        self.output = output


        self.conv1 = nn.Conv1d(self.in_planes, 64, kernel_size=3, stride=1, padding=0)
        self.bn1 = nn.BatchNorm1d(64)
        self.pool1 = nn.MaxPool1d(kernel_size=3,stride=2,padding=1)

        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, stride=1,padding=0)
        self.bn2 = nn.BatchNorm1d(128)
        self.pool2 = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        self.conv3 = nn.Conv1d(128, 256, kernel_size=3, stride=1, padding=0)
        self.bn3 = nn.BatchNorm1d(256)
        self.pool3 = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        self.relu = nn.ReLU(inplace=True)

        self.drop = nn.Dropout(0.5)

    def forward(self, x):

        out = self.pool1(self.relu(self.bn1(self.conv1(x))))

        out = self.pool2(self.relu(self.bn2(self.conv2(out))))

        if self.output == 256:
            out = self.pool3(self.relu(self.bn3(self.conv3(out))))

        return out


def CNNNet18(in_planes,out_planes):
    return CNNNet(in_planes, output=out_planes)

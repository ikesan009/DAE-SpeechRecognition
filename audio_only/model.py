# coding: utf-8

"""
###audio recognition model###

audiofile:
    1ch
    16kHz
    66kbps
    1.16s
    
model:
    #temporalConv#
    input:[batch, 18560]
    ↓
    reshape:[batch, 1, 18560]
    ↓
    fronted1D:[batch, 64, 4640]
    ↓
    ↓conv1d:[batch, 64, 4640]
    ↓BatchNorm1d
    ↓ReLU
    ↓
    resnet18:[batch*29, 512]
    ↓
    ↓layer1:[batch, 64, 4640]
    ↓↓conv1d:[batch, 64, 4640]
    ↓↓BatchNorm1d
    ↓↓ReLU
    ↓↓conv1d:[batch, 64, 4640]
    ↓↓BatchNorm1d
    ↓↓residual
    ↓↓ReLU
    ↓【x2】
    ↓↓
    ↓layer2:[batch, 128, 2320]
    ↓↓conv1d:[batch, 128, 2320]
    ↓↓BatchNorm1d
    ↓↓conv1d:[batch, 128, 2320]
    ↓↓BatchNorm1d
    ↓↓ReLU
    ↓↓conv1d:[batch, 128, 2320]
    ↓↓BatchNorm1d
    ↓↓residual
    ↓↓ReLU
    ↓【x2】
    ↓↓
    ↓layer3:[batch, 256, 1160]
    ↓↓conv1d:[batch, 256, 1160]
    ↓↓BatchNorm1d
    ↓↓conv1d:[batch, 256, 1160]
    ↓↓BatchNorm1d
    ↓↓ReLU
    ↓↓conv1d:[batch, 256, 1160]
    ↓↓BatchNorm1d
    ↓↓residual
    ↓↓ReLU
    ↓【x2】
    ↓↓
    ↓layer4:[batch, 512, 580]
    ↓↓conv1d:[batch, 512, 580]
    ↓↓BatchNorm1d
    ↓↓conv1d:[batch, 512, 580]
    ↓↓BatchNorm1d
    ↓↓ReLU
    ↓↓conv1d:[batch, 512, 580]
    ↓↓BatchNorm1d
    ↓↓residual
    ↓↓ReLU
    ↓【x2】
    ↓↓
    ↓avgpool:[batch, 512, 29]
    ↓reshape:[batch, 29, 512]
    ↓reshape:[batch*29, 512]
    ↓fc:[batch*29, 512]
    ↓
    reshape:[batch, 29, 512]
    reshape:[batch, 512, 29]
    ↓
    backend_conv1:[batch, 2048, 1]
    ↓
    ↓conv1d:[batch, 1024, 13]
    ↓BatchNorm1d
    ↓ReLU
    ↓maxpool:[batch, 1024, 7]
    ↓conv1d:[batch, 2048, 2]
    ↓BatchNorm1d
    ↓ReLU
    ↓
    mean:[batch, 2048]
    ↓
    backend_conv2:[batch, 500]
    ↓
    ↓Linear:[batch, 512]
    ↓BatchNorm1d
    ↓ReLU
    ↓Linear:[batch, 500]
    ↓
    output:[batch, 500]

    #backendGRU#
    resnet18:[batch*29, 512]
    ↓
    reshape:[batch, 29, 512]
    ↓
    gru:[batch, 29, 500]
    ↓
    ↓nn.GRU:[batch, 29, 1024]
    ↓fc:[batch, 29, 500]
    ↓
    output:[batch, 29, 500]
"""

import math
import numpy as np


import torch
import torch.nn as nn
from torch.autograd import Variable


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv1d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm1d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool1d(kernel_size=20, padding=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                n = m.kernel_size[0] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.transpose(1, 2)
        x = x.contiguous()
        x = x.view(-1, x.size(2))
        x = self.fc(x)
        return x


class GRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, every_frame=True):
        super(GRU, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.every_frame = every_frame

        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size*2, num_classes)

    def forward(self, x):
        h0 = Variable(torch.zeros(self.num_layers*2, x.size(0), self.hidden_size))
        out, _ = self.gru(x, h0)
        if self.every_frame:
            out = self.fc(out)  # predicitions based on every time step
        else:
            out = self.fc(out[:, -1, :])  # predictions based on the last time step
        return out


class Lipreading(nn.Module):
    def __init__(self, mode, inputDim=256, hiddenDim=512, nClasses=500, frameLen=29, every_frame=True):
        super(Lipreading, self).__init__()
        self.mode = mode
        self.inputDim = inputDim
        self.hiddenDim = hiddenDim
        self.nClasses = nClasses
        self.frameLen = frameLen
        self.every_frame = every_frame
        self.nLayers = 2
        # frontend1D
        self.fronted1D = nn.Sequential(
                nn.Conv1d(1, 64, kernel_size=80, stride=4, padding=38, bias=False),
                nn.BatchNorm1d(64),
                nn.ReLU(True)
                )
        # resnet
        self.resnet18 = ResNet(BasicBlock, [2, 2, 2, 2], num_classes=self.inputDim)
        # backend_conv
        self.backend_conv1 = nn.Sequential(
            nn.Conv1d(self.inputDim, 2*self.inputDim, 5, 2, 0, bias=False),
            nn.BatchNorm1d(2*self.inputDim),
            nn.ReLU(True),
            nn.MaxPool1d(2, 2),
            nn.Conv1d(2*self.inputDim, 4*self.inputDim, 5, 2, 0, bias=False),
            nn.BatchNorm1d(4*self.inputDim),
            nn.ReLU(True),
                )
        self.backend_conv2 = nn.Sequential(
            nn.Linear(4*self.inputDim, self.inputDim),
            nn.BatchNorm1d(self.inputDim),
            nn.ReLU(True),
            nn.Linear(self.inputDim, self.nClasses)
        )
        # backend_gru
        self.gru = GRU(self.inputDim, self.hiddenDim, self.nLayers, self.nClasses, self.every_frame)
        # initialize
        self._initialize_weights()

    def forward(self, x):
        x = x.view(-1, 1, x.size(1))
        x = self.fronted1D(x)
        x = x.contiguous()
        x = self.resnet18(x)
        if self.mode == 'temporalConv':
            x = x.view(-1, self.frameLen, self.inputDim)
            x = x.transpose(1, 2)
            x = self.backend_conv1(x)
            x = torch.mean(x, 2)
            x = self.backend_conv2(x)
        elif self.mode == 'backendGRU' or self.mode == 'finetuneGRU':
            x = x.view(-1, self.frameLen, self.inputDim)
            x = self.gru(x)
        else:
            raise Exception('No model is selected')
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()

            elif isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()

            elif isinstance(m, nn.Conv1d):
                n = m.kernel_size[0] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()

            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


def lipreading(mode, inputDim=256, hiddenDim=512, nClasses=500, frameLen=25, every_frame=True):
        model = Lipreading(mode, inputDim=inputDim, hiddenDim=hiddenDim, nClasses=nClasses, frameLen=frameLen, every_frame=every_frame)
        return model
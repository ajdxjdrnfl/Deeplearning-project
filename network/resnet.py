#!/usr/bin/env python
# coding: utf-8

import torch
import torch.nn as nn



def conv3x3(input_channels, num_channels, stride=1, groups=1, dilation=1):
    return nn.Conv2d(input_channels, num_channels, kernel_size=3, stride=stride, padding=dilation, groups=groups, bias=False, dilation=dilation )

def conv1x1(input_channels, num_channels, stride=1):
    return nn.Conv2d(input_channels, num_channels, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion=1
    def __init__(self, input_channels, num_channels, stride=1, downsample=None, groups=1, base_width=64, dilation=1, norm_layer=None):
        
        super(BasicBlock, self).__init__()      
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasickBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dialation > 1 not supported in BasickBlock")
        self.conv1 = conv3x3(input_channels, num_channels, stride)
        self.bn1 = norm_layer(num_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(num_channels, num_channels)
        self.bn2 = norm_layer(num_channels)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        # downsample for example : conv1x1
        if self.downsample is not None:
            identity = self.downsample(x)
        
        out += identity
        out = self.relu(out)    
        
        return out


class BottleNeck(nn.Module):
    
    expansion = 4
    def __init__(self, input_channels, num_channels, stride=1, downsample=None, groups=1, base_width=64, dilation=1, norm_layer=None):
        super(BottleNeck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(num_channels * (base_width/64.)) * groups

        self.conv1 = conv1x1(input_channels, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, num_channels * self.expansion)
        self.bn3 = norm_layer(num_channels * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)
        
        out += identity
        out = self.relu(out)
        return out

class ResNetLayer(nn.Module):

    def __init__(self, block, layers, zero_init_residual = False, groups=1, width_per_group=64, replace_stride_with_dilation=None, norm_layer=None):
        super(ResNetLayer, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self.norm_layer = norm_layer

        self.input_channels = 64
        self.dilation = 1

        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group

        self.conv1 = nn.Conv2d(3, self.input_channels, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(self.input_channels)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 =  self._make_layer(block, 64, layers[0])
        self.layer2 =  self._make_layer(block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, dilate = replace_stride_with_dilation[2])
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, BottleNeck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    
    def _make_layer(self, block, num_channels, blocks, stride=1, dilate=False):
        norm_layer = self.norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        
        if stride != 1 or self.input_channels != num_channels*block.expansion:
            downsample =nn.Sequential(conv1x1(self.input_channels, num_channels*block.expansion, stride), norm_layer(num_channels * block.expansion))
        
        layers = []
        layers.append(block(self.input_channels, num_channels, stride, downsample, self.groups, self.base_width, previous_dilation, norm_layer))
        self.input_channels = num_channels * block.expansion

        for _ in range(1, blocks):
            layers.append(block(self.input_channels, num_channels, groups=self.groups, base_width = self.base_width, dilation=self.dilation, norm_layer=norm_layer))
        return nn.Sequential(*layers)
    
    def _forward_impl(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        return x

    def forward(self, x):
        return self._forward_impl(x)
    
def resnetlayer18():
    """ return a ResNet 18 object
    """
    return ResNetLayer(BasicBlock, [2, 2, 2, 2])

def resnetlayer34():
    """ return a ResNet 34 object
    """
    return ResNetLayer(BasicBlock, [3, 4, 6, 3])

class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes = 1000, zero_init_residual = False, groups=1, width_per_group=64, replace_stride_with_dilation=None, norm_layer=None):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self.norm_layer = norm_layer

        self.input_channels = 64
        self.dilation = 1

        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
     
        self.conv1 = nn.Conv2d(3, self.input_channels, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(self.input_channels)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
   
        self.layer1 =  self._make_layer(block, 64, layers[0])
        
        self.layer2 =  self._make_layer(block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0])
       
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, dilate = replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, BottleNeck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    
    def _make_layer(self, block, num_channels, blocks, stride=1, dilate=False):
        norm_layer = self.norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        
        if stride != 1 or self.input_channels != num_channels*block.expansion:
            downsample =nn.Sequential(conv1x1(self.input_channels, num_channels*block.expansion, stride), norm_layer(num_channels * block.expansion),)
        
        layers = []
        layers.append(block(self.input_channels, num_channels, stride, downsample, self.groups, self.base_width, previous_dilation, norm_layer))
        self.input_channels = num_channels * block.expansion

        for _ in range(1, blocks):
            layers.append(block(self.input_channels, num_channels, groups=self.groups, base_width = self.base_width, dilation=self.dilation, norm_layer=norm_layer))
        return nn.Sequential(*layers)
    
    def _forward_impl(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def forward(self, x):
        return self._forward_impl(x)


def resnet18(num_classes):
    """ return a ResNet 18 object
    """
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes)

def resnet34(num_classes):
    """ return a ResNet 34 object
    """
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes=num_classes)

def resnet50(num_classes):
    """ return a ResNet 50 object
    """
    return ResNet(BottleNeck, [3, 4, 6, 3], num_classes=num_classes)

def resnet101(num_classes):
    """ return a ResNet 101 object
    """
    return ResNet(BottleNeck, [3, 4, 23, 3], num_classes=num_classes)

def resnet152(num_classes):
    """ return a ResNet 152 object
    """
    return ResNet(BottleNeck, [3, 8, 36, 3], num_classes=num_classes)
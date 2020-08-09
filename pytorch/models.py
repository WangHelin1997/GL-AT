import os
import sys
import math
import time
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp
from torch.nn.parameter import Parameter

from torchlibrosa.stft import Spectrogram, LogmelFilterBank
from torchlibrosa.augmentation import SpecAugmentation
from pytorch_utils import do_mixup, interpolate, pad_framewise_output
 

def init_layer(layer):
    """Initialize a Linear or Convolutional layer. """
    nn.init.xavier_uniform_(layer.weight)
 
    if hasattr(layer, 'bias'):
        if layer.bias is not None:
            layer.bias.data.fill_(0.)
            
    
def init_bn(bn):
    """Initialize a Batchnorm layer. """
    bn.bias.data.fill_(0.)
    bn.weight.data.fill_(1.)


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        
        super(ConvBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels=in_channels, 
                              out_channels=out_channels,
                              kernel_size=(3, 3), stride=(1, 1),
                              padding=(1, 1), bias=False)
                              
        self.conv2 = nn.Conv2d(in_channels=out_channels, 
                              out_channels=out_channels,
                              kernel_size=(3, 3), stride=(1, 1),
                              padding=(1, 1), bias=False)
                              
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.init_weight()
        
    def init_weight(self):
        init_layer(self.conv1)
        init_layer(self.conv2)
        init_bn(self.bn1)
        init_bn(self.bn2)

        
    def forward(self, input, pool_size=(2, 2), pool_type='avg'):
        
        x = input
        x = F.relu_(self.bn1(self.conv1(x)))
        x = F.relu_(self.bn2(self.conv2(x)))
        if pool_type == 'max':
            x = F.max_pool2d(x, kernel_size=pool_size)
        elif pool_type == 'avg':
            x = F.avg_pool2d(x, kernel_size=pool_size)
        elif pool_type == 'avg+max':
            x1 = F.avg_pool2d(x, kernel_size=pool_size)
            x2 = F.max_pool2d(x, kernel_size=pool_size)
            x = x1 + x2
        else:
            raise Exception('Incorrect argument!')
        
        return x

class Cnn10(nn.Module):
    def __init__(self, sample_rate, window_size, hop_size, mel_bins, fmin, 
        fmax, classes_num):
        
        super(Cnn10, self).__init__()

        window = 'hann'
        center = True
        pad_mode = 'reflect'
        ref = 1.0
        amin = 1e-10
        top_db = None

        # Spectrogram extractor
        self.spectrogram_extractor = Spectrogram(n_fft=window_size, hop_length=hop_size, 
            win_length=window_size, window=window, center=center, pad_mode=pad_mode, 
            freeze_parameters=True)

        # Logmel feature extractor
        self.logmel_extractor = LogmelFilterBank(sr=sample_rate, n_fft=window_size, 
            n_mels=mel_bins, fmin=fmin, fmax=fmax, ref=ref, amin=amin, top_db=top_db, 
            freeze_parameters=True)

        # Spec augmenter
        self.spec_augmenter = SpecAugmentation(time_drop_width=64, time_stripes_num=2, 
            freq_drop_width=8, freq_stripes_num=2)

        self.bn0 = nn.BatchNorm2d(64)

        self.conv_block1 = ConvBlock(in_channels=1, out_channels=64)
        self.conv_block2 = ConvBlock(in_channels=64, out_channels=128)
        self.conv_block3 = ConvBlock(in_channels=128, out_channels=256)
        self.conv_block4 = ConvBlock(in_channels=256, out_channels=512)

        self.fc1 = nn.Linear(512, 512, bias=True)
        self.fc_audioset = nn.Linear(512, classes_num, bias=True)
        
        self.init_weight()

    def init_weight(self):
        init_bn(self.bn0)
        init_layer(self.fc1)
        init_layer(self.fc_audioset)
 
    def forward(self, input, mixup_lambda=None):
        """
        Input: (batch_size, data_length)"""

        x = self.spectrogram_extractor(input)   # (batch_size, 1, time_steps, freq_bins)
        x = self.logmel_extractor(x)    # (batch_size, 1, time_steps, mel_bins)
        
        x = x.transpose(1, 3)
        x = self.bn0(x)
        x = x.transpose(1, 3)
        
        if self.training:
            x = self.spec_augmenter(x)

        # Mixup on spectrogram
        if self.training and mixup_lambda is not None:
            x = do_mixup(x, mixup_lambda)
        
        x = self.conv_block1(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block2(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block3(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block4(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = torch.mean(x, dim=3)
        feature_map = x
        
        (x1, _) = torch.max(x, dim=2)
        x2 = torch.mean(x, dim=2)
        x = x1 + x2
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu_(self.fc1(x))
        embedding = F.dropout(x, p=0.5, training=self.training)
        clipwise_output = torch.sigmoid(self.fc_audioset(x))
        
        output_dict = {'clipwise_output': clipwise_output, 'feature_map': feature_map}

        return output_dict
    
class Cnn10_local(nn.Module):
    def __init__(self, sample_rate, window_size, hop_size, mel_bins, fmin, 
        fmax, classes_num, N, length):
        
        super(Cnn10_local, self).__init__()
        
        self.N = N
        self.length = length
        self.length_all = 10
        self.duration = int(sample_rate*self.length)
        self.duration_all = int(sample_rate*self.length_all)
        self.local_net = Cnn10(sample_rate, window_size, hop_size, mel_bins, fmin, 
        fmax, classes_num)
        self.local_net.load_state_dict(torch.load('/data/dean/panns/audioset_tagging_cnn/pytorch/Cnn10_mAP=0.380.pth')['model'])
        self.fc1 = nn.Linear(512, 512, bias=True)
        self.fc_audioset = nn.Linear(512, classes_num, bias=True)
        for name, module in self.local_net._modules.items():
            if name == 'fc1':
                self.fc1 = module
            if name == 'fc_audioset':
                self.fc_audioset = module
 
    def forward(self, input, global_prob, feature_map, mixup_lambda=None):
        """
        Input: (batch_size, data_length)""" #64, 1, 1001, 64
        x = input #64, 320000
        sorted, indices = torch.sort(global_prob, dim=1, descending=True)
        indices = indices[:,:self.N]#bs,N
        feature_map = feature_map.transpose(1, 2)
        feature_map = F.dropout(feature_map, p=0.5, training=self.training)
        embedding_L = F.relu_(self.fc1(feature_map))
        embedding_L = F.dropout(embedding_L, p=0.5, training=self.training)
        frame_prob = torch.sigmoid(self.fc_audioset(embedding_L))
#         frame_prob = torch.clamp(frame_prob, 1e-7, 1 - 1e-7)#bs,T,527
        
        frame_prob = frame_prob.transpose(1,2)#bs,527,T
        maps = torch.zeros(frame_prob.size(0),self.N,frame_prob.size(2)).cuda()#bs,N,T
        for i in range(indices.size(0)):
                maps[i] = torch.index_select(frame_prob[i],0,indices[i])
        local_regions = self.region_select(maps,x)
        
        local_ = local_regions.view(local_regions.size(0)*local_regions.size(1),local_regions.size(2))
        local_prob = self.local_net(local_,mixup_lambda)['clipwise_output']
        local_prob = local_prob.view(local_regions.size(0),local_regions.size(1),-1)
        (local_prob1, _) = torch.max(local_prob, dim=1)
        local_prob2 = torch.mean(local_prob, dim=1)
        local_prob = 0.5*(local_prob1 + local_prob2)
#         local_prob = torch.clamp(local_prob, 1e-7, 1 - 1e-7)  # bs,527
        
        prob = torch.cat([global_prob[:,None,:],local_prob[:,None,:]],1)
        (prob1, _) = torch.max(prob, dim=1)
        prob2 = torch.mean(prob,dim=1)
        prob = 0.5*(prob1 + prob2)
#         prob = torch.clamp(prob, 1e-7, 1 - 1e-7)
        
        output_dict = {'local_prob': local_prob, 'prob': prob}

        return output_dict
    
    def region_select(self, maps, x):
        local_regions = torch.zeros(x.size(0),self.N, self.duration).cuda()#bs,N,T,F
        over_range = int(self.duration*maps.size(-1)/self.duration_all)//4
        for i in range(maps.size(0)):
            for j in range(maps.size(1)):
                map_ = maps[i,j]
                (_,max_index)=torch.max(map_,dim=0)
                index_ = max_index
                max_index = int(self.duration_all*max_index/len(map_))
                index_l = max_index-self.duration//2
                index_r = max_index+self.duration//2
                if index_r >= self.duration_all:
                    local_regions[i,j,:] = x[i,self.duration_all-self.duration:self.duration_all]
#                     print(self.duration_all-self.duration,self.duration_all)
                    maps[i,:,-2*over_range:]=0.
                elif index_l < 0:
                    local_regions[i,j,:] = x[i,:self.duration]
#                     print(0,self.duration)
                    maps[i,:,:2*over_range]=0.
                else:
                    local_regions[i,j,:] = x[i,index_l:index_r]
#                     print(index_l,index_r)
                    maps[i,:,index_-over_range:index_+over_range]=0.
        return local_regions

def _resnet_conv3x3(in_planes, out_planes):
    #3x3 convolution with padding
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=1,
                     padding=1, groups=1, bias=False, dilation=1)


def _resnet_conv1x1(in_planes, out_planes):
    #1x1 convolution
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, bias=False)


class _ResnetBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(_ResnetBasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('_ResnetBasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in _ResnetBasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1

        self.stride = stride

        self.conv1 = _resnet_conv3x3(inplanes, planes)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = _resnet_conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

        self.init_weights()

    def init_weights(self):
        init_layer(self.conv1)
        init_bn(self.bn1)
        init_layer(self.conv2)
        init_bn(self.bn2)
        nn.init.constant_(self.bn2.weight, 0)

    def forward(self, x):
        identity = x

        if self.stride == 2:
            out = F.avg_pool2d(x, kernel_size=(2, 2))
        else:
            out = x

        out = self.conv1(out)
        out = self.bn1(out)
        out = self.relu(out)
        out = F.dropout(out, p=0.1, training=self.training)

        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.downsample is not None:
            identity = self.downsample(identity)

        out += identity
        out = self.relu(out)

        return out


class _ResnetBottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(_ResnetBottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        self.stride = stride
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = _resnet_conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = _resnet_conv3x3(width, width)
        self.bn2 = norm_layer(width)
        self.conv3 = _resnet_conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

        self.init_weights()

    def init_weights(self):
        init_layer(self.conv1)
        init_bn(self.bn1)
        init_layer(self.conv2)
        init_bn(self.bn2)
        init_layer(self.conv3)
        init_bn(self.bn3)
        nn.init.constant_(self.bn3.weight, 0)

    def forward(self, x):
        identity = x

        if self.stride == 2:
            x = F.avg_pool2d(x, kernel_size=(2, 2))

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = F.dropout(out, p=0.1, training=self.training)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(identity)

        out += identity
        out = self.relu(out)

        return out


class _ResNet(nn.Module):
    def __init__(self, block, layers, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(_ResNet, self).__init__()

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group

        self.layer1 = self._make_layer(block, 64, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            if stride == 1:
                downsample = nn.Sequential(
                    _resnet_conv1x1(self.inplanes, planes * block.expansion),
                    norm_layer(planes * block.expansion),
                )
                init_layer(downsample[0])
                init_bn(downsample[1])
            elif stride == 2:
                downsample = nn.Sequential(
                    nn.AvgPool2d(kernel_size=2), 
                    _resnet_conv1x1(self.inplanes, planes * block.expansion),
                    norm_layer(planes * block.expansion),
                )
                init_layer(downsample[1])
                init_bn(downsample[2])

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        return x

class ResNet38(nn.Module):
    def __init__(self, sample_rate, window_size, hop_size, mel_bins, fmin, 
        fmax, classes_num):
        
        super(ResNet38, self).__init__()

        window = 'hann'
        center = True
        pad_mode = 'reflect'
        ref = 1.0
        amin = 1e-10
        top_db = None

        # Spectrogram extractor
        self.spectrogram_extractor = Spectrogram(n_fft=window_size, hop_length=hop_size, 
            win_length=window_size, window=window, center=center, pad_mode=pad_mode, 
            freeze_parameters=True)

        # Logmel feature extractor
        self.logmel_extractor = LogmelFilterBank(sr=sample_rate, n_fft=window_size, 
            n_mels=mel_bins, fmin=fmin, fmax=fmax, ref=ref, amin=amin, top_db=top_db, 
            freeze_parameters=True)

        # Spec augmenter
        self.spec_augmenter = SpecAugmentation(time_drop_width=64, time_stripes_num=2, 
            freq_drop_width=8, freq_stripes_num=2)

        self.bn0 = nn.BatchNorm2d(64)

        self.conv_block1 = ConvBlock(in_channels=1, out_channels=64)
        # self.conv_block2 = ConvBlock(in_channels=64, out_channels=64)

        self.resnet = _ResNet(block=_ResnetBasicBlock, layers=[3, 4, 6, 3], zero_init_residual=True)

        self.conv_block_after1 = ConvBlock(in_channels=512, out_channels=2048)

        self.fc1 = nn.Linear(2048, 2048)
        self.fc_audioset = nn.Linear(2048, classes_num, bias=True)

        self.init_weights()

    def init_weights(self):
        init_bn(self.bn0)
        init_layer(self.fc1)
        init_layer(self.fc_audioset)


    def forward(self, input, mixup_lambda=None):
        """
        Input: (batch_size, data_length)"""

        x = self.spectrogram_extractor(input)   # (batch_size, 1, time_steps, freq_bins)
        x = self.logmel_extractor(x)    # (batch_size, 1, time_steps, mel_bins)
        
        x = x.transpose(1, 3)
        x = self.bn0(x)
        x = x.transpose(1, 3)
        
        if self.training:
            x = self.spec_augmenter(x)

        # Mixup on spectrogram
        if self.training and mixup_lambda is not None:
            x = do_mixup(x, mixup_lambda)
        
        x = self.conv_block1(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training, inplace=True)
        x = self.resnet(x)
        x = F.avg_pool2d(x, kernel_size=(2, 2))
        x = F.dropout(x, p=0.2, training=self.training, inplace=True)
        x = self.conv_block_after1(x, pool_size=(1, 1), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training, inplace=True)
        x = torch.mean(x, dim=3)
        feature_map=x
        
        (x1, _) = torch.max(x, dim=2)
        x2 = torch.mean(x, dim=2)
        x = x1 + x2
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu_(self.fc1(x))
        embedding = F.dropout(x, p=0.5, training=self.training)
        clipwise_output = torch.sigmoid(self.fc_audioset(x))
        
        output_dict = {'clipwise_output': clipwise_output, 'feature_map': feature_map}

        return output_dict
    
class ResNet38_local(nn.Module):
    def __init__(self, sample_rate, window_size, hop_size, mel_bins, fmin, 
        fmax, classes_num, N, length):
        
        super(ResNet38_local, self).__init__()
        
        self.N = N
        self.length = length
        self.length_all = 10
        self.duration = int(sample_rate*self.length)
        self.duration_all = int(sample_rate*self.length_all)
        self.local_net = ResNet38(sample_rate, window_size, hop_size, mel_bins, fmin, fmax, classes_num)
        self.local_net.load_state_dict(torch.load('/data/dean/panns/audioset_tagging_cnn/pytorch/ResNet38_mAP=0.434.pth')['model'])
        self.fc1 = nn.Linear(2048, 2048)
        self.fc_audioset = nn.Linear(2048, classes_num, bias=True)
        for name, module in self.local_net._modules.items():
            if name == 'fc1':
                self.fc1 = module
            if name == 'fc_audioset':
                self.fc_audioset = module
 
    def forward(self, input, global_prob, feature_map, mixup_lambda=None):
        """
        Input: (batch_size, data_length)""" #64, 1, 1001, 64
        x = input #64, 320000
        sorted, indices = torch.sort(global_prob, dim=1, descending=True)
        indices = indices[:,:self.N]#bs,N
        feature_map = feature_map.transpose(1, 2)
        feature_map = F.dropout(feature_map, p=0.5, training=self.training)
        embedding_L = F.relu_(self.fc1(feature_map))
        embedding_L = F.dropout(embedding_L, p=0.5, training=self.training)
        frame_prob = torch.sigmoid(self.fc_audioset(embedding_L))
#         frame_prob = torch.clamp(frame_prob, 1e-7, 1 - 1e-7)#bs,T,527
        
        frame_prob = frame_prob.transpose(1,2)#bs,527,T
        maps = torch.zeros(frame_prob.size(0),self.N,frame_prob.size(2)).cuda()#bs,N,T
        for i in range(indices.size(0)):
                maps[i] = torch.index_select(frame_prob[i],0,indices[i])
        local_regions = self.region_select(maps,x)
        
        local_ = local_regions.view(local_regions.size(0)*local_regions.size(1),local_regions.size(2))
        local_prob = self.local_net(local_,mixup_lambda)['clipwise_output']
        local_prob = local_prob.view(local_regions.size(0),local_regions.size(1),-1)
        (local_prob1, _) = torch.max(local_prob, dim=1)
        local_prob2 = torch.mean(local_prob, dim=1)
        local_prob = 0.5*(local_prob1 + local_prob2)
#         local_prob = torch.clamp(local_prob, 1e-7, 1 - 1e-7)  # bs,527
        
        prob = torch.cat([global_prob[:,None,:],local_prob[:,None,:]],1)
        (prob1, _) = torch.max(prob, dim=1)
        prob2 = torch.mean(prob,dim=1)
        prob = 0.5*(prob1 + prob2)
#         prob = torch.clamp(prob, 1e-7, 1 - 1e-7)
        
        output_dict = {'local_prob': local_prob, 'prob': prob}

        return output_dict
    
    def region_select(self, maps, x):
        local_regions = torch.zeros(x.size(0),self.N, self.duration).cuda()#bs,N,T,F
        over_range = int(self.duration*maps.size(-1)/self.duration_all)//4
        for i in range(maps.size(0)):
            for j in range(maps.size(1)):
                map_ = maps[i,j]
                (_,max_index)=torch.max(map_,dim=0)
                index_ = max_index
                max_index = int(self.duration_all*max_index/len(map_))
                index_l = max_index-self.duration//2
                index_r = max_index+self.duration//2
                if index_r >= self.duration_all:
                    local_regions[i,j,:] = x[i,self.duration_all-self.duration:self.duration_all]
#                     print(self.duration_all-self.duration,self.duration_all)
                    maps[i,:,-2*over_range:]=0.
                elif index_l < 0:
                    local_regions[i,j,:] = x[i,:self.duration]
#                     print(0,self.duration)
                    maps[i,:,:2*over_range]=0.
                else:
                    local_regions[i,j,:] = x[i,index_l:index_r]
#                     print(index_l,index_r)
                    maps[i,:,index_-over_range:index_+over_range]=0.
        return local_regions
    
class ConvPreWavBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        
        super(ConvPreWavBlock, self).__init__()
        
        self.conv1 = nn.Conv1d(in_channels=in_channels, 
                              out_channels=out_channels,
                              kernel_size=3, stride=1,
                              padding=1, bias=False)
                              
        self.conv2 = nn.Conv1d(in_channels=out_channels, 
                              out_channels=out_channels,
                              kernel_size=3, stride=1, dilation=2, 
                              padding=2, bias=False)
                              
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.bn2 = nn.BatchNorm1d(out_channels)

        self.init_weight()
        
    def init_weight(self):
        init_layer(self.conv1)
        init_layer(self.conv2)
        init_bn(self.bn1)
        init_bn(self.bn2)

        
    def forward(self, input, pool_size):
        
        x = input
        x = F.relu_(self.bn1(self.conv1(x)))
        x = F.relu_(self.bn2(self.conv2(x)))
        x = F.max_pool1d(x, kernel_size=pool_size)
        
        return x

class Wavegram_Logmel_Cnn14(nn.Module):
    def __init__(self, sample_rate, window_size, hop_size, mel_bins, fmin, 
        fmax, classes_num):
        
        super(Wavegram_Logmel_Cnn14, self).__init__()

        window = 'hann'
        center = True
        pad_mode = 'reflect'
        ref = 1.0
        amin = 1e-10
        top_db = None

        self.pre_conv0 = nn.Conv1d(in_channels=1, out_channels=64, kernel_size=11, stride=5, padding=5, bias=False)
        self.pre_bn0 = nn.BatchNorm1d(64)
        self.pre_block1 = ConvPreWavBlock(64, 64)
        self.pre_block2 = ConvPreWavBlock(64, 128)
        self.pre_block3 = ConvPreWavBlock(128, 128)
        self.pre_block4 = ConvBlock(in_channels=4, out_channels=64)

        # Spectrogram extractor
        self.spectrogram_extractor = Spectrogram(n_fft=window_size, hop_length=hop_size, 
            win_length=window_size, window=window, center=center, pad_mode=pad_mode, 
            freeze_parameters=True)

        # Logmel feature extractor
        self.logmel_extractor = LogmelFilterBank(sr=sample_rate, n_fft=window_size, 
            n_mels=mel_bins, fmin=fmin, fmax=fmax, ref=ref, amin=amin, top_db=top_db, 
            freeze_parameters=True)

        # Spec augmenter
        self.spec_augmenter = SpecAugmentation(time_drop_width=64, time_stripes_num=2, 
            freq_drop_width=8, freq_stripes_num=2)

        self.bn0 = nn.BatchNorm2d(64)

        self.conv_block1 = ConvBlock(in_channels=1, out_channels=64)
        self.conv_block2 = ConvBlock(in_channels=128, out_channels=128)
        self.conv_block3 = ConvBlock(in_channels=128, out_channels=256)
        self.conv_block4 = ConvBlock(in_channels=256, out_channels=512)
        self.conv_block5 = ConvBlock(in_channels=512, out_channels=1024)
        self.conv_block6 = ConvBlock(in_channels=1024, out_channels=2048)

        self.fc1 = nn.Linear(2048, 2048, bias=True)
        self.fc_audioset = nn.Linear(2048, classes_num, bias=True)
        
        self.init_weight()

    def init_weight(self):
        init_layer(self.pre_conv0)
        init_bn(self.pre_bn0)
        init_bn(self.bn0)
        init_layer(self.fc1)
        init_layer(self.fc_audioset)
 
    def forward(self, input, mixup_lambda=None):
        """
        Input: (batch_size, data_length)"""
        a1 = F.relu_(self.pre_bn0(self.pre_conv0(input[:, None, :])))
        a1 = self.pre_block1(a1, pool_size=4)
        a1 = self.pre_block2(a1, pool_size=4)
        a1 = self.pre_block3(a1, pool_size=4)
        a1 = a1.reshape((a1.shape[0], -1, 32, a1.shape[-1])).transpose(2, 3)
        a1 = self.pre_block4(a1, pool_size=(2, 1))

        x = self.spectrogram_extractor(input)   # (batch_size, 1, time_steps, freq_bins)
        x = self.logmel_extractor(x)    # (batch_size, 1, time_steps, mel_bins)
        
        x = x.transpose(1, 3)
        x = self.bn0(x)
        x = x.transpose(1, 3)

#         if self.training:
#             x = self.spec_augmenter(x)

        # Mixup on spectrogram
        if self.training and mixup_lambda is not None:
            x = do_mixup(x, mixup_lambda)
            a1 = do_mixup(a1, mixup_lambda)
        
        x = self.conv_block1(x, pool_size=(2, 2), pool_type='avg')
        x = torch.cat((x, a1), dim=1)
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block2(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block3(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block4(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block5(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block6(x, pool_size=(1, 1), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = torch.mean(x, dim=3)

        feature_map = x
        (x1, _) = torch.max(x, dim=2)
        x2 = torch.mean(x, dim=2)
        x = x1 + x2
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu_(self.fc1(x))
        embedding = F.dropout(x, p=0.5, training=self.training)
        clipwise_output = torch.sigmoid(self.fc_audioset(x))
        
        output_dict = {'clipwise_output': clipwise_output, 'feature_map': feature_map}

        return output_dict

class Wavegram_Logmel_Cnn14_local(nn.Module):
    def __init__(self, sample_rate, window_size, hop_size, mel_bins, fmin, 
        fmax, classes_num, N, length):
        
        super(Wavegram_Logmel_Cnn14_local, self).__init__()
        
        self.N = N
        self.length = length
        self.length_all = 10
        self.duration = int(sample_rate*self.length)
        self.duration_all = int(sample_rate*self.length_all)
        self.local_net = Wavegram_Logmel_Cnn14(sample_rate, window_size, hop_size, mel_bins, fmin, 
        fmax, classes_num)
        self.local_net.load_state_dict(torch.load('/data/dean/panns/audioset_tagging_cnn/pytorch/Wavegram_Logmel_Cnn14_mAP=0.439.pth')['model'])
        self.fc1 = nn.Linear(2048, 2048, bias=True)
        self.fc_audioset = nn.Linear(2048, classes_num, bias=True)
        for name, module in self.local_net._modules.items():
            if name == 'fc1':
                self.fc1 = module
            if name == 'fc_audioset':
                self.fc_audioset = module
 
    def forward(self, input, global_prob, feature_map, mixup_lambda=None):
        """
        Input: (batch_size, data_length)""" #64, 1, 1001, 64
        x = input #64, 320000
        sorted, indices = torch.sort(global_prob, dim=1, descending=True)
        indices = indices[:,:self.N]#bs,N
        feature_map = feature_map.transpose(1, 2)
        feature_map = F.dropout(feature_map, p=0.5, training=self.training)
        embedding_L = F.relu_(self.fc1(feature_map))
        embedding_L = F.dropout(embedding_L, p=0.5, training=self.training)
        frame_prob = torch.sigmoid(self.fc_audioset(embedding_L))
#         frame_prob = torch.clamp(frame_prob, 1e-7, 1 - 1e-7)#bs,T,527
        
        frame_prob = frame_prob.transpose(1,2)#bs,527,T
        maps = torch.zeros(frame_prob.size(0),self.N,frame_prob.size(2)).cuda()#bs,N,T
        for i in range(indices.size(0)):
                maps[i] = torch.index_select(frame_prob[i],0,indices[i])
        local_regions = self.region_select(maps,x)
        
        local_ = local_regions.view(local_regions.size(0)*local_regions.size(1),local_regions.size(2))
        local_prob = self.local_net(local_,mixup_lambda)['clipwise_output']
        local_prob = local_prob.view(local_regions.size(0),local_regions.size(1),-1)
        (local_prob1, _) = torch.max(local_prob, dim=1)
        local_prob2 = torch.mean(local_prob, dim=1)
        local_prob = 0.5*(local_prob1 + local_prob2)
#         local_prob = torch.clamp(local_prob, 1e-7, 1 - 1e-7)  # bs,527
        
        prob = torch.cat([global_prob[:,None,:],local_prob[:,None,:]],1)
        (prob1, _) = torch.max(prob, dim=1)
        prob2 = torch.mean(prob,dim=1)
        prob = 0.5*(prob1 + prob2)
#         prob = torch.clamp(prob, 1e-7, 1 - 1e-7)
        
        output_dict = {'local_prob': local_prob, 'prob': prob}

        return output_dict
    
    def region_select(self, maps, x):
        local_regions = torch.zeros(x.size(0),self.N, self.duration).cuda()#bs,N,T,F
        over_range = int(self.duration*maps.size(-1)/self.duration_all)//4
        for i in range(maps.size(0)):
            for j in range(maps.size(1)):
                map_ = maps[i,j]
                (_,max_index)=torch.max(map_,dim=0)
                index_ = max_index
                max_index = int(self.duration_all*max_index/len(map_))
                index_l = max_index-self.duration//2
                index_r = max_index+self.duration//2
                if index_r >= self.duration_all:
                    local_regions[i,j,:] = x[i,self.duration_all-self.duration:self.duration_all]
#                     print(self.duration_all-self.duration,self.duration_all)
                    maps[i,:,-2*over_range:]=0.
                elif index_l < 0:
                    local_regions[i,j,:] = x[i,:self.duration]
#                     print(0,self.duration)
                    maps[i,:,:2*over_range]=0.
                else:
                    local_regions[i,j,:] = x[i,index_l:index_r]
#                     print(index_l,index_r)
                    maps[i,:,index_-over_range:index_+over_range]=0.
        return local_regions

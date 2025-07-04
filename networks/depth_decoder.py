# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import numpy as np
import torch
import torch.nn as nn

from collections import OrderedDict
from layers import *


class DepthDecoder(nn.Module):
    def __init__(self, num_ch_enc, scales=range(4), num_output_channels=1, use_skips=True, decoder_channel_scale=100):
        super(DepthDecoder, self).__init__()

        # 深度图每个像素代表的是物体到相机xy平面的距离，单位为mm。所以通道数默认为1
        self.num_output_channels = num_output_channels
        self.use_skips = use_skips
        # 这里Decoder上采样的模型默认采用nearest
        self.upsample_mode = 'nearest'
        self.scales = scales

        # encoder各通道数
        self.num_ch_enc = num_ch_enc
        # decoder 通道数
        if decoder_channel_scale == 200:
            self.num_ch_dec = np.array([16, 32, 64, 128, 256])      # 原始的
        elif decoder_channel_scale == 100:
            self.num_ch_dec = np.array([8, 16, 32, 64, 128])      # 缩小两倍
        elif decoder_channel_scale == 50:
            self.num_ch_dec = np.array([4, 8, 16, 32, 64])        # 缩小四倍

        # decoder
        self.convs = OrderedDict()  # 用于存储深度解码器中的不同的卷积块
        for i in range(4, -1, -1):
            # upconv_0
            num_ch_in = self.num_ch_enc[-1] if i == 4 else self.num_ch_dec[i + 1]
            num_ch_out = self.num_ch_dec[i]
            # 将当前阶段的特征图上采样，以获得深度图更高的分辨率
            self.convs[("upconv", i, 0)] = ConvBlock(num_ch_in, num_ch_out)

            # upconv_1
            num_ch_in = self.num_ch_dec[i]
            # 跳跃连接则是将编码器中对应层的特征图与解码器中经过上采样得到的特征图进行连接，以保留更丰富的语境信息
            if self.use_skips and i > 0:
                # 将当前阶段的上采样后的特征图与前一阶段的特征图进行连接。这是为了保留更丰富的上下文信息，因为前一阶段的特征图可能包含了更高级别的语义信息
                num_ch_in += self.num_ch_enc[i - 1]
            num_ch_out = self.num_ch_dec[i]
            self.convs[("upconv", i, 1)] = ConvBlock(num_ch_in, num_ch_out)

        for s in self.scales:
            self.convs[("dispconv", s)] = Conv3x3(self.num_ch_dec[s], self.num_output_channels)

        # 允许按顺序迭代所有包含的模块
        self.decoder = nn.ModuleList(list(self.convs.values()))
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_features):
        self.outputs = {}

        # decoder
        x = input_features[-1]
        for i in range(4, -1, -1):
            x = self.convs[("upconv", i, 0)](x)
            x = [upsample(x)]  # 对上一步的结果进行上采样
            if self.use_skips and i > 0:
                x += [input_features[i - 1]]
            x = torch.cat(x, 1)
            x = self.convs[("upconv", i, 1)](x)
            if i in self.scales:
                self.outputs[("disp", i)] = self.sigmoid(self.convs[("dispconv", i)](x))

        return self.outputs

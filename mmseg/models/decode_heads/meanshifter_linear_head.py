# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
from mmcv.cnn import ConvModule
from mmcv.cnn.utils.weight_init import trunc_normal_init

from ..builder import HEADS
from .decode_head import BaseDecodeHead


@HEADS.register_module()
class MeanshifterLinearHead(BaseDecodeHead):
    """Fully Convolution Networks for Semantic Segmentation.

    This head is implemented of `FCNNet <https://arxiv.org/abs/1411.4038>`_.

    Args:
        num_convs (int): Number of convs in the head. Default: 2.
        kernel_size (int): The kernel size for convs in the head. Default: 3.
        concat_input (bool): Whether concat the input and output of convs
            before classification layer.
        dilation (int): The dilation rate for convs in the head. Default: 1.
    """

    def __init__(self,
                 in_channels,
                 meanshift_channels=192,
                 meanshift_layers=3,
                 init_std=0.02,
                 num_convs=0,
                 kernel_size=1,
                 concat_input=False,
                 dilation=1,
                 **kwargs):
        assert num_convs >= 0 and dilation > 0 and isinstance(dilation, int)
        super(MeanshifterLinearHead, self).__init__(in_channels=in_channels, **kwargs)
        self.meanshift_channels = meanshift_channels
        self.meanshift_layers = meanshift_layers
        self.scale = meanshift_channels ** -0.5
        self.q = nn.Linear(in_channels, self.meanshift_channels)
        #self.q = nn.Conv2d(in_channels, self.meanshift_channels, kernel_size=1)

        self.init_std = init_std

        self.num_convs = num_convs
        self.concat_input = concat_input
        self.kernel_size = kernel_size
        if num_convs == 0:
            assert self.in_channels == self.channels

        conv_padding = (kernel_size // 2) * dilation
        convs = []
        for i in range(num_convs):
            _in_channels = self.in_channels if i == 0 else self.channels
            convs.append(
                ConvModule(
                    _in_channels,
                    self.channels,
                    kernel_size=kernel_size,
                    padding=conv_padding,
                    dilation=dilation,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg))

        if len(convs) == 0:
            self.convs = nn.Identity()
        else:
            self.convs = nn.Sequential(*convs)
        if self.concat_input:
            self.conv_cat = ConvModule(
                self.in_channels + self.channels,
                self.channels,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg)

    def init_weights(self):
        trunc_normal_init(self.q, std=self.init_std)

    def _forward_feature(self, inputs):
        """Forward function for feature maps before classifying each pixel with
        ``self.cls_seg`` fc.

        Args:
            inputs (list[Tensor]): List of multi-level img features.

        Returns:
            feats (Tensor): A tensor of shape (batch_size, self.channels,
                H, W) which is feature map for last layer of decoder head.
        """
        x = self._transform_inputs(inputs)
        feats = self.convs(x)
        if self.concat_input:
            feats = self.conv_cat(torch.cat([x, feats], dim=1))
        return feats

    def forward(self, inputs, meanshift_iterations=None):
    #def forward(self, inputs):
        """Forward function."""
        iter = meanshift_iterations if meanshift_iterations else self.meanshift_layers
        x = self._forward_feature(inputs)
        
        b, c, h, w = x.shape
        x = x.permute(0, 2, 3, 1).contiguous().view(b, h*w, c)

        #output = self.cls_seg(output)
        #return output
        
        #print('*-'*100)
        for i in range(iter):
        #for i in range(self.meanshift_layers):
            q = self.q(x)
            #print(q.transpose(-2, -1).shape)
            #print(f"-----> mphka {i+1} fora")
            #print('*-'*100)
            similarity = (q @ q.transpose(-2, -1)) * self.scale
            similarity = similarity.softmax(dim=-1)
            x = similarity @ x
            #pass
        x = x.permute(0, 2, 1).contiguous().view(b, -1, h, w)
        output = self.cls_seg(x)
        return output

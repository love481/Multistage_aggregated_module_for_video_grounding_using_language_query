import torch
from torch import nn

class FrameAvgPool(nn.Module):

    def __init__(self, cfg=None):
        super(FrameAvgPool, self).__init__()
        kernel_size = 8
        stride = 8
        self.avg_pool = nn.AvgPool1d(kernel_size, stride, int(kernel_size/2))

    def forward(self, visual_input):
        vis_h = self.avg_pool(visual_input)
        return vis_h

class FrameMaxPool(nn.Module):

    def __init__(self, stride):
        super(FrameMaxPool, self).__init__()
        self.max_pool = nn.MaxPool1d(stride)

    def forward(self, visual_input):
        vis_h = self.max_pool(visual_input)
        return vis_h

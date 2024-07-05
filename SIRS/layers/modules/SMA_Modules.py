##############################################
### DATE: 20230911
### AUTHOR: zzc
### TODO: SMA modules

import torch
import torch.nn as nn
import torch.nn.functional as F

class SMA(nn.Module):
    def __init__(self, opt = {}):
        super(SMA, self).__init__()
        print('Initializing SSA module...')
        # self.alpha = 0.5
        self.weighting = opt['SSA']['weighting']
        self.downsample = opt['SSA']['downsample']
        self.norm = opt['SSA']['normalization']
        assert self.weighting in ['step', 'linear', 'power']
        assert self.downsample in ['interpolation', 'maxPooling', 'averagePooling', 'convolution2d', 'None']
        assert self.norm in ['batchNorm', 'groupNorm', 'None']
        if self.downsample == 'convolution2d':
            self.DS1 = nn.Conv1d(in_channels=64, out_channels=self.embed_dim, kernel_size=3, padding=1, stride=2)
            self.DS2 = nn.Conv1d(in_channels=128, out_channels=self.embed_dim, kernel_size=3, padding=1, stride=2)
            self.DS3 = nn.Conv1d(in_channels=256, out_channels=self.embed_dim, kernel_size=3, padding=1, stride=2)
            self.DS4 = nn.Conv1d(in_channels=512, out_channels=self.embed_dim, kernel_size=3, padding=1, stride=2)
            self.act = nn.ELU()
        elif self.downsample == 'maxPooling':
            self.DS1 = nn.AdaptiveMaxPool2d((64, 64))
            self.DS2 = nn.AdaptiveMaxPool2d((32, 32))
            self.DS3 = nn.AdaptiveMaxPool2d((16, 16))
            self.DS4 = nn.AdaptiveMaxPool2d((8, 8))
        elif self.downsample == 'averagePooling':
            self.DS1 = nn.AdaptiveAvgPool2d((64, 64))
            self.DS2 = nn.AdaptiveAvgPool2d((32, 32))
            self.DS3 = nn.AdaptiveAvgPool2d((16, 16))
            self.DS4 = nn.AdaptiveAvgPool2d((8, 8))
        if self.norm == 'batchNorm':
            print('Initializing batch normalizaiton...')
            self.norm1 = nn.BatchNorm2d(64)
            self.norm2 = nn.BatchNorm2d(128)
            self.norm3 = nn.BatchNorm2d(256)
            self.norm4 = nn.BatchNorm2d(512)
        elif self.norm == 'groupNorm':
            print('Initializing group normalizaiton...')
            self.norm1 = nn.GroupNorm(16, 64)
            self.norm2 = nn.GroupNorm(16, 128)
            self.norm3 = nn.GroupNorm(16, 256)
            self.norm4 = nn.GroupNorm(16, 512)

    def forward(self, features, preds):

        ## [[b, 64, 64, 64] [b, 128, 32, 32] [b, 256, 16, 16] [b, 512, 8, 8]]
        ## [b, cls, 256, 256]
        
        f1, f2, f3, f4 = features
        att = torch.sum(preds[:, 1:, :, :], dim=1).unsqueeze(1)
        if self.downsample == 'interpolation':
            c1 = f1 * F.interpolate(att, scale_factor=1/4, mode='bilinear', align_corners=False)
            c2 = f2 * F.interpolate(att, scale_factor=1/8, mode='bilinear', align_corners=False)
            c3 = f3 * F.interpolate(att, scale_factor=1/16, mode='bilinear', align_corners=False)
            c4 = f4 * F.interpolate(att, scale_factor=1/32, mode='bilinear', align_corners=False)
        elif self.downsample == 'maxPooling' or self.downsample == 'averagePooling':
            # print('yes!!!')
            c1 = f1 * self.DS1(att)
            c2 = f2 * self.DS2(att)
            c3 = f3 * self.DS3(att)
            c4 = f4 * self.DS4(att)
        elif self.downsample == 'None':
            # print('yes111')
            c1 = f1
            c2 = f2
            c3 = f3
            c4 = f4
        if self.norm == 'batchNorm' or self.norm == 'groupNorm':
            # print('yes222')
            n1 = self.norm1(c1)
            n2 = self.norm2(c2)
            n3 = self.norm3(c3)
            n4 = self.norm4(c4)
            return [n1, n2, n3, n4]
        elif self.norm == 'None':
            return [c1, c2, c3, c4]
########################################################
### TODO: for testing decoder
import yaml
import numpy as np
import cv2
from thop import profile
    
if __name__ == '__main__':
    # load model options
    with open('../../option/RSITMD_IRSeg.yaml', 'r') as handle:
        options = yaml.load(handle, Loader=yaml.FullLoader)
    # options['model']['SSA']['downsample'] = 'interpolation'
    options['model']['SSA']['downsample'] = 'averagePooling'
    # options['model']['SSA']['downsample'] = 'maxPooling'
    options['model']['SSA']['normalization'] = 'batchNorm'
    # options['model']['SSA']['normalization'] = 'groupNorm'
    print(options['model']['SSA'])
    # [b, 64, 64, 64] [b, 128, 32, 32] [b, 256, 16, 16] [b, 512, 8, 8]
    feats = [torch.randn([1, 64, 64, 64]), torch.randn([1, 128, 32, 32]), \
        torch.randn([1, 256, 16, 16]), torch.randn([1, 512, 8, 8])]
    preds = torch.randn([1, 15, 256, 256])
    preds[:, :, 30:60, 30:80] = 50*preds[:, :, 30:60, 30:80]
    decoder = SMA(options['model'])
    flops, params = profile(decoder, (feats, preds))
    print('flops: ', flops, 'params: ', params)
    print('flops: %.2f M, params: %.2f M' % (flops / 1000000.0, params / 1000000.0))    
    # output = decoder(feats, preds)
    # for i, feat in enumerate(output):
    #     print(feat.size())
    #     array = feat.detach().numpy()
    #     array = np.mean(array, axis=1).transpose(1, 2, 0)
    #     print('array: ', array.shape)
    #     # cv2.imwrite('./{}.png'.format(i), array)
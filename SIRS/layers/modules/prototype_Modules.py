##############################################
### DATE: 20230911
### AUTHOR: zzc
### TODO: prototype modules

import sys
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.nn.init
import torchvision.models as models
from torch.autograd import Variable
from torch.nn.utils.clip_grad import clip_grad_norm
import numpy as np
from collections import OrderedDict
from torchvision.models.resnet import resnet18
import torch.nn.functional as F
from layers import seq2vec
import math
import copy

def l2norm(X, dim, eps=1e-8):
    """L2-normalize columns of X
    """
    norm = torch.pow(X, 2).sum(dim=dim, keepdim=True).sqrt() + eps
    X = torch.div(X, norm)
    return X

class ExtractFeature(nn.Module):
    def __init__(self, opt = {}, finetune=True, backbone='resnet'):
        super(ExtractFeature, self).__init__()

        if backbone == 'resnet':
            print('Apply backbone resnet-18...')
            self.resnet = resnet18(pretrained=True)
            for param in self.resnet.parameters():
                param.requires_grad = finetune
        elif backbone == 'swin':
            print('Apply backbone swin_xxt_v2...')
            from layers.modules.swin_transformer import build_swin_xxt_v2
            self.swin = build_swin_xxt_v2()
            for param in self.swin.parameters():
                param.requires_grad = finetune
        else:
            print('No backbone fitted, please check!')
            sys.exit()
        self.backbone = backbone
    def forward(self, img):
        if self.backbone == 'resnet':
            x = self.resnet.conv1(img)
            x = self.resnet.bn1(x)
            x = self.resnet.relu(x)
            x = self.resnet.maxpool(x)

            f1 = self.resnet.layer1(x)
            f2 = self.resnet.layer2(f1)
            f3 = self.resnet.layer3(f2)
            f4 = self.resnet.layer4(f3)

            # torch.Size([10, 192, 64, 64])
            # torch.Size([10, 768, 64, 64])
            # torch.Size([10, 512])
        elif self.backbone == 'swin':
            x = self.swin.patch_embed(img)
            if self.swin.ape:
                x = x + self.swin.absolute_pos_embed
            x = self.swin.pos_drop(x)
                
            f1 = self.swin.layers[0](x)
            f2 = self.swin.layers[1](f1)
            f3 = self.swin.layers[2](f2)
            f4 = self.swin.layers[3](f3)
            
            f1 = f1.permute(0, 2, 1)
            f2 = f2.permute(0, 2, 1)
            f3 = f3.permute(0, 2, 1)
            f4 = f4.permute(0, 2, 1)

            # x = self.swin.norm(x)  # B L C
        return [f1, f2, f3, f4]

class Visual_base(nn.Module):
    def __init__(self, opt = {}):
        super(Visual_base, self).__init__()

        self.embed_dim = opt['embed']['embed_dim']

        self.linear = nn.Linear(in_features=512, out_features=self.embed_dim)

    def forward(self, features):

        f4 = features[3]

        # batch * 512
        feature = f4.view(f4.shape[0], 512, -1)
        solo_feature = self.linear(torch.mean(feature,dim=-1))

        # torch.Size([10, 192, 64, 64])
        # torch.Size([10, 768, 64, 64])
        # torch.Size([10, 512])
        return solo_feature

class Visual_base_multi(nn.Module):
    def __init__(self, opt = {}, sample = 'max', mapping = 'interpolation'):
        super(Visual_base_multi, self).__init__()

        self.embed_dim = opt['embed']['embed_dim']
        assert sample in ['mean', 'max', 'median']
        self.sample = sample
        assert mapping in ['linear', 'conv1d', 'interpolation']
        self.mapping = mapping
        if mapping == 'linear':
            self.mapping1 = nn.Linear(in_features=64, out_features=self.embed_dim)
            self.mapping2 = nn.Linear(in_features=128, out_features=self.embed_dim)
            self.mapping3 = nn.Linear(in_features=256, out_features=self.embed_dim)
            self.mapping4 = nn.Linear(in_features=512, out_features=self.embed_dim)
        elif mapping == 'conv1d':
            self.mapping1 = nn.Conv1d(in_channels=64, out_channels=self.embed_dim, kernel_size=1)
            self.mapping2 = nn.Conv1d(in_channels=128, out_channels=self.embed_dim, kernel_size=1)
            self.mapping3 = nn.Conv1d(in_channels=256, out_channels=self.embed_dim, kernel_size=1)
            self.mapping4 = nn.Conv1d(in_channels=512, out_channels=self.embed_dim, kernel_size=1)
        elif mapping == 'interpolation':
            self.mapping1 = nn.Upsample(size=(self.embed_dim), mode='nearest')
            self.mapping2 = nn.Upsample(size=(self.embed_dim), mode='nearest')
            self.mapping3 = nn.Upsample(size=(self.embed_dim), mode='nearest')
            self.mapping4 = nn.Upsample(size=(self.embed_dim), mode='nearest')

    def forward(self, features):
        # [b, 64, 64, 64] [b, 128, 32, 32] [b, 256, 16, 16] [b, 512, 8, 8]
        c2, c3, c4, c5 = features

        # [batch * 512]
        if self.mapping == 'linear':
            if self.sample == 'mean':
                c2 = self.mapping1(torch.mean(c2.view(c2.shape[0], 64, -1),dim=-1))
                c3 = self.mapping2(torch.mean(c3.view(c3.shape[0], 128, -1),dim=-1))
                c4 = self.mapping3(torch.mean(c4.view(c4.shape[0], 256, -1),dim=-1))
                c5 = self.mapping4(torch.mean(c5.view(c5.shape[0], 512, -1),dim=-1))
            elif self.sample == 'max':
                c2 = self.mapping1(torch.max(c2.view(c2.shape[0], 64, -1),dim=-1)[0])
                c3 = self.mapping2(torch.max(c3.view(c3.shape[0], 128, -1),dim=-1)[0])
                c4 = self.mapping3(torch.max(c4.view(c4.shape[0], 256, -1),dim=-1)[0])
                c5 = self.mapping4(torch.max(c5.view(c5.shape[0], 512, -1),dim=-1)[0])
            elif self.sample == 'median':
                c2 = self.mapping1(torch.median(c2.view(c2.shape[0], 64, -1),dim=-1)[0])
                c3 = self.mapping2(torch.median(c3.view(c3.shape[0], 128, -1),dim=-1)[0])
                c4 = self.mapping3(torch.median(c4.view(c4.shape[0], 256, -1),dim=-1)[0])
                c5 = self.mapping4(torch.median(c5.view(c5.shape[0], 512, -1),dim=-1)[0])
        elif self.mapping == 'conv1d':
            if self.sample == 'mean':
                c2 = torch.mean(self.mapping1(c2.view(c2.shape[0], 64, -1)), dim=-1)
                c3 = torch.mean(self.mapping2(c3.view(c3.shape[0], 128, -1)), dim=-1)
                c4 = torch.mean(self.mapping3(c4.view(c4.shape[0], 256, -1)), dim=-1)
                c5 = torch.mean(self.mapping4(c5.view(c5.shape[0], 512, -1)), dim=-1)
            elif self.sample == 'max':
                c2 = torch.max(self.mapping1(c2.view(c2.shape[0], 64, -1)), dim=-1)[0]
                c3 = torch.max(self.mapping2(c3.view(c3.shape[0], 128, -1)), dim=-1)[0]
                c4 = torch.max(self.mapping3(c4.view(c4.shape[0], 256, -1)), dim=-1)[0]
                c5 = torch.max(self.mapping4(c5.view(c5.shape[0], 512, -1)), dim=-1)[0]
            elif self.sample == 'median':
                c2 = torch.median(self.mapping1(c2.view(c2.shape[0], 64, -1)), dim=-1)[0]
                c3 = torch.median(self.mapping2(c3.view(c3.shape[0], 128, -1)), dim=-1)[0]
                c4 = torch.median(self.mapping3(c4.view(c4.shape[0], 256, -1)), dim=-1)[0]
                c5 = torch.median(self.mapping4(c5.view(c5.shape[0], 512, -1)), dim=-1)[0]
        elif self.mapping == 'interpolation':
            if self.sample == 'mean':
                c2 = torch.mean(self.mapping1(c2.view(c2.shape[0], 64, -1).transpose(1, 2)), dim=1)
                c3 = torch.mean(self.mapping2(c3.view(c3.shape[0], 128, -1).transpose(1, 2)), dim=1)
                c4 = torch.mean(self.mapping3(c4.view(c4.shape[0], 256, -1).transpose(1, 2)), dim=1)
                c5 = torch.mean(self.mapping4(c5.view(c5.shape[0], 512, -1).transpose(1, 2)), dim=1)
            elif self.sample == 'max':
                c2 = torch.max(self.mapping1(c2.view(c2.shape[0], 64, -1).transpose(1, 2)), dim=1)[0]
                c3 = torch.max(self.mapping2(c3.view(c3.shape[0], 128, -1).transpose(1, 2)), dim=1)[0]
                c4 = torch.max(self.mapping3(c4.view(c4.shape[0], 256, -1).transpose(1, 2)), dim=1)[0]
                c5 = torch.max(self.mapping4(c5.view(c5.shape[0], 512, -1).transpose(1, 2)), dim=1)[0]
            elif self.sample == 'median':
                c2 = torch.median(self.mapping1(c2.view(c2.shape[0], 64, -1).transpose(1, 2)), dim=1)[0]
                c3 = torch.median(self.mapping2(c3.view(c3.shape[0], 128, -1).transpose(1, 2)), dim=1)[0]
                c4 = torch.median(self.mapping3(c4.view(c4.shape[0], 256, -1).transpose(1, 2)), dim=1)[0]
                c5 = torch.median(self.mapping4(c5.view(c5.shape[0], 512, -1).transpose(1, 2)), dim=1)[0]
        # [b, 512] [b, 512] [b, 512] [b, 512]
        return [c2, c3, c4, c5]

class FPN(nn.Module):
    def __init__(self, opt = {}):
        super(FPN, self).__init__()
        self.embed_dim = opt['embed']['embed_dim']

        # Top layer
        self.toplayer = nn.Conv2d(512, self.embed_dim, kernel_size=1, stride=1, padding=0)  # Reduce channels

        # Lateral layers
        self.latlayer1 = nn.Conv2d(256, self.embed_dim, kernel_size=1, stride=1, padding=0)
        self.latlayer2 = nn.Conv2d(128, self.embed_dim, kernel_size=1, stride=1, padding=0)
        self.latlayer3 = nn.Conv2d(64, self.embed_dim, kernel_size=1, stride=1, padding=0)
        
        # Smooth Layers
        self.smooth1 = nn.Linear(in_features=64, out_features=self.embed_dim)
        self.smooth2 = nn.Linear(in_features=128, out_features=self.embed_dim)
        self.smooth3 = nn.Linear(in_features=256, out_features=self.embed_dim)
        self.smooth4 = nn.Linear(in_features=512, out_features=self.embed_dim)

    def _upsample_add(self, x, y):
        _,_,H,W = y.size()
        return F.upsample(x, size=(H,W), mode='bilinear') + y

    def forward(self, x):
        # [b, 64, 64, 64] [b, 128, 32, 32] [b, 256, 16, 16] [b, 512, 8, 8]
        # Bottom-up
        [c2, c3, c4, c5] = x
        
        # Top-down
        p5 = self.toplayer(c5)
        p4 = self._upsample_add(p5, self.latlayer1(c4))
        p3 = self._upsample_add(p4, self.latlayer2(c3))
        p2 = self._upsample_add(p3, self.latlayer3(c2))
        # [b, 512, 64, 64] [b, 512, 32, 32] [b, 512, 16, 16] [b, 512, 8, 8]
        
        # Mapping
        p5 = self.smooth4(torch.mean(p5.view(p5.shape[0], 512, -1), dim=-1))
        p4 = self.smooth3(torch.mean(p4.view(p4.shape[0], 256, -1), dim=-1))
        p3 = self.smooth2(torch.mean(p3.view(p3.shape[0], 128, -1), dim=-1))
        p2 = self.smooth1(torch.mean(p2.view(p2.shape[0], 64, -1), dim=-1))
        
        return [p2, p3, p4, p5]

class Skipthoughts_Embedding_Module(nn.Module):
    def __init__(self, vocab, opt, out_dropout=-1):
        super(Skipthoughts_Embedding_Module, self).__init__()
        self.opt = opt
        self.vocab_words = vocab

        self.seq2vec = seq2vec.factory(self.vocab_words, self.opt['seq2vec'], self.opt['seq2vec']['dropout'])

        self.to_out = nn.Linear(in_features=2400, out_features=self.opt['embed']['embed_dim'])
        self.dropout = out_dropout

    def forward(self, input_text ):
        x_t_vec = self.seq2vec(input_text)
        out = F.relu(self.to_out(x_t_vec))
        if self.dropout >= 0:
            out = F.dropout(out, self.dropout)

        return out 

def cosine_similarity(x1, x2, dim=-1, eps=1e-8):
    """Returns cosine similarity between x1 and x2, computed along dim."""
    w12 = torch.sum(x1 * x2, dim)
    w1 = torch.norm(x1, 2, dim)
    w2 = torch.norm(x2, 2, dim)
    return (w12 / (w1 * w2).clamp(min=eps)).squeeze()

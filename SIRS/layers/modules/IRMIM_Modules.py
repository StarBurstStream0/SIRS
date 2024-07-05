##############################################
### DATE: 20230911
### AUTHOR: zzc
### TODO: IRMIM modules

import torch
import torch.nn as nn
import torch.nn.init
from torchvision.models.resnet import resnet18
import torch.nn.functional as F
from layers import seq2vec
import math

########################################################
### TODO: for MIM decoder

from typing import List
from timm.models.layers import trunc_normal_

def is_pow2n(x):
    return x > 0 and (x & (x - 1) == 0)

class UNetBlock(nn.Module):
    def __init__(self, cin, cout, bn2d):
        """
        a UNet block with 2x up sampling
        """
        super().__init__()
        self.up_sample = nn.ConvTranspose2d(cin, cin, kernel_size=4, stride=2, padding=1, bias=True)
        self.conv = nn.Sequential(
            nn.Conv2d(cin, cin, kernel_size=3, stride=1, padding=1, bias=False), bn2d(cin), nn.ReLU6(inplace=True),
            nn.Conv2d(cin, cout, kernel_size=3, stride=1, padding=1, bias=False), bn2d(cout),
        )
    
    def forward(self, x):
        x = self.up_sample(x)
        return self.conv(x)

class LightDecoder(nn.Module):
    def __init__(self, up_sample_ratio, width=768, sbn=True):   # todo: the decoder's width follows a simple halfing rule; you can change it to any other rule
        super().__init__()
        self.width = width
        assert is_pow2n(up_sample_ratio)
        n = round(math.log2(up_sample_ratio))
        channels = [self.width // 2 ** i for i in range(n + 1)] # todo: the decoder's width follows a simple halfing rule; you can change it to any other rule
        bn2d = nn.SyncBatchNorm if sbn else nn.BatchNorm2d
        self.dec = nn.ModuleList([UNetBlock(cin, cout, bn2d) for (cin, cout) in zip(channels[:-1], channels[1:])])
        self.proj = nn.Conv2d(channels[-1], 3, kernel_size=1, stride=1, bias=True)
        
        self.initialize()
    
    def forward(self, to_dec: List[torch.Tensor]):
        x = 0
        for i, d in enumerate(self.dec):
            if i < len(to_dec) and to_dec[i] is not None:
                x = x + to_dec[i]
            x = self.dec[i](x)
        return self.proj(x)
    
    def extra_repr(self) -> str:
        return f'width={self.width}'
    
    def initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv2d):
                trunc_normal_(m.weight, std=.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.)
            elif isinstance(m, (nn.LayerNorm, nn.BatchNorm1d, nn.BatchNorm2d, nn.SyncBatchNorm)):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

########################################################

def l2norm(X, dim, eps=1e-8):
    """L2-normalize columns of X
    """
    norm = torch.pow(X, 2).sum(dim=dim, keepdim=True).sqrt() + eps
    X = torch.div(X, norm)
    return X

class ExtractFeature(nn.Module):
    def __init__(self, opt = {}, finetune=True):
        super(ExtractFeature, self).__init__()

        print('Apply backbone resnet-18...')
        self.resnet = resnet18(pretrained=True)
        # from timm.models import create_model
        # self.resnet = create_model('resnet18', pretrained=True)
        for param in self.resnet.parameters():
            param.requires_grad = finetune

    def forward(self, img):
        x = self.resnet.conv1(img)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        f1 = self.resnet.layer1(x)
        f2 = self.resnet.layer2(f1)
        f3 = self.resnet.layer3(f2)
        f4 = self.resnet.layer4(f3)

        # [b, 64, 64, 64] [b, 128, 32, 32] [b, 256, 16, 16] [b, 512, 8, 8]

        return [f1, f2, f3, f4]
    
    def get_downsample_ratio(self):
        return 32
    
    def get_feature_map_channels(self):
        # `self.feature_info` is maintained by `timm`
        # return [info['num_chs'] for info in self.resnet.feature_info[1:]]
        return [64, 128, 256, 512]

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
    def __init__(self, opt = {}):
        super(Visual_base_multi, self).__init__()

        self.embed_dim = opt['embed']['embed_dim']
        self.sample = opt['VFM']['sample']
        self.mapping = opt['VFM']['mapping']
        assert self.sample in ['mean', 'max', 'median']
        assert self.mapping in ['linear', 'conv1d', 'interpolation']
        if self.mapping == 'linear':
            self.mapping1 = nn.Linear(in_features=64, out_features=self.embed_dim)
            self.mapping2 = nn.Linear(in_features=128, out_features=self.embed_dim)
            self.mapping3 = nn.Linear(in_features=256, out_features=self.embed_dim)
            self.mapping4 = nn.Linear(in_features=512, out_features=self.embed_dim)
        elif self.mapping == 'conv1d':
            self.mapping1 = nn.Conv1d(in_channels=64, out_channels=self.embed_dim, kernel_size=1)
            self.mapping2 = nn.Conv1d(in_channels=128, out_channels=self.embed_dim, kernel_size=1)
            self.mapping3 = nn.Conv1d(in_channels=256, out_channels=self.embed_dim, kernel_size=1)
            self.mapping4 = nn.Conv1d(in_channels=512, out_channels=self.embed_dim, kernel_size=1)
        elif self.mapping == 'interpolation':
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

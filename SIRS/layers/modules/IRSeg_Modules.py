##############################################
### DATE: 20230911
### AUTHOR: zzc
### TODO: IRSeg modules

import torch
import torch.nn as nn
import torch.nn.init
from torchvision.models.resnet import resnet18, resnet50
import torch.nn.functional as F
from layers import seq2vec

########################################################
### TODO: for Seg decoder
import functools

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class seg_fpn_decoder(nn.Module):
    
    def __init__(self, opt, num_blocks=[1,1,1,1]):
        super(seg_fpn_decoder, self).__init__()
        self.in_planes = 64
        self.num_classes = opt['decoder']['num_classes']
        self.channels = opt['channels']

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        BatchNorm = nn.BatchNorm2d

        # Bottom-up layers
        self.layer1 = self._make_layer(Bottleneck, self.channels[0], num_blocks[0], stride=1)
        self.layer2 = self._make_layer(Bottleneck, self.channels[1], num_blocks[1], stride=2)
        self.layer3 = self._make_layer(Bottleneck, self.channels[2], num_blocks[2], stride=2)
        self.layer4 = self._make_layer(Bottleneck, self.channels[3], num_blocks[3], stride=2)

        # Top layer
        self.toplayer = nn.Conv2d(self.channels[-1], 256, kernel_size=1, stride=1, padding=0)  # Reduce channels

        # Smooth layers
        self.smooth1 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.smooth2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.smooth3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)

        # Lateral layers
        self.latlayer1 = nn.Conv2d(self.channels[2], 256, kernel_size=1, stride=1, padding=0)
        self.latlayer2 = nn.Conv2d(self.channels[1], 256, kernel_size=1, stride=1, padding=0)
        self.latlayer3 = nn.Conv2d(self.channels[0], 256, kernel_size=1, stride=1, padding=0)

		# Semantic branch
        self.semantic_branch = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(128, self.num_classes, kernel_size=1, stride=1, padding=0)
        # num_groups, num_channels
        self.gn1 = nn.GroupNorm(128, 128) 
        self.gn2 = nn.GroupNorm(256, 256)
        
    def _make_layer(self, Bottleneck, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(Bottleneck(self.in_planes, planes, stride))
            self.in_planes = planes * Bottleneck.expansion
        return nn.Sequential(*layers)
    
    def _upsample_add(self, x, y):
        '''Upsample and add two feature maps.
        Args:
            x: (Variable) top feature map to be upsampled.
            y: (Variable) lateral feature map.
        Returns:
            (Variable) added feature map.
        Note in PyTorch, when input size is odd, the upsampled feature map
        with `F.upsample(..., scale_factor=2, mode='nearest')`
        maybe not equal to the lateral feature map size.
        e.g.
        original input size: [N,_,15,15] ->
        conv2d feature map size: [N,_,8,8] ->
        upsampled feature map size: [N,_,16,16]
        So we choose bilinear upsample which supports arbitrary output sizes.
        '''
        _,_,H,W = y.size()
        return F.interpolate(x, size=(H,W), mode='bilinear', align_corners=True) + y

    def _upsample(self, x, h, w):
        return F.interpolate(x, size=(h, w), mode='bilinear', align_corners=True)

    def forward(self, low_level_features):
        # Bottom-up using backbone
        # c1 = low_level_features[0]
        # c2 = low_level_features[1]
        # c3 = low_level_features[2]
        # c4 = low_level_features[3]
        # c5 = low_level_features[4]
        
        # [b, 64, 64, 64] [b, 128, 32, 32] [b, 256, 16, 16] [b, 512, 8, 8]
        c2 = low_level_features[0]
        c3 = low_level_features[1]
        c4 = low_level_features[2]
        c5 = low_level_features[3]
        
        # Bottom-up
        #c1 = F.relu(self.bn1(self.conv1(x)))
        #c1 = F.max_pool2d(c1, kernel_size=3, stride=2, padding=1)
        #c2 = self.layer1(c1)
        #c3 = self.layer2(c2)
        #c4 = self.layer3(c3)
        #c5 = self.layer4(c4)


        # Top-down
        p5 = self.toplayer(c5) # [b, 256, 8, 8]
        p4 = self._upsample_add(p5, self.latlayer1(c4)) # [b, 256, 16, 16]
        p3 = self._upsample_add(p4, self.latlayer2(c3)) # [b, 256, 32, 32]
        p2 = self._upsample_add(p3, self.latlayer3(c2)) # [b, 256, 64, 64]


        # Smooth
        p4 = self.smooth1(p4) # [b, 256, 16, 16]
        p3 = self.smooth2(p3) # [b, 256, 32, 32]
        p2 = self.smooth3(p2) # [b, 256, 64, 64]


        # Semantic
        _, _, h, w = p2.size() # h: 64, w: 64
        # 256->256
        s5 = self._upsample(F.relu(self.gn2(self.conv2(p5))), h, w) # [b, 256, 64, 64]
        # 256->256
        s5 = self._upsample(F.relu(self.gn2(self.conv2(s5))), h, w) # [b, 256, 64, 64]
        # 256->128
        s5 = self._upsample(F.relu(self.gn1(self.semantic_branch(s5))), h, w) # [b, 128, 64, 64]

        # 256->256
        s4 = self._upsample(F.relu(self.gn2(self.conv2(p4))), h, w) # [b, 256, 64, 64]
        # 256->128
        s4 = self._upsample(F.relu(self.gn1(self.semantic_branch(s4))), h, w) # [b, 128, 64, 64]

        # 256->128
        s3 = self._upsample(F.relu(self.gn1(self.semantic_branch(p3))), h, w) # [b, 128, 64, 64]

        s2 = F.relu(self.gn1(self.semantic_branch(p2))) # [b, 128, 64, 64]
        return self._upsample(self.conv3(s2 + s3 + s4 + s5), 4 * h, 4 * w)

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

        if 'backbone' in opt.keys() and opt['backbone'] == 'resnet18':
            print('Apply backbone resnet-18...')
            self.resnet = resnet18(pretrained=True)
        elif 'backbone' in opt.keys() and opt['backbone'] == 'resnet50':
            print('Apply backbone resnet-50...')
            self.resnet = resnet50(pretrained=True)
        else:
            print('Apply backbone resnet-18...')
            self.resnet = resnet18(pretrained=True)
        # self.resnet = resnet18()
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
        self.channels = opt['channels']
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
        self.channels = opt['channels']
        self.embed_dim = opt['embed']['embed_dim']
        self.sample = opt['VFM']['sample']
        self.mapping = opt['VFM']['mapping']
        assert self.sample in ['mean', 'max', 'median']
        assert self.mapping in ['linear', 'conv1d', 'interpolation']
        if self.mapping == 'linear':
            self.mapping1 = nn.Linear(in_features=self.channels[0], out_features=self.embed_dim)
            self.mapping2 = nn.Linear(in_features=self.channels[1], out_features=self.embed_dim)
            self.mapping3 = nn.Linear(in_features=self.channels[2], out_features=self.embed_dim)
            self.mapping4 = nn.Linear(in_features=self.channels[3], out_features=self.embed_dim)
        elif self.mapping == 'conv1d':
            # self.mapping1 = nn.Conv1d(in_channels=64, out_channels=self.embed_dim, kernel_size=1)
            # self.mapping2 = nn.Conv1d(in_channels=128, out_channels=self.embed_dim, kernel_size=1)
            # self.mapping3 = nn.Conv1d(in_channels=256, out_channels=self.embed_dim, kernel_size=1)
            # self.mapping4 = nn.Conv1d(in_channels=512, out_channels=self.embed_dim, kernel_size=1)
            self.mapping1 = nn.Conv1d(in_channels=self.channels[0], out_channels=self.embed_dim, kernel_size=3, padding=1)
            self.mapping2 = nn.Conv1d(in_channels=self.channels[1], out_channels=self.embed_dim, kernel_size=3, padding=1)
            self.mapping3 = nn.Conv1d(in_channels=self.channels[2], out_channels=self.embed_dim, kernel_size=3, padding=1)
            self.mapping4 = nn.Conv1d(in_channels=self.channels[3], out_channels=self.embed_dim, kernel_size=3, padding=1)
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
                c2 = self.mapping1(torch.mean(c2.view(c2.shape[0], self.channels[0], -1),dim=-1))
                c3 = self.mapping2(torch.mean(c3.view(c3.shape[0], self.channels[1], -1),dim=-1))
                c4 = self.mapping3(torch.mean(c4.view(c4.shape[0], self.channels[2], -1),dim=-1))
                c5 = self.mapping4(torch.mean(c5.view(c5.shape[0], self.channels[3], -1),dim=-1))
            elif self.sample == 'max':
                c2 = self.mapping1(torch.max(c2.view(c2.shape[0], self.channels[0], -1),dim=-1)[0])
                c3 = self.mapping2(torch.max(c3.view(c3.shape[0], self.channels[1], -1),dim=-1)[0])
                c4 = self.mapping3(torch.max(c4.view(c4.shape[0], self.channels[2], -1),dim=-1)[0])
                c5 = self.mapping4(torch.max(c5.view(c5.shape[0], self.channels[3], -1),dim=-1)[0])
            elif self.sample == 'median':
                c2 = self.mapping1(torch.median(c2.view(c2.shape[0], self.channels[0], -1),dim=-1)[0])
                c3 = self.mapping2(torch.median(c3.view(c3.shape[0], self.channels[1], -1),dim=-1)[0])
                c4 = self.mapping3(torch.median(c4.view(c4.shape[0], self.channels[2], -1),dim=-1)[0])
                c5 = self.mapping4(torch.median(c5.view(c5.shape[0], self.channels[3], -1),dim=-1)[0])
        elif self.mapping == 'conv1d':
            if self.sample == 'mean':
                c2 = torch.mean(self.mapping1(c2.view(c2.shape[0], self.channels[0], -1)), dim=-1)
                c3 = torch.mean(self.mapping2(c3.view(c3.shape[0], self.channels[1], -1)), dim=-1)
                c4 = torch.mean(self.mapping3(c4.view(c4.shape[0], self.channels[2], -1)), dim=-1)
                c5 = torch.mean(self.mapping4(c5.view(c5.shape[0], self.channels[3], -1)), dim=-1)
            elif self.sample == 'max':
                c2 = torch.max(self.mapping1(c2.view(c2.shape[0], self.channels[0], -1)), dim=-1)[0]
                c3 = torch.max(self.mapping2(c3.view(c3.shape[0], self.channels[1], -1)), dim=-1)[0]
                c4 = torch.max(self.mapping3(c4.view(c4.shape[0], self.channels[2], -1)), dim=-1)[0]
                c5 = torch.max(self.mapping4(c5.view(c5.shape[0], self.channels[3], -1)), dim=-1)[0]
            elif self.sample == 'median':
                c2 = torch.median(self.mapping1(c2.view(c2.shape[0], self.channels[0], -1)), dim=-1)[0]
                c3 = torch.median(self.mapping2(c3.view(c3.shape[0], self.channels[1], -1)), dim=-1)[0]
                c4 = torch.median(self.mapping3(c4.view(c4.shape[0], self.channels[2], -1)), dim=-1)[0]
                c5 = torch.median(self.mapping4(c5.view(c5.shape[0], self.channels[3], -1)), dim=-1)[0]
        elif self.mapping == 'interpolation':
            if self.sample == 'mean':
                c2 = torch.mean(self.mapping1(c2.view(c2.shape[0], self.channels[0], -1).transpose(1, 2)), dim=1)
                c3 = torch.mean(self.mapping2(c3.view(c3.shape[0], self.channels[1], -1).transpose(1, 2)), dim=1)
                c4 = torch.mean(self.mapping3(c4.view(c4.shape[0], self.channels[2], -1).transpose(1, 2)), dim=1)
                c5 = torch.mean(self.mapping4(c5.view(c5.shape[0], self.channels[3], -1).transpose(1, 2)), dim=1)
            elif self.sample == 'max':
                c2 = torch.max(self.mapping1(c2.view(c2.shape[0], self.channels[0], -1).transpose(1, 2)), dim=1)[0]
                c3 = torch.max(self.mapping2(c3.view(c3.shape[0], self.channels[1], -1).transpose(1, 2)), dim=1)[0]
                c4 = torch.max(self.mapping3(c4.view(c4.shape[0], self.channels[2], -1).transpose(1, 2)), dim=1)[0]
                c5 = torch.max(self.mapping4(c5.view(c5.shape[0], self.channels[3], -1).transpose(1, 2)), dim=1)[0]
            elif self.sample == 'median':
                c2 = torch.median(self.mapping1(c2.view(c2.shape[0], self.channels[0], -1).transpose(1, 2)), dim=1)[0]
                c3 = torch.median(self.mapping2(c3.view(c3.shape[0], self.channels[1], -1).transpose(1, 2)), dim=1)[0]
                c4 = torch.median(self.mapping3(c4.view(c4.shape[0], self.channels[2], -1).transpose(1, 2)), dim=1)[0]
                c5 = torch.median(self.mapping4(c5.view(c5.shape[0], self.channels[3], -1).transpose(1, 2)), dim=1)[0]
        # [b, 512] [b, 512] [b, 512] [b, 512]
        return [c2, c3, c4, c5]

class FPN(nn.Module):
    def __init__(self, opt = {}):
        super(FPN, self).__init__()
        self.embed_dim = opt['embed']['embed_dim']
        self.channels = opt['channels']

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


##################################################################################
### TODO: check ssa time cost


class seg_fpn_decoder_check(nn.Module):
    
    def __init__(self, opt, num_blocks=[1,1,1,1]):
        super(seg_fpn_decoder_check, self).__init__()
        self.in_planes = 64
        self.num_classes = opt['decoder']['num_classes']
        self.channels = opt['channels']

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1, bias=False)
        # self.bn1 = nn.BatchNorm2d(64)

        BatchNorm = nn.BatchNorm2d

        # Bottom-up layers
        self.layer1 = self._make_layer(Bottleneck, self.channels[0], num_blocks[0], stride=1)
        self.layer2 = self._make_layer(Bottleneck, self.channels[1], num_blocks[1], stride=2)
        self.layer3 = self._make_layer(Bottleneck, self.channels[2], num_blocks[2], stride=2)
        self.layer4 = self._make_layer(Bottleneck, self.channels[3], num_blocks[3], stride=2)

        # Top layer
        self.toplayer = nn.Conv2d(self.channels[-1], 256, kernel_size=1, stride=1, padding=0)  # Reduce channels

        # Smooth layers
        self.smooth1 = nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0)
        self.smooth2 = nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0)
        self.smooth3 = nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0)

        # Lateral layers
        self.latlayer1 = nn.Conv2d(self.channels[2], 256, kernel_size=1, stride=1, padding=0)
        self.latlayer2 = nn.Conv2d(self.channels[1], 256, kernel_size=1, stride=1, padding=0)
        self.latlayer3 = nn.Conv2d(self.channels[0], 256, kernel_size=1, stride=1, padding=0)

		# Semantic branch
        self.semantic_branch = nn.Conv2d(256, 128, kernel_size=1, stride=1, padding=0)
        self.conv2 = nn.Conv2d(256, 256, kernel_size=2, stride=1, padding=0)
        self.conv3 = nn.Conv2d(128, self.num_classes, kernel_size=1, stride=1, padding=0)
        # num_groups, num_channels
        self.gn1 = nn.GroupNorm(128, 128) 
        self.gn2 = nn.GroupNorm(256, 256)
        
    def _make_layer(self, Bottleneck, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(Bottleneck(self.in_planes, planes, stride))
            self.in_planes = planes * Bottleneck.expansion
        return nn.Sequential(*layers)
    
    def _upsample_add(self, x, y):
        '''Upsample and add two feature maps.
        Args:
            x: (Variable) top feature map to be upsampled.
            y: (Variable) lateral feature map.
        Returns:
            (Variable) added feature map.
        Note in PyTorch, when input size is odd, the upsampled feature map
        with `F.upsample(..., scale_factor=2, mode='nearest')`
        maybe not equal to the lateral feature map size.
        e.g.
        original input size: [N,_,15,15] ->
        conv2d feature map size: [N,_,8,8] ->
        upsampled feature map size: [N,_,16,16]
        So we choose bilinear upsample which supports arbitrary output sizes.
        '''
        _,_,H,W = y.size()
        return F.interpolate(x, size=(H,W), mode='bilinear', align_corners=True) + y

    def _upsample(self, x, h, w):
        return F.interpolate(x, size=(h, w), mode='bilinear', align_corners=True)

    def forward(self, low_level_features):
        # print('hahaha')
        # Bottom-up using backbone
        # c1 = low_level_features[0]
        # c2 = low_level_features[1]
        # c3 = low_level_features[2]
        # c4 = low_level_features[3]
        # c5 = low_level_features[4]
        
        # [b, 64, 64, 64] [b, 128, 32, 32] [b, 256, 16, 16] [b, 512, 8, 8]
        c2 = low_level_features[0]
        c3 = low_level_features[1]
        c4 = low_level_features[2]
        c5 = low_level_features[3]
        
        # Bottom-up
        #c1 = F.relu(self.bn1(self.conv1(x)))
        #c1 = F.max_pool2d(c1, kernel_size=3, stride=2, padding=1)
        #c2 = self.layer1(c1)
        #c3 = self.layer2(c2)
        #c4 = self.layer3(c3)
        #c5 = self.layer4(c4)


        # Top-down
        p5 = self.toplayer(c5) # [b, 256, 8, 8]
        # p4 = self._upsample_add(p5, self.latlayer1(c4)) # [b, 256, 16, 16]
        # p3 = self._upsample_add(p4, self.latlayer2(c3)) # [b, 256, 32, 32]
        # p2 = self._upsample_add(p3, self.latlayer3(c2)) # [b, 256, 64, 64]
        p4 = self.latlayer1(c4)
        p3 = self.latlayer2(c3)
        p2 = self.latlayer3(c2)


        # Smooth
        # p4 = self.smooth1(p4) # [b, 256, 16, 16]
        # p3 = self.smooth2(p3) # [b, 256, 32, 32]
        # p2 = self.smooth3(p2) # [b, 256, 64, 64]


        # Semantic
        _, _, h, w = p2.size() # h: 64, w: 64
        # 256->256
        # s5 = self._upsample(F.relu(self.gn2(self.conv2(p5))), h, w) # [b, 256, 64, 64]
        # 256->256
        # s5 = self._upsample(F.relu(self.gn2(self.conv2(s5))), h, w) # [b, 256, 64, 64]
        # 256->128
        # s5 = self._upsample(F.relu(self.gn1(self.semantic_branch(s5))), h, w) # [b, 128, 64, 64]

        # 256->256
        # s4 = self._upsample(F.relu(self.gn2(self.conv2(p4))), h, w) # [b, 256, 64, 64]
        # 256->128
        # s4 = self._upsample(F.relu(self.gn1(self.semantic_branch(s4))), h, w) # [b, 128, 64, 64]
        s4 = self._upsample(F.relu(self.gn1(self.semantic_branch(p4))), h, w) # [b, 128, 64, 64]

        # 256->128
        s3 = self._upsample(F.relu(self.gn1(self.semantic_branch(p3))), h, w) # [b, 128, 64, 64]

        s2 = F.relu(self.gn1(self.semantic_branch(p2))) # [b, 128, 64, 64]
        # return self._upsample(self.conv3(s2 + s3 + s4 + s5), 4 * h, 4 * w)
        return self._upsample(self.conv3(s2 + s3 + s4), 4 * h, 4 * w)


########################################################
### TODO: for testing decoder
import yaml
from thop import profile

# if __name__ == '__main__':
#     # load model options
#     with open('../../option/RSITMD_IRSeg.yaml', 'r') as handle:
#         options = yaml.load(handle, Loader=yaml.FullLoader)
#     # [b, 64, 64, 64] [b, 128, 32, 32] [b, 256, 16, 16] [b, 512, 8, 8]
#     feats = [torch.zeros([1, 64, 64, 64]), torch.zeros([1, 128, 32, 32]), \
#         torch.zeros([1, 256, 16, 16]), torch.zeros([1, 512, 8, 8])]
#     decoder = seg_fpn_decoder(options['model'])
#     output = decoder(feats)
#     print(output.size())

if __name__ == '__main__':
    # load model options
    with open('../../option/RSITMD_IRSeg.yaml', 'r') as handle:
        options = yaml.load(handle, Loader=yaml.FullLoader)
    options['model']['VFM']['sample'] = 'mean'
    options['model']['VFM']['mapping'] = 'linear'
    print(options['model']['VFM'])
    # [b, 64, 64, 64] [b, 128, 32, 32] [b, 256, 16, 16] [b, 512, 8, 8]
    feats = [torch.randn([1, 64, 64, 64]), torch.randn([1, 128, 32, 32]), \
        torch.randn([1, 256, 16, 16]), torch.randn([1, 512, 8, 8])]
    preds = torch.randn([1, 15, 256, 256])
    preds[:, :, 30:60, 30:80] = 50*preds[:, :, 30:60, 30:80]
    # decoder = Visual_base_multi(options['model'])
    decoder = seg_fpn_decoder(options['model'])
    flops, params = profile(decoder, (feats, preds))
    print('flops: ', flops, 'params: ', params)
    print('flops: %.2f M, params: %.2f M' % (flops / 1000000.0, params / 1000000.0))    
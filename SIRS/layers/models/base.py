##############################################
### DATE: 20230911
### AUTHOR: zzc
### TODO: base model

import torch
import torch.nn as nn
import torch.distributed as dist
import torch.nn.init
import torchvision.models as models
from torch.autograd import Variable
from torch.nn.utils.clip_grad import clip_grad_norm
import numpy as np
from collections import OrderedDict
from layers.modules.base_Modules import *
import copy

class BaseModel(nn.Module):
    def __init__(self, opt={}, vocab_words=[]):
        super(BaseModel, self).__init__()

        # img feature
        self.extract_feature = ExtractFeature(opt = opt)

        # vsa feature
        self.visual_feature = Visual_base(opt = opt)

        # text feature
        self.text_feature = Skipthoughts_Embedding_Module(
            vocab= vocab_words,
            opt = opt
        )

        # self.cross_attention_s = CrossAttention(opt = opt)

        # self.vgmf_gate = VGMF_Fusion(opt = opt)

        self.Eiters = 0

    def forward(self, img, text, text_lens):

        # extract features
        features = self.extract_feature(img)

        # mvsa featrues
        visual_feature = self.visual_feature(features)

        # text features
        text_feature = self.text_feature(text)

        # VGMF
        # Ft = self.cross_attention_s(mvsa_feature, text_feature)

        # sim dual path
        text_feature = text_feature.unsqueeze(dim=0).expand(visual_feature.shape[0], -1, -1)
        visual_feature = visual_feature.unsqueeze(dim=1).expand(-1, text_feature.shape[1], -1)
        dual_sim = cosine_similarity(visual_feature, text_feature)

        return dual_sim


def factory(opt, vocab_words, cuda=True, data_parallel=True):
    opt = copy.copy(opt)

    model = BaseModel(opt, vocab_words)

    if data_parallel:
        model = nn.DataParallel(model).cuda()
        if not cuda:
            raise ValueError

    if cuda:
        model.cuda()

    return model

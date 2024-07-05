##############################################
### DATE: 20230911
### AUTHOR: zzc
### TODO: IRSeg model

import torch.nn as nn
from layers.modules.IRSeg_Modules import *
from layers.modules.SMA_Modules import *
import copy
from typing import List
from timm.models.layers import trunc_normal_

import utils

# _cur_active: torch.Tensor = None   
def _get_active_ex_or_ii(H, W, _cur_active, returning_active_ex=True):
    h_repeat, w_repeat = H // _cur_active.shape[-2], W // _cur_active.shape[-1]
    active_ex = _cur_active.repeat_interleave(h_repeat, dim=2).repeat_interleave(w_repeat, dim=3)
    return active_ex if returning_active_ex else active_ex.squeeze(1).nonzero(as_tuple=True)  # ii: bi, hi, wi


def sp_bn_forward(self, x: torch.Tensor, cur_active: torch.Tensor):
    ii = _get_active_ex_or_ii(H=x.shape[2], W=x.shape[3], _cur_active=cur_active, returning_active_ex=False)
    
    bhwc = x.permute(0, 2, 3, 1)
    nc = bhwc[ii]                               # select the features on non-masked positions to form a flatten feature `nc`
    nc = super(type(self), self).forward(nc)    # use BN1d to normalize this flatten feature `nc`
    
    bchw = torch.zeros_like(bhwc)
    bchw[ii] = nc
    bchw = bchw.permute(0, 3, 1, 2)
    return bchw

class SparseBatchNorm2d(nn.BatchNorm1d):
    forward = sp_bn_forward     # hack: override the forward function; see `sp_bn_forward` above for more details

class BaseModel(nn.Module):
    def __init__(self, opt={}, vocab_words=[], input_size=256, decoder=seg_fpn_decoder):
        super(BaseModel, self).__init__()
        
        self.opt = opt
        self.Eiters = 0
        
        ## IR part
        # img feature
        self.extract_feature = ExtractFeature(opt = opt)
        if 'backbone' in opt.keys() and opt['backbone'] == 'resnet18':
            opt['channels'] = [64, 128, 256, 512]
        elif 'backbone' in opt.keys() and opt['backbone'] == 'resnet50':
            opt['channels'] = [256, 512, 1024, 2048]
        else:
            opt['channels'] = [64, 128, 256, 512]
        # vsa feature
        if 'VFM' in opt.keys():
            self.visual_feature = Visual_base_multi(opt = opt)
        else:
            self.visual_feature = Visual_base(opt = opt)
        # text feature
        self.text_feature = Skipthoughts_Embedding_Module(
            vocab= vocab_words,
            opt = opt
        )
        
        if 'SSA' in opt.keys():
            self.ss_attention = SMA(opt)
            self.ssa = True
        else:
            self.ssa = False
        
        ## Segmentation part     
        if 'decoder' in opt.keys():
            # self.decoder = seg_fpn_decoder(opt=opt)
            self.decoder = seg_fpn_decoder_check(opt=opt)
            self.dec = True
        else:
            self.dec = False
            

    def forward(self, img_bchw, text, text_lens):
        
        if self.training:
        
            feat_bcffs = self.extract_feature(img_bchw)
            
            ## Seg loss
            if self.dec == True:
                outputs = self.decoder(feat_bcffs)
            
            ## IR loss
            # SSA feature
            if self.ssa == True:
                feat_bcffs = self.ss_attention(feat_bcffs, outputs)
            # visual featrues
            visual_feature = self.visual_feature(feat_bcffs)
            # text features
            text_feature = self.text_feature(text)
            # sim dual path
            if isinstance(visual_feature, list):
                text_feature = text_feature.unsqueeze(dim=0).expand(visual_feature[0].shape[0], -1, -1)
                visual_feature = [v_f.unsqueeze(dim=1).expand(-1, text_feature.shape[1], -1) for v_f in visual_feature]
                dual_sim = [cosine_similarity(v_f, text_feature) for v_f in visual_feature]
                dual_sim = sum(dual_sim) / len(dual_sim)
            else:            
                text_feature = text_feature.unsqueeze(dim=0).expand(visual_feature.shape[0], -1, -1)
                visual_feature = visual_feature.unsqueeze(dim=1).expand(-1, text_feature.shape[1], -1)
                dual_sim = cosine_similarity(visual_feature, text_feature)
                

            # return {'MIM_loss': MIM_loss, 'IR_loss': IR_loss}
            if self.dec == True:
                return (outputs, dual_sim)
            else:
                return (None, dual_sim)
        else:
            feat_bcffs = self.extract_feature(img_bchw)
            
            # SSA feature
            if self.ssa == True:
                outputs = self.decoder(feat_bcffs)
                feat_bcffs = self.ss_attention(feat_bcffs, outputs)
            
            # mvsa featrues
            visual_feature = self.visual_feature(feat_bcffs)

            # text features
            text_feature = self.text_feature(text)

            # sim dual path
            if isinstance(visual_feature, list):
                text_feature = text_feature.unsqueeze(dim=0).expand(visual_feature[0].shape[0], -1, -1)
                visual_feature = [v_f.unsqueeze(dim=1).expand(-1, text_feature.shape[1], -1) for v_f in visual_feature]
                dual_sim = [cosine_similarity(v_f, text_feature) for v_f in visual_feature]
                dual_sim = sum(dual_sim) / len(dual_sim)
            else:            
                text_feature = text_feature.unsqueeze(dim=0).expand(visual_feature.shape[0], -1, -1)
                visual_feature = visual_feature.unsqueeze(dim=1).expand(-1, text_feature.shape[1], -1)
                dual_sim = cosine_similarity(visual_feature, text_feature)
                
            return (None, dual_sim)
        
    def test(self, img_bchw, text=0, text_lens=0):
        
        ## Seg loss
        feat_bcffs = self.extract_feature(img_bchw)
        outputs = self.decoder(feat_bcffs)
        return (outputs, None)
    
    def vis(self, img_bchw, text=0, text_lens=0):
        
        ## Seg loss
        feat_bcffs = self.extract_feature(img_bchw)
        # feat_bcffs = self.ss_attention(feat_bcffs, outputs)
        # SSA feature
        if self.ssa == True:
            # print('yes')
            outputs = self.decoder(feat_bcffs)
            # feat_bcffs = self.ss_attention(feat_bcffs, outputs)
        visual_feature = self.visual_feature(feat_bcffs)
        return feat_bcffs

    def mask(self, batch: int, device, generator=None):
        h, w = self.fmap_h, self.fmap_w
        idx = torch.rand(batch, h * w, generator=generator).argsort(dim=1)
        idx = idx[:, :self.len_keep].to(device)  # (batch, len_keep)
        return torch.zeros(batch, h * w, dtype=torch.bool, device=device).scatter_(dim=1, index=idx, value=True).view(batch, 1, h, w)
    
    def patchify(self, bchw):
        p = self.downsample_raito
        h, w = self.fmap_h, self.fmap_w
        B, C = bchw.shape[:2]
        bchw = bchw.reshape(shape=(B, C, h, p, w, p))
        bchw = torch.einsum('bchpwq->bhwpqc', bchw)
        bln = bchw.reshape(shape=(B, h * w, C * p ** 2))  # (B, f*f, 3*downsample_raito**2)
        return bln

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

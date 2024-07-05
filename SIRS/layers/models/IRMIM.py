##############################################
### DATE: 20230911
### AUTHOR: zzc
### TODO: IRMIM model

import torch.nn as nn
from layers.modules.IRMIM_Modules import *
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
    def __init__(self, opt={}, vocab_words=[], input_size=256, decoder=LightDecoder):
        super(BaseModel, self).__init__()
        
        # self.margin = float(opt['optim']['margin'])
        # self.max_violation = opt['optim']['max_violation']
        # opt = opt['model']
        self.opt = opt
        self.Eiters = 0
        
        ## IR part
        # img feature
        self.extract_feature = ExtractFeature(opt = opt)
        # vsa feature
        self.visual_feature = Visual_base_multi(opt = opt)
        # text feature
        self.text_feature = Skipthoughts_Embedding_Module(
            vocab= vocab_words,
            opt = opt
        )
        
        ## MIM part     
        self.downsample_raito = self.extract_feature.get_downsample_ratio()
        self.decoder = decoder(self.downsample_raito, sbn=False)   
        self.fmap_h, self.fmap_w = input_size // self.downsample_raito, input_size // self.downsample_raito
        self.len_keep = round(self.fmap_h * self.fmap_w * (1 - opt['MIM']['mask_ratio']))
        e_widths, d_width = self.extract_feature.get_feature_map_channels(), self.decoder.width
        self.hierarchy = len(e_widths)
        self.densify_norms = nn.ModuleList()
        self.densify_projs = nn.ModuleList()
        self.mask_tokens = nn.ParameterList()
        e_widths: List[int]
        for i in range(self.hierarchy):
            e_width = e_widths.pop()
            # create mask token
            p = nn.Parameter(torch.zeros(1, e_width, 1, 1))
            trunc_normal_(p, mean=0, std=.02, a=-.02, b=.02)
            self.mask_tokens.append(p)
            
            # create densify norm
            densify_norm = SparseBatchNorm2d(e_width)
            self.densify_norms.append(densify_norm)
            
            # create densify proj
            if i == 0 and e_width == d_width:
                densify_proj = nn.Identity()    # todo: NOTE THAT CONVNEXT-S WOULD USE THIS, because it has a width of 768 that equals to the decoder's width 768
                print(f'[SparK.__init__, densify {i+1}/{self.hierarchy}]: use nn.Identity() as densify_proj')
            else:
                kernel_size = 1 if i <= 0 else 3
                densify_proj = nn.Conv2d(e_width, d_width, kernel_size=kernel_size, stride=1, padding=kernel_size // 2, bias=True)
                print(f'[SparK.__init__, densify {i+1}/{self.hierarchy}]: densify_proj(ksz={kernel_size}, #para={sum(x.numel() for x in densify_proj.parameters()) / 1e6:.2f}M)')
            self.densify_projs.append(densify_proj)
            
            # todo: the decoder's width follows a simple halfing rule; you can change it to any other rule
            d_width //= 2
            

    def forward(self, img_bchw, text, text_lens):
        
        if self.training:
            ## MIM loss
            mask_b1ff: torch.BoolTensor = self.mask(img_bchw.shape[0], img_bchw.device)  # (B, 1, H/dr=f, W/dr=f)
            mask_b1hw = mask_b1ff.repeat_interleave(self.downsample_raito, 2).repeat_interleave(self.downsample_raito, 3)  # (B, 1, H, W)
            masked_bchw = img_bchw * mask_b1hw
            
            feat_bcffs = self.extract_feature(masked_bchw)
            feat_bcffs.reverse()
            
            cur_active = mask_b1ff     # (B, 1, H/dr=f, W/dr=f)
            to_decoder = []
            for i, feat_bcff in enumerate(feat_bcffs):  # from the smallest feature map to the largest
                if feat_bcff is not None:
                    bcff = self.densify_norms[i](feat_bcff, cur_active)
                    mask_tokens = self.mask_tokens[i].expand_as(bcff)
                    bcff = torch.where(cur_active.expand_as(bcff), bcff, mask_tokens)   # fill in empty (non-active) positions with [mask] tokens
                    bcff: torch.Tensor = self.densify_projs[i](bcff)
                to_decoder.append(bcff)
                cur_active = cur_active.repeat_interleave(2, dim=2).repeat_interleave(2, dim=3)  # dilate the mask map, from (B, 1, f, f) to (B, 1, H, W)
            
            rec_bchw = self.decoder(to_decoder)
            inp, rec = self.patchify(img_bchw), self.patchify(rec_bchw)   # inp and rec: (B, L = f*f, N = C*downsample_raito**2)
            mean = inp.mean(dim=-1, keepdim=True)
            var = (inp.var(dim=-1, keepdim=True) + 1e-6) ** .5
            inp = (inp - mean) / var
            # l2_loss = ((rec - inp) ** 2).mean(dim=2, keepdim=False)    # (B, L, C) ==mean==> (B, L)
            
            # non_active = mask_b1ff.logical_not().int().view(mask_b1ff.shape[0], -1)  # (B, 1, f, f) => (B, L)
            # MIM_loss = l2_loss.mul_(non_active).sum() / (non_active.sum() + 1e-8)  # loss only on masked (non-active) patches
            
            ## IR loss
            feat_bcffs.reverse()

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
                
            # IR_loss = utils.calcul_loss(dual_sim, img_bchw.shape[0], self.margin, max_violation = self.max_violation)
            
            # return {'MIM_loss': MIM_loss, 'IR_loss': IR_loss}
            return (inp, rec, mask_b1ff), dual_sim
        else:
            feat_bcffs = self.extract_feature(img_bchw)
            
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
                
            return dual_sim

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

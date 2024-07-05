##############################################
### DATE: 20230918
### AUTHOR: zzc
### TODO: infer single query
### REQ: 

import sys
import os
import cv2
import data
import numpy as np
import engine
import argparse
import yaml
import torch
from layers.models import IRSeg as models
# from layers.models import AMFMN as models
# from layers.models import GaLR as models
from vocab import deserialize_vocab
from torch.autograd import Variable
import matplotlib.pyplot as plt
import pandas as pd
import time 

# weights_path = '/root/autodl-tmp/zzc_backup/archive/IRSeg_series/rsitmd_irseg_v12_base/0/IRSeg_best.pth.tar'
# weights_path = '/root/autodl-tmp/zzc_backup/archive/IRSeg_series/rsitmd_irseg_v11_addPort/2/IRSeg_best.pth.tar'
config_path = '/root/autodl-tmp/zzc_backup/code/private/SIRS/option/RSITMD_IRSeg_noSS.yaml'
# config_path = '/root/autodl-tmp/zzc_backup/code/private/SIRS/option/RSITMD_IRSeg.yaml'
# config_path = '/root/autodl-tmp/zzc_backup/code/private/SIRS/option/RSITMD_AMFMN.yaml'
# config_path = '/root/autodl-tmp/zzc_backup/code/private/SIRS/option/RSITMD_GaLR.yaml'

data_type = 'test'

def get_parameter_number(model, name=''):
    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # return {'Total': total_num // (1024*1024), 'MB', 'Trainable': trainable_num / 1024}
    if name != '':
        print('Model {}:'.format(name))
    print('Total: ', format(total_num / (1024*1024), '.2f'), 'M')
    print('Trainable: ', format(trainable_num / (1024*1024), '.2f'), 'M')

def parser_options():
    # Hyper Parameters setting
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_opt', default=config_path, type=str,
                        help='path to a yaml options file')
    # parser.add_argument('--text_sim_path', default='data/ucm_precomp/train_caps.npy', type=str,help='path to t2t sim matrix')
    opt = parser.parse_args()

    # load model options
    with open(opt.path_opt, 'r') as handle:
        # options = yaml.load(handle)
        options = yaml.load(handle, Loader=yaml.FullLoader)
    return options

options = parser_options()

# make vocab
vocab = deserialize_vocab(options['dataset']['vocab_path'])
vocab_word = sorted(vocab.word2idx.items(), key=lambda x: x[1], reverse=False)
vocab_word = [tup[0] for tup in vocab_word]

model = models.factory(options['model'],
                        vocab_word,
                        cuda=True, 
                        data_parallel=False)

get_parameter_number(model, model.__class__.__name__)

time.sleep(5)

# options['optim']['resume'] = weights_path


# if options['optim']['resume']:
#     if os.path.isfile(options['optim']['resume']):
#         print("=> loading checkpoint '{}'".format(options['optim']['resume']))
#         checkpoint = torch.load(options['optim']['resume'])
#         start_epoch = checkpoint['epoch']
#         # best_rsum = checkpoint['best_rsum']
#         model.load_state_dict(checkpoint['model'])
#         print("=> loaded checkpoint '{}' (epoch {})"
#                 .format(options['optim']['resume'], start_epoch))


##############################################
### DATE: 20230911
### AUTHOR: zzc
### TODO: infer query results and segmentation results
### REQ: remember to refine data augmentation in data.py for testloader

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
from vocab import deserialize_vocab
from torch.autograd import Variable
import matplotlib.pyplot as plt
import pandas as pd

import time

data_type = 'test'

# weights_path = '/root/autodl-tmp/zzc_backup/archive/IRSeg_series/rsitmd_irseg_v12_noSeg/0/IRSeg_best.pth.tar'
# weights_path = '/root/autodl-tmp/zzc_backup/archive/IRSeg_series/rsitmd_irseg_v12_base/0/IRSeg_best.pth.tar'
# weights_path = '/root/autodl-tmp/zzc_backup/archive/IRSeg_series/rsitmd_irseg_v11_addPort/2/IRSeg_best.pth.tar'
# config_path = '/home/zzc/code/private/SIRS/option/RSITMD_IRSeg_noSS.yaml'
# config_path = '/root/autodl-tmp/zzc_backup/archive/IRSeg_series/rsitmd_irseg_v11_addPort/options.yaml'
# config_path = '/root/autodl-tmp/zzc_backup/archive/IRSeg_series/rsitmd_irseg_v10_SSA_mPBN/options.yaml'
# config_path = '/root/autodl-tmp/zzc_backup/archive/IRSeg_series/rsitmd_irseg_v10_SSA_mPGN/options.yaml'
config_path = '/home/zzc/archive/IRSeg_series/rsitmd_irseg_v10_SSA_aP/options.yaml'
# config_path = '/home/zzc/archive/IRSeg_series/rsicd_irseg_v1_noSeg/options.yaml'

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
# model = models.factory(options,
                        vocab_word,
                        cuda=True, 
                        data_parallel=False)

# optionally resume from a checkpoint
# options['optim']['resume'] = '/root/autodl-tmp/zzc_backup/archive/IRSeg_series/rsitmd_irseg_v7_vis/0/IRSeg_best.pth.tar'
# options['optim']['resume'] = '/root/autodl-tmp/zzc_backup/archive/IRSeg_series/rsitmd_irseg_v11_addPort/3/IRSeg_best.pth.tar'
# options['optim']['resume'] = weights_path



if 'weights_path' in vars():
    options['optim']['resume'] = weights_path
    if os.path.isfile(options['optim']['resume']):
        print("=> loading checkpoint '{}'".format(options['optim']['resume']))
        checkpoint = torch.load(options['optim']['resume'])
        start_epoch = checkpoint['epoch']
        # best_rsum = checkpoint['best_rsum']
        model.load_state_dict(checkpoint['model'])
        print("=> loaded checkpoint '{}' (epoch {})"
                .format(options['optim']['resume'], start_epoch))

options['dataset']['batch_size_val'] = 1
# test_loader = data.get_test_loader(vocab, options)
test_loader, _ = data.get_precomp_loader(data_type, vocab,
                    options['dataset']['batch_size_val'], False, options['dataset']['test_workers'], opt=options)

model.eval()

samples_num = len(test_loader.dataset)

input_visual = np.zeros((samples_num, 3, 256, 256))
input_text_all = np.zeros((samples_num, 47), dtype=np.int64)
input_text_lengeth_all = [0]*samples_num
images_list = []
captions_list = []

# for i, val_data in enumerate(test_loader):
#     # print('train_data: ', train_data)
#     tmps, captions, lengths, ids= val_data
#     if options['model']['name'] == 'IRSeg' and 'decoder' in options['model'].keys():
#         image, target = tmps
#         target = Variable(target)
#         if torch.cuda.is_available():
#             target = target.cuda()
#     else:
#         image = tmps
#         image = Variable(image)
#     input_text = Variable(captions)

#     if torch.cuda.is_available():
#         image = image.cuda()
#         input_text = input_text.cuda()
        
#     # print('image: ', image.size())
start = time.time()
# seg_result = engine.validate_seg(test_loader, model, options['model']['decoder']['num_classes'], 0)
ave_score = engine.validate_v2(test_loader, model)
end = time.time()
period = end - start
fps = samples_num / period

print('-------------------------------------------------')
print('Total samples number {}'.format(samples_num))
print("Total time cost %.2fs" % (period))
print("FPS %.2f" % (fps))
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

k = 5
data_type = 'val'
image_dir = '/root/autodl-tmp/zzc_backup/data/RSITMD/images'
vis_dir = '/root/autodl-tmp/zzc_backup/data/RSITMD/vis_IRSeg_{}'.format(data_type)
if not os.path.exists(vis_dir):
    os.makedirs(vis_dir)
mix_dir = '/root/autodl-tmp/zzc_backup/data/RSITMD/mix_IRSeg_{}'.format(data_type)
if not os.path.exists(mix_dir):
    os.makedirs(mix_dir)

i2t_path = './output/SIRS_{}_i2t_dict.yaml'.format(data_type)
t2i_path = './output/SIRS_{}_t2i_dict.yaml'.format(data_type)
# weights_path = '/root/autodl-tmp/zzc_backup/archive/IRSeg_series/rsitmd_irseg_v12_noSeg/0/IRSeg_best.pth.tar'
# weights_path = '/root/autodl-tmp/zzc_backup/archive/IRSeg_series/rsitmd_irseg_v12_base/0/IRSeg_best.pth.tar'
weights_path = '/root/autodl-tmp/zzc_backup/archive/IRSeg_series/rsitmd_irseg_v11_addPort/2/IRSeg_best.pth.tar'
# config_path = '/root/autodl-tmp/zzc_backup/code/private/SIRS/option/RSITMD_IRSeg_noSS.yaml'
config_path = '/root/autodl-tmp/zzc_backup/code/private/SIRS/option/RSITMD_IRSeg.yaml'

def shard_dis(images, captions, model, shard_size=64, lengths=None):
    """compute image-caption pairwise distance during validation and test"""

    n_im_shard = (len(images) - 1) // shard_size + 1
    n_cap_shard = (len(captions) - 1) // shard_size + 1

    d = np.zeros((len(images), len(captions)))

    for i in range(n_im_shard):
        im_start, im_end = shard_size*i, min(shard_size*(i+1), len(images))

#        print("======================")
#        print("im_start:",im_start)
#        print("im_end:",im_end)

        for j in range(n_cap_shard):
            sys.stdout.write('\r>> shard_distance batch (%d,%d)' % (i,j))
            cap_start, cap_end = shard_size * j, min(shard_size * (j + 1), len(captions))

            im = Variable(torch.from_numpy(images[im_start:im_end]), volatile=True).float().cuda()
            s = Variable(torch.from_numpy(captions[cap_start:cap_end]), volatile=True).cuda()
            l = lengths[cap_start:cap_end]

            (_, sim) = model(im, s,l)
            sim = sim.squeeze()
            d[im_start:im_end, cap_start:cap_end] = sim.data.cpu().numpy()
    sys.stdout.write('\n')
    return d
    
mask_color_dict_forVis = {
    ### black
    'Background':    [255, 255, 255], 
    ### orange
    'Plane':         [255, 165, 0],
    ### cyan
    'Boat':          [0, 255, 255],
    ### yellow
    'StorageTank':   [255, 255, 0],
    ### MediumBlue
    'Pond':          [0, 0, 205],
    ### DodgerBlue
    'River':         [30, 144, 255],
    ### Gold
    'Beach':         [255, 215, 0],
    ### OliveDrab
    'Playground':    [107, 142, 35],
    ### RoyalBlue
    'SwimmingPool':  [65, 105, 225],
    ### GreenYellow
    'Court':         [173, 255, 47],
    ### Gold4
    'BaseballField': [139, 117, 0],
    ### Tomato
    'Center':        [255, 99, 71],
    ### white
    'Church':        [255, 255, 255],
    ### HotPink
    'Stadium':       [255, 105, 180],
    ### grey51
    'Bridge':        [130, 130, 130],
}

mask_color_dict_forMix = {
    ### black
    'Background':    [0, 0, 0], 
    ### orange
    'Plane':         [255, 165, 0],
    ### cyan
    'Boat':          [0, 255, 255],
    ### yellow
    'StorageTank':   [255, 255, 0],
    ### MediumBlue
    'Pond':          [0, 0, 205],
    ### DodgerBlue
    'River':         [30, 144, 255],
    ### Gold
    'Beach':         [255, 215, 0],
    ### OliveDrab
    'Playground':    [107, 142, 35],
    ### RoyalBlue
    'SwimmingPool':  [65, 105, 225],
    ### GreenYellow
    'Court':         [173, 255, 47],
    ### Gold4
    'BaseballField': [139, 117, 0],
    ### Tomato
    'Center':        [255, 99, 71],
    ### white
    'Church':        [255, 255, 255],
    ### HotPink
    'Stadium':       [255, 105, 180],
    ### grey51
    'Bridge':        [130, 130, 130],
}

def get_RSITMD_labels(mask_color_dict):
    labels = []
    for item in mask_color_dict:
        labels.append(mask_color_dict[item])
    return np.array(labels)

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

def decode_segmap(label_mask, plot=False, dict_=None):
    """Decode segmentation class labels into a color image
    Args:
        label_mask (np.ndarray): an (M,N) array of integer values denoting
          the class label at each spatial location.
        plot (bool, optional): whether to show the resulting color image
          in a figure.
    Returns:
        (np.ndarray, optional): the resulting decoded color image.
    """

    n_classes = 15
    label_colors = get_RSITMD_labels(dict_)

    # print('label_mask: ', label_mask.shape)
    r = label_mask.copy()
    g = label_mask.copy()
    b = label_mask.copy()
    for ll in range(0, n_classes):
        # r[label_mask == ll] = label_colors[ll, 0]
        # g[label_mask == ll] = label_colors[ll, 1]
        # b[label_mask == ll] = label_colors[ll, 2]
        b[label_mask == ll] = label_colors[ll, 0]
        g[label_mask == ll] = label_colors[ll, 1]
        r[label_mask == ll] = label_colors[ll, 2]
    rgb = np.zeros((label_mask.shape[0], label_mask.shape[1], 3))
    # rgb[:, :, 0] = r / 255.0
    # rgb[:, :, 1] = g / 255.0
    # rgb[:, :, 2] = b / 255.0
    rgb[:, :, 0] = r
    rgb[:, :, 1] = g
    rgb[:, :, 2] = b
    if plot:
        plt.imshow(rgb)
        plt.show()
    else:
        return rgb

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
options['optim']['resume'] = weights_path



if options['optim']['resume']:
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

input_visual = np.zeros((len(test_loader.dataset), 3, 256, 256))
input_text_all = np.zeros((len(test_loader.dataset), 47), dtype=np.int64)
input_text_lengeth_all = [0]*len(test_loader.dataset)
images_list = []
captions_list = []

for i, val_data in enumerate(test_loader):
    # print('train_data: ', train_data)
    tmps, captions, lengths, ids= val_data
    if options['model']['name'] == 'IRSeg' and 'decoder' in options['model'].keys():
        image, target = tmps
        target = Variable(target)
        if torch.cuda.is_available():
            target = target.cuda()
    else:
        image = tmps
        image = Variable(image)
    input_text = Variable(captions)

    if torch.cuda.is_available():
        image = image.cuda()
        input_text = input_text.cuda()
        
    # print('image: ', image.size())
        
    #### Seg part 
    with torch.no_grad():
        (output, _) = model.test(image, input_text, lengths)
        
    pred = output.data.cpu().numpy()
    target = target.cpu().numpy()
    pred = np.argmax(pred, axis=1)
    # pred = pred[:, np.newaxis, :, :]
    pred = pred[0]
    # print('pred: ', pred.shape)
    # Add batch sample into evaluator

    # show result
    pred_rgb_vis = decode_segmap(pred, dict_=mask_color_dict_forVis)
    pred_rgb_mix = decode_segmap(pred, dict_=mask_color_dict_forMix)
    # pred_rgb = np.array(pred_rgb).transpose(2, 0, 1)
    # print('pred_rgb: ', pred_rgb_vis.shape)
    if data_type == 'test':
        image_name = str(test_loader.dataset.images[ids[0]])[2:-1]
    else:
        image_name = str(test_loader.dataset.images[ids[0] // 5])[2:-1]
    vis_name = image_name.replace('tif', 'png')
    save_path = os.path.join(vis_dir, vis_name.replace('tif', 'png'))
    cv2.imwrite(save_path, pred_rgb_vis)
    
    image_path = os.path.join(image_dir, image_name)
    image_ = cv2.imread(image_path)
    # print('image: ', image_.shape)
    # dst = cv2.addWeighted(image, 0.7, pred_rgb, 0.3, 0)  # 图片img1所占比重0.7;图片img2所占比重0.3
    dst = image_*0.7 + pred_rgb_mix*0.3
    save_path = os.path.join(mix_dir, vis_name.replace('tif', 'png'))
    cv2.imwrite(save_path, dst)
    
    #### IR part
    if data_type == 'test':
        image_name = str(test_loader.dataset.images[ids[0]])[2:-1]
    else:
        image_name = str(test_loader.dataset.images[ids[0] // 5])[2:-1]
    caption_name = test_loader.dataset.captions[ids[0]]
    images_list.append(image_name)
    captions_list.append(caption_name)
    for (id, img, cap, key,l) in zip(ids, (image.cpu().numpy().copy()), (captions.cpu().numpy().copy()), image, lengths):
        input_visual[id] = img
        input_text_all[id, :captions.size(1)] = cap
        input_text_lengeth_all[id] = l

images_list = [images_list[i] for i in range(0, len(images_list), 5)]
input_visual = np.array([input_visual[i] for i in range(0, len(input_visual), 5)])
d = shard_dis(input_visual, input_text_all, model, lengths=input_text_lengeth_all )

print('d: ', d.shape)
i2t_dict = {}
t2i_dict = {}

for i, image in enumerate(images_list):
    scores = np.array(d[i])
    idx = np.argsort(-scores)
    results = [captions_list[idx[j]].decode('utf-8') for j in range(k)]
    i2t_dict[image] = results

for i, caption in enumerate(captions_list):
    scores = np.array(d[:, i])
    idx = np.argsort(-scores)
    results = [images_list[idx[j]] for j in range(k)]
    t2i_dict[caption.decode('utf-8')] = results
    
import yaml

with open(i2t_path, 'w') as file:
    yaml.dump(i2t_dict, file)
with open(t2i_path, 'w') as file:
    yaml.dump(t2i_dict, file)
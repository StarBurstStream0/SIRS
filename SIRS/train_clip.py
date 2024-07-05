##############################################
### DATE: 20230911
### AUTHOR: zzc
### TODO: train model

import os,random,copy
import torch
import torch.nn as nn
import argparse
import yaml
import shutil
import tensorboard_logger as tb_logger
import logging
import click

import utils
import data
import engine

from vocab import deserialize_vocab

import numpy as np

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

import clip

def parser_options():
    # Hyper Parameters setting
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_opt', default='option/RSITMD_IRSeg.yaml', type=str,
                        help='path to a yaml options file')
    # parser.add_argument('--text_sim_path', default='data/ucm_precomp/train_caps.npy', type=str,help='path to t2t sim matrix')
    opt = parser.parse_args()

    # load model options
    with open(opt.path_opt, 'r') as handle:
        # options = yaml.load(handle)
        options = yaml.load(handle, Loader=yaml.FullLoader)

    return options

def main(options):
    # choose model
    if options['model']['name'] == "AMFMN":
        from layers.models import AMFMN as models
    elif options['model']['name'] == "base":
        from layers.models import base as models
    elif options['model']['name'] == "prototype":
        from layers.models import prototype as models
    elif options['model']['name'] == "SISTR":
        from layers.models import prototype as models
    elif options['model']['name'] == "IRMIM":
        from layers.models import IRMIM as models
    elif options['model']['name'] == "IRSeg":
        from layers.models import IRSeg as models
    else:
        raise NotImplementedError

    # make ckpt save dir
    if not os.path.exists(options['logs']['ckpt_save_path']):
        os.makedirs(options['logs']['ckpt_save_path'])

    #######################################################################################
    # make vocab
    vocab = deserialize_vocab(options['dataset']['vocab_path'])
    vocab_word = sorted(vocab.word2idx.items(), key=lambda x: x[1], reverse=False)
    vocab_word = [tup[0] for tup in vocab_word]

    # Create dataset, model, criterion and optimizer
    train_loader, val_loader, weights = data.get_loaders(vocab, options)
    test_loader = data.get_test_loader(vocab, options)
    
    model = models.factory(options['model'],
    # model = models.factory(options,
                           vocab_word,
                           cuda=True, 
                           data_parallel=False)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)
    
    
    
    #######################################################################################

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                                 lr=options['optim']['lr'])

    print('Model has {} parameters'.format(utils.params_count(model)))

    # optionally resume from a checkpoint
    if options['optim']['resume']:
        if os.path.isfile(options['optim']['resume']):
            print("=> loading checkpoint '{}'".format(options['optim']['resume']))
            checkpoint = torch.load(options['optim']['resume'])
            start_epoch = checkpoint['epoch']
            best_rsum = checkpoint['best_rsum']
            model.load_state_dict(checkpoint['model'])

            # Eiters is used to show logs as the continuation of another
            # training
            model.Eiters = checkpoint['Eiters']

            print("=> loaded checkpoint '{}' (epoch {}, best_rsum {})"
                  .format(options['optim']['resume'], start_epoch, best_rsum))
            rsum, all_scores =  engine.validate(val_loader, model)
            print(all_scores)
        else:
            print("=> no checkpoint found at '{}'".format(options['optim']['resume']))
    else:
        start_epoch = 0

    # Train the Model
    ave_scores = []

    for epoch in range(start_epoch, options['optim']['epochs']):

        utils.adjust_learning_rate(options, optimizer, epoch)

        # train for one epoch
        engine.train(train_loader, model, optimizer, epoch, opt=options, weights=weights)

        # evaluate on validation set
        # if epoch % options['logs']['eval_step'] == 0:
        if epoch % options['logs']['eval_step'] == 0 \
            and epoch >= options['optim']['epochs']-options['optim']['last_num_epochs_val']:
            ave_score = engine.validate_v2(test_loader, model)
            utils.log_to_txt(
                contexts='score: ' + str(ave_score),
                filename=options['logs']['ckpt_save_path']+ options['model']['name'] + "_" + options['dataset']['datatype'] +".txt"
            )
            ave_scores.append(ave_score)
            if options['model']['name'] == 'IRSeg' and 'decoder' in options['model'].keys():
                seg_result = engine.validate_seg(test_loader, model, options['model']['decoder']['num_classes'], epoch)
                utils.log_to_txt(
                    contexts=str(seg_result),
                    filename=options['logs']['ckpt_save_path']+ options['model']['name'] + "_" + options['dataset']['datatype'] +".txt"
                )
    # save ckpt
    if options['optim']['weight_save'] and epoch == options['optim']['epochs']-1:
        utils.save_checkpoint(
            {
            'epoch': epoch + 1,
            'arch': 'baseline',
            'model': model.state_dict(),
            # 'best_rsum': best_rsum,
            'options': options,
            'Eiters': model.Eiters,
        },
            True,
            filename='ckpt_{}_{}.pth.tar'.format(options['model']['name'] ,epoch),
            prefix=options['logs']['ckpt_save_path'],
            model_name=options['model']['name']
        )
    
    ave_scores = np.mean(np.array(ave_scores), axis=0)
    currscore = (ave_scores[5] + ave_scores[6] + ave_scores[7] + ave_scores[0] + ave_scores[1] + ave_scores[2])/6.0

    print("Current {}th fold.".format(options['k_fold']['current_num']))
    average_scores = "Average score:\n ------\n r1i:{} r5i:{} r10i:{} medri:{} meanri:{}\n r1t:{} r5t:{} r10t:{} medrt:{} meanrt:{}\n average:{}\n ------\n".format(
        ave_scores[0], ave_scores[1], ave_scores[2], ave_scores[3], ave_scores[4], \
        ave_scores[5], ave_scores[6], ave_scores[7], ave_scores[8], ave_scores[9], \
        currscore
    )
    print(average_scores)

    utils.log_to_txt(
        contexts= "Epoch:{} ".format(epoch+1) + average_scores,
        filename=options['logs']['ckpt_save_path']+ options['model']['name'] + "_" + options['dataset']['datatype'] +".txt"
    )
        

def generate_random_samples(options):
    # load all anns
    caps = utils.load_from_txt(options['dataset']['data_path']+'train_caps.txt')
    fnames = utils.load_from_txt(options['dataset']['data_path']+'train_filename.txt')

    # merge
    assert len(caps) // 5 == len(fnames)
    all_infos = []
    for img_id in range(len(fnames)):
        cap_id = [img_id * 5 ,(img_id+1) * 5]
        all_infos.append([caps[cap_id[0]:cap_id[1]], fnames[img_id]])

    # shuffle
    random.shuffle(all_infos)

    # split_trainval
    percent = 0.8
    train_infos = all_infos[:int(len(all_infos)*percent)]
    val_infos = all_infos[int(len(all_infos)*percent):]

    # save to txt
    train_caps = []
    train_fnames = []
    for item in train_infos:
        for cap in item[0]:
            train_caps.append(cap)
        train_fnames.append(item[1])
    utils.log_to_txt(train_caps, options['dataset']['data_path']+'train_caps_verify.txt',mode='w')
    utils.log_to_txt(train_fnames, options['dataset']['data_path']+'train_filename_verify.txt',mode='w')

    val_caps = []
    val_fnames = []
    for item in val_infos:
        for cap in item[0]:
            val_caps.append(cap)
        val_fnames.append(item[1])
    utils.log_to_txt(val_caps, options['dataset']['data_path']+'val_caps_verify.txt',mode='w')
    utils.log_to_txt(val_fnames, options['dataset']['data_path']+'val_filename_verify.txt',mode='w')

    print("Generate random samples to {} complete.".format(options['dataset']['data_path']))

def generate_sceneGuided_samples(options):
    print('Start to generate samples with scene-guided strategy...')
    
    # if os.path.exists(options['dataset']['data_path']+'train_caps_verify.txt') and \
    #     os.path.exists(options['dataset']['data_path']+'train_filename_verify.txt') and \
    #     os.path.exists(options['dataset']['data_path']+'val_caps_verify.txt') and \
    #     os.path.exists(options['dataset']['data_path']+'val_filename_verify.txt'):
    #     print('Files already exists! keep them...')
    #     return 0
    
    # load all anns
    caps = utils.load_from_txt(options['dataset']['data_path']+'train_caps.txt')
    fnames = utils.load_from_txt(options['dataset']['data_path']+'train_filename.txt')

    # collect unuique scenes from file names and merge
    scenes = {}
    for img_id, fname in enumerate(fnames):
        cap_id = [img_id * 5 ,(img_id+1) * 5]
        scene = fname.split('_')[0]
        if scene not in scenes:
            scenes[scene] = []
        else:
            scenes[scene].append([caps[cap_id[0]:cap_id[1]], fname])

    # shuffle
    for scene in scenes:
        random.shuffle(scenes[scene])

    # split_trainval
    percent = 0.8
    train_infos = []
    val_infos = []
    for scene in scenes:
        train_infos.extend(scenes[scene][:int(len(scenes[scene])*percent)])
        val_infos.extend(scenes[scene][int(len(scenes[scene])*percent):])
        
    # shuffle again
    random.shuffle(train_infos)
    random.shuffle(val_infos)

    # save to txt
    train_caps = []
    train_fnames = []
    for item in train_infos:
        for cap in item[0]:
            train_caps.append(cap)
        train_fnames.append(item[1])
    utils.log_to_txt(train_caps, options['dataset']['data_path']+'train_caps_verify.txt',mode='w')
    utils.log_to_txt(train_fnames, options['dataset']['data_path']+'train_filename_verify.txt',mode='w')

    val_caps = []
    val_fnames = []
    for item in val_infos:
        for cap in item[0]:
            val_caps.append(cap)
        val_fnames.append(item[1])
    utils.log_to_txt(val_caps, options['dataset']['data_path']+'val_caps_verify.txt',mode='w')
    utils.log_to_txt(val_fnames, options['dataset']['data_path']+'val_filename_verify.txt',mode='w')

    print("Generate random samples to {} complete.".format(options['dataset']['data_path']))
    
def generate_allTraining_samples(options):
    # load all anns
    caps = utils.load_from_txt(options['dataset']['data_path']+'train_caps.txt')
    fnames = utils.load_from_txt(options['dataset']['data_path']+'train_filename.txt')

    # merge
    assert len(caps) // 5 == len(fnames)
    all_infos = []
    for img_id in range(len(fnames)):
        cap_id = [img_id * 5 ,(img_id+1) * 5]
        all_infos.append([caps[cap_id[0]:cap_id[1]], fnames[img_id]])

    # shuffle
    random.shuffle(all_infos)

    # split_trainval
    train_infos = all_infos

    # save to txt
    train_caps = []
    train_fnames = []
    for item in train_infos:
        for cap in item[0]:
            train_caps.append(cap)
        train_fnames.append(item[1])
    utils.log_to_txt(train_caps, options['dataset']['data_path']+'train_caps_verify.txt',mode='w')
    utils.log_to_txt(train_fnames, options['dataset']['data_path']+'train_filename_verify.txt',mode='w')

    print("Generate all training samples to {} complete.".format(options['dataset']['data_path']))

def update_options_savepath(options, k):
    updated_options = copy.deepcopy(options)

    updated_options['k_fold']['current_num'] = k
    updated_options['logs']['ckpt_save_path'] = options['logs']['ckpt_save_path'] + \
                                                options['k_fold']['experiment_name'] + "/" + str(k) + "/"
    return updated_options

if __name__ == '__main__':
    options = parser_options()
    
    # save the configuration
    if not os.path.exists(options['logs']['ckpt_save_path'] + options['k_fold']['experiment_name']):
        os.makedirs(options['logs']['ckpt_save_path'] + options['k_fold']['experiment_name'])
    opt_path = options['logs']['ckpt_save_path'] + options['k_fold']['experiment_name'] + '/' + 'options.yaml'
    with open(opt_path, 'w', encoding='utf-8', ) as f:
        yaml.dump(options, f, encoding='utf-8', allow_unicode=True)

    # make logger
    tb_logger.configure(options['logs']['ckpt_save_path'] + options['k_fold']['experiment_name'], flush_secs=5)
    logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)

    # k_fold verify
    for k in range(options['k_fold']['current_num'], options['k_fold']['nums']):
        print("=========================================")
        print("Start {}th fold".format(k))

        # generate random train and val samples
        # generate_random_samples(options)
        if options['dataset']['data_split'] == 'all':
            generate_allTraining_samples(options)
        elif options['dataset']['data_split'] == 'trainval' and options['dataset']['datatype'] == 'ucm':
            generate_random_samples(options)
        elif options['dataset']['data_split'] == 'trainval':
            generate_sceneGuided_samples(options)

        # update save path
        update_options = update_options_savepath(options, k)

        # run experiment
        main(update_options)

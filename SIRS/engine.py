##############################################
### DATE: 20230911
### AUTHOR: zzc
### TODO: processing training and validation

import time
import torch
import numpy as np
import sys
from torch.autograd import Variable
import utils
import tensorboard_logger as tb_logger
import logging
from torch.nn.utils.clip_grad import clip_grad_norm

from metrics import Evaluator

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

def train(train_loader, model, optimizer, epoch, opt={}, logger = None, weights = None):

    # extract value
    grad_clip = opt['optim']['grad_clip']
    max_violation = opt['optim']['max_violation']
    margin = opt['optim']['margin']
    # loss_name = opt['model']['name'] + "_" + opt['dataset']['datatype']
    print_freq = opt['logs']['print_freq']

    # switch to train mode
    model.train()
    batch_time = utils.AverageMeter()
    data_time = utils.AverageMeter()
    train_logger = utils.LogCollector()

    end = time.time()
    params = list(model.parameters())
    for i, train_data in enumerate(train_loader):
        # print('train_data: ', train_data)
        tmps, captions, lengths, ids= train_data
        
        if opt['model']['name'] == 'IRSeg' and 'decoder' in opt['model'].keys():
            assert weights is not None
            images, targets = tmps
            targets = Variable(targets)
            if torch.cuda.is_available():
                targets = targets.cuda()
        else:
            images = tmps
            images = Variable(images)

        batch_size = images.size(0)
        # measure data loading time
        data_time.update(time.time() - end)
        model.logger = train_logger

        input_text = Variable(captions)
        if torch.cuda.is_available():
            images = images.cuda()
            input_text = input_text.cuda()

        # losses = model(input_visual, input_text, lengths)
        results = model(images, input_text, lengths)
        losses = {}
        if opt['model']['name'] == 'IRMIM':
            MIM_res = results[0]
            losses['MIM_loss'] = utils.cal_MIM_loss(MIM_res)
        elif opt['model']['name'] == 'IRSeg' and 'decoder' in opt['model'].keys():
            Seg_res = results[0]
            loss_func = utils.SegmentationLosses(weight=weights, cuda=True).build_loss(mode='ce')
            losses['Seg_loss'] = loss_func(Seg_res, targets.long()) * opt['optim']['loss_weight']['SS']
            # losses['Seg_loss'] = loss_func(Seg_res, targets.long())
        IR_res = results[-1]
        losses['IR_loss'] = utils.cal_IR_loss(IR_res, images.size(0), margin, max_violation=max_violation) \
                * opt['optim']['loss_weight']['IR']
        # losses['IR_loss'] = utils.cal_IR_loss(IR_res, images.size(0), margin, max_violation=max_violation)
        
        torch.cuda.synchronize()

        if grad_clip > 0:
            clip_grad_norm(params, grad_clip)

        loss = 0
        for item in losses:
            train_logger.update(item, losses[item].cpu().data.numpy())
            loss += losses[item]
        loss /= len(losses)

        optimizer.zero_grad()
        loss.backward()
        torch.cuda.synchronize()
        optimizer.step()
        torch.cuda.synchronize()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % print_freq == 0:
            logging.info(
                'Epoch: [{0}][{1}/{2}]\t'
                'Time {batch_time.val:.3f}\t'
                '{elog}\t'
                .format(epoch, i, len(train_loader),
                        batch_time=batch_time,
                        elog=str(train_logger)))

            utils.log_to_txt(
                'Epoch: [{0}][{1}/{2}]\t'
                'Time {batch_time.val:.3f}\t'
                '{elog}\t'
                    .format(epoch, i, len(train_loader),
                            batch_time=batch_time,
                            elog=str(train_logger)),
                opt['logs']['ckpt_save_path']+ opt['model']['name'] + "_" + opt['dataset']['datatype'] +".txt"
            )
        if logger != None:
            logger.log_value('epoch', epoch, step=model.Eiters)
            logger.log_value('step', i, step=model.Eiters)
            logger.log_value('batch_time', batch_time.val, step=model.Eiters)
            train_logger.tb_log(logger, step=model.Eiters)
        else:
            tb_logger.log_value('epoch', epoch, step=model.Eiters)
            tb_logger.log_value('step', i, step=model.Eiters)
            tb_logger.log_value('batch_time', batch_time.val, step=model.Eiters)
            train_logger.tb_log(tb_logger, step=model.Eiters)

def validate(val_loader, model):

    model.eval()
    val_logger = utils.LogCollector()
    model.logger = val_logger

    start = time.time()
    input_visual = np.zeros((len(val_loader.dataset), 3, 256, 256))
    input_text = np.zeros((len(val_loader.dataset), 47), dtype=np.int64)
    input_text_lengeth = [0]*len(val_loader.dataset)
    for i, val_data in enumerate(val_loader):

        images, captions, lengths, ids = val_data
        
        for (id, img, cap, key,l) in zip(ids, (images.numpy().copy()), (captions.numpy().copy()), images , lengths):
            input_visual[id] = img
            input_text[id, :captions.size(1)] = cap
            input_text_lengeth[id] = l


    input_visual = np.array([input_visual[i] for i in range(0, len(input_visual), 5)])

    d = utils.shard_dis(input_visual, input_text, model , lengths=input_text_lengeth )

    end = time.time()
    print("calculate similarity time:", end - start)

    (r1i, r5i, r10i, medri, meanri), _ = utils.acc_i2t2(d)
    logging.info("Image to text: %.1f, %.1f, %.1f, %.1f, %.1f" %
                 (r1i, r5i, r10i, medri, meanri))
    (r1t, r5t, r10t, medrt, meanrt), _ = utils.acc_t2i2(d)
    logging.info("Text to image: %.1f, %.1f, %.1f, %.1f, %.1f" %
                 (r1t, r5t, r10t, medrt, meanrt))
    currscore = (r1t + r5t + r10t + r1i + r5i + r10i)/6.0

    all_score = "r1i:{} r5i:{} r10i:{} medri:{} meanri:{}\n r1t:{} r5t:{} r10t:{} medrt:{} meanrt:{}\n sum:{}\n ------\n".format(
        r1i, r5i, r10i, medri, meanri, r1t, r5t, r10t, medrt, meanrt, currscore
    )
  
    tb_logger.log_value('r1i', r1i, step=model.Eiters)
    tb_logger.log_value('r5i', r5i, step=model.Eiters)
    tb_logger.log_value('r10i', r10i, step=model.Eiters)
    tb_logger.log_value('medri', medri, step=model.Eiters)
    tb_logger.log_value('meanri', meanri, step=model.Eiters)
    tb_logger.log_value('r1t', r1t, step=model.Eiters)
    tb_logger.log_value('r5t', r5t, step=model.Eiters)
    tb_logger.log_value('r10t', r10t, step=model.Eiters)
    tb_logger.log_value('medrt', medrt, step=model.Eiters)
    tb_logger.log_value('meanrt', meanrt, step=model.Eiters)
    tb_logger.log_value('rsum', currscore, step=model.Eiters)

    return currscore, all_score


def validate_test(val_loader, model):
    model.eval()
    val_logger = utils.LogCollector()
    model.logger = val_logger

    start = time.time()
    input_visual = np.zeros((len(val_loader.dataset), 3, 256, 256))
    input_text = np.zeros((len(val_loader.dataset), 47), dtype=np.int64)
    input_text_lengeth = [0] * len(val_loader.dataset)
    for i, val_data in enumerate(val_loader):

        images, captions, lengths, ids = val_data

        for (id, img, cap, key, l) in zip(ids, (images.numpy().copy()), (captions.numpy().copy()), images, lengths):
            input_visual[id] = img
            input_text[id, :captions.size(1)] = cap
            input_text_lengeth[id] = l

    input_visual = np.array([input_visual[i] for i in range(0, len(input_visual), 5)])

    d = utils.shard_dis(input_visual, input_text, model, lengths=input_text_lengeth)

    end = time.time()
    print("calculate similarity time:", end - start)

    return d

def validate_all(loader, model, type = 'val', logger = None):

    model.eval()
    val_logger = utils.LogCollector()
    model.logger = val_logger
    
    logging.info('Start to validate on {} set...'.format(type))

    start = time.time()
    input_visual = np.zeros((len(loader.dataset), 3, 256, 256))
    input_text = np.zeros((len(loader.dataset), 47), dtype=np.int64)
    input_text_lengeth = [0]*len(loader.dataset)
    for i, data in enumerate(loader):

        images, captions, lengths, ids = data
        
        for (id, img, cap, key,l) in zip(ids, (images.numpy().copy()), (captions.numpy().copy()), images , lengths):
            input_visual[id] = img
            input_text[id, :captions.size(1)] = cap
            input_text_lengeth[id] = l


    input_visual = np.array([input_visual[i] for i in range(0, len(input_visual), 5)], dtype=np.float32)

    d = utils.shard_dis(input_visual, input_text, model , lengths=input_text_lengeth )

    end = time.time()
    print("calculate similarity time:", end - start)

    (r1i, r5i, r10i, medri, meanri), _ = utils.acc_i2t2(d)
    logging.info("Image to text: %.1f, %.1f, %.1f, %.1f, %.1f" %
                 (r1i, r5i, r10i, medri, meanri))
    (r1t, r5t, r10t, medrt, meanrt), _ = utils.acc_t2i2(d)
    logging.info("Text to image: %.1f, %.1f, %.1f, %.1f, %.1f" %
                 (r1t, r5t, r10t, medrt, meanrt))
    currscore = (r1t + r5t + r10t + r1i + r5i + r10i)/6.0

    all_score = "r1i:{} r5i:{} r10i:{} medri:{} meanri:{}\n r1t:{} r5t:{} r10t:{} medrt:{} meanrt:{}\n sum:{}\n ------\n".format(
        r1i, r5i, r10i, medri, meanri, r1t, r5t, r10t, medrt, meanrt, currscore
    )
    
    if logger != None:
        logger.log_value('r1i', r1i, step=model.Eiters)
        logger.log_value('r5i', r5i, step=model.Eiters)
        logger.log_value('r10i', r10i, step=model.Eiters)
        logger.log_value('medri', medri, step=model.Eiters)
        logger.log_value('meanri', meanri, step=model.Eiters)
        logger.log_value('r1t', r1t, step=model.Eiters)
        logger.log_value('r5t', r5t, step=model.Eiters)
        logger.log_value('r10t', r10t, step=model.Eiters)
        logger.log_value('medrt', medrt, step=model.Eiters)
        logger.log_value('meanrt', meanrt, step=model.Eiters)
        logger.log_value('rsum', currscore, step=model.Eiters)
    else:
        tb_logger.log_value('r1i', r1i, step=model.Eiters)
        tb_logger.log_value('r5i', r5i, step=model.Eiters)
        tb_logger.log_value('r10i', r10i, step=model.Eiters)
        tb_logger.log_value('medri', medri, step=model.Eiters)
        tb_logger.log_value('meanri', meanri, step=model.Eiters)
        tb_logger.log_value('r1t', r1t, step=model.Eiters)
        tb_logger.log_value('r5t', r5t, step=model.Eiters)
        tb_logger.log_value('r10t', r10t, step=model.Eiters)
        tb_logger.log_value('medrt', medrt, step=model.Eiters)
        tb_logger.log_value('meanrt', meanrt, step=model.Eiters)
        tb_logger.log_value('rsum', currscore, step=model.Eiters)

    return currscore, all_score

def validate_v2(val_loader, model):

    model.eval()
    val_logger = utils.LogCollector()
    model.logger = val_logger

    start = time.time()
    input_visual = np.zeros((len(val_loader.dataset), 3, 256, 256))
    input_text = np.zeros((len(val_loader.dataset), 47), dtype=np.int64)
    input_text_lengeth = [0]*len(val_loader.dataset)
    # for i, val_data in enumerate(val_loader):

    #     images, captions, lengths, ids = val_data
    for i, val_data in enumerate(val_loader):
        # print('train_data: ', train_data)
        tmps, captions, lengths, ids= val_data
        # if len(tmps) > 1:
        if isinstance(tmps, tuple):
            images, targets = tmps
        else:
            images = tmps
        for (id, img, cap, key,l) in zip(ids, (images.numpy().copy()), (captions.numpy().copy()), images , lengths):
            input_visual[id] = img
            input_text[id, :captions.size(1)] = cap
            input_text_lengeth[id] = l


    input_visual = np.array([input_visual[i] for i in range(0, len(input_visual), 5)])

    d = utils.shard_dis(input_visual, input_text, model , lengths=input_text_lengeth )

    end = time.time()
    print("calculate similarity time:", end - start)

    (r1i, r5i, r10i, medri, meanri), _ = utils.acc_i2t2(d)
    logging.info("Image to text: %.1f, %.1f, %.1f, %.1f, %.1f" %
                 (r1i, r5i, r10i, medri, meanri))
    (r1t, r5t, r10t, medrt, meanrt), _ = utils.acc_t2i2(d)
    logging.info("Text to image: %.1f, %.1f, %.1f, %.1f, %.1f" %
                 (r1t, r5t, r10t, medrt, meanrt))


    return [r1i, r5i, r10i, medri, meanri, r1t, r5t, r10t, medrt, meanrt]


def validate_seg(val_loader, model, num_class, epoch):
    model.eval()
    evaluator = Evaluator(num_class)
    # self.evaluator.reset()
    test_loss = 0.0
    
    for i, val_data in enumerate(val_loader):
        # print('train_data: ', train_data)
        tmps, captions, lengths, ids= val_data
        image, target = tmps
        image = Variable(image)
        target = Variable(target)
        input_text = Variable(captions)

        if torch.cuda.is_available():
            image = image.cuda()
            target = target.cuda()
            input_text = input_text.cuda()
        with torch.no_grad():
            (output, _) = model.test(image, input_text, lengths)
        pred = output.data.cpu().numpy()
        target = target.cpu().numpy()
        pred = np.argmax(pred, axis=1)
        pred = pred[:, np.newaxis, :, :]
        # Add batch sample into evaluator
        evaluator.add_batch(target, pred)

    # Fast test during the training
    Acc = evaluator.Pixel_Accuracy()
    Acc_class = evaluator.Pixel_Accuracy_Class()
    mIoU = evaluator.Mean_Intersection_over_Union()
    FWIoU = evaluator.Frequency_Weighted_Intersection_over_Union()
    OverallAccuracy = evaluator.OverallAccuracy()
    F1_Score = evaluator.F1Score()
    # print('OverallAccuracy: ', OverallAccuracy)
    # print('F1_Score: ', F1_Score)
    print('Validation:')
    print("Acc:{:.5f}, Acc_class:{:.5f}, mIoU:{:.5f}, fwIoU:{:.5f}, OA:{:.5f}, F1_Score:{:.5f}".format(Acc, Acc_class, mIoU, FWIoU, OverallAccuracy, F1_Score))
    return {'Acc': Acc, 'Acc_class': Acc_class, 'mIoU': mIoU, 'FWIoU': FWIoU, 'OA': OverallAccuracy, 'F1_Score': F1_Score}
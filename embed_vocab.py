'''
Train the question generation model
'''

import argparse
import math
import time
import random
from tqdm import tqdm
import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
import transformer.Constants as Constants
from dataset import QGDataset, paired_collate_fn
from transformer.Model import Transformer
from transformer.Optim import ScheduledOptim
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
torch.backends.cudnn.enabled = True


def cal_performance(pred, gold, smoothing=False):
    '''

    :param pred:
    :param gold:
    :param smoothing:
    :return:
    '''
    ''' use label smoothing if needed'''

    loss = cal_loss(pred, gold, smoothing)

    pred = pred.max(1)[1]
    # print('pred:', pred[:10])
    gold = gold.contiguous().view(-1)
    non_pad_mask = gold.ne(Constants.PAD)
    n_correct = pred.eq(gold)
    n_correct = n_correct.masked_select(non_pad_mask).sum().item()

    return loss, n_correct, pred


def cal_loss(pred, gold, smoothing):
    '''

    :param pred:
    :param gold:
    :param smoothing:
    :return:
    '''
    ''' Calculate cross entropy loss, apply label smoothing if needed. '''

    gold = gold.contiguous().view(-1)

    if smoothing:
        eps = 0.1
        n_class = pred.size(1)

        one_hot = torch.zeros_like(pred).scatter(1, gold.view(-1, 1), 1)
        one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
        log_prb = F.log_softmax(pred, dim=1)

        non_pad_mask = gold.ne(Constants.PAD)
        loss = -(one_hot * log_prb).sum(dim=1)
        loss = loss.masked_select(non_pad_mask).sum()  # average later
    else:
        loss = F.cross_entropy(pred, gold, ignore_index=Constants.PAD, reduction='sum')

    return loss


def train_epoch(model, training_data, optimizer, device, smoothing):
    '''

    :param model:
    :param training_data:
    :param optimizer:
    :param device:
    :param smoothing:
    :return:
    '''

    ''' Train epoch'''

    model.train()

    total_loss = 0
    n_word_total = 0
    n_word_correct = 0

    for batch in tqdm(
            training_data, mininterval=2,
            desc='  - (Training)   ', leave=False):

        # prepare data
        src1_seq, src1_pos, \
        src2_seq, src2_pos,\
        src3_seq, src3_pos,\
        src4_seq, src4_pos,\
        src1_emo, \
        src2_emo,  \
        src3_emo,  \
        src4_emo,  \
        tgt_seq, tgt_pos = map(lambda x: x.to(device), batch)

        gold = tgt_seq[:, 1:]

        src = src1_seq, src1_pos, src2_seq, src2_pos, src3_seq, src3_pos, src4_seq, src4_pos, \
              src1_emo, src2_emo,  src3_emo,  src4_emo


        # forward
        optimizer.zero_grad()
        pred = model(src, tgt_seq, tgt_pos)

        # backward
        loss, n_correct, prediction = cal_performance(pred, gold, smoothing=smoothing)
        loss.backward()

        # update parameters
        optimizer.step_and_update_lr()

        # note keeping
        total_loss += loss.item()

        non_pad_mask = gold.ne(Constants.PAD)
        n_word = non_pad_mask.sum().item()
        n_word_total += n_word
        n_word_correct += n_correct

    loss_per_word = total_loss/n_word_total
    accuracy = n_word_correct/n_word_total
    return loss_per_word, accuracy


def eval_epoch(model, validation_data, device, opt, epoch_i):
    '''

    :param model:
    :param validation_data:
    :param device:
    :return:
    '''

    ''' Evaluate each epoch'''

    model.eval()

    total_loss = 0
    n_word_total = 0
    n_word_correct = 0

    data = torch.load(opt.data)
    tgt_index2word = {idx: word for word, idx in data['dict']['tgt'].items()}
    word_list = []
    
    with torch.no_grad():
        for batch in tqdm(validation_data, mininterval=2, desc='  - (Validation) ', leave=False):

            # prepare data
            src1_seq, src1_pos, \
            src2_seq, src2_pos, \
            src3_seq, src3_pos, \
            src4_seq, src4_pos, \
            src1_emo, \
            src2_emo,  \
            src3_emo,  \
            src4_emo, \
            tgt_seq, tgt_pos = map(lambda x: x.to(device), batch)

            gold = tgt_seq[:, 1:]

            src = src1_seq, src1_pos, src2_seq, src2_pos, src3_seq, src3_pos, src4_seq, src4_pos, \
                  src1_emo,  src2_emo,  src3_emo,  src4_emo,\

            # forward
            # optimizer.zero_grad()
            pred = model(src, tgt_seq, tgt_pos)

            # tgt_index2word = validation_data.tgt_idx2word()

            loss, n_correct, prediction = cal_performance(pred, gold, smoothing=False)

            # print('batch_prediction:', prediction.shape)
            prediction = prediction.data.cpu().numpy()

            for pred in prediction:
                word = tgt_index2word[pred]
                word_list.append(word)

            # note keeping
            total_loss += loss.item()

            non_pad_mask = gold.ne(Constants.PAD)
            n_word = non_pad_mask.sum().item()
          
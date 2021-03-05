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
from dataset import SEGDataset, paired_collate_fn
from transformer.Model import Transformer
from transformer.Optim import ScheduledOptim
import os



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
            n_word_total += n_word
            n_word_correct += n_correct


    loss_per_word = total_loss/n_word_total
    accuracy = n_word_correct/n_word_total
    return loss_per_word, accuracy


def train(model, training_data, validation_data, optimizer, device, opt, seed):
    '''

    :param model:
    :param training_data:
    :param validation_data:
    :param optimizer:
    :param device:
    :param opt:
    :return:
    '''

    ''' Start training '''

    log_train_file = None
    log_valid_file = None

    if opt.log:
        log_train_file = opt.log + str(seed) + '_train.log'
        log_valid_file = opt.log + str(seed) + '_valid.log'

        print('[Info] Training performance will be written to file: {} and {}'.format(
            log_train_file, log_valid_file))
        if not os.path.exists(opt.log):
            os.mkdir(opt.log)
        with open(log_train_file, 'w') as log_tf, open(log_valid_file, 'w') as log_vf:
            log_tf.write('epoch, loss,ppl, accuracy\n')
            log_vf.write('epoch, loss,ppl, accuracy\n')

    valid_accus = []
    for epoch_i in range(opt.epoch):
        print('[ Epoch', epoch_i, ']')

        start = time.time()
        train_loss, train_accu = train_epoch(
            model, training_data, optimizer, device, smoothing=opt.label_smoothing)
        print('\n')
        print('  - (Training)   ppl: {ppl: 8.5f}, accuracy: {accu:3.3f} lr: {lr:.4f}%, '\
              'elapse: {elapse:3.3f} min'.format(
                  ppl=math.exp(min(train_loss, 100)), accu=100*train_accu,
                  elapse=(time.time()-start)/60, lr=opt.lr))
        start = time.time()
        valid_loss, valid_accu = eval_epoch(model, validation_data, device, opt, epoch_i)
        print('\n')
        print('  - (Validation) ppl: {ppl: 8.5f}, accuracy: {accu:3.3f}  lr: {lr:.4f}%, '\
                'elapse: {elapse:3.3f} min'.format(
                    ppl=math.exp(min(valid_loss, 100)), accu=100*valid_accu,
                    elapse=(time.time()-start)/60, lr=opt.lr))

        valid_accus += [valid_accu]

        model_state_dict = model.state_dict()
        checkpoint = {
            'model': model_state_dict,
            'settings': opt,
            'epoch': epoch_i}

        if opt.save_model:
            if opt.save_mode == 'all':
                model_name = opt.save_model + str(seed) + '_%.3f.ckpt' % math.exp(min(valid_loss, 100))
                torch.save(checkpoint, model_name)
            elif opt.save_mode == 'best':
                model_name = opt.save_model + str(seed) + '_%.3f.ckpt' % math.exp(min(valid_loss, 100))
                # if math.exp(min(valid_loss, 100)) >= max(math.exp(min(valid_loss, 100))):
                #     torch.save(checkpoint, model_name)
                if valid_accu >= max(valid_accus):
                    torch.save(checkpoint, model_name)
                    print('\n')
                    print('    - [Info] The checkpoint file has been updated.')

        if log_train_file and log_valid_file:
            with open(log_train_file, 'a') as log_tf, open(log_valid_file, 'a') as log_vf:
                log_tf.write('{epoch},{loss: 8.5f},{ppl: 8.5f},{accu:3.3f}\n'.format(
                    epoch=epoch_i, loss=train_loss,
                    ppl=math.exp(min(train_loss, 100)), accu=100*train_accu))
                log_vf.write('{epoch},{loss: 8.5f},{ppl: 8.5f},{accu:3.3f}, {lr: .4f}\n'.format(
                    epoch=epoch_i, loss=valid_loss,
                    ppl=math.exp(min(valid_loss, 100)), accu=100*valid_accu, lr=opt.lr))


def main():
    ''' Main function '''
    parser = argparse.ArgumentParser()

    parser.add_argument('-data', type=str, default='data/final_gcn_data')

    parser.add_argument('-epoch', type=int, default=20)
    parser.add_argument('-batch_size', type=int, default=64)
    parser.add_argument('-gpu', type=int, default=0)
    parser.add_argument('-d_word_vec', type=int, default=300)
    parser.add_argument('-d_model', type=int, default=300)
    parser.add_argument('-d_inner', type=int, default=2048)
    parser.add_argument('-d_k', type=int, default=64)
    parser.add_argument('-d_v', type=int, default=64)

    parser.add_argument('-n_head', type=int, default=6)
    parser.add_argument('-n_layers', type=int, default=2)
    parser.add_argument('-n_warmup_steps', type=int, default=400)

    parser.add_argument('-dropout', type=float, default=0.1)
    parser.add_argument('-embs_share_weight', action='store_true')
    parser.add_argument('-proj_share_weight', action='store_true')
    parser.add_argument('-finetune', type=bool, default=True)
    parser.add_argument('-usepretrained', type=bool, default=True)
    parser.add_argument('-use_pretrained_model', type=bool, default=False)
    parser.add_argument('-lr', type=float, default=0.005)

    parser.add_argument('-log', default='log/')
    parser.add_argument('-save_model', default='ckpt/')
    parser.add_argument('-save_mode', type=str, choices=['all', 'best'], default='best')

    parser.add_argument('-no_cuda', action='store_true')
    parser.add_argument('-label_smoothing', action='store_true')

    # gcn setting
    parser.add_argument('-hidden_size', type=int, default=300)
    parser.add_argument('-flat_glimpses', type=int, default=1)

    opt = parser.parse_args()
    opt.cuda = not opt.no_cuda
    opt.d_word_vec = opt.d_word_vec
    os.environ["CUDA_VISIBLE_DEVICES"] = str(opt.gpu)
    torch.backends.cudnn.enabled = True
    #========= Loading Dataset =========#
    data = torch.load(opt.data)
    opt.max_src_token_seq_len = data['settings'].max_src_token_seq_len
    opt.max_tgt_token_seq_len = data['settings'].max_tgt_token_seq_len

    device = torch.device('cuda' if opt.cuda else 'cpu')

    seed = random.randint(0, 999)
    # seed = 1152
    opt.seed = seed

    # load the data from dataset
    training_data, validation_data = prepare_dataloaders(data, opt)

    opt.src_vocab_size = training_data.dataset.src_vocab_size
    opt.tgt_vocab_size = training_data.dataset.tgt_vocab_size

    # initialize the model
    transformer = Transformer(opt, opt.src_vocab_size, opt.tgt_vocab_size).to(device)



    if opt.embs_share_weight:
        assert training_data.dataset.src_word2idx == training_data.dataset.tgt_word2idx, \
            'The src/tgt word2idx table are different but asked to share word embedding.'


    optimizer = ScheduledOptim(optim.Adam(
            filter(lambda x: x.requires_grad, transformer.parameters()),
            betas=(0.9, 0.98), eps=1e-09), opt)

    print(opt)

    train(transformer, training_data, validation_data, optimizer, device, opt, seed)


def prepare_dataloaders(data, opt):
    # ========= Preparing DataLoader =========#
    train_loader = torch.utils.data.DataLoader(
        SEGDataset(
            src_word2idx=data['dict']['src'],
            tgt_word2idx=data['dict']['tgt'],
            src1_insts=data['train']['src1'],
            src2_insts=data['train']['src2'],
            src3_insts=data['train']['src3'],
            src4_insts=data['train']['src4'],
            src1_emo=data['train']['adj1'],
            src2_emo=data['train']['adj2'],
            src3_emo=data['train']['adj3'],
            src4_emo=data['train']['adj4'],
            tgt_insts=data['train']['tgt'][0]),
        num_workers=0,
        batch_size=opt.batch_size,
        collate_fn=paired_collate_fn,
        shuffle=True)

    valid_loader = torch.utils.data.DataLoader(
        SEGDataset(
            src_word2idx=data['dict']['src'],
            tgt_word2idx=data['dict']['tgt'],
            src1_insts=data['valid']['src1'],
            src2_insts=data['valid']['src2'],
            src3_insts=data['valid']['src3'],
            src4_insts=data['valid']['src4'],
            src1_emo=data['valid']['adj1'],
            src2_emo=data['valid']['adj2'],
            src3_emo=data['valid']['adj3'],
            src4_emo=data['valid']['adj4'],
            tgt_insts=data['valid']['tgt'][0]),
        num_workers=0,
        batch_size=opt.batch_size,
        collate_fn=paired_collate_fn)

    return train_loader, valid_loader


if __name__ == '__main__':
    main()

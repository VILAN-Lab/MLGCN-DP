import numpy as np
import torch
import torch.utils.data

from transformer import Constants

def paired_collate_fn(insts):
    src1_insts, src2_insts, src3_insts, src4_insts, \
    scr1_emo, scr2_emo, scr3_emo, scr4_emo, tgt_insts = list(zip(*insts))

    src1_insts = collate_fn(src1_insts, 'src1')
    src2_insts = collate_fn(src2_insts, 'src2')
    src3_insts = collate_fn(src3_insts, 'src3')
    src4_insts = collate_fn(src4_insts, 'src4')

    scr1_emo = collate_fn_emotion(scr1_emo)
    scr2_emo = collate_fn_emotion(scr2_emo)
    scr3_emo = collate_fn_emotion(scr3_emo)
    scr4_emo = collate_fn_emotion(scr4_emo)


    tgt_insts = collate_fn(tgt_insts, 'src4')

    return (*src1_insts, *src2_insts, *src3_insts, *src4_insts, *scr1_emo,
            *scr2_emo, *scr3_emo, *scr4_emo, *tgt_insts)

def collate_fn_emotion(insts):
    ''' Pad the instance to the max seq length in batch '''

    # max_len = max(len(inst) for inst in insts)
    max_len = 1
    # print('insts:', insts)
    batch_seq = np.array([inst for inst in insts])

    # batch_pos = np.array(
    #     [pos_i+1 if w_i != Constants.PAD else 0
    #      for pos_i, w_i in enumerate(batch_seq)])
    # # print(batch_seq)
    batch_seq = torch.LongTensor(batch_seq)
    # batch_pos = torch.LongTensor(batch_pos)

    return batch_seq,

def collate_fn(insts, sent):
    ''' Pad the instance to the max seq length in batch '''

    # max_len = max(len(inst) for inst in insts)
    if sent == 'src1':
        max_len = 20
        batch_seq = np.array([inst[:max_len] + [Constants.PAD] * (max_len - len(inst[:max_len]))
                              for inst in insts])
    elif sent == 'src2':
        max_len = 20
        batch_seq = np.array([inst[:max_len] + [Constants.PAD] * (max_len - len(inst[:max_len])) + [4]
                              for inst in insts])
    elif sent == 'src3':
        max_len = 20
        batch_seq = np.array([inst[:max_len] + [Constants.PAD] * (max_len - len(inst[:max_len])) + [4, 4]
                              for inst in insts])
    else:
        max_len = 20

        batch_seq = np.array([inst[:max_len] + [Constants.PAD] * (max_len - len(inst[:max_len])) + [4, 4, 4]
            for inst in insts])

    batch_pos = np.array([
        [pos_i+1 if w_i != Constants.PAD else 0
         for pos_i, w_i in enumerate(inst)] for inst in batch_seq])
    # print(batch_seq)
    batch_seq = torch.LongTensor(batch_seq)
    batch_pos = torch.LongTensor(batch_pos)

    return batch_seq, batch_pos


class SEGDataset(torch.utils.data.Dataset):
    def __init__(self, src_word2idx, tgt_word2idx,
                 src1_insts=None,
                 src2_insts=None,
                 src3_insts=None,
                 src4_insts=None,
                 src1_emo=None,
                 src2_emo=None,
                 src3_emo=None,
                 src4_emo=None,
                 tgt_insts=None):

        # assert src_insts
        # print(len(src1_insts), len(tgt_insts))
        assert not tgt_insts or (len(src1_insts) == len(tgt_insts))

        src_idx2word = {idx: word for word, idx in src_word2idx.items()}
        self._src_word2idx = src_word2idx
        self._src_idx2word = src_idx2word

        self._src1_insts = src1_insts
        self._src2_insts = src2_insts
        self._src3_insts = src3_insts
        self._src4_insts = src4_insts

        self.src1_emo = src1_emo
        self.src2_emo = src2_emo
        self.src3_emo = src3_emo
        self.src4_emo = src4_emo

        tgt_idx2word = {idx: word for word, idx in tgt_word2idx.items()}
        self._tgt_word2idx = tgt_word2idx
        self._tgt_idx2word = tgt_idx2word
        self._tgt_insts = tgt_insts

    @property
    def n_insts(self):
        ''' Property for dataset size '''
        return len(self._src1_insts)

    @property
    def src_vocab_size(self):
        ''' Property for vocab size '''
        return len(self._src_word2idx)

    @property
    def tgt_vocab_size(self):
        ''' Property for vocab size '''
        return len(self._tgt_word2idx)

    @property
    def src_word2idx(self):
        ''' Property for word dictionary '''
        return self._src_word2idx

    @property
    def tgt_word2idx(self):
        ''' Property for word dictionary '''
        return self._tgt_word2idx

    @property
    def src_idx2word(self):
        ''' Property for index dictionary '''
        return self._src_idx2word

    def tgt_idx2word(self):
        ''' Property for index dictionary '''
        return self._tgt_idx2word

    def __len__(self):
        return self.n_insts

    def __getitem__(self, idx):
        if self._tgt_insts:
            return self._src1_insts[idx],\
                   self._src2_insts[idx],\
                   self._src3_insts[idx],\
                   self._src4_insts[idx],\
                   self.src1_emo[idx],\
                   self.src2_emo[idx],\
                   self.src3_emo[idx],\
                   self.src4_emo[idx],\
                   self._tgt_insts[idx]

        return self._src1_insts[idx], \
               self._src2_insts[idx],\
               self._src3_insts[idx],\
               self._src4_insts[idx], \
               self.src1_emo[idx], \
               self.src2_emo[idx], \
               self.src3_emo[idx], \
               self.src4_emo[idx],\

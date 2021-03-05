'''Generate question from trained model batch by batch '''

import torch
import torch.utils.data
import argparse
from tqdm import tqdm
import numpy as np
from transformer import Constants
# import time
from dataset import SEGDataset
from transformer.Story_Generator import StoryGenerator
# from data_preprocess import read_instances_from_file, convert_instance_to_idx_seq

def paired_collate_fn(insts):
    src1_insts, src2_insts, src3_insts, src4_insts, \
    scr1_emo, scr2_emo, scr3_emo, scr4_emo = list(zip(*insts))

    src1_insts = collate_fn(src1_insts, 'src1')
    src2_insts = collate_fn(src2_insts, 'src2')
    src3_insts = collate_fn(src3_insts, 'src3')
    src4_insts = collate_fn(src4_insts, 'src4')

    scr1_emo = collate_fn_emotion(scr1_emo)
    scr2_emo = collate_fn_emotion(scr2_emo)
    scr3_emo = collate_fn_emotion(scr3_emo)
    scr4_emo = collate_fn_emotion(scr4_emo)


    return (*src1_insts, *src2_insts, *src3_insts, *src4_insts, *scr1_emo,
            *scr2_emo, *scr3_emo, *scr4_emo)

def collate_fn_emotion(insts):


    batch_seq = np.array([inst for inst in insts])

    batch_seq = torch.LongTensor(batch_seq)

    return batch_seq,

def collate_fn(insts, sent):
    ''' Pad the instance to the max seq length in batch '''


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




def main():
    '''Main Function'''

    parser = argparse.ArgumentParser(description='generate_story.py')

    parser.add_argument('-model', type=str, default='ckpt/1152_17.149.ckpt',
                        help='Path to model .pt file')
    # parser.add_argument('-src', type=str, default='data/test/test.post',
    #                     help='Source sequence to decode (one line per sequence)')
    parser.add_argument('-data', type=str, default='data/final_gcn_data',
                        help='Source sequence to decode (one line per sequence)')
    parser.add_argument('-output', default='story_generation/',
                        help="""Path to output the predictions (each line will
                        be the decoded sequence""")
    parser.add_argument('-beam_size', type=int, default=5,
                        help='Beam size')
    parser.add_argument('-finetune', type=bool, default=True)
    parser.add_argument('-usepretrained', type=bool, default=True)
    parser.add_argument('-batch_size', type=int, default=5,
                        help='Batch size')
    parser.add_argument('-n_best', type=int, default=1,
                        help="""If verbose is set, will output the n_best
                        decoded sentences""")
    parser.add_argument('-no_cuda', action='store_true')
    parser.add_argument('-gpu',type=int, default=0, help='Choose which gpu')
    parser.add_argument('-hidden_size', type=int, default=300)
    parser.add_argument('-flat_glimpses', type=int, default=1)

    opt = parser.parse_args()
    opt.cuda = not opt.no_cuda
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = str(opt.gpu)
    torch.backends.cudnn.enabled = True
    preprocess_data = torch.load(opt.data)
    tgt_vocab = preprocess_data['dict']['tgt']
    tgt_idx2word = {idx: word for word, idx in tgt_vocab.items()}
    print('loading the dataset...')
    test_loader = torch.utils.data.DataLoader(
        SEGDataset(
            src_word2idx=preprocess_data['dict']['src'],
            tgt_word2idx=preprocess_data['dict']['tgt'],
            src1_insts=preprocess_data['test']['src1'],
            src2_insts=preprocess_data['test']['src2'],
            src3_insts=preprocess_data['test']['src3'],
            src4_insts=preprocess_data['test']['src4'],
            src1_emo=preprocess_data['test']['adj1'],
            src2_emo=preprocess_data['test']['adj2'],
            src3_emo=preprocess_data['test']['adj3'],
            src4_emo=preprocess_data['test']['adj4']),
        num_workers=0,
        batch_size=opt.batch_size,
        collate_fn=paired_collate_fn)
    # seed = 1151
    seed = opt.model.split("/")[1].split("_")[0]
    print('seed = ', seed)
    q_gen = StoryGenerator(opt)
    if not os.path.exists(opt.output ):
        os.mkdir(opt.output)
    with open(opt.output + str(seed), 'w') as f:
        for batch in tqdm(test_loader):
            all_hyp, all_scores = q_gen.generate_question_batch(*batch)
            for idx_seqs in all_hyp:
                for idx_seq in idx_seqs:
                    new_pred = []
                    for idx in idx_seq:
                        word = tgt_idx2word[idx]
                        if word in ['</s>']:  #'.</s>', '!</s>',
                            continue
                        else:
                            new_pred.append(word)
                    pred_line = ' '.join([word for word in new_pred])
                    if pred_line[-1] == "!":
                        f.write(pred_line.lower() + ' \n')
                    else:
                        f.write(pred_line.lower() + ' .\n')                    
    print('[Info] Finished.')


if __name__ == "__main__":
    main()

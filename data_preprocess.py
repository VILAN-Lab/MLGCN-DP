''' preprocess data '''
import argparse
import torch
import transformer.Constants as Constants
import json
from relation_to_adj_matrix import tree_to_adj, head_to_tree
from tqdm import tqdm
import numpy as np


def read_instances_from_file(inst_file, max_sent_len, keep_case):
    ''' Convert file into word seq lists and vocab '''

    word_insts = []
    trimmed_sent_count = 0
    with open(inst_file, encoding='latin') as f:
        for sents in f:
            if not keep_case:
                sents = sents.lower()
            sents = sents.strip().split("\t")

            word_list = []
            for words in sents:
                if len(words) > max_sent_len:
                    trimmed_sent_count += 1
                word_inst = words[:max_sent_len]

                if word_inst:
                    word_list += [Constants.BOS_WORD + ' ' + word_inst + ' ' + Constants.EOS_WORD]
                # else:
                #     word_list += [None]

            word_insts.append(word_list)
    print('[Info] Get {} instances from {}'.format(len(word_insts), inst_file))

    if trimmed_sent_count > 0:
        print('[Warning] {} instances are trimmed to the max sentence length {}.'
              .format(trimmed_sent_count, max_sent_len))
    # print(word_insts)
    # print('word_insts[0]', word_insts[0])
    return word_insts


def build_vocab_idx(word_insts, min_word_count, max_word_count):
    ''' Trim vocab by number of occurence '''
    print('word_inst[0]', word_insts[99])

    # full_vocab = set(w for sent in word_insts for w in sent[0].split())
    # print(full_vocab[0])
    # print('[Info] Original Vocabulary size =', len(full_vocab))

    word2idx = {
        Constants.BOS_WORD: Constants.BOS,
        Constants.EOS_WORD: Constants.EOS,
        Constants.PAD_WORD: Constants.PAD,
        Constants.UNK_WORD: Constants.UNK}

    word_count = {}

    for sent in word_insts:
        for s in sent[:4]:
            for word in s.split():
                if word in word_count:
                    word_count[word] += 1
                else:
                    word_count[word] = 1

    ignored_word_count = 0
    for word, count in word_count.items():
        if word not in word2idx:
            if count > min_word_count:
                word2idx[word] = len(word2idx)
            else:
                ignored_word_count += 1

    print('[Info] Trimmed vocabulary size = {},'.format(len(word2idx)),
          'each with minimum occurrence = {}'.format(min_word_count))
    print("[Info] Ignored word count = {}".format(ignored_word_count))

    # print(word2idx)

    return word2idx


def convert_instance_to_idx_seq(word_insts, word2idx):
    ''' Mapping words to idx sequence. '''
    first = []
    second = []
    third = []
    four = []
    # print(word_insts[0])
    for sents in word_insts:
        # print('sents:', len(sents))
        # if len(sents) != 4:
        #     print(sents)
        #     break
        for i, s in enumerate(sents):
            sent_list = []
            s = s.replace(' .', '').replace("'", '')
            for w in s.split(' '):
                if w in word2idx:
                    sent_list.append(word2idx[w])
                else:
                    continue
            if i == 0:
                first.append(sent_list)
            elif i == 1:
                second.append(sent_list)
            elif i == 2:
                third.append(sent_list)
            elif i == 3:
                four.append(sent_list)
            else:
                continue
    #
    # print('1', len(first))
    # print('2', len(second))
    # print('3', len(third))
    # print('4', len(four))

    return first, second, third, four


def convert_emotion_to_idx_seq(all_emotion, word2idx):
    first = []
    second = []
    third = []
    four = []
    for i, emotion_score in enumerate(all_emotion):
        for emo in emotion_score:
            if i == 0:
                first.append(word2idx[emo])
            elif i == 1:
                second.append(word2idx[emo])
            elif i == 2:
                third.append(word2idx[emo])
            elif i == 3:
                four.append(word2idx[emo])
            else:
                continue

    return first, second, third, four


def convert_topic_to_idx_seq(word_insts, word2idx):
    ''' Mapping words to idx sequence. '''
    return [[word2idx.get(w, Constants.UNK) for w in s[1:-1]] for s in word_insts]

# def read_emotion_file(path):
#     with open(path, 'r') as f:
#         lines = f.readlines()
#         emotion_score = []
#         for line in lines:
#             line = line.strip()
#             emotion_score.append(line[2:-2])
#     print(emotion_score[:5])
#     return emotion_score

def relation_to_adj_matrix(relation, name, sent):
    # print(relation)
    if name == 'train':

        head = head_to_tree(relation)

        if sent == 'sent1':
            adj_mat = tree_to_adj(20, head, sent)
        elif sent == 'sent2':
            adj_mat = tree_to_adj(21, head, sent)
        elif sent == 'sent3':
            adj_mat = tree_to_adj(22, head, sent)
        else:
            adj_mat = tree_to_adj(23, head, sent)

    elif name == 'val':

        head = head_to_tree(relation)

        if sent == 'sent1':
            adj_mat = tree_to_adj(20, head, sent)
        elif sent == 'sent2':
            adj_mat = tree_to_adj(21, head, sent)
        elif sent == 'sent3':
            adj_mat = tree_to_adj(22, head, sent)
        else:
            adj_mat = tree_to_adj(23, head, sent)
    else:

        head = head_to_tree(relation)

        if sent == 'sent1':
            adj_mat = tree_to_adj(20, head, sent)
        elif sent == 'sent2':
            adj_mat = tree_to_adj(21, head, sent)
        elif sent == 'sent3':
            adj_mat = tree_to_adj(22, head, sent)
        else:
            adj_mat = tree_to_adj(23, head, sent)

    return adj_mat


def main():
    ''' Main function '''

    parser = argparse.ArgumentParser()
    parser.add_argument('-train_src', type=str, default='data/train/train.post')
    parser.add_argument('-train_tgt', type=str, default='data/train/train.response')
    parser.add_argument('-valid_src', type=str, default='data/val/val.post')
    parser.add_argument('-valid_tgt', type=str, default='data/val/val.response')
    parser.add_argument('-test_src',  type=str, default='data/test/test.post')
    parser.add_argument('-test_tgt',  type=str, default='data/test/test.response')

    parser.add_argument('-save_data', type=str, default='data/final_gcn_data')
    #parser.add_argument('-max_len', '--max_word_seq_len', type=int, default=700)
    parser.add_argument('-max_src_len', '--max_src_word_seq_len', type=int, default=200)
    parser.add_argument('-max_tgt_len', '--max_tgt_word_seq_len', type=int, default=60)

    parser.add_argument('-max_word_count', type=int, default=1000000)
    parser.add_argument('-min_word_count', type=int, default=0)
    parser.add_argument('-keep_case', action='store_true')
    parser.add_argument('-share_vocab', action='store_true')
    parser.add_argument('-vocab', default=None)

    opt = parser.parse_args()
    opt.max_src_token_seq_len = opt.max_src_word_seq_len + 2  # include the <s> and </s>
    opt.max_tgt_token_seq_len = opt.max_tgt_word_seq_len + 2  # include the <s> and </s>

    # Training set
    train_src_word_insts = read_instances_from_file(
        opt.train_src, opt.max_src_word_seq_len, opt.keep_case)
    #
    train_tgt_word_insts = read_instances_from_file(
        opt.train_tgt, opt.max_tgt_word_seq_len, opt.keep_case)

    if len(train_src_word_insts) != len(train_tgt_word_insts):
        print('[Warning] The training instance count is not equal.')
        min_inst_count = min(len(train_src_word_insts), len(train_tgt_word_insts))
        train_src_word_insts = train_src_word_insts[:min_inst_count]
        train_tgt_word_insts = train_tgt_word_insts[:min_inst_count]
    #
    #- Remove empty instances
    train_src_word_insts, train_tgt_word_insts = list(zip(*[
        (s, t) for s, t in zip(train_src_word_insts, train_tgt_word_insts) if s and t]))

    # Validation set
    valid_src_word_insts = read_instances_from_file(
        opt.valid_src, opt.max_src_word_seq_len, opt.keep_case)
    valid_tgt_word_insts = read_instances_from_file(
        opt.valid_tgt, opt.max_tgt_word_seq_len, opt.keep_case)

    if len(valid_src_word_insts) != len(valid_tgt_word_insts):
        print('[Warning] The validation instance count is not equal.')
        min_inst_count = min(len(valid_src_word_insts), len(valid_tgt_word_insts))
        valid_src_word_insts = valid_src_word_insts[:min_inst_count]
        valid_tgt_word_insts = valid_tgt_word_insts[:min_inst_count]

    #- Remove blank or empty  instances
    valid_src_word_insts, valid_tgt_word_insts = list(zip(*[
        (s, t) for s, t in zip(valid_src_word_insts, valid_tgt_word_insts) if s and t]))

    # test set
    test_src_word_insts = read_instances_from_file(opt.test_src, opt.max_src_word_seq_len, opt.keep_case)
    test_tgt_word_insts = read_instances_from_file(opt.test_tgt, opt.max_tgt_word_seq_len, opt.keep_case)
    #
    if len(test_src_word_insts) != len(test_tgt_word_insts):
        print('[Warning] The validation instance count is not equal.')
        min_inst_count = min(len(test_src_word_insts), len(test_tgt_word_insts))
        test_src_word_insts = test_src_word_insts[:min_inst_count]
        test_tgt_word_insts = test_tgt_word_insts[:min_inst_count]

    # - Remove blank or empty  instances
    test_src_word_insts, test_tgt_word_insts = list(zip(*[
        (s, t) for s, t in zip(test_src_word_insts, test_tgt_word_insts) if s and t]))

    # Build source and target vocabulary

    if opt.vocab:
        predefined_data = torch.load(opt.vocab)
        assert 'dict' in predefined_data

        print('[Info] Pre-defined vocabulary found.')
        src_word2idx = predefined_data['dict']['src']
        tgt_word2idx = predefined_data['dict']['tgt']
    else:
        if opt.share_vocab:
            print('[Info] Build shared vocabulary for source and target.')
            word2idx = build_vocab_idx(
                train_src_word_insts + train_tgt_word_insts, opt.min_word_count)
            src_word2idx = tgt_word2idx = word2idx
        else:
            print('[Info] Build vocabulary for source.')
            src_word2idx = build_vocab_idx(train_src_word_insts + valid_src_word_insts ,
                                          opt.min_word_count, opt.max_word_count)
            print('[Info] Build vocabulary for target.')
            tgt_word2idx = build_vocab_idx(valid_tgt_word_insts + train_tgt_word_insts ,
                                           1, 100000)

    # print('src_word2idx:', src_word2idx)
    # print('tgt_word2idx:', tgt_word2idx)
    # word to index
    print('[Info] Convert source word instances into sequences of word index.')
    train_src_insts = convert_instance_to_idx_seq(train_src_word_insts, src_word2idx)
    valid_src_insts = convert_instance_to_idx_seq(valid_src_word_insts, src_word2idx)
    test_src_insts = convert_instance_to_idx_seq(test_src_word_insts, src_word2idx)


    print('[Info] Convert target word instances into sequences of word index.')
    train_tgt_insts = convert_instance_to_idx_seq(train_tgt_word_insts, tgt_word2idx)
    valid_tgt_insts = convert_instance_to_idx_seq(valid_tgt_word_insts, tgt_word2idx)
    test_tgt_insts = convert_instance_to_idx_seq(test_tgt_word_insts, tgt_word2idx)


    train_relation = json.load(open('data/dependency_relation_train.json', 'r'))
    val_relation = json.load(open('data/dependency_relation_val.json', 'r'))
    test_relation = json.load(open('data/dependency_relation_test.json', 'r'))


    print(len(train_relation), len(val_relation), len(test_relation))
    relation = [train_relation, val_relation, test_relation]
    # relation = [test_relation]
    mode = ['train', 'val', 'test']

    # print('test:', test_relation['sent1'][252][0],test_relation['sent2'][252][0]
    #       ,test_relation['sent3'][252][0],test_relation['sent4'][252][0])

    train_adj, val_adj, test_adj = [], [], []

    for data, name in zip(relation, mode):
        adj1, adj2, adj3, adj4 = [], [], [], []

        for sent1, sent2, sent3, sent4 in tqdm(zip(data['sent1'], data['sent2'], data['sent3'], data['sent4'])):
            # print('sent1:', sent1[0])
            adj1.append(relation_to_adj_matrix(sent1[0], name, 'sent1'))
            adj2.append(relation_to_adj_matrix(sent2[0], name, 'sent2'))
            adj3.append(relation_to_adj_matrix(sent3[0], name, 'sent3'))
            adj4.append(relation_to_adj_matrix(sent4[0], name, 'sent4'))

            # adj1.append(np.ones([20,20],dtype=np.float))
            # adj2.append(np.ones([20,20],dtype=np.float))
            # adj3.append(np.ones([20,20],dtype=np.float))
            # adj4.append(np.ones([20,20],dtype=np.float))
        if name == 'train':
            train_adj.append(adj1)
            train_adj.append(adj2)
            train_adj.append(adj3)
            train_adj.append(adj4)

        elif name == 'val':
            val_adj.append(adj1)
            val_adj.append(adj2)
            val_adj.append(adj3)
            val_adj.append(adj4)

        else:
            test_adj.append(adj1)
            test_adj.append(adj2)
            test_adj.append(adj3)
            test_adj.append(adj4)

    print('train_adj:', len(train_adj), train_adj[0][0])
    print(len(val_adj))
    print(len(test_adj))

    data = {
        'settings': opt,
        'dict': {
            'src': src_word2idx,
            'tgt': tgt_word2idx},
        'train': {
            'src1': train_src_insts[0],
            'src2': train_src_insts[1],
            'src3': train_src_insts[2],
            'src4': train_src_insts[3],
            'adj1': train_adj[0],
            'adj2': train_adj[1],
            'adj3': train_adj[2],
            'adj4': train_adj[3],
            'tgt': train_tgt_insts},


        'valid': {
            'src1': valid_src_insts[0],
            'src2': valid_src_insts[1],
            'src3': valid_src_insts[2],
            'src4': valid_src_insts[3],
            'adj1': val_adj[0],
            'adj2': val_adj[1],
            'adj3': val_adj[2],
            'adj4': val_adj[3],
            'tgt': valid_tgt_insts},


        'test': {
            'src1': test_src_insts[0],
            'src2': test_src_insts[1],
            'src3': test_src_insts[2],
            'src4': test_src_insts[3],
            'adj1': test_adj[0],
            'adj2': test_adj[1],
            'adj3': test_adj[2],
            'adj4': test_adj[3],
            'tgt': test_tgt_insts},
        }

    print('[Info] Dumping the processed data to pickle file', opt.save_data)
    torch.save(data, opt.save_data)
    print('[Info] Finish.')


if __name__ == '__main__':
    main()

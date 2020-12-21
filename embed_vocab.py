from __future__ import division
''' @author vishwajeet'''

''' Get 300 dimension embedding vector for each word in vocab'''


import six
import sys
import numpy as np
import argparse
import torch
from utils.logging import logger

def get_vocabs(dict_path):
    '''

    :param dict_path:
    :return:
    '''
    fields = torch.load(dict_path)['dict']
    # print(fields.keys())  dict_keys(['src', 'tgt'])
    vocs = []
    for side in ['src', 'tgt']:
        # if _old_style_vocab(fields):
        #     vocab = next((v for n, v in fields if n == side), None)
        # else:
        try:
            vocab = fields[side]
            print(vocab)
        except AttributeError:
            vocab = fields[side]

        vocs.append(vocab)

    print(len(vocab))

    enc_vocab, dec_vocab = vocs[0], vocs[1]

    print("From: %s" % dict_path)
    print("\t* source vocab: %d words" % len(enc_vocab))
    print("\t* target vocab: %d words" % len(dec_vocab))

    return enc_vocab, dec_vocab


def read_embeddings(file_enc, skip_lines=0):
    '''

    :param file_enc:
    :param skip_lines:
    :return:
    '''
    embs = dict()
    with open(file_enc, 'rb') as f:
        for i, line in enumerate(f):
            if i < skip_lines:
                continue
            if not line:
                break
            if len(line) == 0:
                # is this reachable?
                continue

            l_split = line.decode('utf8').strip().split(' ')
            if len(l_split) == 2:
                continue
            embs[l_split[0]] = [float(em) for em in l_split[1:]]
    return embs


def match_embeddings(vocab, emb, opt):
    '''

    :param vocab:
    :param emb:
    :param opt:
    :return:
    '''
    dim = len(six.next(six.itervalues(emb)))
    print('embedding_dim:', dim)

    filtered_embeddings = np.zeros((len(vocab), dim))
    count = {"match": 0, "miss": 0}
    for w, w_id in vocab.items():
        if w in emb:
            filtered_embeddings[w_id] = emb[w]
            count['match'] += 1
        else:
            if opt.verbose:
                logger.info(u"not found:\t{}".format(w), file=sys.stderr)
            count['miss'] += 1

    return torch.Tensor(filtered_embeddings), count


def main():

    parser = argparse.ArgumentParser(description='embeddings_to_torch.py')
    parser.add_argument('-emb_file_enc', type=str, default='data/embedding/glove.6B.300d.txt',
                        help="source Embeddings from this file")
    parser.add_argument('-emb_file_dec', type=str, default='data/embedding/glove.6B.300d.txt',
                        help="target Embeddings from this file")
    parser.add_argument('-output_file', type=str, default='data/embedding/embedding',
                        help="Output file for the prepared data")
    parser.add_argument('-dict_file', type=str, default='data/final_gcn_data',
                        help="Dictionary file")
    parser.add_argument('-verbose', action="store_true", default=False)
    parser.add_argument('-skip_lines', type=int, default=0,
                        help="Skip first lines of the embedding file")
    parser.add_argument('-type', choices=["GloVe", "word2vec"],
                        default="GloVe")
    opt = parser.parse_args()

    enc_vocab, dec_vocab = get_vocabs(opt.dict_file)

    skip_lines = 1 if opt.type == "word2vec" else opt.skip_lines
    src_vectors = read_embeddings(opt.emb_file_enc, skip_lines)
    logger.info("Got {} encoder embeddings from {}".format(
        len(src_vectors), opt.emb_file_enc))

    tgt_vectors = read_embeddings(opt.emb_file_dec)
    logger.info("Got {} decoder embeddings from {}".format(
        len(tgt_vectors), opt.emb_file_dec))

    filtered_enc_embeddings, enc_count = match_embeddings(
        enc_vocab, src_vectors, opt)
    filtered_dec_embeddings, dec_count = match_embeddings(
        dec_vocab, tgt_vectors, opt)
    logger.info("\nMatching: ")
    match_percent = [_['match'] / (_['match'] + _['miss']) * 100
                     for _ in [enc_count, dec_count]]
    logger.info("\t* enc: %d match, %d missing, (%.2f%%)"
                % (enc_count['match'],
                   enc_count['miss'],
                   match_percent[0]))
    logger.info("\t* dec: %d match, %d missing, (%.2f%%)"
                % (dec_count['match'],
                   dec_count['miss'],
                   match_percent[1]))

    logger.info("\nFiltered embeddings:")
    logger.info("\t* enc: %s" % str(filtered_enc_embeddings.size()))
    logger.info("\t* dec: %s" % str(filtered_dec_embeddings.size()))

    enc_output_file = opt.output_file + "_enc.pt"
    dec_output_file = opt.output_file + "_dec.pt"
    logger.info("\nSaving embedding as:\n\t* enc: %s\n\t* dec: %s"
                % (enc_output_file, dec_output_file))
    torch.save(filtered_enc_embeddings, enc_output_file)
    torch.save(filtered_dec_embeddings, dec_output_file)
    logger.info("\nDone.")


if __name__ == "__main__":
    #init_logger('embeddings_to_torch.log')
    main()

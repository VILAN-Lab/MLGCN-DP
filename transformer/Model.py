''' Define the Transformer model '''
import torch
import torch.nn as nn
import numpy as np
import transformer.Constants as Constants
from transformer.Layers import EncoderLayer, DecoderLayer
# from utils.util import load_embeddings
import torch
import torch.nn as nn
from torch.nn.utils.weight_norm import weight_norm
from transformer.hiarerchical_GCN_for_SEG import GCN_Module
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.ticker as ticker

def get_non_pad_mask(seq):
    assert seq.dim() == 2
    return seq.ne(Constants.PAD).type(torch.float).unsqueeze(-1)

def get_sinusoid_encoding_table(n_position, d_hid, padding_idx=None):
    ''' Sinusoid position encoding table '''

    def cal_angle(position, hid_idx):
        return position / np.power(10000, 2 * (hid_idx // 2) / d_hid)

    def get_posi_angle_vec(position):
        return [cal_angle(position, hid_j) for hid_j in range(d_hid)]


    sinusoid_table = np.array([get_posi_angle_vec(pos_i) for pos_i in range(n_position)])

    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    if padding_idx is not None:
        # zero vector for padding dimension
        sinusoid_table[padding_idx] = 0.

    return torch.FloatTensor(sinusoid_table)

def get_attn_key_pad_mask(seq_k, seq_q):
    ''' For masking out the padding part of key sequence. '''

    # Expand to fit the shape of key query attention matrix.
    len_q = seq_q.size(1)

    padding_mask = seq_k.eq(Constants.PAD)
    padding_mask = padding_mask.unsqueeze(1).expand(-1, len_q, -1)  # b x lq x lk

    return padding_mask

def get_subsequent_mask(seq):
    ''' For masking out the subsequent info. '''

    sz_b, len_s = seq.size()
    subsequent_mask = torch.triu(
        torch.ones((len_s, len_s), device=seq.device, dtype=torch.uint8), diagonal=1)
    subsequent_mask = subsequent_mask.unsqueeze(0).expand(sz_b, -1, -1)  # b x ls x ls

    return subsequent_mask.eq(1)


class Encoder(nn.Module):
    ''' A encoder model with self attention mechanism. '''

    def __init__(
            self, opt,
            n_src_vocab, src_len_max_seq, d_word_vec,
            n_layers, n_head, d_k, d_v,
            d_model, d_inner, dropout=0.1):

        super(Encoder, self).__init__()
        self.opt = opt
        n_position = src_len_max_seq + 1

        self.src_word_emb = nn.Embedding(
            n_src_vocab, d_word_vec, padding_idx=Constants.PAD)
        print(d_word_vec)



        print('usepretrained:', opt.usepretrained)
        if opt.usepretrained:
            src_embed = torch.load('data/embedding_enc.pt')

            self.src_word_emb.weight = nn.Parameter(src_embed)
            self.src_word_emb.weight.requires_grad = opt.finetune

        self.position_enc = nn.Embedding.from_pretrained(
            get_sinusoid_encoding_table(n_position, d_word_vec, padding_idx=0),
            freeze=True)

        self.layer_stack = nn.ModuleList([
            EncoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
            for _ in range(n_layers)])

        self.fc = nn.Linear(d_word_vec + d_model, d_model)

        self.attention = NewAttention(d_model, d_model, d_model)

    def forward(self, src_seq, src_pos, src_emo=None, enc_out=None, return_attns=False):

        enc_slf_attn_list = []

        # -- Prepare masks
        slf_attn_mask = get_attn_key_pad_mask(seq_k=src_seq, seq_q=src_seq)
        non_pad_mask = get_non_pad_mask(src_seq)
        # print(self.src_word_emb(src_seq).size())

        src_emb = self.src_word_emb(src_seq)
        src_emo = self.src_word_emb(src_emo)

        # -- Forward
        if enc_out is None:
            src_emb = src_emb
        else:
            src_emb = self.fc(torch.cat((src_emb, enc_out), dim=-1))

        att = self.attention(src_emb, src_emo)
        src_emb = att * src_emb

        enc_output = src_emb + self.position_enc(src_pos)

        for enc_layer in self.layer_stack:
            enc_output, enc_slf_attn = enc_layer(
                enc_output,
                non_pad_mask=non_pad_mask,
                slf_attn_mask=slf_attn_mask)
            if return_attns:
                enc_slf_attn_list += [enc_slf_attn]

        if return_attns:
            return enc_output, enc_slf_attn_list
        return enc_output,

# def visual_decoder(z, attn):
#
#     # variables = [q for q in que]
#     # labels = variables[0]
#     #
#     attn = attn.cpu().numpy()
#     print('z = %d' % z, 'layers = %d' % len(attn), 'n words = %d' % len(attn[0]))
#
#     src_1 = 'Henry received an overdraft alert from his credit card company'
#     src_2 = 'There was a transaction for $400 dollars in his account'
#     src_3 = 'Henry called to report an unauthorized payment'
#     src_4 = "The representative told him that it was for a speeding ticket"
#     src_raw_context = src_1 + ' ' + src_2
#     length = len(src_raw_context.split())
#     print('length:', length, len(attn))
#     # print(attn.size())
#
#     for k, att in enumerate(attn, 0):  # (80 * 80)
#
#
#         print('z/k = %d/%d' % (z, k))
#         print('att:', att.shape)  # (1, 19)
#         print("att:", att)
#         df = pd.DataFrame(att, columns=src_raw_context.split())
#
#         fig = plt.figure()
#
#         ax = fig.add_subplot(111)
#
#         cax = ax.matshow(df, interpolation='nearest', cmap='hot_r')
#         fig.colorbar(cax)
#
#         tick_spacing = 1
#         ax.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
#         ax.yaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
#
#         # ax.set_xticklabels([''] + list(df.columns))
#         # ax.set_yticklabels([''] + list(df.index))
#         # fontdict = {'rotation': 'vertical'}    #设置文字旋转
#         fontdict = {'rotation': 90}  # 或者这样设置文字旋转
#         # ax.set_xticklabels([''] + list(df.columns), rotation=90)  #或者直接设置到这里
#         # Axes.set_xticklabels(labels, fontdict=None, minor=False, **kwargs)
#         ax.set_xticklabels([''] + list(df.columns), fontdict=fontdict)
#         ax.set_yticklabels([''] + list(df.index))
#
#         plt.show()

class Decoder(nn.Module):
    ''' A decoder model with self attention mechanism. '''

    def __init__(
            self, opt,
            n_tgt_vocab, tgt_len_max_seq, d_word_vec,
            n_layers, n_head, d_k, d_v,
            d_model, d_inner, dropout=0.1):

        super(Decoder, self).__init__()
        n_position = tgt_len_max_seq + 1

        self.tgt_word_emb = nn.Embedding(
            n_tgt_vocab, d_word_vec, padding_idx=Constants.PAD)

        if opt.usepretrained:
            src_embed = torch.load('data/embedding/embedding_dec.pt')

            self.tgt_word_emb.weight = nn.Parameter(src_embed)
            self.tgt_word_emb.weight.requires_grad = opt.finetune

        self.position_enc = nn.Embedding.from_pretrained(
            get_sinusoid_encoding_table(n_position, d_word_vec, padding_idx=0),
            freeze=True)

        self.layer_stack = nn.ModuleList([
            DecoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
            for _ in range(n_layers)])

        # self.linear = nn.Linear(200, 512)


    def forward(self, tgt_seq, tgt_pos, src_seq, enc_output, return_attns=False):

        dec_slf_attn_list, dec_enc_attn_list = [], []

        # -- Prepare masks
        non_pad_mask = get_non_pad_mask(tgt_seq)

        slf_attn_mask_subseq = get_subsequent_mask(tgt_seq)
        slf_attn_mask_keypad = get_attn_key_pad_mask(seq_k=tgt_seq, seq_q=tgt_seq)
        slf_attn_mask = (slf_attn_mask_keypad + slf_attn_mask_subseq).gt(0)

        dec_enc_attn_mask = get_attn_key_pad_mask(seq_k=src_seq, seq_q=tgt_seq)

        # -- Forward
        dec_output = self.tgt_word_emb(tgt_seq) + self.position_enc(tgt_pos)

        for dec_layer in self.layer_stack:
            dec_output, dec_slf_attn, dec_enc_attn = dec_layer(
                dec_output, enc_output,
                non_pad_mask=non_pad_mask,
                slf_attn_mask=slf_attn_mask,
                dec_enc_attn_mask=dec_enc_attn_mask)
            # print("decoder attention weight")
            # print("g",g)
            # visual_decoder(g, dec_enc_attn)


            if return_attns:
                dec_slf_attn_list += [dec_slf_attn]
                dec_enc_attn_list += [dec_enc_attn]

        if return_attns:
            return dec_output, dec_slf_attn_list, dec_enc_attn_list
        return dec_output,

class FCNet(nn.Module):
    """Simple class for non-linear fully connect network
    """
    def __init__(self, dims):
        super(FCNet, self).__init__()

        layers = []
        for i in range(len(dims)-2):
            in_dim = dims[i]
            out_dim = dims[i+1]
            layers.append(weight_norm(nn.Linear(in_dim, out_dim), dim=None))
            layers.append(nn.ReLU())
        layers.append(weight_norm(nn.Linear(dims[-2], dims[-1]), dim=None))
        layers.append(nn.ReLU())

        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)


class NewAttention(nn.Module):
    def __init__(self, v_dim, q_dim, num_hid, dropout=0.2):
        super(NewAttention, self).__init__()

        self.v_proj = FCNet([v_dim, num_hid])
        self.q_proj = FCNet([q_dim, num_hid])
        self.dropout = nn.Dropout(dropout)
        self.linear = weight_norm(nn.Linear(q_dim, 1), dim=None)

    def forward(self, v, q):
        """
        v: [batch, k, vdim]
        q: [batch, qdim]
        """
        logits = self.logits(v, q)
        w = nn.functional.softmax(logits, 1)
        return w

    def logits(self, v, q):
        batch, k, _ = v.size()
        v_proj = self.v_proj(v)  # [batch, k, qdim]
        q_proj = self.q_proj(q).unsqueeze(1).repeat(1, k, 1)
        joint_repr = v_proj * q_proj
        joint_repr = self.dropout(joint_repr)
        logits = self.linear(joint_repr)
        return logits

class Attention(nn.Module):
    def __init__(self, v_dim, q_dim, num_hid, dropout=0.2):
        super(Attention, self).__init__()

        self.v_proj = FCNet([v_dim, num_hid])
        self.q_proj = FCNet([q_dim, num_hid])
        self.dropout = nn.Dropout(dropout)
        self.linear = weight_norm(nn.Linear(q_dim, 1), dim=None)
        self.lstm = nn.LSTM(v_dim, v_dim, 1, batch_first=True)

    def forward(self, v, q):
        """
        v: [batch, k, vdim]
        q: [batch, qdim]
        """
        logits = self.logits(v, q)
        w = nn.functional.softmax(logits, 1)
        return w

    def logits(self, v, q):
        batch, k, _ = v.size()
        v_proj = self.v_proj(v)
        q, _ = self.lstm(q)  # torch.Size([64, 20, 300])
        # q[:, -1, :] = q[:, -1] = [64, 300]
        q_proj = q[:, -1].unsqueeze(1).repeat(1, k, 1)  # (batch, 19, 300)
        joint_repr = v_proj * q_proj
        joint_repr = self.dropout(joint_repr)
        logits = self.linear(joint_repr)
        return logits

class Transformer(nn.Module):
    ''' A sequence to sequence model with attention mechanism. '''

    def __init__(self, opt, n_src_vocab, n_tgt_vocab):

        super(Transformer, self).__init__()

        src_len_max_seq = 2003
        tgt_len_max_seq = 103
        d_word_vec = opt.d_word_vec
        d_model = opt.d_model
        d_inner = opt.d_inner
        n_layers = opt.n_layers
        n_head = opt.n_head
        d_k = opt.d_k
        d_v = opt.d_v
        dropout = opt.dropout
        tgt_emb_prj_weight_sharing = False
        emb_src_tgt_weight_sharing = False

        self.gcn = GCN_Module(opt, n_src_vocab)

        self.decoder = Decoder(opt,
                                n_tgt_vocab=n_tgt_vocab, tgt_len_max_seq=tgt_len_max_seq,
                                d_word_vec=d_word_vec, d_model=d_model, d_inner=d_inner,
                                n_layers=n_layers, n_head=n_head, d_k=d_k, d_v=d_v,
                                dropout=dropout)

        self.tgt_word_prj = nn.Linear(2048, n_tgt_vocab, bias=False)
        # self.tgt_word_prj = nn.Linear(d_model * 2, n_tgt_vocab, bias=False)

        nn.init.xavier_normal_(self.tgt_word_prj.weight)

        # assert d_model == d_word_vec, \
        'To facilitate the residual connections, \
         the dimensions of all module outputs shall be the same.'

        if tgt_emb_prj_weight_sharing:
            # Share the weight matrix between target word embedding & the final logit dense layer
            self.tgt_word_prj.weight = self.decoder.tgt_word_emb.weight
            self.x_logit_scale = (d_model ** -0.5)
        else:
            self.x_logit_scale = 1.

        if emb_src_tgt_weight_sharing:
            # Share the weight matrix between source & target word embeddings
            assert n_src_vocab == n_tgt_vocab, \
            "To share word embedding table, the vocabulary size of src/tgt shall be the same."
            self.encoder.src_word_emb.weight = self.decoder.tgt_word_emb.weight

        self.fc = nn.Linear(d_model, 2048)
        # self.relu = nn.ReLU()
        # self.fc = nn.Linear(d_model * 2, d_model * 2)


    def forward(self, src, tgt_seq, tgt_pos):

        src1_seq, src1_pos, src2_seq, src2_pos, src3_seq, src3_pos, src4_seq, src4_pos,\
        adj1, adj2, adj3, adj4 = src

        tgt_seq, tgt_pos = tgt_seq[:, :-1], tgt_pos[:, :-1]

        enc_output = self.gcn(src1_seq, src2_seq, src3_seq, src4_seq, adj1, adj2, adj3, adj4)
        # print('enc_output:', src4_seq.shape, enc_output.shape)

        # src_seq = torch.cat([src1_seq,src2_seq,src3_seq,src4_seq], dim=-1)
        dec_output, *_ = self.decoder(tgt_seq, tgt_pos, src4_seq, enc_output)

        seq_logit = self.tgt_word_prj(self.fc(dec_output)) * self.x_logit_scale

        return seq_logit.view(-1, seq_logit.size(2))  # (-1, vocab_size)


import torch.nn as nn
from transformer.SubLayers import MultiHeadAttention, PositionwiseFeedForward


#
# class EncoderLayer(nn.Module):
#     def __init__(self, stacked_layer, d_model, d_inner, n_head, d_k, d_v, dropout=0.1):
#         super(EncoderLayer, self).__init__()
#
#         self.self_att = Self_Attention(d_model, d_inner, n_head, d_k, d_v, dropout=0.1)
#         self.guided_att = nn.ModuleList([Guided_Attention(d_model, d_inner, n_head, d_k, d_v, dropout=0.1)
#                                          for _ in range(stacked_layer)])
#
#     def forward(self, q, k, v, non_pad_mask=None, slf_attn_mask=None):
class EncoderLayer(nn.Module):

    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.slf_attn = MultiHeadAttention(
            n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)

    def forward(self, enc_input, non_pad_mask=None, slf_attn_mask=None):
        enc_output, enc_slf_attn = self.slf_attn(
            enc_input, enc_input, enc_input, mask=slf_attn_mask)
        enc_output *= non_pad_mask

        enc_output = self.pos_ffn(enc_output)
        enc_output *= non_pad_mask

        return enc_output, enc_slf_attn

class SelfAttention(nn.Module):

    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1):
        super(SelfAttention, self).__init__()
        self.slf_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)

    def forward(self, enc_input, non_pad_mask=None, slf_attn_mask=None):

        enc_output, enc_slf_attn = self.slf_attn(enc_input, enc_input, enc_input, mask=slf_attn_mask)
        enc_output *= non_pad_mask

        enc_output = self.pos_ffn(enc_output)
        enc_output *= non_pad_mask

        return enc_output


class GuidedAttention(nn.Module):

    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1):
        super(GuidedAttention, self).__init__()

        self.slf_attn1 = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.slf_attn2 = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)

        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)

    def forward(self, q, k, v, non_pad_mask=None, slf_attn_mask=None):

        q, enc_slf_attn1 = self.slf_attn1(q, q, q, mask=slf_attn_mask)  # self-attention

        q, enc_slf_attn2 = self.slf_attn2(q, k, v, mask=slf_attn_mask)  # Guided-attention

        q *= non_pad_mask
        enc_output = self.pos_ffn(q)
        enc_output *= non_pad_mask

        return enc_output


class DecoderLayer(nn.Module):

    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1):
        super(DecoderLayer, self).__init__()
        self.slf_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.enc_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)

    def forward(self, dec_input, enc_output, non_pad_mask=None, slf_attn_mask=None, dec_enc_attn_mask=None):

        dec_output, dec_slf_attn = self.slf_attn(dec_input, dec_input, dec_input, mask=slf_attn_mask)
        dec_output *= non_pad_mask

        dec_output, dec_enc_attn = self.enc_attn(dec_output, enc_output, enc_output, mask=dec_enc_attn_mask)
        dec_output *= non_pad_mask

        dec_output = self.pos_ffn(dec_output)
        dec_output *= non_pad_mask

        return dec_output, dec_slf_attn, dec_enc_attn

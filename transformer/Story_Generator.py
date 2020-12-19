''' This module will handle the question generation with beam search. '''

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformer.Model import Transformer
from transformer.Beam import Beam


class StoryGenerator(object):
    ''' Load with trained model and Generate question using  beam search '''

    def __init__(self, opt):
        self.opt = opt
        self.device = torch.device('cuda' if opt.cuda else 'cpu')

        #checkpoint = torch.load(opt.model)
        checkpoint = torch.load(opt.model, map_location=self.device)
        model_opt = checkpoint['settings']

        self.model_opt = model_opt

        model = Transformer(opt=self.model_opt,
            n_src_vocab=model_opt.src_vocab_size,
            n_tgt_vocab=model_opt.tgt_vocab_size)

        model.load_state_dict(checkpoint['model'])
        print('[Info] Trained model state loaded.')

        model.word_prob_prj = nn.LogSoftmax(dim=1)

        model = model.to(self.device)

        self.model = model
        self.model.eval()

    def generate_question_batch(self, src1_seq, src1_pos, src2_seq, src2_pos, src3_seq, src3_pos, src4_seq, src4_pos,
                                src1_emo, src2_emo,  src3_emo,  src4_emo):
        '''
        :param src_seq:
        :param src_pos:
        :return:
        '''
        ''' Generate question batach by batch'''


        def get_inst_idx_to_tensor_position_map(inst_idx_list):
            return {inst_idx: tensor_position for tensor_position, inst_idx in enumerate(inst_idx_list)}

        def collect_active_part(beamed_tensor, curr_active_inst_idx, n_prev_active_inst, n_bm):

            _, *d_hs = beamed_tensor.size()
            n_curr_active_inst = len(curr_active_inst_idx)
            new_shape = (n_curr_active_inst * n_bm, *d_hs)

            beamed_tensor = beamed_tensor.view(n_prev_active_inst, -1)
            beamed_tensor = beamed_tensor.index_select(0, curr_active_inst_idx)
            beamed_tensor = beamed_tensor.view(*new_shape)

            return beamed_tensor

        def collate_active_info(src_seq, src_enc, inst_idx_to_position_map, active_inst_idx_list):
            '''

            :param src_seq:
            :param src_enc:
            :param inst_idx_to_position_map:
            :param active_inst_idx_list:
            :return:
            '''

            n_prev_active_inst = len(inst_idx_to_position_map)
            active_inst_idx = [inst_idx_to_position_map[k] for k in active_inst_idx_list]
            active_inst_idx = torch.LongTensor(active_inst_idx).to(self.device)

            active_src_seq = collect_active_part(src_seq, active_inst_idx, n_prev_active_inst, n_bm)
            active_src_enc = collect_active_part(src_enc, active_inst_idx, n_prev_active_inst, n_bm)
            active_inst_idx_to_position_map = get_inst_idx_to_tensor_position_map(active_inst_idx_list)

            return active_src_seq, active_src_enc, active_inst_idx_to_position_map

        def beam_decode_step(inst_dec_beams, len_dec_seq, src_seq, enc_output, inst_idx_to_position_map, n_bm):
            '''
            :param inst_dec_beams:
            :param len_dec_seq:
            :param src_seq:
            :param enc_output:
            :param inst_idx_to_position_map:
            :param n_bm:
            :return:
            '''

            def prepare_beam_dec_seq(inst_dec_beams, len_dec_seq):
                dec_partial_seq = [b.get_current_state() for b in inst_dec_beams if not b.done]
                dec_partial_seq = torch.stack(dec_partial_seq).to(self.device)
                dec_partial_seq = dec_partial_seq.view(-1, len_dec_seq)
                return dec_partial_seq

            def prepare_beam_dec_pos(len_dec_seq, n_active_inst, n_bm):
                dec_partial_pos = torch.arange(1, len_dec_seq + 1, dtype=torch.long, device=self.device)
                dec_partial_pos = dec_partial_pos.unsqueeze(0).repeat(n_active_inst * n_bm, 1)
                return dec_partial_pos

            def predict_word(dec_seq, dec_pos, src_seq, enc_output, n_active_inst, n_bm):

                dec_output, *_ = self.model.decoder(dec_seq, dec_pos, src_seq, enc_output)

                dec_output = dec_output[:, -1, :]  # Pick the last step: (bh * bm) * d_h
                word_prob = F.log_softmax(self.model.tgt_word_prj(self.model.fc(dec_output)), dim=1)

                word_prob = word_prob.view(n_active_inst, n_bm, -1)

                return word_prob

            def collect_active_inst_idx_list(inst_beams, word_prob, inst_idx_to_position_map):
                active_inst_idx_list = []
                for inst_idx, inst_position in inst_idx_to_position_map.items():
                    is_inst_complete = inst_beams[inst_idx].advance(word_prob[inst_position])
                    if not is_inst_complete:
                        active_inst_idx_list += [inst_idx]

                return active_inst_idx_list

            n_active_inst = len(inst_idx_to_position_map)

            dec_seq = prepare_beam_dec_seq(inst_dec_beams, len_dec_seq)
            dec_pos = prepare_beam_dec_pos(len_dec_seq, n_active_inst, n_bm)

            word_prob = predict_word(dec_seq, dec_pos, src_seq, enc_output, n_active_inst, n_bm)

            active_inst_idx_list = collect_active_inst_idx_list(
                inst_dec_beams, word_prob, inst_idx_to_position_map)

            return active_inst_idx_list

        def collect_hypothesis_and_scores(inst_dec_beams, n_best):
            all_hyp, all_scores = [], []
            for inst_idx in range(len(inst_dec_beams)):
                scores, tail_idxs = inst_dec_beams[inst_idx].sort_scores()
                all_scores += [scores[:n_best]]

                hyps = [inst_dec_beams[inst_idx].get_hypothesis(i) for i in tail_idxs[:n_best]]
                all_hyp += [hyps]
            return all_hyp, all_scores

        with torch.no_grad():

            src1_seq = src1_seq.to(self.device)
            src2_seq = src2_seq.to(self.device)
            src3_seq = src3_seq.to(self.device)
            src_seq = src4_seq.to(self.device)

            src1_emo = src1_emo.to(self.device)
            src2_emo = src2_emo.to(self.device)
            src3_emo = src3_emo.to(self.device)
            src4_emo = src4_emo.to(self.device)
            src_enc = self.model.gcn(src1_seq, src2_seq, src3_seq, src4_seq,  src1_emo, src2_emo, src3_emo, src4_emo)  # (batch, 20, 300)
            n_bm = self.opt.beam_size  # 5
            n_inst, len_s, d_h = src_enc.size()  # (batch, 20, 300)
            src_seq = src_seq.repeat(1, n_bm).view(n_inst * n_bm, len_s)  # (batch * 5, 20)

            # (batch, 20*5, 300) --> (batch * 5, 20, 300)
            src_enc = src_enc.repeat(1, n_bm, 1).view(n_inst * n_bm, len_s, d_h)

            inst_dec_beams = [Beam(n_bm, device=self.device) for _ in range(n_inst)]

            active_inst_idx_list = list(range(n_inst))  # [0, 1, 2, ..., batch-1]
            inst_idx_to_position_map = get_inst_idx_to_tensor_position_map(active_inst_idx_list)

            for len_dec_seq in range(1, 15):

                active_inst_idx_list = beam_decode_step(inst_dec_beams, len_dec_seq,
                                                        src_seq, src_enc, inst_idx_to_position_map, n_bm)

                if not active_inst_idx_list:
                    break

                src_seq, src_enc, inst_idx_to_position_map = collate_active_info(
                    src_seq, src_enc, inst_idx_to_position_map, active_inst_idx_list)

        batch_hyp, batch_scores = collect_hypothesis_and_scores(inst_dec_beams, self.opt.n_best)

        return batch_hyp, batch_scores

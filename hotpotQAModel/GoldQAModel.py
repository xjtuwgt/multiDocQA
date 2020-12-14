from torch import Tensor as T
from argparse import Namespace
import torch
from modelUtils.longformerUtils import LongformerEncoder
from hotpotLossUtils.focalLossUtils import MultiClassFocalLoss, PairwiseCEFocalLoss
from modelUtils.moduleUtils import MLP
from modelUtils.moduleUtils import MASK_VALUE
from torch.nn import CrossEntropyLoss
from modelUtils.longformerUtils import get_hotpotqa_longformer_tokenizer
import logging
########################################################################################################################
########################################################################################################################
def compute_smooth_sigmoid(scores: T, smooth_factor=1e-7):
    prob = torch.sigmoid(scores)
    prob = torch.clamp(prob, smooth_factor, 1.0 - smooth_factor)
    return prob
def compute_smooth_reverse_sigmoid(prob: T):
    scores = torch.log(prob/(1.0 - prob))
    return scores
########################################################################################################################
class LongformerGoldQAModel(torch.nn.Module):
    def __init__(self, args: Namespace, fix_encoder=False):
        super().__init__()
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        self.tokenizer = get_hotpotqa_longformer_tokenizer(model_name=args.pretrained_cfg_name)
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        longEncoder = LongformerEncoder.init_encoder(cfg_name=args.pretrained_cfg_name, projection_dim=args.project_dim,
                                                     hidden_dropout=args.input_drop, attn_dropout=args.attn_drop,
                                                     seq_project=args.seq_project)
        longEncoder.resize_token_embeddings(len(self.tokenizer))
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        if args.frozen_layer_num > 0:
            modules = [longEncoder.embeddings, *longEncoder.encoder.layer[:args.frozen_layer_num]]
            for module in modules:
                for param in module.parameters():
                    param.requires_grad = False
            logging.info('Frozen the first {} layers'.format(args.frozen_layer_num))
        # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        self.longformer = longEncoder #### LongFormer encoder
        self.hidden_size = longEncoder.get_out_size()
        self.answer_type_outputs = MLP(d_input=self.hidden_size, d_mid=4 * self.hidden_size, d_out=3) ## yes, no, span question score
        self.answer_span_outputs = MLP(d_input=self.hidden_size, d_mid=4 * self.hidden_size, d_out=2) ## span prediction score
        self.sent_mlp = MLP(d_input=self.hidden_size, d_mid=4 * self.hidden_size, d_out=1) ## support sentence prediction
        self.fix_encoder = fix_encoder
        ####+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        self.hparams = args
        ####+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        self.graph_training = (self.hparams.with_graph_training == 1)
        ####+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        self.mask_value = MASK_VALUE
        ####+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    @staticmethod
    def get_representation(sub_model: LongformerEncoder, ids: T, attn_mask: T, global_attn_mask: T,
                           fix_encoder: bool = False) -> (T, T, T):
        sequence_output = None
        if ids is not None:
            if fix_encoder:
                with torch.no_grad():
                    sequence_output, _, _ = sub_model.forward(input_ids=ids,
                                                                                      attention_mask=attn_mask,
                                                                                      global_attention_mask=global_attn_mask)
                if sub_model.training:
                    sequence_output.requires_grad_(requires_grad=True)
            else:
                sequence_output, _, _ = sub_model.forward(input_ids=ids,
                                                                                  attention_mask=attn_mask,
                                                                                  global_attention_mask=global_attn_mask)
        return sequence_output
    ##++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def forward(self, sample):
        ctx_encode_ids, ctx_attn_mask, ctx_global_attn_mask = sample['ctx_encode'], sample['ctx_attn_mask'], sample['ctx_global_mask']
        sequence_output = self.get_representation(self.longformer, ctx_encode_ids, ctx_attn_mask, ctx_global_attn_mask, self.fix_encoder)
        return sequence_output
    ##++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def answer_type_prediction(self, cls_emb: T):
        scores = self.answer_type_outputs.forward(cls_emb).squeeze(dim=-1)
        return scores
    def answer_span_prediction(self, sequence_output: T):
        logits = self.answer_span_outputs.forward(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)
        return start_logits, end_logits
    ##++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def supp_sent_prediction(self, sent_embed: T, query_embed: T):
        ##++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        sent_score = self.sent_mlp.forward(sent_embed).squeeze(dim=-1)
        #####+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        return sent_score
    ##++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def answer_span_loss(self, start_logits: T, end_logits: T, start_positions: T, end_positions: T):
        if len(start_positions.size()) > 1:
            start_positions = start_positions.squeeze(-1)
        if len(end_positions.size()) > 1:
            end_positions = end_positions.squeeze(-1)
        # sometimes the start/end positions are outside our model inputs, we ignore these terms
        ignored_index = start_logits.size(1)
        start_positions.clamp_(0, ignored_index)
        end_positions.clamp_(0, ignored_index)

        loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
        start_loss = loss_fct(start_logits, start_positions)
        end_loss = loss_fct(end_logits, end_positions)
        total_loss = (start_loss + end_loss) / 2
        return total_loss
    ##++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def answer_type_loss(self, answer_type_logits: T, true_labels: T):
        if len(true_labels.shape) > 1:
            true_labels = true_labels.squeeze(dim=-1)
        no_span_num = (true_labels > 0).sum().data.item()
        answer_type_loss_fct = MultiClassFocalLoss(num_class=3)
        yn_loss = answer_type_loss_fct.forward(answer_type_logits, true_labels)
        return yn_loss, no_span_num, true_labels

    def supp_sent_loss(self, sent_scores: T, sent_label: T, sent_mask: T):
        supp_loss_fct = PairwiseCEFocalLoss()
        supp_sent_loss = supp_loss_fct.forward(scores=sent_scores, targets=sent_label, target_len=sent_mask)
        return supp_sent_loss
    ##++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def multi_loss_computation(self, output_scores: dict, sample: dict):
        answer_type_scores = output_scores['answer_type_score']
        answer_type_labels = sample['yes_no']
        answer_type_loss_score, no_span_num, answer_type_labels = self.answer_type_loss(answer_type_logits=answer_type_scores,
                                                                                  true_labels=answer_type_labels)
        ################################################################################################################
        answer_start_positions, answer_end_positions = sample['ans_start'], sample['ans_end']
        start_logits, end_logits = output_scores['answer_span_score']
        ################################################################################################################
        if no_span_num > 0:
            device = start_logits.device
            seq_num = start_logits.shape[1]
            ans_batch_idx = (answer_type_labels > 0).nonzero().squeeze()
            no_span_start_positions, no_span_end_positions = answer_start_positions[ans_batch_idx].squeeze(), answer_end_positions[ans_batch_idx].squeeze()
            start_logits_back = torch.full((no_span_num, seq_num), fill_value=-10.0, device=device)
            end_logits_back = torch.full((no_span_num, seq_num), fill_value=-10.0, device=device)
            start_logits_back[torch.arange(0, no_span_num), no_span_start_positions] = 10.0
            end_logits_back[torch.arange(0, no_span_num), no_span_end_positions] = 10.0
            start_logits[ans_batch_idx] = start_logits_back
            end_logits[ans_batch_idx] = end_logits_back
        ################################################################################################################
        answer_span_loss_score = self.answer_span_loss(start_logits=start_logits, end_logits=end_logits,
                                                 start_positions=answer_start_positions, end_positions=answer_end_positions)
        ################################################################################################################
        sent_scores = output_scores['sent_score']
        sent_label, sent_lens = sample['sent_labels'], sample['sent_lens']
        sent_mask = sent_lens.masked_fill(sent_lens > 0, 1)
        supp_sent_loss_score = self.supp_sent_loss(sent_scores=sent_scores, sent_label=sent_label, sent_mask=sent_mask)
        ################################################################################################################
        return {'answer_type_loss': answer_type_loss_score, 'span_loss': answer_span_loss_score,
                'sent_loss': supp_sent_loss_score}
    ####################################################################################################################
    def score_computation(self, sample):
        sequence_output = self.forward(sample=sample)
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        cls_embed = sequence_output[:, 0, :]
        query_embed = sequence_output[:, 1, :]
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        sent_positions = sample['sent_start']
        sent_num = sent_positions.shape[1]
        batch_size = sent_positions.shape[0]
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        sent_batch_idx = torch.arange(0, batch_size, device=sequence_output.device).view(batch_size, 1).repeat(1, sent_num)
        sent_embed = sequence_output[sent_batch_idx, sent_positions]
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        answer_type_scores = self.answer_type_prediction(cls_emb=cls_embed)
        start_logits, end_logits = self.answer_span_prediction(sequence_output=sequence_output)
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        sent_scores = self.supp_sent_prediction(sent_embed=sent_embed, query_embed=query_embed)
        ##++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        ctx_attn_mask, ctx_global_attn_mask, special_marker = sample['ctx_attn_mask'], sample['ctx_global_mask'], sample['marker']
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        start_logits = start_logits.masked_fill(ctx_attn_mask == 0, self.mask_value)
        start_logits = start_logits.masked_fill(special_marker == 1, self.mask_value)
        end_logits = end_logits.masked_fill(ctx_attn_mask == 0, self.mask_value)
        end_logits = end_logits.masked_fill(special_marker == 1, self.mask_value)
        ##++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        output_score = {'answer_type_score': answer_type_scores, 'answer_span_score': (start_logits, end_logits),
                        'sent_score': sent_scores}
        return output_score
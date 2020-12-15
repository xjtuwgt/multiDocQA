from torch import Tensor as T
from argparse import Namespace
import torch
from modelUtils.longformerUtils import LongformerEncoder
from hotpotLossUtils.focalLossUtils import PairwiseCEFocalLoss
from modelUtils.moduleUtils import MLP
from modelUtils.moduleUtils import MASK_VALUE
import torch.nn.functional as F
import pytorch_lightning as pl
from modelUtils.longformerUtils import get_hotpotqa_longformer_tokenizer
from modelUtils.moduleUtils import TransformerModule
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
########################################################################################################################
class LongformerRetrievalModel(torch.nn.Module):
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
        self.doc_mlp = MLP(d_input=self.hidden_size, d_mid=4 * self.hidden_size, d_out=1) ## support document prediction
        self.sent_mlp = MLP(d_input=self.hidden_size, d_mid=4 * self.hidden_size, d_out=1) ## support sentence prediction
        self.fix_encoder = fix_encoder
        ####+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        self.hparams = args
        ####+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        self.graph_training = self.hparams.with_graph_training == 1
        ####+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        self.with_graph = self.hparams.with_graph == 1
        if self.with_graph:
            self.graph_encoder = TransformerModule(layer_num=self.hparams.layer_number, d_model=self.hidden_size,
                                                   heads=self.hparams.heads)
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
    def supp_doc_sent_prediction(self, sent_embed: T, doc_embed: T, query_embed: T):
        sent_score = self.sent_mlp.forward(sent_embed).squeeze(dim=-1)
        doc_score = self.doc_mlp.forward(doc_embed).squeeze(dim=-1)
        #####+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        return sent_score, doc_score
    ##++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def supp_doc_loss(self, doc_scores: T, doc_label: T, doc_mask: T):
        supp_loss_fct = PairwiseCEFocalLoss()
        supp_doc_loss = supp_loss_fct.forward(scores=doc_scores, targets=doc_label, target_len=doc_mask)
        return supp_doc_loss
    ##++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def supp_sent_loss(self, sent_scores: T, sent_label: T, sent_mask: T):
        supp_loss_fct = PairwiseCEFocalLoss()
        supp_sent_loss = supp_loss_fct.forward(scores=sent_scores, targets=sent_label, target_len=sent_mask)
        return supp_sent_loss
    ##++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def multi_loss_computation(self, output_scores: dict, sample: dict):
        doc_scores = output_scores['doc_score']
        doc_label, doc_lens = sample['doc_labels'], sample['doc_lens']
        doc_mask = doc_lens.masked_fill(doc_lens > 0, 1)
        supp_doc_loss_score = self.supp_doc_loss(doc_scores=doc_scores, doc_label=doc_label, doc_mask=doc_mask)
        ################################################################################################################
        sent_scores = output_scores['sent_score']
        sent_label, sent_lens = sample['sent_labels'], sample['sent_lens']
        sent_mask = sent_lens.masked_fill(sent_lens > 0, 1)
        supp_sent_loss_score = self.supp_sent_loss(sent_scores=sent_scores, sent_label=sent_label, sent_mask=sent_mask)
        ################################################################################################################
        return {'doc_loss': supp_doc_loss_score, 'sent_loss': supp_sent_loss_score}
    ####################################################################################################################
    def score_computation(self, sample):
        sequence_output = self.forward(sample=sample)
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        query_embed = sequence_output[:, 1, :]  ### query start position
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        doc_positions, sent_positions = sample['doc_start'], sample['sent_end']
        batch_size, doc_num = doc_positions.shape
        sent_num = sent_positions.shape[1]
        ##++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        sent_batch_idx = torch.arange(0, batch_size, device=sequence_output.device).view(batch_size, 1).repeat(1, sent_num)
        sent_embed = sequence_output[sent_batch_idx, sent_positions]
        doc_batch_idx = torch.arange(0, batch_size, device=sequence_output.device).view(batch_size, 1).repeat(1, doc_num)
        doc_embed = sequence_output[doc_batch_idx, doc_positions]
        ##++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        doc_lens, sent_lens = sample['doc_lens'], sample['sent_lens']
        doc_mask = (doc_lens == 0)
        sent_mask = (sent_lens == 0)
        if self.with_graph:
            doc_embed = doc_embed.transpose(0,1)
            doc_embed = self.graph_encoder.forward(doc_embed, doc_mask).transpose(0,1)
            sent_embed = sent_embed.transpose(0,1)
            sent_embed = self.graph_encoder.forward(sent_embed, sent_mask).transpose(0,1)
        ##++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        sent_scores, doc_scores = self.supp_doc_sent_prediction(sent_embed=sent_embed,
                                                                                 doc_embed=doc_embed, query_embed=query_embed)
        ##++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        if self.graph_training:
            doc_lens, sent_lens, doc2sent_map = sample['doc_lens'], sample['sent_lens'], sample['s2d_map']
            sent_scores, doc_scores = self.hierarchical_score(doc_scores=doc_scores, sent_scores=sent_scores,
                                                             sent_lens=sent_lens, doc_lens=doc_lens, doc2sent_map=doc2sent_map)
        ##++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        output_score = {'doc_score': doc_scores, 'sent_score': sent_scores}
        return output_score

    def hierarchical_score(self, doc_scores: T, sent_scores: T, doc_lens: T, sent_lens: T, doc2sent_map: T):
        doc_scores = doc_scores.masked_fill(doc_lens == 0, self.mask_value)
        doc_attn = F.softmax(doc_scores, dim=-1)
        sent_sigmoid_scores = compute_smooth_sigmoid(scores=sent_scores)
        batch_size, sent_num = doc2sent_map.shape
        sent_row_idxes = torch.arange(0, batch_size).unsqueeze(-1).repeat(1, sent_num)
        doc2sent_attn = doc_attn[sent_row_idxes, doc2sent_map]
        sent_sigmoid_scores = sent_sigmoid_scores * doc2sent_attn
        rev_sent_scores = compute_smooth_reverse_sigmoid(prob=sent_sigmoid_scores)
        rev_sent_scores = rev_sent_scores.masked_fill(sent_lens ==0, self.mask_value)
        return rev_sent_scores, doc_scores

########################################################################################################################
class LongformerDocRetrievalModel(pl.LightningModule):
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
        self.doc_mlp = MLP(d_input=self.hidden_size, d_mid=4 * self.hidden_size, d_out=1) ## support document prediction
        self.fix_encoder = fix_encoder
        ####+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        self.hparams = args
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
                    sequence_output, _, _ = sub_model.forward(input_ids=ids, attention_mask=attn_mask,
                                                              global_attention_mask=global_attn_mask)
                if sub_model.training:
                    sequence_output.requires_grad_(requires_grad=True)
            else:
                sequence_output, _, _ = sub_model.forward(input_ids=ids, attention_mask=attn_mask,
                                                          global_attention_mask=global_attn_mask)
        return sequence_output
    ##++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def seq_encoder(self, sample):
        ctx_encode_ids, ctx_attn_mask, ctx_global_attn_mask = sample['ctx_encode'], sample['ctx_attn_mask'], sample['ctx_global_mask']
        sequence_output = self.get_representation(self.longformer, ctx_encode_ids, ctx_attn_mask, ctx_global_attn_mask, self.fix_encoder)
        return sequence_output
    def forward(self, sample):
        output_score = self.score_computation(sample=sample)
        loss_out_put = self.multi_loss_computation(output_scores=output_score, sample=sample)
        if self.training:
            return loss_out_put
        else:
            return loss_out_put, output_score
    ##++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def supp_doc_prediction(self, doc_embed: T):
        doc_score = self.doc_mlp.forward(doc_embed).squeeze(dim=-1)
        return doc_score
    ##++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def supp_doc_loss(self, doc_scores: T, doc_label: T, doc_mask: T):
        supp_loss_fct = PairwiseCEFocalLoss()
        supp_doc_loss = supp_loss_fct.forward(scores=doc_scores, targets=doc_label, target_len=doc_mask)
        return supp_doc_loss
    ##++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    ##++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def multi_loss_computation(self, output_scores: dict, sample: dict):
        doc_scores = output_scores['doc_score']
        doc_label, doc_lens = sample['doc_labels'], sample['doc_lens']
        doc_mask = doc_lens.masked_fill(doc_lens > 0, 1)
        supp_doc_loss_score = self.supp_doc_loss(doc_scores=doc_scores, doc_label=doc_label, doc_mask=doc_mask)
        ################################################################################################################
        return {'doc_loss': supp_doc_loss_score}
    ####################################################################################################################
    def score_computation(self, sample):
        sequence_output = self.seq_encoder(sample=sample)
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        doc_start_positions, doc_end_positions = sample['doc_start'], sample['doc_end']
        batch_size, doc_num = doc_start_positions.shape
        ##++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        doc_batch_idx = torch.arange(0, batch_size, device=sequence_output.device).view(batch_size, 1).repeat(1, doc_num)
        doc_start_embed = sequence_output[doc_batch_idx, doc_start_positions]
        doc_end_embed = sequence_output[doc_batch_idx, doc_end_positions]
        doc_embed = (doc_start_embed + doc_end_embed)/2.0
        ##++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        doc_scores = self.supp_doc_prediction(doc_embed=doc_embed)
        ##++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        output_score = {'doc_score': doc_scores}
        return output_score
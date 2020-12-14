import os
import sys
PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))
import pandas as pd
import logging
from argparse import Namespace
from pandas import DataFrame
from dataUtils.ioutils import loadJSONData
from hotpotEvaluation.hotpot_evaluate_v1 import json_eval, exact_match_score, f1_score
from torch import Tensor as T
from datetime import date, datetime
import torch
import torch.nn.functional as F
import swifter
MAX_ANSWER_DECODE_LEN = 30

########################################################################################################################
def get_date_time():
    today = date.today()
    str_today = today.strftime('%b_%d_%Y')
    now = datetime.now()
    current_time = now.strftime("%H_%M_%S")
    date_time_str = str_today + '_' + current_time
    return date_time_str
########################################################################################################################
def log_metrics(mode, metrics):
    '''
    Print the evaluation logs
    '''
    for metric in metrics:
        logging.info('{} {}: {:.4f}'.format(mode, metric, metrics[metric]))
########################################################################################################################
def sp_score(prediction, gold):
    cur_sp_pred = set(prediction)
    gold_sp_pred = set(gold)
    tp, fp, fn = 0, 0, 0
    for e in cur_sp_pred:
        if e in gold_sp_pred:
            tp += 1
        else:
            fp += 1
    for e in gold_sp_pred:
        if e not in cur_sp_pred:
            fn += 1
    prec = 1.0 * tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = 1.0 * tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = 2 * prec * recall / (prec + recall) if prec + recall > 0 else 0.0
    em = 1.0 if fp + fn == 0 else 0.0
    return em, prec, recall, f1
def recall_computation(prediction, gold):
    gold_set = set(gold)
    gold_count = len(gold)
    tp = 0
    for pred in prediction:
        if pred in gold_set:
            tp = tp + 1
    recall = 1.0 * tp /gold_count
    return recall

def answer_score(prediction, gold):
    em = exact_match_score(prediction, gold)
    f1, prec, recall = f1_score(prediction, gold)
    return em, f1, prec, recall
########################################################################################################################
def answer_type_evaluation(type_scores: T, true_labels: T):
    type_predicted_labels = torch.argmax(type_scores, dim=-1).squeeze()
    true_labels = true_labels.squeeze()
    type_accuracy = (type_predicted_labels == true_labels).sum()*1.0/true_labels.shape[0]
    return type_accuracy
########################################################################################################################
def answer_type_test(type_scores: T):
    type_predicted_labels = torch.argmax(type_scores, dim=-1)
    type_predicted_labels = type_predicted_labels.detach().tolist()
    ans_type_map = {0: 'span', 1: 'yes', 2: 'no'}
    type_predicted_labels = [ans_type_map[_] for _ in type_predicted_labels]
    return type_predicted_labels
########################################################################################################################
def support_doc_test(doc_scores: T, doc_mask: T, top_k=2):
    batch_size, doc_num = doc_scores.shape[0], doc_scores.shape[1]
    assert top_k <= doc_num
    scores = torch.sigmoid(doc_scores)
    masked_doc_scores = scores.masked_fill(doc_mask == 0, -1)  ### mask
    argsort_doc = torch.argsort(masked_doc_scores, dim=1, descending=True)
    ####################################################################################################################
    pred_docs = []
    for idx in range(batch_size):
        pred_idxes_i = argsort_doc[idx].detach().tolist()
        pred_docs_i = pred_idxes_i[:top_k]
        pred_docs.append(pred_docs_i)
        # ==============================================================================================================
    doc_res = {'pred_doc': pred_docs}
    return doc_res
########################################################################################################################
def supp_sent_test(sent_scores: T, sent_mask: T, doc_index: T, sent_index: T, top_k=2, threshold=0.9):
    batch_size, sent_num = sent_scores.shape[0], sent_scores.shape[1]
    assert top_k <= sent_num
    scores = torch.sigmoid(sent_scores)
    masked_sent_scores = scores.masked_fill(sent_mask == 0, -1)  ### mask
    argsort_sent = torch.argsort(masked_sent_scores, dim=1, descending=True)
    ####################################################################################################################
    pred_sents = []
    pred_pairs = []
    for idx in range(batch_size):
        #################################################
        doc_index_i = doc_index[idx].detach().tolist()
        sent_index_i = sent_index[idx].detach().tolist()
        doc_sent_idx_pair_i = list(zip(doc_index_i, sent_index_i))
        #################################################
        pred_idxes_i = argsort_sent[idx].detach().tolist()
        pred_sents_i = pred_idxes_i[:top_k]
        for i in range(top_k, sent_num):
            if masked_sent_scores[idx, pred_idxes_i[i]] > threshold * masked_sent_scores[idx, pred_idxes_i[top_k-1]]:
                pred_sents_i.append(pred_idxes_i[i])
        pred_pair_i = [doc_sent_idx_pair_i[_] for _ in pred_sents_i]
        pred_sents.append(pred_sents_i)
        pred_pairs.append(pred_pair_i)
        # ==============================================================================================================
    sent_res = {'pred_sent': pred_sents, 'pred_pair': pred_pairs}
    return sent_res
########################################################################################################################
def answer_span_evaluation(start_scores: T, end_scores: T, sent_start_positions: T, sent_end_positions: T, sent_mask: T):
    batch_size, seq_len = start_scores.shape[0], start_scores.shape[1]
    start_prob = torch.sigmoid(start_scores)
    end_prob = torch.sigmoid(end_scores)
    sent_number = sent_start_positions.shape[1]
    if len(sent_start_positions.shape) > 1:
        sent_start_positions = sent_start_positions.unsqueeze(dim=-1)
    if len(sent_end_positions.shape) > 1:
        sent_end_positions = sent_end_positions.unsqueeze(dim=-1)
    answer_span_pairs = []
    for batch_idx in range(batch_size):
        max_score_i = -1e9
        max_pair_idx = None
        for sent_idx in range(sent_number):
            if sent_mask[batch_idx][sent_idx] > 0:##for possile sentence
                sent_start_i, sent_end_i = sent_start_positions[batch_idx][sent_idx], sent_end_positions[batch_idx][sent_idx]
                sent_start_score_i = start_prob[batch_idx][sent_start_i:(sent_end_i + 1)]
                sent_end_score_i = end_prob[batch_idx][sent_start_i:(sent_end_i + 1)]
                max_sent_core_i, ans_start_idx, ans_end_idx = answer_span_evaluation_in_sentence(start_scores=sent_start_score_i, end_scores=sent_end_score_i)
                temp_ans_start_idx = ans_start_idx + sent_start_i
                temp_ans_end_idx = ans_end_idx + sent_start_i ## the former added the end idx. FIXED
                if max_score_i < max_sent_core_i:
                    max_pair_idx = (temp_ans_start_idx.detach().item(), temp_ans_end_idx.detach().item())
                    max_score_i = max_sent_core_i ## missing in former coding. FIXED
        assert max_pair_idx is not None, 'max score {}'.format(max_score_i)
        answer_span_pairs.append({'idx_pair': max_pair_idx, 'score': max_score_i})
    assert len(answer_span_pairs) == batch_size
    return answer_span_pairs
########################################################################################################################
def answer_span_evaluation_in_sentence(start_scores: T, end_scores: T, max_ans_decode_len: int = MAX_ANSWER_DECODE_LEN, debug=False):
    assert start_scores.shape[0] == end_scores.shape[0]
    sent_len = start_scores.shape[0]
    score_matrix = start_scores.unsqueeze(1) * end_scores.unsqueeze(0)
    if debug:
        print(score_matrix)
    score_matrix = torch.triu(score_matrix)
    if debug:
        print(score_matrix)
    if max_ans_decode_len < sent_len:
        trip_len = sent_len - max_ans_decode_len
        mask_upper_tri = torch.triu(torch.ones((trip_len, trip_len))).to(start_scores.device)
        mask_upper_tri = F.pad(mask_upper_tri, [max_ans_decode_len, 0, 0, max_ans_decode_len])
        score_matrix = score_matrix.masked_fill(mask_upper_tri==1, 0)
    if debug:
        for i in range(sent_len):
            print('{}-th row {}'.format(i, score_matrix[i]))
    max_idx = torch.argmax(score_matrix)
    start_idx, end_idx = max_idx // sent_len, max_idx % sent_len
    start_idx, end_idx = start_idx.detach().item(), end_idx.detach().item()
    score = score_matrix[start_idx][end_idx].detach().item()
    return score, start_idx, end_idx
########################################################################################################################
def add_id_context(data: DataFrame, data_path: str, data_name: str):
    golden_data = loadJSONData(PATH=data_path, json_fileName=data_name)
    golden_data['e_id'] = range(0, golden_data.shape[0])
    merge_data = pd.concat([data.set_index('e_id'), golden_data.set_index('e_id')], axis=1, join='inner')
    pred_data_col_names = ['_id', 'context', 'ans_type_pred', 'ss_ds_pair', 'ans_span_pred']
    data = merge_data[pred_data_col_names]
    golden_data_col_names = ['supporting_facts', 'level', 'question', 'context', 'answer', '_id', 'type']
    golden_data = merge_data[golden_data_col_names]
    return data, golden_data
########################################################################################################################
def LeadBoardEvaluation(data: DataFrame, args: Namespace):
    ##++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    data, golden_data = add_id_context(data=data, data_path=args.orig_data_path, data_name=args.orig_dev_data_name)
    ##++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def process_row(row):
        answer_type_prediction = row['ans_type_pred']
        supp_sent_prediction_pair = row['ss_ds_pair']
        answer_span_prediction = row['ans_span_pred']
        context_docs = row['context']
        if answer_type_prediction == 'span':
            answer_prediction = answer_span_prediction.strip()
        else:
            answer_prediction = answer_type_prediction

        for x in supp_sent_prediction_pair:
            assert x[0] < len(context_docs), "x[0] {} \n x[1]={} \n doc len = {} \n {}".format(x[0], x[1], len(context_docs), row['_id'])
        supp_title_sent_id = [(context_docs[x[0]][0], x[1]) for x in supp_sent_prediction_pair]
        return answer_prediction, supp_title_sent_id

    pred_names = ['answer', 'sp']
    data[pred_names] = data.apply(lambda row: pd.Series(process_row(row)), axis=1)
    res_names = ['_id', 'answer', 'sp']

    predicted_data = data[res_names]
    id_list = predicted_data['_id'].tolist()
    answer_list = predicted_data['answer'].tolist()
    sp_list = predicted_data['sp'].tolist()
    answer_id_dict = dict(zip(id_list, answer_list))
    sp_id_dict = dict(zip(id_list, sp_list))
    ####################################################################################################################
    predicted_data_dict = {'answer': answer_id_dict, 'sp': sp_id_dict}
    # golden_data, _ = HOTPOT_DevData_Distractor()
    golden_data_dict = golden_data.to_dict(orient='records')
    metrics = json_eval(prediction=predicted_data_dict, gold=golden_data_dict)
    res_data_frame = pd.DataFrame.from_dict(predicted_data_dict)
    return metrics, res_data_frame

########################################################################################################################
def supp_doc_evaluation(doc_scores: T, doc_labels: T, doc_mask: T, top_k=2):
    batch_size, doc_num = doc_scores.shape[0], doc_scores.shape[1]
    assert top_k <= doc_num
    scores = torch.sigmoid(doc_scores)
    masked_doc_scores = scores.masked_fill(doc_mask == 0, -1)  ### mask
    argsort_doc = torch.argsort(masked_doc_scores, dim=1, descending=True)
    ####################################################################################################################
    doc_metric_logs = []
    topk_logs = []
    for idx in range(batch_size):
        pred_idxes_i = argsort_doc[idx].detach().tolist()
        pred_docs_i = pred_idxes_i[:top_k]
        # ==============================================================================================================
        gold_docs_i = (doc_labels[idx] > 0).nonzero(as_tuple=False).squeeze().detach().tolist()
        topk_metric = dict()
        for t in range((top_k+1), doc_num):
            top_t_docs = pred_idxes_i[:t]
            top_t_recall = recall_computation(prediction=top_t_docs, gold=gold_docs_i)
            topk_metric['recall_{}'.format(t)] = top_t_recall
        topk_logs.append(topk_metric)
        # ==============================================================================================================
        em_i, prec_i, recall_i, f1_i = sp_score(prediction=pred_docs_i, gold=gold_docs_i)
        doc_metric_logs.append({
            'doc_sp_em': em_i,
            'doc_sp_f1': f1_i,
            'doc_sp_prec': prec_i,
            'doc_sp_recall': recall_i
        })
    return doc_metric_logs, topk_logs
########################################################################################################################
def supp_sent_evaluation(sent_scores: T, sent_labels: T, sent_mask: T, top_k=2, threshold=0.9):
    batch_size, sent_num = sent_scores.shape[0], sent_scores.shape[1]
    assert top_k <= sent_num
    scores = torch.sigmoid(sent_scores)
    masked_sent_scores = scores.masked_fill(sent_mask == 0, -1)  ### mask
    argsort_sent = torch.argsort(masked_sent_scores, dim=1, descending=True)
    ####################################################################################################################
    sent_metric_logs = []
    for idx in range(batch_size):
        pred_idxes_i = argsort_sent[idx].detach().tolist()
        pred_sents_i = pred_idxes_i[:top_k]
        for i in range(top_k, sent_num):
            if masked_sent_scores[idx, pred_idxes_i[i]] > threshold * masked_sent_scores[idx, pred_idxes_i[top_k-1]]:
                pred_sents_i.append(pred_idxes_i[i])
        # ==============================================================================================================
        gold_sents_i = (sent_labels[idx] > 0).nonzero(as_tuple=False).squeeze().detach().tolist()
        # ==============================================================================================================
        em_i, prec_i, recall_i, f1_i = sp_score(prediction=pred_sents_i, gold=gold_sents_i)
        sent_metric_logs.append({
            'sent_sp_em': em_i,
            'sent_sp_f1': f1_i,
            'sent_sp_prec': prec_i,
            'sent_sp_recall': recall_i
        })
    return sent_metric_logs

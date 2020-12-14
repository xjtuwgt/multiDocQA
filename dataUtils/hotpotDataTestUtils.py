import os
import sys
PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))
from dataUtils.ioutils import loadJSONData
from modelUtils.longformerUtils import get_hotpotqa_longformer_tokenizer
from dataUtils.ioutils import HOTPOT_DevData_Distractor, HOTPOT_TrainData
import torch
from hotpotEvaluation.hotpotEvaluationUtils import answer_score
import collections

distractor_wiki_path = '../data/hotpotqa/distractor_qa'
dev_processed_data_name = 'hotpot_dev_distractor_wiki_encoded.json'
test_processed_data_name = 'hotpot_test_distractor_wiki_encoded.json'
train_processed_data_name = 'hotpot_train_distractor_wiki_encoded.json'

##max_doc_len = 3908

def consistent_checker():
    tokenizer = get_hotpotqa_longformer_tokenizer()
    encoded_data = loadJSONData(PATH=distractor_wiki_path, json_fileName=dev_processed_data_name)
    orig_data, _ = HOTPOT_DevData_Distractor()
    # orig_data, _ = HOTPOT_TrainData()
    col_names = []
    for col in encoded_data.columns:
        col_names.append(col)
        # print(col)
    #
    # doc_label
    # doc_ans_label
    # doc_num
    # doc_len
    # doc_start
    # doc_end
    # head_idx
    # tail_idx
    # sent_label
    # sent_ans_label
    # sent_num
    # sent_len
    # sent_start
    # sent_end
    # sent2doc
    # sentIndoc
    # doc_sent_num
    # ctx_encode
    # ctx_len
    # attn_mask
    # global_attn
    # token2sent
    # ans_mask
    # ans_pos_tups
    # ans_start
    # ans_end
    # answer_type
    def support_doc_checker(row, orig_row):
        doc_label = row['doc_label']
        answer_type = row['answer_type']
        ans_orig = orig_row['answer']

        print(doc_label)
        doc_label = row['doc_ans_label']
        print(doc_label)
        doc_idxes = [x[0] for x in enumerate(doc_label) if x[1] > 0]
        doc_labels = [doc_label[x] for x in doc_idxes]
        print(doc_labels)
        # flag = (doc_labels[0] == doc_labels[1]) and (doc_labels[0] == 1) and answer_type[0] > 0
        flag = answer_type[0] > 1 and ans_orig.strip() not in {'no'}
        # flag = (doc_labels[0] != doc_labels[1])


        orig_context = orig_row['context']
        ctx_titles = [orig_context[x][0] for x in doc_idxes]
        print('decode support doc title {}'.format(ctx_titles))
        support_fact = orig_row['supporting_facts']
        supptitle = list(set([x[0] for x in support_fact]))
        print('supp doc title {}'.format(supptitle))
        print('*' * 75)
        ctx_encode = row['ctx_encode']
        ctx_encode = torch.LongTensor(ctx_encode)
        doc_start = row['doc_start']
        doc_end = row['doc_end']
        for i in range(len(doc_label)):
            print('decode doc: \n{}'.format(tokenizer.decode(ctx_encode[doc_start[i]:(doc_end[i] + 1)])))
            print('orig_doc : \n{}'.format(orig_row['context'][i]))
            print('-' * 75)

        print(tokenizer.decode(ctx_encode[doc_start]))
        print(tokenizer.decode(ctx_encode[doc_end]))
        print(len(doc_label))

        return len(ctx_encode), flag

    def support_sent_checker(row, orig_row):
        sent_label = row['sent_label']
        sent_idxes = [x[0] for x in enumerate(sent_label) if x[1] > 0]
        sent2doc = row['sent2doc']
        sentIndoc = row['sentIndoc']

        sentidxPair = list(zip(sent2doc, sentIndoc))
        suppPair = [sentidxPair[x] for x in sent_idxes]

        orig_context = orig_row['context']
        decode_supp_sent = [(orig_context[x[0]][0], x[1]) for x in suppPair]
        print('decode supp sent {}'.format(decode_supp_sent))
        support_fact = orig_row['supporting_facts']
        print(support_fact)
        print('*'*75)

        sent_start = row['sent_start']
        sent_end = row['sent_end']
        sent_pair = list(zip(sent_start, sent_end))
        supp_sent_pair = [sent_pair[x] for x in sent_idxes]
        ctx_encode = row['ctx_encode']
        ctx_encode = torch.LongTensor(ctx_encode)
        decode_supp_sent_text = [tokenizer.decode(ctx_encode[x[0]:(x[1] + 1)]) for x in supp_sent_pair]
        print('decode sents:\n{}'.format('\n'.join(decode_supp_sent_text)))
        orig_supp_sent = [orig_context[x[0]][1][x[1]] for x in suppPair]
        print('orig sents:\n{}'.format('\n'.join(orig_supp_sent)))
        print('*' * 75)
        return len(sent_start)

    def answer_checker(row, orig_row):
        orig_answer = orig_row['answer']
        ans_tups = row['ans_pos_tups']
        print(len(ans_tups))
        ctx_encode = row['ctx_encode']
        ctx_encode = torch.LongTensor(ctx_encode)
        ans_start = row['ans_start'][0]
        ans_end = row['ans_end'][0]
        decode_answer = tokenizer.decode(ctx_encode[ans_start:(ans_end + 1)])
        print('ori answer: {}'.format(orig_answer.strip()))
        print('dec answer: {}'.format(decode_answer.strip()))
        em, f1, prec, recall = answer_score(prediction=decode_answer, gold=orig_answer)
        print('em {} f1 {} prec {} recall {}'.format(em, f1, prec, recall))
        print('*' * 75)
        return em, f1, prec, recall, len(ans_tups)

    def doc_sent_ans_consistent(row, orig_row):
        answer_start = row['ans_start']
        answer_end = row['ans_end']

        sent_start = row['sent_start']
        sent_end = row['sent_end']

        doc_start = row['doc_start']
        doc_end = row['doc_end']

        doc_ans_label = row['doc_ans_label']
        doc_with_ans_idx = [x[0] for x in enumerate(doc_ans_label) if x[1] > 1]
        sent_ans_label = row['sent_ans_label']
        sent_with_ans_idx = [x[0] for x in enumerate(sent_ans_label) if x[1] > 1]

        ctx_encode = row['ctx_encode']
        ctx_encode = torch.LongTensor(ctx_encode)

        answer_type = row['answer_type']
        if answer_type[0] == 0:
            ans_doc_start = doc_start[doc_with_ans_idx[0]]
            ans_doc_end = doc_end[doc_with_ans_idx[0]]

            ans_sent_start = sent_start[sent_with_ans_idx[0]]
            ans_sent_end = sent_end[sent_with_ans_idx[0]]
            # print('ans {}\n sent {}\n doc{}'.format((answer_start, answer_end),
            #                                          (ans_sent_start, ans_sent_end),
            #                                          (ans_doc_start, ans_doc_end)))

            flag1 = (answer_start[0] >= ans_sent_start) and (answer_end[0] <= ans_sent_end)
            flag2 = (answer_start[0] >= ans_doc_start) and (answer_end[0] <= ans_doc_end)
            flag3 = (ans_sent_start >= ans_doc_start) and (ans_sent_end <= ans_doc_end)

            # print('ans {} sent {} doc {}'.format(flag1, flag2, flag3))
            # if not (flag1 and flag2 and flag3):
            #     print('wrong preprocess')
            print('ans {}\n sent {}\n doc{}\n'.format(tokenizer.decode(ctx_encode[answer_start[0]:(answer_end[0]+1)]),
                                                      tokenizer.decode(ctx_encode[ans_sent_start:(ans_sent_end+1)]),
                                                      tokenizer.decode(ctx_encode[ans_doc_start:(ans_doc_end+1)])))

    # em_score = 0.0
    # f1_score = 0.0
    # ans_count_array = []
    # for row_idx, row in encoded_data.iterrows():
    #     # support_doc_checker(row, orig_data.iloc[row_idx])
    #     # support_sent_checker(row, orig_data.iloc[row_idx])
    #     em, f1, prec, recall, ans_count = answer_checker(row, orig_data.iloc[row_idx])
    #     em_score = em_score + em
    #     f1_score = f1_score + f1
    #     ans_count_array.append(ans_count)
    # print('em {} f1 {}'.format(em_score/encoded_data.shape[0], f1_score/encoded_data.shape[0]))
    # occurrences = dict(collections.Counter(ans_count_array))
    # for key, value in occurrences.items():
    #     print('{}\t{}'.format(key, value*1.0/encoded_data.shape[0]))
    # print(occurrences)
    #########################################
    # max_len = 0
    # equal_count = 0
    # for row_idx, row in encoded_data.iterrows():
    #     doc_len, equal_flag = support_doc_checker(row, orig_data.iloc[row_idx])
    #     if equal_flag:
    #         equal_count = equal_count + 1
    #     if max_len < doc_len:
    #         max_len = doc_len
    #     # sent_len = support_sent_checker(row, orig_data.iloc[row_idx])
    #     # if max_len < sent_len:
    #     #     max_len = sent_len
    # print(max_len)
    # print(equal_count, equal_count * 1.0/encoded_data.shape[0])
    #########################################
    for row_idx, row in encoded_data.iterrows():
        doc_sent_ans_consistent(row, orig_data.iloc[row_idx])


if __name__ == '__main__':
    consistent_checker()
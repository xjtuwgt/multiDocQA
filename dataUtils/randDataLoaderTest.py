import os
import sys
PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))
from dataUtils.ioutils import loadJSONData
from modelUtils.longformerUtils import get_hotpotqa_longformer_tokenizer
from torch.utils.data import DataLoader
from dataUtils.randHotpotQADataSet import HotpotTrainDataset, HotpotDevDataset
from dataUtils.ioutils import HOTPOT_DevData_Distractor, HOTPOT_TrainData, GOLD_HOTPOT_DevData_Distractor
import torch
from hotpotEvaluation.hotpotEvaluationUtils import answer_score
import collections

distractor_wiki_path = '../data/hotpotqa/distractor_qa'
dev_processed_data_name = 'hotpot_dev_distractor_wiki_tokenized.json'
train_processed_data_name = 'hotpot_train_distractor_wiki_tokenized.json'

def get_train_data_loader(data_frame, tokenizer, shuffle)-> DataLoader:
    train_data = HotpotTrainDataset(data_frame=data_frame, tokenizer=tokenizer,  max_doc_num=10, shuffle=shuffle)
    dataloader = DataLoader(dataset=train_data, batch_size=1,
        shuffle=False,
        drop_last=False,
        pin_memory=True,
        num_workers=max(1, 6),
        collate_fn=HotpotTrainDataset.collate_fn)
    return dataloader


def get_val_data_loader(data_frame, tokenizer) -> DataLoader:
    dev_data = HotpotDevDataset(data_frame=data_frame, tokenizer=tokenizer, max_doc_num=2)
    dataloader = DataLoader(
        dataset=dev_data,
        batch_size=1,
        shuffle=False,
        num_workers=max(1, 6),
        collate_fn=HotpotDevDataset.collate_fn
    )
    return dataloader


def consistent_checker():
    tokenizer = get_hotpotqa_longformer_tokenizer()
    # encoded_data = loadJSONData(PATH=distractor_wiki_path, json_fileName=dev_processed_data_name)
    encoded_data = loadJSONData(PATH=distractor_wiki_path, json_fileName=train_processed_data_name)
    # orig_data_frame, _ = HOTPOT_DevData_Distractor()
    orig_data_frame, _ = HOTPOT_TrainData()
    encoded_data['e_id'] = range(0, encoded_data.shape[0])
    # data_loader = get_val_data_loader(data_frame=encoded_data, tokenizer=tokenizer)
    data_loader = get_train_data_loader(data_frame=encoded_data, tokenizer=tokenizer, shuffle=False)

    def answer_checker(row, orig_row):
        ans_start = row['ans_start'][0]
        ans_end = row['ans_end'][0]
        ctx_encode = row['ctx_encode'][0]
        answer = orig_row['answer']
        print('orig answer: {}\ndeco answer: {}'.format(answer,
                                                        tokenizer.decode(ctx_encode[ans_start:(ans_end + 1)])))
        print('*' * 75)

    def doc_sent_checker(row, orig_row):
        doc_label = row['doc_labels'][0]
        doc_start = row['doc_start'][0]
        doc_end = row['doc_end'][0]
        ctx_encode = row['ctx_encode'][0]
        doc_num = doc_label.shape[0]
        pos_doc_idx = (doc_label > 0).nonzero().detach().tolist()
        supp_docs = orig_row['supporting_facts']

        for doc_idx in pos_doc_idx:
            print('doc {}'.format(tokenizer.decode(ctx_encode[doc_start[doc_idx]:(doc_end[doc_idx] + 1)])))
        print(doc_label, doc_num)
        print('=' * 75)

        sent_label = row['sent_labels'][0]
        sent_start = row['sent_start'][0]
        sent_end = row['sent_end'][0]
        ctx_encode = row['ctx_encode'][0]
        sent_num = sent_label.shape[0]
        print(sent_label)
        pos_sent_idx = (sent_label > 0).nonzero().detach().tolist()
        print(pos_sent_idx)
        # for i in range(sent_num):
        #     print('sent end {}'.format(tokenizer.decode(ctx_encode[sent_end[i]:(sent_end[i] + 1)])))

        print('+' * 75)
        s2d_map = row['s2d_map'][0]
        print('sent_2_doc {} {}'.format(s2d_map, row['sent_lens'][0].shape))
        sInd_map = row['sInd_map'][0]
        # print(sInd_map)
        print('+' * 75)

        context = orig_row['context']

        for sent_idx in pos_sent_idx:
            print('Sent\n {}'.format(tokenizer.decode(ctx_encode[sent_start[sent_idx]:(sent_end[sent_idx] + 1)])))
            # print('Pair {}'.format((s2d_map[sent_idx], sInd_map[sent_idx])))
            doc_idx = s2d_map[sent_idx][0].detach().item()
            sent_idx = sInd_map[sent_idx][0].detach().item()
            print('Sent pair\n {}'.format((context[doc_idx][0], sent_idx, context[doc_idx][1][sent_idx])))
        print(supp_docs)
        print('*' * 75)

    for batch_idx, batch in enumerate(data_loader):
        row = batch
        orig_row = orig_data_frame.iloc[batch_idx]
        doc_sent_checker(row, orig_row)

        if batch_idx >= 100:
            break


if __name__ == '__main__':
    consistent_checker()
import argparse
import torch
from modelUtils.longformerUtils import PRE_TAINED_LONFORMER_BASE

def parse_args(args=None):
    parser = argparse.ArgumentParser(
        description='Training and Testing Long Sequence Reason Model')
    parser.add_argument('--do_train', default=True, action='store_true')
    parser.add_argument('--do_valid', default=True, action='store_true')
    parser.add_argument('--do_test', action='store_true')
    parser.add_argument('--cuda', default=False, action='store_true')
    ##++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    parser.add_argument('--orig_data_path', type=str, default='../data/hotpotqa')
    parser.add_argument('--orig_dev_data_name', type=str, default='hotpot_dev_distractor_v1.json')
    ##++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    parser.add_argument('--data_path', type=str, default='../data/hotpotqa/distractor_qa')
    parser.add_argument('--train_data_name', type=str, default='hotpot_train_distractor_wiki_encoded.json')
    parser.add_argument('--train_data_filtered', type=int, default=0) # 0: no filter, 1: filter out easy, 2: filter out easy, medium
    parser.add_argument('--training_shuffle', default=0, type=int)  ## whether re-order training data
    parser.add_argument('--valid_data_name', type=str, default='hotpot_dev_distractor_wiki_encoded.json')
    parser.add_argument('--test_data_name', type=str, default='hotpot_test_distractor_wiki_encoded.json')
    ##++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    parser.add_argument('--gamma', default=2.0, type=float, help='parameter for focal loss')
    parser.add_argument('--alpha', default=1.0, type=float, help='parameter for focal loss')
    ##++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    parser.add_argument('--pretrained_cfg_name', default=PRE_TAINED_LONFORMER_BASE, type=str)
    parser.add_argument('--hop_model_name', default='DotProduct', type=str)  # 'DotProduct', 'BiLinear'
    parser.add_argument('--frozen_layer_num', default=0, type=int, help='number of layers for document encoder frozen during training')
    parser.add_argument('--project_dim', default=0, type=int)
    parser.add_argument('--seq_project', default=True, action='store_true', help='whether perform sequence projection')
    parser.add_argument('--global_mask_name', default='query_doc_sent', type=str) ## query, query_doc, query_doc_sent
    parser.add_argument('--max_sent_num', default=150, type=int)
    parser.add_argument('--max_doc_num', default=10, type=int)
    parser.add_argument('--max_ctx_len', default=4096, type=int)
    ##++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    parser.add_argument('--sent_threshold', default=0.9, type=float)
    parser.add_argument('--doc_threshold', default=0.9, type=float)
    parser.add_argument('--span_weight', default=0.2, type=float)
    parser.add_argument('--pair_score_weight', default=1.0, type=float)
    ##++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    parser.add_argument('--input_drop', default=0.1, type=float)
    parser.add_argument('--attn_drop', default=0.1, type=float)
    parser.add_argument('--heads', default=8, type=float)
    parser.add_argument('--with_graph', default=0, type=int)
    parser.add_argument('--task_name', default='doc_sent_ans', type=str) ## doc, doc_sent, doc_sent_ans
    parser.add_argument('--with_graph_training', default=0, type=int)
    ##++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    parser.add_argument('--grad_clip_value', default=1.0, type=float)
    parser.add_argument('-cpu', '--cpu_num', default=24, type=int) ### for data_loader
    parser.add_argument('-init', '--init_checkpoint', default=None, type=str)
    parser.add_argument('-save', '--checkpoint_path', default='../checkpoints', type=str)
    parser.add_argument('-log', '--log_path', default='../hotpot_logs', type=str)
    parser.add_argument('--log_name', default='HotPotQALog', type=str)
    ##++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    parser.add_argument('--accumulate_grad_batches', default=8, type=int)
    parser.add_argument('--train_batch_size', default=1, type=int)
    parser.add_argument('--max_epochs', default=6, type=int)
    if torch.cuda.is_available():
        parser.add_argument('--gpus', default=4, type=int)
    else:
        parser.add_argument('--gpus', default=0, type=int)
    parser.add_argument('--gpu_list', default=None, type=str)##'0,1,2,3'
    parser.add_argument('--warmup_steps', default=2000, type=int)
    parser.add_argument('--total_steps', default=-1, type=int)
    parser.add_argument('--accelerator', default='ddp', type=str)
    parser.add_argument('--precision', default=32, type=int) ## 16, 32
    ##++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    parser.add_argument('--weight_decay', default=1e-5, type=float)
    parser.add_argument('-lr', '--learning_rate', default=4e-5, type=float) # 1e-5 level
    parser.add_argument('--adam_epsilon', default=1e-8, type=float)  # 1e-5 level
    ##++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    parser.add_argument('--save_checkpoint_steps', default=10000, type=int)
    parser.add_argument('--val_check_interval', default=2000, type=float) ##check every 0.1 epoch
    parser.add_argument('--log_steps', default=50, type=int, help='train log every xx steps')
    parser.add_argument('--test_batch_size', default=16, type=int, help='valid/test batch size')
    parser.add_argument('--rand_seed', default=42, type=int, help='random seed')
    ##++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    return parser.parse_args(args)
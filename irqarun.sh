#!/bin/sh

python -u -c 'import torch; print(torch.__version__)'

CODE_PATH=ircodes
DATA_PATH=data/hotpotqa/distractor_qa
ORI_DATA_PATH=data/hotpotqa
CHECK_POINT_PATH=ir_checkpoints
LOG_PATH=ir_hotpot_logs


HOP_SCORE=$1
EPOCH=$2
BATCH_SIZE=$3
LEARNING_RATE=$4
SENT_THRETH=$5
FROZEN=$6
TRAIN_DA_TYPE=$7
TRAIN_SHUFFLE=$8
SPAN_WEIGHT=$9
PAIR_SCORE_WEIGHT=${10}
WITH_GRAPH_TRAIN=${11}
TASK_NAME=${12}
LOG_NAME=${13}
SEED=${14}
ACCU_BATCH_SIZE=${15}

echo "Start Training......"

CUDA_VISIBLE_DEVICES=0,1,2,3 python -u $CODE_PATH/irrun.py --do_train \
    --data_path $DATA_PATH\
    --orig_data_path $ORI_DATA_PATH\
    --hop_model_name $HOP_SCORE\
    --max_epochs $EPOCH\
    --checkpoint_path $CHECK_POINT_PATH\
    --log_path $LOG_PATH\
    --accumulate_grad_batches $ACCU_BATCH_SIZE\
    --train_batch_size $BATCH_SIZE\
    --learning_rate $LEARNING_RATE\
    --sent_threshold $SENT_THRETH\
    --frozen_layer_num $FROZEN\
    --train_data_filtered $TRAIN_DA_TYPE\
    --training_shuffle $TRAIN_SHUFFLE\
    --span_weight $SPAN_WEIGHT\
    --pair_score_weight $PAIR_SCORE_WEIGHT\
    --with_graph_training $WITH_GRAPH_TRAIN\
    --task_name $TASK_NAME\
    --log_name $LOG_NAME\
    --rand_seed $SEED
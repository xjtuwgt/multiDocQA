import os
import sys
PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))
from randcodes.randhyperparaSettings import parse_args
from dataUtils.ioutils import create_dir_if_not_exist, set_logger
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from hotpotQAModel.RandQAModel import LongformerRandHotPotQAModel
from pytorch_lightning import loggers as pl_loggers
import logging
import torch
from codes.gpu_utils import gpu_setting
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def trainer_builder(args):
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    logging.info("PyTorch Lighting Trainer constructing...")
    tb_logger = pl_loggers.TensorBoardLogger(save_dir=args.log_path, name=args.log_name + '_log')
    ####################################################################################################################
    checkpoint_callback = ModelCheckpoint(monitor='valid_loss',
                                          dirpath=args.checkpoint_path,
                                          filename='full_doc_hotpotQA-{epoch:02d}-{val_loss:4f}')
    ####################################################################################################################
    if args.gpus > 0:
        gpu_list_str = args.gpu_list
        gpu_ids = [int(x) for x in gpu_list_str.split(',')]
        trainer = pl.Trainer(logger=tb_logger,
                             gradient_clip_val=args.grad_clip_value,
                             gpus=gpu_ids,
                             callbacks=[checkpoint_callback],
                             val_check_interval=args.val_check_interval,
                             accumulate_grad_batches=args.accumulate_grad_batches,
                             accelerator=args.accelerator,
                             precision=args.precision,
                             num_nodes=1,
                             log_every_n_steps=args.log_steps,
                             max_epochs=args.max_epochs)
    else:
        trainer = pl.Trainer(logger=tb_logger,
                             gradient_clip_val=args.grad_clip_value,
                             val_check_interval=args.val_check_interval,
                             accumulate_grad_batches=args.accumulate_grad_batches,
                             log_every_n_steps=args.log_steps,
                             max_epochs=args.max_epochs)
    return trainer

def logger_builder(args):
    if args.checkpoint_path is not None:
        create_dir_if_not_exist(save_path=args.checkpoint_path, sub_folder=args.log_name)
    if args.log_path is not None:
        create_dir_if_not_exist(save_path=args.log_path, sub_folder=args.log_name)
    set_logger(args=args)
    if torch.cuda.is_available():
        if args.gpus > 0:
            free_gpu_ids, used_memory = gpu_setting(num_gpu=args.gpus)
            logging.info('{} gpus with used memory = {}, gpu ids = {}'.format(len(free_gpu_ids), used_memory, free_gpu_ids))
            if args.gpus > len(free_gpu_ids):
                gpu_list_str = ','.join([str(_) for _ in free_gpu_ids])
                args.gpus = len(free_gpu_ids)
            else:
                gpu_list_str = ','.join([str(free_gpu_ids[i]) for i in range(args.gpus)])
            args.gpu_list = gpu_list_str
            logging.info('gpu list = {}'.format(gpu_list_str))

def main(args):
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    pl.seed_everything(seed=args.rand_seed)
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    logging.info('*' * 75)
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    abs_orig_data_path = os.path.abspath(args.orig_data_path)
    abs_data_path = os.path.abspath(args.data_path)
    abs_log_path = os.path.abspath(args.log_path)
    abs_checkpoint_path = os.path.abspath(args.checkpoint_path)
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    logging.info('original paths:\n'
                 '\tabs orig data:{}\n\tabs data:{}\n\tabs log:{}\n\tabs checkpoint{}\n'.format(abs_orig_data_path,
                                                                                              abs_data_path,
                                                                                              abs_log_path,
                                                                                              abs_checkpoint_path))
    args.orig_data_path = abs_orig_data_path
    args.data_path = abs_data_path
    args.log_path = abs_log_path
    args.checkpoint_path = abs_checkpoint_path
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    logging.info('Building HotPotQA reasoning module...')
    hotpotQA_model = LongformerRandHotPotQAModel(args=args)
    logging.info('Building reasoning module completed')
    hotpotQA_model.prepare_data()
    hotpotQA_model.setup(stage='fit')
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    logging.info('Model Parameter Configuration:')
    for name, param in hotpotQA_model.named_parameters():
        logging.info('Parameter {}: {}, require_grad = {}'.format(name, str(param.size()), str(param.requires_grad)))
    logging.info('*' * 75)
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    logging.info("Model hype-parameter information...")
    for key, value in vars(args).items():
        logging.info('Hype-parameter\t{} = {}'.format(key, value))
    logging.info('*' * 75)
    ####################################################################################################################
    trainer = trainer_builder(args=args)
    ####################################################################################################################
    return trainer, hotpotQA_model

if __name__ == '__main__':
    ####################################################################################################################
    torch.autograd.set_detect_anomaly(True)
    ####################################################################################################################
    args = parse_args()
    ####################################################################################################################
    logger_builder(args=args)
    ####################################################################################################################
    trainer, hotpotQA_model = main(args=args)
    trainer.fit(model=hotpotQA_model)
    ####################################################################################################################

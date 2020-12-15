import os
import sys
PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))
from ircodes.irhyperparaSettings import parse_args
from dataUtils.ioutils import create_dir_if_not_exist, set_logger
import pytorch_lightning as pl
from hotpotQAModel.RetrievalModel import LongformerDocRetrievalModel
from modelTrain.IRTrainFunction import configure_optimizers, training_epoch_ir
import logging
import torch
from torch.utils.data import DataLoader
from modelUtils.gpu_utils import gpu_setting
from dataUtils.fullHotpotQADataSet import HotpotTrainDataset, HotpotDevDataset, HotpotTestDataset
from dataUtils.ioutils import loadJSONData
from torch.nn import DataParallel
from time import time
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def trainer_builder(args):
    logging.info("Trainer constructing...")
    if args.gpus > 0:
        gpu_list_str = args.gpu_list
        gpu_ids = [int(x) for x in gpu_list_str.split(',')]
        device = torch.device("cuda:%d" % gpu_ids[0])
        device_ids = gpu_ids
        logging.info('GPU setting')
        args.cuda = True
    else:
        device = torch.device("cpu")
        device_ids = None
        logging.info('CPU setting')
        args.cuda = False
    fix_encoder = args.frozen_layer_num == 12
    hotpotIR_model = LongformerDocRetrievalModel(args=args, fix_encoder=fix_encoder).to(device)
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    logging.info('Building reasoning module completed')
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    train_data_loader, dev_data_loader = prepare_data(model=hotpotIR_model, args=args)
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    args.total_steps = (
            (len(train_data_loader.dataset) // args.train_batch_size)
            // args.accumulate_grad_batches
            * float(args.max_epochs)
    )
    logging.info('Loading data completed')
    logging.info('Total steps = {}'.format(args.total_steps))
    if device_ids is not None:
        hotpotIR_model = DataParallel(hotpotIR_model, device_ids=device_ids)
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    optimizer, scheduler = configure_optimizers(model=hotpotIR_model, args=args)
    return hotpotIR_model, train_data_loader, dev_data_loader, optimizer[0], scheduler[0]

def logger_builder(args):
    if args.checkpoint_path is not None:
        create_dir_if_not_exist(save_path=args.checkpoint_path, sub_folder=args.log_name)
    if args.log_path is not None:
        create_dir_if_not_exist(save_path=args.log_path, sub_folder=args.log_name)
    set_logger(args=args)
    logging.info('Logging have been set')
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

def prepare_data(model, args):
    logging.info('Data preparing...')
    train_data_frame = loadJSONData(PATH=args.data_path, json_fileName=args.train_data_name)
    train_data_frame['e_id'] = range(0, train_data_frame.shape[0])
    train_data = HotpotTrainDataset(data_frame=train_data_frame, tokenizer=model.tokenizer)
    dev_data_frame = loadJSONData(PATH=args.data_path, json_fileName=args.valid_data_name)
    dev_data_frame['e_id'] = range(0, dev_data_frame.shape[0])
    dev_data = HotpotDevDataset(data_frame=dev_data_frame, tokenizer=model.tokenizer)
    train_data_loader, dev_data_loader = train_dataloader(train_data=train_data, args=args), val_dataloader(
        dev_data=dev_data, args=args)
    return train_data_loader, dev_data_loader


def train_dataloader(train_data, args) -> DataLoader:
    dataloader = DataLoader(dataset=train_data, batch_size=args.train_batch_size,
                            shuffle=True,
                            drop_last=True,
                            pin_memory=True,
                            num_workers=max(1, args.cpu_num // 2),
                            collate_fn=HotpotTrainDataset.collate_fn)
    return dataloader


def val_dataloader(dev_data, args) -> DataLoader:
    dataloader = DataLoader(
        dataset=dev_data,
        batch_size=args.test_batch_size,
        shuffle=False,
        num_workers=max(1, args.cpu_num // 2),
        collate_fn=HotpotDevDataset.collate_fn
    )
    return dataloader

def main(args):
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    pl.seed_everything(seed=args.rand_seed)
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    logging.info('*' * 75)
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
    hotpotIR_model, train_data_loader, dev_data_loader, optimizer, scheduler = trainer_builder(args=args)
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    logging.info('Model Parameter Configuration:')
    for name, param in hotpotIR_model.named_parameters():
        logging.info('Parameter {}: {}, require_grad = {}'.format(name, str(param.size()), str(param.requires_grad)))
    logging.info('*' * 75)
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    logging.info("Model hype-parameter information...")
    for key, value in vars(args).items():
        logging.info('Hype-parameter\t{} = {}'.format(key, value))
    logging.info('*' * 75)
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    logging.info('Start training...')
    start_time = time()
    min_val_loss, final_val_loss = training_epoch_ir(model=hotpotIR_model, optimizer=optimizer, dev_dataloader=dev_data_loader,
                    train_dataloader=train_data_loader, scheduler=scheduler, args=args)
    logging.info('Completed training in {:.4f} seconds'.format(time() - start_time))
    logging.info('Min val loss {}, final val loss{}'.format(min_val_loss, final_val_loss))
    ##++++++++++++++++++++++++++++++++++++++++++++++++++++

if __name__ == '__main__':
    ####################################################################################################################
    torch.autograd.set_detect_anomaly(True)
    ####################################################################################################################
    args = parse_args()
    ####################################################################################################################
    logger_builder(args=args)
    ####################################################################################################################
    main(args=args)
    ####################################################################################################################

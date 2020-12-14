import torch
import logging
from time import time
from dataUtils.ioutils import save_check_point
from hotpotEvaluation.hotpotEvaluationUtils import log_metrics, supp_doc_evaluation
from transformers import AdamW, get_linear_schedule_with_warmup

def configure_optimizers(model, args):
    "Prepare optimizer and schedule (linear warmup and decay)"
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if
                       (p.requires_grad) and (not any(nd in n for nd in no_decay))],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if
                       (p.requires_grad) and (any(nd in n for nd in no_decay))],
            "weight_decay": 0.0,
        }
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=args.total_steps
    )
    scheduler = {
        'scheduler': scheduler,
        'interval': 'step',
        'frequency': 1
    }
    return [optimizer], [scheduler]

def training_ir_warm_up(model, optimizer, train_dataloader, dev_dataloader, args):
    warm_up_steps = args.warm_up_steps
    start_time = time()
    step = 0
    training_logs = []
    logging.info('Starting warm up...')
    logging.info('*' * 75)
    #########
    model.train()
    model.zero_grad()
    #########
    for batch_idx, train_sample in enumerate(train_dataloader):
        log = train_step_ir(model=model, optimizer=optimizer, batch=train_sample, args=args)
        step = step + 1
        training_logs.append(log)
        if step % args.log_steps == 0:
            metrics = {}
            for metric in training_logs[0].keys():
                metrics[metric] = sum([log[metric] for log in training_logs]) / len(training_logs)
            log_metrics('Training average', metrics)
            logging.info('Training in {} ({}, {}) steps takes {:.4f} seconds'.format(step, 'warm_up', batch_idx + 1,
                                                                                     time() - start_time))
            training_logs = []
        if step >= warm_up_steps:
            logging.info('Warm up completed in {:.4f} seconds'.format(time() - start_time))
            logging.info('*' * 75)
            break
    logging.info('Evaluating on Valid Dataset...')
    metric_dict = validation_epoch_ir(model=model, test_data_loader=dev_dataloader, args=args)
    logging.info('*' * 75)
    valid_loss = metric_dict['valid_loss']
    logging.info('Validation loss = {}'.format(valid_loss))
    logging.info('*' * 75)
    for key, value in metric_dict.items():
        if key.endswith('metric'):
            logging.info('{} prediction after warm up'.format(key))
            log_metrics('Valid', value)
        logging.info('*' * 75)

def training_epoch_ir(model, optimizer, scheduler, train_dataloader, dev_dataloader, args):
    warm_up_steps = args.warm_up_steps
    if warm_up_steps > 0:
        training_ir_warm_up(model=model, optimizer=optimizer, train_dataloader=train_dataloader, dev_dataloader=dev_dataloader, args=args)
        logging.info('*' * 75)
        current_learning_rate = optimizer.param_groups[-1]['lr']
        learning_rate = current_learning_rate * 0.5
        optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate, weight_decay=args.weight_decay)
        logging.info('Update learning rate from {} to {}'.format(current_learning_rate, learning_rate))

    start_time = time()
    train_loss = 0.0
    min_valid_loss = 1e9
    step = 0
    training_logs = []
    for epoch in range(1, args.epoch + 1):
        for batch_idx, batch in enumerate(train_dataloader):
            log = train_step_ir(model=model, optimizer=optimizer, batch=batch, args=args)
            # ##+++++++++++++++++++++++++++++++++++++++++++++++
            scheduler.step()
            # ##+++++++++++++++++++++++++++++++++++++++++++++++
            step = step + 1
            training_logs.append(log)
            ##+++++++++++++++++++++++++++++++++++++++++++++++
            if step % args.save_checkpoint_steps == 0:
                save_path = save_check_point(model=model, optimizer=optimizer, step=step, loss=train_loss, args=args)
                logging.info('Saving the mode in {}'.format(save_path))
            if step % args.log_steps == 0:
                metrics = {}
                for metric in training_logs[0].keys():
                    metrics[metric] = sum([log[metric] for log in training_logs]) / len(training_logs)
                log_metrics('Training average', step, metrics)
                train_loss = metrics['al_loss']
                logging.info('Training in {} ({}, {}) steps takes {:.4f} seconds'.format(step, epoch, batch_idx + 1,
                                                                                         time() - start_time))
                training_logs = []

            if args.do_valid and step % args.valid_steps == 0:
                logging.info('*' * 75)
                logging.info('Evaluating on Valid Dataset...')
                metric_dict = validation_epoch_ir(model=model, test_data_loader=dev_dataloader, args=args)
                logging.info('*' * 75)
                valid_loss = metric_dict['valid_loss']
                if valid_loss < min_valid_loss:
                    min_valid_loss = valid_loss
                    save_path = save_check_point(model=model, optimizer=optimizer, step=step, loss=min_valid_loss,
                                                 args=args)
                    logging.info('Saving the mode in {}'.format(save_path))
                logging.info('Current valid loss: {}'.format(min_valid_loss))
                logging.info('*' * 75)
                for key, value in metric_dict.items():
                    if key.endswith('metrics'):
                        logging.info('{} prediction at step {}'.format(key, step))
                        log_metrics('Valid', value)
                logging.info('*' * 75)
    return min_valid_loss

def train_step_ir(model, batch, optimizer, args):
    model.train()
    model.zero_grad()
    optimizer.zero_grad()
    if args.cuda:
        sample = dict()
        for key, value in batch.items():
            sample[key] = value.cuda()
    else:
        sample = batch
    output_scores = model.score_computation(sample=sample)
    loss_res = model.multi_loss_computation(sample=sample, output_scores=output_scores)
    supp_doc_loss = loss_res['doc_loss']
    loss = supp_doc_loss
    loss.mean().backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_value)
    optimizer.step()
    torch.cuda.empty_cache()
    log = {
        'loss': loss.mean().item()
    }
    return log

def validation_epoch_ir(model, test_data_loader, args):
    model.eval()
    total_steps = len(test_data_loader)
    start_time = time()
    doc_metric_logs = []
    topk_metric_logs = []
    loss_out_put = []
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_data_loader):
            out_put = validation_step_ir(model=model, batch=batch, args=args)
            loss_out_put.append(out_put['valid_loss'])
            doc_metric_logs += out_put['doc_metric']
            topk_metric_logs += out_put['topk_metric']
            if batch_idx % args.test_log_steps == 0:
                logging.info('Evaluating the Model... {}/{} in {:.4f} seconds'.format(batch_idx, total_steps, time() - start_time))

    avg_loss = sum(loss_out_put)/len(loss_out_put)
    doc_metrics = {}
    for key in doc_metric_logs[0].keys():
        doc_metrics[key] = sum([log[key] for log in doc_metric_logs]) / len(doc_metric_logs)
    topk_metrics = {}
    for key in topk_metric_logs[0].keys():
        topk_metrics[key] = sum([log[key] for log in topk_metric_logs]) / len(topk_metric_logs)
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    return {'valid_loss': avg_loss, 'doc_metric': doc_metrics, 'topk_metric': topk_metrics}

def validation_step_ir(model, batch, args):
    if args.cuda:
        sample = dict()
        for key, value in batch.items():
            sample[key] = value.cuda()
    else:
        sample = batch
    output_scores = model.score_computation(sample=sample)
    loss_res = model.multi_loss_computation(sample=sample, output_scores=output_scores)

    supp_doc_loss = loss_res['doc_loss']
    loss = supp_doc_loss
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    doc_scores = output_scores['doc_score']
    doc_labels, doc_mask = batch['doc_labels'], batch['doc_lens']
    doc_metric_logs, topk_metric_logs = supp_doc_evaluation(doc_scores=doc_scores, doc_labels=doc_labels,
                                                            doc_mask=doc_mask, top_k=2)
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    output = {'valid_loss': loss.mean().detach().item(), 'doc_metric': doc_metric_logs, 'topk_metric': topk_metric_logs}
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    return output


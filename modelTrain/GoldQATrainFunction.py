import torch
import logging
from time import time
from pandas import DataFrame
import os
from dataUtils.ioutils import save_check_point
from hotpotEvaluation.hotpotEvaluationUtils import log_metrics, answer_type_test, supp_sent_test, \
    answer_span_evaluation, get_date_time, LeadBoardEvaluation
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

def training_qa_warm_up(model, optimizer, train_dataloader, dev_dataloader, args):
    warm_up_steps = args.warmup_steps
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
        log = train_step_qa(model=model, optimizer=optimizer, batch=train_sample, args=args)
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
    metric_dict = validation_epoch_qa(model=model, test_data_loader=dev_dataloader, args=args)
    logging.info('*' * 75)
    valid_loss = metric_dict['valid_loss']
    logging.info('Validation loss = {}'.format(valid_loss))
    logging.info('*' * 75)
    for key, value in metric_dict.items():
        if key.endswith('metric'):
            logging.info('{} prediction after warm up'.format(key))
            log_metrics('Valid', value)
        logging.info('*' * 75)

def training_epoch_qa(model, optimizer, scheduler, train_dataloader, dev_dataloader, args):
    warm_up_steps = args.warmup_steps
    if warm_up_steps > 0:
        training_qa_warm_up(model=model, optimizer=optimizer, train_dataloader=train_dataloader, dev_dataloader=dev_dataloader, args=args)
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
            log = train_step_qa(model=model, optimizer=optimizer, batch=batch, args=args)
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
                log_metrics('Training average', metrics)
                train_loss = metrics['train_loss']
                logging.info('Training in {} ({}, {}) steps takes {:.4f} seconds'.format(step, epoch, batch_idx + 1,
                                                                                         time() - start_time))
                training_logs = []

            if args.do_valid and step % args.valid_steps == 0:
                logging.info('*' * 75)
                logging.info('Evaluating on Valid Dataset...')
                metric_dict = validation_epoch_qa(model=model, test_data_loader=dev_dataloader, args=args)
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

    logging.info('*' * 75)
    logging.info('Evaluating on Valid Dataset...')
    metric_dict = validation_epoch_qa(model=model, test_data_loader=dev_dataloader, args=args)
    logging.info('*' * 75)
    final_valid_loss = metric_dict['valid_loss']
    logging.info('Current valid loss: {}'.format(final_valid_loss))
    logging.info('*' * 75)
    for key, value in metric_dict.items():
        if key.endswith('metrics'):
            logging.info('{} prediction at final step'.format(key))
            log_metrics('Valid', value)
    logging.info('*' * 75)
    save_path = save_check_point(model=model, optimizer=optimizer, step='final', loss=final_valid_loss,
                                 args=args)
    logging.info('Saving the final mode in {}'.format(save_path))
    return min_valid_loss, final_valid_loss

def train_step_qa(model, batch, optimizer, args):
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
    ans_type_loss, answer_span_loss, supp_sent_loss = loss_res['answer_type_loss'], \
                                                      loss_res['span_loss'], loss_res['sent_loss']
    train_loss = supp_sent_loss + ans_type_loss + \
                 answer_span_loss * args.span_weight
    loss = train_loss
    loss.mean().backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_value)
    optimizer.step()
    torch.cuda.empty_cache()
    log = {
        'train_loss': loss.mean().item(),
        'ans_type_loss': answer_span_loss.mean().item(),
        'ans_span_loss': answer_span_loss.mean().item(),
        'sent_loss': supp_sent_loss.mean().item()
    }
    return log

def validation_epoch_qa(model, test_data_loader, args):
    model.eval()
    total_steps = len(test_data_loader)
    start_time = time()
    loss_out_put = []
    valid_dict_outputs = []
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_data_loader):
            out_put = validation_step_qa(model=model, batch=batch, args=args)
            loss_out_put.append(out_put['valid_loss'])
            valid_dict_outputs.append(out_put['valid_dict_output'])
            if batch_idx % args.test_log_steps == 0:
                logging.info('Evaluating the Model... {}/{} in {:.4f} seconds'.format(batch_idx, total_steps, time() - start_time))
    #################################################################
    total_sample_number = 0.0
    answer_type_predictions = []
    sent_predictions = []
    answer_span_predictions = []
    example_ids = []
    for batch_idx, output in enumerate(valid_dict_outputs):
        total_sample_number = total_sample_number + output['batch_size']
        example_ids = example_ids + output['ids']
        answer_type_predictions = answer_type_predictions + output['answer_type']
        answer_span_predictions = answer_span_predictions + output['answer_pred']
        sent_res_i = output['sent_pred']
        sent_predictions = sent_predictions + sent_res_i['pred_pair']
    ################################################################################################################
    logging.info('Leadboard evaluation...')
    result_dict = {'ans_type_pred': answer_type_predictions,
                   'ans_span_pred': answer_span_predictions,
                   'ss_ds_pair': sent_predictions,
                   'e_id': example_ids}  ## for detailed results checking
    res_data_frame = DataFrame(result_dict)
    lead_metrics, result_df = LeadBoardEvaluation(data=res_data_frame, args=args)
    metric_name = get_date_time() + '_joint_f1_' + str(lead_metrics['joint_f1'])
    logging.info('Leader board evaluation completed over {} records'.format(result_df.shape[0]))
    log_metrics(mode='Evaluation', metrics=lead_metrics)
    logging.info('*' * 75)
    save_result_name = os.path.join(args.log_path, metric_name + '.json')
    result_df.to_json(save_result_name)
    avg_loss = sum(loss_out_put) / len(loss_out_put)
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    return {'valid_loss': avg_loss, 'metric': lead_metrics, 'pred_results': result_df}

def validation_step_qa(model, batch, args):
    if args.cuda:
        sample = dict()
        for key, value in batch.items():
            sample[key] = value.cuda()
    else:
        sample = batch
    output_scores = model.score_computation(sample=sample)
    loss_res = model.multi_loss_computation(sample=sample, output_scores=output_scores)

    ans_type_loss, answer_span_loss, supp_sent_loss = loss_res['answer_type_loss'], \
                                                      loss_res['span_loss'], loss_res['sent_loss']
    valid_loss = supp_sent_loss + ans_type_loss + \
                 answer_span_loss * args.span_weight
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    answer_type_scores = output_scores['answer_type_score']
    type_predicted_labels = answer_type_test(type_scores=answer_type_scores)
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    sent_scores = output_scores['sent_score']
    sent_lens = batch['sent_lens']
    doc_idxes, sent_idxes = batch['s2d_map'], batch['sInd_map']
    batch_size = doc_idxes.shape[0]
    sent_pred_res = supp_sent_test(sent_scores=sent_scores, sent_mask=sent_lens, doc_index=doc_idxes,
                                   sent_index=sent_idxes)
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    answer_start_logit, answer_end_logit = output_scores['answer_span_score']
    sent_start, sent_end, sent_mask = batch['sent_start'], batch['sent_end'], batch['sent_lens']
    text_encode = batch['ctx_encode']
    answer_span_pairs = answer_span_evaluation(start_scores=answer_start_logit, end_scores=answer_end_logit,
                                               sent_start_positions=sent_start, sent_end_positions=sent_end,
                                               sent_mask=sent_mask)
    assert len(answer_span_pairs) == batch_size
    predicted_answers = [
        model.tokenizer.decode(token_ids=text_encode[batch_idx][x['idx_pair'][0]:(x['idx_pair'][1] + 1)],
                              skip_special_tokens=True) for batch_idx, x in enumerate(answer_span_pairs)]
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    example_ids = batch['id'].squeeze().detach().tolist()
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    valid_dictionary = {'batch_size': answer_type_scores.shape[0], 'ids': example_ids,
                        'answer_type': type_predicted_labels, 'sent_pred': sent_pred_res,
                        'answer_pred': predicted_answers}
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    dict_for_log = {'valid_loss': valid_loss.mean().detach().item(), 'ans_type_loss': ans_type_loss.mean().detach().item(),
                    'ans_span_loss': answer_span_loss.mean().detach().item(), 'sent_loss': supp_sent_loss.mean().detach().item()}
    output = {'valid_loss': valid_loss.mean().detach().item(), 'log': dict_for_log, 'valid_dict_output': valid_dictionary}
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    return output
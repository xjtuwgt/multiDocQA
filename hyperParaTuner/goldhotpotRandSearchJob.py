import sys
import os

PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))
import shutil
from hyperParaTuner.randSearch import RandomSearchJob

def remove_all_files(dirpath):
    for filename in os.listdir(dirpath):
        filepath = os.path.join(dirpath, filename)
        try:
            shutil.rmtree(filepath)
        except OSError:
            os.remove(filepath)

def HypeParameterSpace():
    hop_score_name = {'name': 'hop_model_name', 'type': 'fixed', 'value': 'DotProduct'}
    learning_rate = {'name': 'learning_rate', 'type': 'choice', 'values': [2e-5, 3e-5, 4e-5, 5e-5]}
    batch_size = {'name': 'batch_size', 'type': 'fixed', 'value': 8}
    accu_grad_batch = {'name': 'accu_grad_size', 'type': 'choice', 'values': [1]}
    sent_threshold = {'name': 'sent_threshold', 'type': 'choice', 'values': [0.925, 0.95]}
    task_name = {'name': 'task_name', 'type': 'choice', 'values': ['doc_sent_ans']} ##
    frozen_layer_num = {'name': 'frozen_layer', 'type': 'choice', 'values': [0]} #1, 2
    answer_span_weight = {'name': 'span_weight', 'type': 'choice', 'values': [0.2, 0.5, 1.0]}
    pair_score_weight = {'name': 'pair_score_weight', 'type': 'choice', 'values': [0]} #0.1, 0.2, 0.5, 1.0
    train_data_filtered = {'name': 'train_data_type', 'type': 'choice', 'values': [0]} # 0, 1, 2
    train_data_shuffler = {'name': 'train_shuffle', 'type': 'choice', 'values': [0]} # 0, 1
    with_graph_training = {'name': 'with_graph_training', 'type': 'choice', 'values': [0]}# 0, 1
    epochs = {'name': 'epoch', 'type': 'choice', 'values': [6]}
    #++++++++++++++++++++++++++++++++++
    search_space = [learning_rate, hop_score_name, with_graph_training, batch_size, epochs,
                    sent_threshold, answer_span_weight, pair_score_weight, accu_grad_batch,
                    task_name, frozen_layer_num, train_data_filtered, train_data_shuffler]
    search_space = dict((x['name'], x) for x in search_space)
    return search_space

def generate_random_search_bash(task_num):
    bash_save_path = '../gold_hotpot_jobs/'
    if os.path.exists(bash_save_path):
        remove_all_files(bash_save_path)
    if bash_save_path and not os.path.exists(bash_save_path):
        os.makedirs(bash_save_path)
    search_space = HypeParameterSpace()
    random_search_job =RandomSearchJob(search_space=search_space)
    for i in range(task_num):
        task_id, parameter_id = random_search_job.single_task_trial(42+i)
        with open(bash_save_path + 'gold_qa_run_' + task_id +'.sh', 'w') as rsh_i:
            command_i = 'bash goldqarun.sh ' + parameter_id
            rsh_i.write(command_i)
    print('{} jobs have been generated'.format(task_num))

if __name__ == '__main__':
    generate_random_search_bash(task_num=5)
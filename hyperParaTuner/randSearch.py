from numpy import random
import numpy as np

class RandomSearchJob(object):
    def __init__(self, search_space: dict):
        self.search_space = search_space
    def single_task_trial(self, rand_seed):
        """
        according to the parameter order in qarun.sh
        """
        parameter_dict = {}
        parameter_list = [[] for _ in range(16)]
        parameter_list[1] = self.rand_search_parameter(self.search_space['hop_model_name'])
        parameter_dict['hop'] = parameter_list[1]
        parameter_list[2] = self.rand_search_parameter(self.search_space['epoch'])
        parameter_dict['epo'] = str(parameter_list[2])
        parameter_list[3] = self.rand_search_parameter(self.search_space['batch_size'])
        parameter_dict['bs'] = str(parameter_list[3])
        parameter_list[4] = self.rand_search_parameter(self.search_space['learning_rate'])
        parameter_dict['lr'] = str(parameter_list[4])
        parameter_list[5] = self.rand_search_parameter(self.search_space['sent_threshold'])
        parameter_dict['sth'] = str(parameter_list[5])
        parameter_list[6] = self.rand_search_parameter(self.search_space['frozen_layer'])
        parameter_dict['fl'] = str(parameter_list[6])
        parameter_list[7] = self.rand_search_parameter(self.search_space['train_data_type'])
        parameter_dict['dt'] = str(parameter_list[7])
        parameter_list[8] = self.rand_search_parameter(self.search_space['train_shuffle'])
        parameter_dict['ts'] = str(parameter_list[8])
        parameter_list[9] = self.rand_search_parameter(self.search_space['span_weight'])
        parameter_dict['sw'] = str(parameter_list[9])
        parameter_list[10] = self.rand_search_parameter(self.search_space['pair_score_weight'])
        parameter_dict['psw'] = str(parameter_list[10])
        parameter_list[11] = self.rand_search_parameter(self.search_space['with_graph_training'])
        parameter_dict['wgt'] = str(parameter_list[11])
        parameter_list[12] = self.rand_search_parameter(self.search_space['task_name'])
        parameter_dict['tn'] = str(parameter_list[12])
        parameter_list[14] = rand_seed
        parameter_dict['rs'] = str(parameter_list[14])
        parameter_list[15] = self.rand_search_parameter(self.search_space['accu_grad_size'])
        parameter_dict['abs'] = str(parameter_list[15])
        task_id = '_'.join([k+'_' + v for k, v in parameter_dict.items()])
        parameter_list[13] = task_id
        parameter_id = ' '.join([str(para) for idx, para in enumerate(parameter_list) if idx > 0])
        return task_id, parameter_id

    def rand_search_parameter(self, space: dict):
        para_type = space['type']
        if para_type == 'fixed':
            return space['value']
        if para_type == 'choice':
            candidates = space['values']
            value = random.choice(candidates, 1)[0]
            return value
        if para_type == 'range':
            log_scale = space.get('log_scale', False)
            low, high = space['bounds']
            if log_scale:
                value = random.uniform(low=np.log(low), high=np.log(high), size=1)[0]
                value = np.exp(value)
            else:
                value = random.uniform(low=low, high=high,size=1)[0]
            return value
        else:
            raise ValueError('Training batch mode %s not supported' % para_type)
import os
import os.path as osp
import logging
from collections import OrderedDict
import json


def parse(opt_path, is_train=True):
    # remove comments starting with '//'
    json_str = ''
    with open(opt_path, 'r') as f:
        for line in f:
            line = line.split('//')[0] + '\n'
            json_str += line
    opt = json.loads(json_str, object_pairs_hook=OrderedDict)

    opt['is_train'] = is_train

    # path
    for key, path in opt['path'].items():
        if path and key in opt['path']:
            opt['path'][key] = os.path.expanduser(path)
    if is_train:
        experiments_root = os.path.join(opt['path']['root'], 'experiments', opt['name'])
        opt['path']['experiments_root'] = experiments_root
        opt['path']['models'] = os.path.join(experiments_root, 'models')
        opt['path']['training_state'] = os.path.join(experiments_root, 'training_state')
        opt['path']['log'] = experiments_root

        # change some options for debug mode
        if 'debug' in opt['name']:
            opt['train']['val_freq'] = 8
            opt['logger']['print_freq'] = 2
            opt['logger']['save_checkpoint_freq'] = 8
            opt['train']['lr_decay_iter'] = 10
    else:  # test
        results_root = os.path.join(opt['path']['root'], 'results', opt['name'])
        opt['path']['results_root'] = results_root
        opt['path']['log'] = results_root

    # export CUDA_VISIBLE_DEVICES
    gpu_list = ','.join(str(x) for x in opt['gpu_ids'])
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_list
    print('export CUDA_VISIBLE_DEVICES=' + gpu_list)

    return opt


class NoneDict(dict):
    def __missing__(self, key):
        return None


# convert to NoneDict, which return None for missing key.
def dict_to_nonedict(opt):
    if isinstance(opt, dict):
        new_opt = dict()
        for key, sub_opt in opt.items():
            new_opt[key] = dict_to_nonedict(sub_opt)
        return NoneDict(**new_opt)
    elif isinstance(opt, list):
        return [dict_to_nonedict(sub_opt) for sub_opt in opt]
    else:
        return opt


def dict2str(opt, indent_l=1):
    '''dict to string for logger'''
    msg = ''
    for k, v in opt.items():
        if isinstance(v, dict):
            msg += ' ' * (indent_l * 2) + k + ':[\n'
            msg += dict2str(v, indent_l + 1)
            msg += ' ' * (indent_l * 2) + ']\n'
        else:
            msg += ' ' * (indent_l * 2) + k + ': ' + str(v) + '\n'
    return msg


def check_resume(opt):
    '''Check resume states and pretrain_model paths'''
    logger = logging.getLogger('base')
    if opt['path']['resume_state']:

        state_idx = osp.basename(opt['path']['resume_state']).split('.')[0]
        opt['path']['pretrain_model'] = osp.join(opt['path']['models'],
                                                   '{}.pth'.format(state_idx))
        logger.info('Set [pretrain_model] to ' + opt['path']['pretrain_model'])
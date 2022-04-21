import os
import argparse
import torch
import utils


def collect_args_main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--experiment', choices=['baseline'])
    parser.add_argument('--experiment_name', type=str, default='_')
    parser.add_argument('--data_dir', type=str, default='data/celeba')
    parser.add_argument('--no_cuda', dest='cuda', action='store_false')
    parser.add_argument('--attribute', type=int, default=31)
    parser.add_argument('--protected_attribute', type=int, default=20)
    parser.add_argument('--test_mode', type=bool, default=False)
    parser.add_argument('--num_training_images', type=int, default=10000)
    parser.add_argument("--debug", action='store_true', help="In debug mode it's not saved the output log")
    parser.add_argument('--validation_check', type=int, default=500)
    parser.add_argument('--print_freq', type=int, default=250)
    parser.set_defaults(cuda=True)

    opt = vars(parser.parse_args())
    opt = create_experiment_setting(opt)
    return opt


def create_experiment_setting(opt):
    attr_list = utils.get_all_attr()
    attr_name = attr_list[opt['attribute']]
    opt['device'] = torch.device('cuda' if opt['cuda'] else 'cpu')
    opt['dtype'] = torch.float32

    orig_save = 'record/'

    if opt['protected_attribute'] != 20:
        orig_save += 'protected' + attr_list[opt['protected_attribute']] + '/'

    utils.make_dir('record')
    utils.make_dir(orig_save)

    if opt['experiment_name'] == '_':
        opt['save_folder'] = os.path.join(orig_save + opt['experiment'],
                                          attr_name)
        utils.make_dir(orig_save + opt['experiment'])
        utils.make_dir(opt['save_folder'])
    else:
        opt['save_folder'] = orig_save + opt['experiment_name'] + '/' + attr_name

        utils.make_dir(orig_save + opt['experiment_name'])
        utils.make_dir(opt['save_folder'])

    optimizer_setting = {
        'optimizer': torch.optim.Adam,
        'lr': 1e-4,
        'weight_decay': 0,
    }
    opt['optimizer_setting'] = optimizer_setting

    if opt['experiment'] == 'baseline':
        batch_size = 64

        params_real_train = {'batch_size': batch_size,
                             'shuffle': True,
                             'num_workers': 4}

        params_real_val = {'batch_size': batch_size,
                           'shuffle': False,
                           'num_workers': 4}

        data_setting = {
            'path': opt['data_dir'],
            'params_real_train': params_real_train,
            'params_real_val': params_real_val,
            'protected_attribute': opt['protected_attribute'],
            'attribute': opt['attribute'],
            'augment': True
        }
        opt['data_setting'] = data_setting

        opt['total_iterations'] = int(3000000 / batch_size)

        print('Total number of iterations: ', str(opt['total_iterations']))

    return opt




from os import path
import sys

from attr_classifier import attribute_classifier
from load_data import *
import parse_args
import itertools


def main(opt):
    print(opt)

    if opt['experiment'] == 'baseline':
        train = create_dataset_actual(
            opt['data_setting']['path'],
            opt['data_setting']['attribute'],
            opt['data_setting']['protected_attribute'],
            opt['data_setting']['params_real_train'],
            opt['data_setting']['augment'],
            CelebaDataset,
            split='train',
            num_training_images=opt['num_training_images'])

        val = create_dataset_actual(
            opt['data_setting']['path'],
            opt['data_setting']['attribute'],
            opt['data_setting']['protected_attribute'],
            opt['data_setting']['params_real_val'],
            False,
            CelebaDataset,
            split='valid')

        test = create_dataset_actual(
            opt['data_setting']['path'],
            opt['data_setting']['attribute'],
            opt['data_setting']['protected_attribute'],
            opt['data_setting']['params_real_val'],
            False,
            CelebaDataset,
            split='test')

    # Train the attribute classifier
    save_path = opt['save_folder'] + '/best.pth'
    save_path_curr = opt['save_folder'] + '/current.pth'

    # if test mode = True -> it's done only the evaluation on the saved checkpoint
    if not opt['test_mode']:
        # if debug the log output is not saved in a file
        if not opt['debug']:
            orig_stdout = sys.stdout
            f = open(opt['save_folder'] + '/out.txt', 'w')
            sys.stdout = f
            print(opt)

        print('Starting to train model...')
        model_path = None

        if path.exists(save_path_curr):
            print('Model exists, resuming training')
            model_path = save_path_curr

        AC = attribute_classifier(opt['device'], opt['dtype'], modelpath=model_path)

        train_loss = 0
        train_iter = itertools.cycle(train)
        start_it = AC.iteration

        print('Training with ' + str(len(train.dataset)) + ' samples')

        for i in range(start_it, opt['total_iterations']):

            AC.iteration = i
            train_loss_ = AC.do_iteration(train_iter)
            train_loss += train_loss_

            if i % opt['print_freq'] == 0 and i > 0:
                print('Training iteration {}:, loss:{}'.format(i, train_loss / i))

            if i % opt['validation_check'] == 0 and i > 0:
                y_all, pred_all = AC.get_scores(val)
                acc, acc_prot_attr_0, acc_prot_attr_1, _, _, _, _, _ = AC.check_metrics(y_all, pred_all)

                print('Accuracy = {}'.format(acc * 100), 'Acc. Prot. Attr. 0 = {}'.format(acc_prot_attr_0 * 100),
                      'Acc. Prot. Attr. 1 = {}'.format(acc_prot_attr_1 * 100))

                min_group_acc = min(acc_prot_attr_0, acc_prot_attr_1)

                if (min_group_acc.cuda() > AC.best_acc):
                    AC.best_acc = min_group_acc
                    AC.save_model(save_path)
                AC.save_model(save_path_curr)

    AC = attribute_classifier(opt['device'], opt['dtype'], modelpath=save_path)

    test_targets, test_scores = AC.get_scores(test)

    print('Test results: ')
    acc, acc_prot_attr_0, acc_prot_attr_1, TPR, TPR_prot_attr_0, TPR_prot_attr_1, DEO, DEODD = AC.check_metrics(
        test_targets, test_scores)
    print('Accuracy = {}'.format(acc * 100), 'Acc. Prot. Attr. 0 = {}'.format(acc_prot_attr_0 * 100),
          'Acc. Prot. Attr. 1 = {}'.format(acc_prot_attr_1 * 100))
    print('TPR = {}'.format(TPR * 100), 'TPR Prot. Attr. 0 = {}'.format(TPR_prot_attr_0 * 100),
          'TPR Prot. Attr. 1 = {}'.format(TPR_prot_attr_1 * 100))
    print('DEO = {}'.format(DEO * 100), 'DEODD = {}'.format(DEODD * 100))

    if not opt['debug']:
        if not opt['test_mode']:
            sys.stdout = orig_stdout
            f.close()


if __name__ == "__main__":
    opt = parse_args.collect_args_main()

    main(opt)

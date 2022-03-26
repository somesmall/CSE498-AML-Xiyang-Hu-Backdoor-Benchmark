'''
logic of load:
1. yaml file, if yaml setting name is given then find the yaml setting
2.
3. argparse overwrite args from yaml file if any in args is not None
(so ANY params in add_args should have NO default value except yaml config and yaml setting name)
4. delete any params in args with value None
'''

import sys, yaml, os

os.chdir(sys.path[0])
sys.path.append('../')
os.getcwd()

import argparse
from pprint import  pformat
import numpy as np
import torch
from utils.aggregate_block.save_path_generate import generate_save_folder
import time
import logging

from utils.aggregate_block.fix_random import fix_random
from utils.aggregate_block.dataset_and_transform_generate import dataset_and_transform_generate
from utils.bd_dataset import prepro_cls_DatasetBD
from torch.utils.data import DataLoader
from utils.backdoor_generate_pindex import generate_pidx_from_label_transform
from utils.aggregate_block.bd_attack_generate import bd_attack_img_trans_generate, bd_attack_label_trans_generate
from copy import deepcopy
from utils.aggregate_block.model_trainer_generate import generate_cls_model, generate_cls_trainer
from utils.aggregate_block.train_settings_generate import argparser_opt_scheduler, argparser_criterion
from utils.save_load_attack import save_attack_result



def add_args(parser):
    """
    parser : argparse.ArgumentParser
    return a parser added with args required by fit
    """
    # Training settings
    # parser.add_argument('--mode', type=str,
    #                     help='classification/detection/segmentation')
    parser.add_argument('--device', type = str)
    parser.add_argument('--attack', type = str, )
    parser.add_argument('--yaml_path', type=str, default='../config/SSBAAttack/default.yaml',
                        help='path for yaml file provide additional default attributes')
    parser.add_argument('--lr_scheduler', type=str,
                        help='which lr_scheduler use for optimizer')
    # only all2one can be use for clean-label
    parser.add_argument('--attack_label_trans', type=str,
                        help='which type of label modification in backdoor attack'
                        )
    parser.add_argument('--pratio', type=float,
                        help='the poison rate '
                        )
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--dataset', type=str,
                        help='which dataset to use'
                        )
    parser.add_argument('--dataset_path', type=str)
    parser.add_argument('--attack_target', type=int,
                        help='target class in all2one attack')
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--img_size', type=list)
    parser.add_argument('--lr', type=float)
    parser.add_argument('--steplr_stepsize', type=int)
    parser.add_argument('--steplr_gamma', type=float)
    parser.add_argument('--num_classes', type=int)
    parser.add_argument('--sgd_momentum', type=float)
    parser.add_argument('--wd', type=float, help='weight decay of sgd')
    parser.add_argument('--steplr_milestones', type=list)
    parser.add_argument('--client_optimizer', type=int)
    parser.add_argument('--random_seed', type=int,
                        help='random_seed')
    parser.add_argument('--frequency_save', type=int,
                        help=' frequency_save, 0 is never')
    parser.add_argument('--model', type=str,
                        help='choose which kind of model')
    parser.add_argument('--save_folder_name', type=str,
                        help='(Optional) should be time str + given unique identification str')
    parser.add_argument('--git_hash', type=str,
                        help='git hash number, in order to find which version of code is used')
    return parser

def main():

    parser = (add_args(argparse.ArgumentParser(description=sys.argv[0])))
    args = parser.parse_args()

    with open(args.yaml_path, 'r') as f:
        defaults = yaml.safe_load(f)

    defaults.update({k:v for k,v in args.__dict__.items() if v is not None})

    args.__dict__ = defaults

    args.terminal_info = sys.argv

    if 'save_folder_name' not in args:
        save_path = generate_save_folder(
            run_info=('afterwards' if 'load_path' in args.__dict__ else 'attack') + '_' + args.attack,
            given_load_file_path=args.load_path if 'load_path' in args else None,
            all_record_folder_path='../record',
        )
    else:
        save_path = '../record/' + args.save_folder_name
        os.mkdir(save_path)

    args.save_path = save_path

    torch.save(args.__dict__, save_path + '/info.pickle')



    # logFormatter = logging.Formatter("%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s")
    logFormatter = logging.Formatter(
        fmt='%(asctime)s [%(levelname)-8s] [%(filename)s:%(lineno)d] %(message)s',
        datefmt='%Y-%m-%d:%H:%M:%S',
    )
    logger = logging.getLogger()
    # logFormatter = logging.Formatter("%(asctime)s [%(levelname)-5.5s] %(message)s")

    fileHandler = logging.FileHandler(save_path + '/' + time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime()) + '.log')
    fileHandler.setFormatter(logFormatter)
    logger.addHandler(fileHandler)

    consoleHandler = logging.StreamHandler()
    consoleHandler.setFormatter(logFormatter)
    logger.addHandler(consoleHandler)

    logger.setLevel(logging.INFO)
    logging.info(pformat(args.__dict__))

    try:
        import wandb
        wandb.init(
            project="bdzoo2",
            entity="chr",
            name=('afterwards' if 'load_path' in args.__dict__ else 'attack') + '_' + os.path.basename(save_path),
            config=args,
        )
        set_wandb = True
    except:
        set_wandb = False
    logging.info(f'set_wandb = {set_wandb}')



    fix_random(int(args.random_seed))



    train_dataset_without_transform, \
                train_img_transform, \
                train_label_transfrom, \
    test_dataset_without_transform, \
                test_img_transform, \
                test_label_transform = dataset_and_transform_generate(args)




    benign_train_dl = DataLoader(
        prepro_cls_DatasetBD(
            full_dataset_without_transform=train_dataset_without_transform,
            poison_idx=np.zeros(len(train_dataset_without_transform)),  # one-hot to determine which image may take bd_transform
            bd_image_pre_transform=None,
            bd_label_pre_transform=None,
            ori_image_transform_in_loading=train_img_transform,
            ori_label_transform_in_loading=train_label_transfrom,
            add_details_in_preprocess=True,
        ),
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True
    )

    benign_test_dl = DataLoader(
        prepro_cls_DatasetBD(
            test_dataset_without_transform,
            poison_idx=np.zeros(len(test_dataset_without_transform)),  # one-hot to determine which image may take bd_transform
            bd_image_pre_transform=None,
            bd_label_pre_transform=None,
            ori_image_transform_in_loading=test_img_transform,
            ori_label_transform_in_loading=test_label_transform,
            add_details_in_preprocess=True,
        ),
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
    )




    train_bd_img_transform, test_bd_img_transform = bd_attack_img_trans_generate(args)

    bd_label_transform = bd_attack_label_trans_generate(args)



    train_pidx = generate_pidx_from_label_transform(
        benign_train_dl.dataset.targets,
        label_transform=bd_label_transform,
        train=True,
        pratio= args.pratio if 'pratio' in args.__dict__ else None,
        p_num= args.p_num if 'p_num' in args.__dict__ else None,
    )
    torch.save(train_pidx,
        args.save_path + '/train_pidex_list.pickle',
    )

    adv_train_ds = prepro_cls_DatasetBD(
        deepcopy(train_dataset_without_transform),
        poison_idx= train_pidx,
        bd_image_pre_transform=train_bd_img_transform,
        bd_label_pre_transform=bd_label_transform,
        ori_image_transform_in_loading=train_img_transform,
        ori_label_transform_in_loading=train_label_transfrom,
        add_details_in_preprocess=True,
    )

    adv_train_dl = DataLoader(
        dataset = adv_train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
    )

    test_pidx = generate_pidx_from_label_transform(
        benign_test_dl.dataset.targets,
        label_transform=bd_label_transform,
        train=False,
    )

    adv_test_dataset = prepro_cls_DatasetBD(
        deepcopy(test_dataset_without_transform),
        poison_idx=test_pidx,
        bd_image_pre_transform=test_bd_img_transform,
        bd_label_pre_transform=bd_label_transform,
        ori_image_transform_in_loading=test_img_transform,
        ori_label_transform_in_loading=test_label_transform,
        add_details_in_preprocess=True,
    )

    adv_test_dataset.subset(
        np.where(test_pidx == 1)[0]
    )

    adv_test_dl = DataLoader(
        dataset = adv_test_dataset,
        batch_size= args.batch_size,
        shuffle= False,
        drop_last= False,
    )



    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    net  = generate_cls_model(
        model_name=args.model,
        num_classes=args.num_classes,
    )

    trainer = generate_cls_trainer(
        net,
        args.attack
    )



    criterion = argparser_criterion(args)

    optimizer, scheduler = argparser_opt_scheduler(net, args)




    if 'load_path' not in args.__dict__:

        trainer.train_with_test_each_epoch(
            train_data = adv_train_dl,
            test_data = benign_test_dl,
            adv_test_data = adv_test_dl,
            end_epoch_num = args.epochs,
            criterion = criterion,
            optimizer = optimizer,
            scheduler = scheduler,
            device = device,
            frequency_save = args.frequency_save,
            save_folder_path = save_path,
            save_prefix = 'attack',
            continue_training_path = None,
        )

    else:

        if 'recover' not in args.__dict__ or args.recover == False :

            print('finetune so use less data, 5% of benign train data')

            benign_train_dl.dataset.subset(
                np.random.choice(
                    np.arange(
                        len(benign_train_dl.dataset)),
                    size=round((len(benign_train_dl.dataset)) / 20),  # 0.05
                    replace=False,
                )
            )

            torch.save(
                list(benign_train_dl.dataset.original_index),
                args.save_path + '/finetune_idx_list.pt',
            )

            trainer.train_with_test_each_epoch(
                train_data=benign_train_dl,
                test_data=benign_test_dl,
                adv_test_data=adv_test_dl,
                end_epoch_num=args.epochs,
                criterion=criterion,
                optimizer=optimizer,
                scheduler=scheduler,
                device=device,
                frequency_save=args.frequency_save,
                save_folder_path=save_path,
                save_prefix='finetune',
                continue_training_path=args.load_path,
                only_load_model=True,
            )

        elif 'recover' in args.__dict__ and args.recover == True :

            trainer.train_with_test_each_epoch(
                train_data=adv_train_dl,
                test_data=benign_test_dl,
                adv_test_data=adv_test_dl,
                end_epoch_num=args.epochs,
                criterion=criterion,
                optimizer=optimizer,
                scheduler=scheduler,
                device=device,
                frequency_save=args.frequency_save,
                save_folder_path=save_path,
                save_prefix='attack',
                continue_training_path=args.load_path,
                only_load_model=False,
            )


    #
    # torch.save(
    #         {
    #             'model_name':args.model,
    #             'model': trainer.model.cpu().state_dict(),
    #             'clean_train': {
    #                 'x' : torch.tensor(nHWC_to_nCHW(benign_train_dl.dataset.data)).float().cpu(),
    #                 'y' : torch.tensor(benign_train_dl.dataset.targets).long().cpu(),
    #             },
    #
    #             'clean_test' : {
    #                 'x' : torch.tensor(nHWC_to_nCHW(benign_test_dl.dataset.data)).float().cpu(),
    #                 'y' : torch.tensor(benign_test_dl.dataset.targets).long().cpu(),
    #             },
    #
    #             'bd_train': {
    #                 'x' : torch.tensor(nHWC_to_nCHW(adv_train_ds.data)).float().cpu(),
    #                 'y' : torch.tensor(adv_train_ds.targets).long().cpu(),
    #             },
    #
    #             'bd_test': {
    #                 'x': torch.tensor(nHWC_to_nCHW(adv_test_dataset.data)).float().cpu(),
    #                 'y' : torch.tensor(adv_test_dataset.targets).long().cpu(),
    #             },
    #         },
    #     f'{save_path}/attack_result.pt'
    # )



    save_attack_result(
        model_name = args.model,
        num_classes = args.num_classes,
        model = trainer.model.cpu().state_dict(),
        data_path = args.dataset_path,
        img_size = args.img_size,
        clean_data = args.dataset,
        bd_train = adv_train_ds,
        bd_test = adv_test_dataset,
        save_path = save_path,
    )

if __name__ == '__main__':
    main()
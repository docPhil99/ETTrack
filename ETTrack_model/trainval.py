import argparse
import ast
import os
from loguru import logger
import torch
import yaml

from src.processor import processor

# Use Deterministic mode and set random seed
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.manual_seed(0)


def get_parser():
    parser = argparse.ArgumentParser(
        description='STAR')
    parser.add_argument('--dataset', default='MOT17') # MOT20 dancetrack sportsmot
    parser.add_argument('--save_dir')
    parser.add_argument('--model_dir')
    parser.add_argument('--config')
    parser.add_argument('--using_cuda', default=True, type=ast.literal_eval)
    parser.add_argument('--test_set', default='TEST_MOT17', type=str, #MOT20
                        help='')
    parser.add_argument('--train_data_path', default='/home/hxd/cv_project/STAR_LASTERv4/datasets/MOT17/train',
                        help='path')
    parser.add_argument('--val_data_path', default='/home/hxd/cv_project/STAR_LASTERv4/datasets/MOT17/val',
                        help='path')
    parser.add_argument('--base_dir', default='.', help='Base directory including these scripts.')
    parser.add_argument('--save_base_dir', default='/home/hxd/cv_project/STAR_LASTERv4/', help='Directory for saving caches and models.')
    parser.add_argument('--phase', default='train', help='Set this value to \'train\' or \'test\'')
    parser.add_argument('--train_model', default='star', help='Your model name')
    parser.add_argument('--load_model', default=1, type=str, help="load pretrained model for test or training")
    parser.add_argument('--model', default='star.STAR')
    parser.add_argument('--seq_length', default=20, type=int)
    parser.add_argument('--obs_length', default=4, type=int)
    parser.add_argument('--pred_length', default=12, type=int)
    parser.add_argument('--batch_around_ped', default=256, type=int)
    parser.add_argument('--batch_size', default=16, type=int)  # 4
    parser.add_argument('--test_batch_size', default=16, type=int) #4
    parser.add_argument('--show_step', default=50, type=int)
    parser.add_argument('--start_test', default=0, type=int)
    parser.add_argument('--sample_num', default=20, type=int)
    parser.add_argument('--num_epochs', default=200, type=int)
    parser.add_argument('--ifshow_detail', default=True, type=ast.literal_eval)
    parser.add_argument('--ifsave_results', default=False, type=ast.literal_eval)
    parser.add_argument('--randomRotate', default=False, type=ast.literal_eval,
                        help="=True:random rotation of each trajectory fragment")
    parser.add_argument('--randomDrop', default=False, type=ast.literal_eval,
                        help="=True:random rotation of each trajectory fragment")
    parser.add_argument('--neighbor_thred', default=1, type=int)
    parser.add_argument('--neighbor_thred_mot20', default=100, type=int)
    parser.add_argument('--learning_rate', default=0.0015, type=float)
    parser.add_argument('--clip', default=1, type=int)


    parser.add_argument('-d_model', type=int, default=521) #521
    parser.add_argument('-d_inner_hid', type=int, default=2048)#2048
    parser.add_argument('-d_k', type=int, default=64)#64
    parser.add_argument('-d_v', type=int, default=64)#64
    parser.add_argument('-warmup', '--n_warmup_steps', type=int, default=4000)
    parser.add_argument('-lr_mul', type=float, default=2.0)
    parser.add_argument('-n_head', type=int, default=8)
    parser.add_argument('-n_layers', type=int, default=6)
    parser.add_argument('-dropout', type=float, default=0.1)
    logger.debug('created parser')
    return parser


def load_arg(p):
    # save arg
    logger.info(f'Attemping to load {p.config}')
    if os.path.exists(p.config):
        logger.info(f'Loading config: {p.config}')
        with open(p.config, 'r') as f:
            default_arg = yaml.load(f, Loader=yaml.FullLoader)
        key = vars(p).keys()
        for k in default_arg.keys():
            if k not in key:
                print('WRONG ARG: {}'.format(k))
                try:
                    assert (k in key)
                except:
                    s = 1
        parser.set_defaults(**default_arg)
        return parser.parse_args()
    else:
        logger.debug('No config file to load')
        return False


def save_arg(args):
    # save arg
    arg_dict = vars(args)
    if not os.path.exists(args.model_dir):
        os.makedirs(args.model_dir)
    logger.info(f"Saving config {args.config}")
    with open(args.config, 'w') as f:
        yaml.dump(arg_dict, f)


if __name__ == '__main__':
    logger.debug(f'Starting trainval with torch version {torch.__version__}')
    parser = get_parser()
    p = parser.parse_args()
   # p.batch_around_ped = 16
    p.save_dir = p.save_base_dir + str(p.test_set) + '/'
    p.model_dir = p.save_base_dir + str(p.test_set) + '/' + p.train_model + '/'
    p.config = p.model_dir + '/config_' + p.phase + '.yaml'

    if not load_arg(p):
        save_arg(p)

    args = load_arg(p)
    torch.cuda.set_device(0)

    trainer = processor(args)

    if args.phase == 'test':
        logger.info('Starting test phase')
        net = trainer.test()
        print("well,done")
    else:
        logger.info('Starting training phase')
        trainer.train()

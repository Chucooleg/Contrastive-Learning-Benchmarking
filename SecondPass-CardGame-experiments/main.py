import os
import sys
import json
import time
import datetime
import pprint
import numpy as np

import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor

import wandb
wandb.login()
from pytorch_lightning.loggers import WandbLogger

from datamodule import GameDataModule
from trainmodule import ContrastiveTrainModule, GenerativeTrainModule

from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser


def load_data(data_path):
    with open(data_path, 'r') as f:
        data = json.load(f)
    print('---------data----------')
    for k in data:
        if not 'datapoints' in k and not 'tokens' in k and not 'gt_idxs' in k:
            print(k, ':', data[k])
        else:
            print(k,'length :', len(data[k]))
    print('-----------------------')
    return data


def validate_data(data):
    for key in (
        'key_support_size',
        'num_attributes', 'num_attr_vals', 
        'train_tokens', 'val_tokens'
    ):
        try:
            assert key in data, f'{key} not found in data'
        except:
            breakpoint()


def load_hparams(args, data):
    with open(args.config_path, 'r') as f:
        hparams = json.load(f)

    if args.mode in ('train', 'param_summary'):
        hparams['key_support_size'] = data['key_support_size']
        hparams['num_attributes'] = data['num_attributes']
        hparams['num_attr_vals'] = data['num_attr_vals']
        hparams['nest_depth_int'] = data['nest_depth_int']
        hparams['query_length_multiplier'] = data['query_length_multiplier']
        hparams['multiple_OR_sets_bool'] = data['multiple_OR_sets_bool']
        hparams['vocab_size'] = data['vocab_size']
        hparams['('] = data['(']
        hparams[')'] = data[')']
        hparams['NULL'] = data['NULL']
        hparams['SEP'] = data['SEP']
        hparams['SOS'] = data['SOS']
        hparams['EOS'] = data['EOS']
        hparams['PAD'] = data['PAD']
        hparams['PLH'] = data['PLH']
        hparams['&'] = data['&']
        hparams['|'] = data['|']

        hparams['max_len_q'] = data['max_len_q'] + 2 # <SOS>---
        hparams['len_k'] = data['len_k']

        assert hparams['model'] in ("contrastive", "generative")
        if hparams['embedding_by_property']:
            assert hparams["encoder"] in ('transformer')
            assert hparams["decoder"] in ('transformer')
        else:
            hparams["encoder"] = 'lookup'
            hparams["decoder"] = 'lookup'

    if hparams['model'] == 'contrastive' and (not 'contrastive_use_infoNCE' in hparams):
        hparams['contrastive_use_infoNCE'] = True
    if hparams['model'] == 'contrastive' and (not 'normalize_dotproduct' in hparams):
        hparams['normalize_dotproduct'] = False

    if args.mode == 'resume_train':
        hparams['max_epochs'] = args.resume_max_epochs

    print('----------hparams----------')
    for k in hparams:
        print(k, ':', hparams[k])
    print('---------------------------')
    return hparams  

def run_test(args, hparams, ckpt_name, trainmodule, datamodule, ckpt_dir_PATH, figsize=(10,15)):

    checkpoint_PATH = os.path.join(ckpt_dir_PATH, ckpt_name) #'last.ckpt'
    checkpoint = torch.load(checkpoint_PATH, map_location=lambda storage, loc: storage)
    trainmodule.load_state_dict(checkpoint['state_dict'])

    trainer = pl.Trainer(
        gpus=[args.gpu],
        min_epochs=1, max_epochs=1, 
        precision=32, 
        log_gpu_memory='all',
        weights_summary = 'full',
        gradient_clip_val=hparams['gradient_clip_val'],
    )
    res = trainer.test(model=trainmodule, datamodule=datamodule)
    
def resume_train(args, hparams, project_name, run_Id, trainmodule, datamodule, ckpt_dir_PATH, wd_logger):
    
    checkpoint_PATH = os.path.join(ckpt_dir_PATH, 'last.ckpt')
    run_PATH = os.path.join(project_name, run_Id) # also from wandb interface '1ih8yza5'

    wandb.restore(checkpoint_PATH, run_path=run_PATH)
    checkpoint = torch.load(checkpoint_PATH, map_location=lambda storage, loc: storage)
    trainmodule.load_state_dict(checkpoint['state_dict'])
 
    checkpoint_callback = ModelCheckpoint(
        monitor='avg_val_accuracy_by_Query',
        dirpath=ckpt_dir_PATH,
        filename='{epoch:02d}-{step:02d}-{val_loss:.2f}',
        save_top_k=3,
        save_last=True,
        mode='max',
    )

    lr_monitor = LearningRateMonitor(logging_interval='step')

    trainer = pl.Trainer(
        gpus=[args.gpu], 
        min_epochs=2, max_epochs=hparams['max_epochs'], 
        check_val_every_n_epoch=hparams['val_every_n_epoch'],
        precision=32, 
        logger=wd_logger,
        log_gpu_memory='all',
        weights_summary = 'full',
        gradient_clip_val=hparams['gradient_clip_val'],
        callbacks=[checkpoint_callback, lr_monitor],
    )
    
    with torch.autograd.detect_anomaly():
        trainer.fit(trainmodule, datamodule)

    wandb.save(os.path.join(ckpt_dir_PATH, 'last.ckpt'))


def run_train(args, hparams, trainmodule, datamodule, ckpt_dir_PATH, wd_logger):

    # checkpoints
    checkpoint_callback = ModelCheckpoint(
        monitor='avg_val_accuracy_by_Query',
        dirpath=ckpt_dir_PATH,
        filename='{epoch:02d}-{step:02d}-{val_loss:.2f}',
        save_top_k=3,
        save_last=True,
        mode='max',
    )

    # monitors
    lr_monitor = LearningRateMonitor(logging_interval='step')

    # trainer
    trainer = pl.Trainer(
        gpus=[args.gpu], 
        min_epochs=2, max_epochs=hparams['max_epochs'], 
        check_val_every_n_epoch=hparams['val_every_n_epoch'],
        precision=32, 
        logger=wd_logger,
        log_gpu_memory='all',
        weights_summary='full',
        gradient_clip_val=hparams['gradient_clip_val'],
        callbacks=[checkpoint_callback, lr_monitor],
        profiler="simple"
        # num_sanity_val_steps=0,
    )

    #fit
    with torch.autograd.detect_anomaly():
        trainer.fit(trainmodule, datamodule)

    wandb.save(os.path.join(ckpt_dir_PATH, 'last.ckpt'))


def validate_inputs(args, hparams):
    assert os.path.exists(args.config_path), 'config_path does not exist'
    assert os.path.exists(args.data_path), 'data_path does not exist' 
    assert os.path.exists(args.checkpoint_dir), f'checkpoint_dir {args.checkpoint_dir} does not exist'
    assert args.mode in ('train', 'resume_train', 'test_full', 'param_summary')
    if args.mode in ('resume_train', 'test_full'):
        assert args.runID, 'missing runID, e.g. 1lygiuq3'
        assert args.project_name, 'missing project_name. e.g. ContrastiveLearning-cardgame-Scaling'
    if args.mode == 'test_full':
        assert args.ckpt_name, 'missing ckpt_name for testing. e.g. last.ckpt'


def main(args):

    game_data = load_data(args.data_path)
    validate_data(game_data)
    hparams = load_hparams(args, game_data)
    validate_inputs(args, hparams)

    # approve model summary before training
    if args.approve_before_training:
        val = input("Continue to build model? (y/n)")
        if not val.lower() in ('yes', 'y'):
            sys.exit(1)

        hparams['key_support_size']

    pl.seed_everything(hparams['seed'])

    # model
    Module = ContrastiveTrainModule if hparams['model'] == 'contrastive' else GenerativeTrainModule
    trainmodule =  Module(hparams, gt_distributions={})
    model_summary = pl.core.memory.ModelSummary(trainmodule, mode='full')
    print(model_summary,'\n')

    if args.mode == 'param_summary':
        sys.exit(1)

    # dataset
    game_datamodule = GameDataModule(
        hparams = hparams,
        raw_data = game_data,
    )   

    # logger
    run_name = 'SimpleShatter;{},seed{};queryOrder{};attr{}-val{}-nest{};{};d_model{};{};params{}K'.format(
        hparams['model'],
        hparams['seed'],
        hparams['simple_shatter_query_order'],
        hparams['num_attributes'], hparams['num_attr_vals'], hparams['nest_depth_int'],
        'embedByProperty' if hparams['embedding_by_property'] else 'lookupTable',
        hparams['d_model'],
        'dot-product' if hparams['dotproduct_bottleneck'] else 'non-linear',
        round(max(model_summary.param_nums)/1000,2))
    # project_name = 'ContrastiveLearning-cardgame-Scaling-SET-FirstPass'
    project_name = 'ContrastiveLearning-simple-Shattering-sampling-tests'
    if args.mode == 'train':
        wd_logger = WandbLogger(name=run_name, project=project_name)
    else:
        wd_logger = WandbLogger(name=run_name, project=project_name, id=args.runID if args.runID else None)
    print('RUN NAME :\n', run_name)

    # approve model summary before training
    if args.approve_before_training:
        val = input("Continue training? (y/n)")
        if not val.lower() in ('yes', 'y'):
            sys.exit(1)

    # check point path
    if args.mode == 'train':
        ts = time.time()
        TIMESTAMP = st = datetime.datetime.fromtimestamp(ts).strftime('%Y%m%d-%H%M%S-')
        ckpt_dir_PATH = os.path.join(args.checkpoint_dir, project_name, TIMESTAMP+run_name)
    else:
        ckpt_dir_PATH = '/'.join(args.config_path.split('/')[:-1])
    print('Checkpoint Path:\n', ckpt_dir_PATH)
    os.makedirs(ckpt_dir_PATH, exist_ok=True)

    # save config
    with open(os.path.join(ckpt_dir_PATH, 'config.json'), 'w') as f:
        json.dump(hparams, f)

    # train
    if args.mode == 'train':
        run_train(
            args, hparams, trainmodule, game_datamodule, ckpt_dir_PATH, wd_logger
        )
    elif args.mode == 'resume_train':
        resume_train(
            args, hparams, project_name, args.runID, trainmodule, game_datamodule, ckpt_dir_PATH, wd_logger
        )
    else: # test
        run_test(args, hparams, args.ckpt_name, trainmodule, game_datamodule, ckpt_dir_PATH, figsize=(10,15))

    return trainmodule, game_datamodule


if __name__ == '__main__':
    # parser
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)

    # settings
    parser.add_argument('--config_path', help='path to config.json')
    parser.add_argument('--data_path', help='path to data json file')
    parser.add_argument('--checkpoint_dir', help='path to save and load checkpoints.')
    parser.add_argument('--mode', help='train, resume_train, test_full')
    parser.add_argument('--resume_max_epochs', default=None, help='must provide if resume training or testing')
    parser.add_argument('--runID', default=None, help='must provide if resume training or testing')
    parser.add_argument('--project_name', default=None, help='must provide if resume training or testing')
    parser.add_argument('--ckpt_name', default=None, help='must provide if resume training or testing')
    parser.add_argument('--gpu', help='gpu id', type=int)
    parser.add_argument(
        '--approve_before_training', help='1/0. Prompt for user to approve model configuration for training.', type=int
    )  

    # args and init
    args = parser.parse_args()

    main(args)
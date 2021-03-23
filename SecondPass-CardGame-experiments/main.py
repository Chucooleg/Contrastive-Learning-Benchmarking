import os
import sys
import json
import time
import datetime
import pathlib
import pprint
import numpy as np
import shutil
import tempfile

import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor

import wandb
from pytorch_lightning.loggers import WandbLogger

from datamodule import GameDataModule
from trainmodule import ContrastiveTrainModule, GenerativeTrainModule

from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser


def _get_aml_run_context():
    import azureml.core
    return azureml.core.Run.get_context()


def wandb_login(aml):
    if aml:
        run = _get_aml_run_context()
        os.environ['WANDB_API_KEY'] = run.get_secret(name='WANDBAPIKEY')
    wandb.login()


def load_aml_dataset(dataset_name):
    import azureml.core
    run = _get_aml_run_context()
    ws = run.experiment.workspace
    dataset = azureml.core.Dataset.get_by_name(ws, name=args.dataset_name)
    mounted_path = tempfile.mkdtemp()
    print(f'will mount dataset to {mounted_path}')

    # mount dataset onto the mounted_path of a Linux-based compute
    mount_context = dataset.mount(mounted_path)

    mount_context.start()
    data_paths = list(pathlib.Path(mounted_path).glob('*.json'))

    assert len(data_paths) == 1, data_paths
    return data_paths[0].absolute()


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


def get_full_config_path(args):
    if args.aml:
        return pathlib.Path(__file__).parent / args.config_path
    else:
        return args.config_path


def load_hparams(args, data):
    with open(get_full_config_path(args), 'r') as f:
        hparams = json.load(f)

    hparams['mode'] = args.mode
    #####################
    hparams['key_support_size'] = data['key_support_size']
    hparams['num_attributes'] = data['num_attributes']
    hparams['num_attr_vals'] = data['num_attr_vals']
    hparams['union_only'] = data['union_only']
    hparams['vocab_size'] = data['vocab_size']

    hparams['vocab_by_property'] = data['vocab_by_property']
    if 'symbol_vocab_token_lookup' in data.keys():
        sym_lookup = data['symbol_vocab_token_lookup']
    else:
        sym_lookup = data
    for k in ('(', ')', 'NULL', 'SEP', 'SOS', 'EOS', 'PAD', 'PLH', '|', '!'):
        hparams[k] = sym_lookup[k]

    hparams['max_len_q'] = data['max_len_q'] 
    hparams['N_pairs'] = data.get('N_pairs', 3)
    hparams['len_k'] = data['len_k']
    if not hparams['vocab_by_property']:
        assert hparams['len_k'] == 1

    assert hparams['model'] in ("contrastive", "generative")
     #####################


    if hparams['model'] == 'contrastive' and (not 'contrastive_use_infoNCE' in hparams):
        hparams['contrastive_use_infoNCE'] = True
    if hparams['model'] == 'contrastive' and (not 'normalize_dotproduct' in hparams):
        hparams['normalize_dotproduct'] = False

    if args.mode == 'resume_train':
        hparams['max_epochs'] = int(args.resume_max_epochs)

    if args.mode in ('resume_train', 'test', 'test_marginal'):
        hparams['resume_checkpoint_dir'] = args.resume_checkpoint_dir 

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
        # log_every_n_steps=1,
    )
    res = trainer.test(model=trainmodule, datamodule=datamodule)
    
def resume_train(args, hparams, project_name, run_Id, trainmodule, datamodule, ckpt_dir_PATH, wd_logger):
    
    checkpoint_PATH = os.path.join(ckpt_dir_PATH, 'last.ckpt')
    shutil.copyfile(checkpoint_PATH, os.path.join(ckpt_dir_PATH, 'last_backup.ckpt'))

    run_PATH = os.path.join(project_name, run_Id) # also from wandb interface '1ih8yza5'

    wandb.restore(checkpoint_PATH, run_path=run_PATH)
    checkpoint = torch.load(checkpoint_PATH, map_location=lambda storage, loc: storage)
    trainmodule.load_state_dict(checkpoint['state_dict'])
 
    checkpoint_callback = ModelCheckpoint(
        monitor='val_KL_loss_per_example',
        dirpath=ckpt_dir_PATH,
        filename='val_KL_loss_{epoch:02d}-{step:02d}-{val_KL_loss_per_example:.5f}',
        save_top_k=2,
        save_last=True,
        mode='min',
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
        # log_every_n_steps=1,
    )
    
    # with torch.autograd.detect_anomaly():
    trainer.fit(trainmodule, datamodule)

    wandb.save(os.path.join(ckpt_dir_PATH, 'last.ckpt'))


def run_train(args, hparams, trainmodule, datamodule, ckpt_dir_PATH, wd_logger):

    # checkpoints
    checkpoint_callback = ModelCheckpoint(
        monitor='val_KL_loss_per_example',
        dirpath=ckpt_dir_PATH,
        filename='val_KL_loss_{epoch:02d}-{step:02d}-{val_KL_loss_per_example:.5f}',
        save_top_k=2,
        save_last=True,
        mode='min',
    )

    # monitors
    lr_monitor = LearningRateMonitor(logging_interval='step')

    # https://discuss.pytorch.org/t/i-have-3-gpu-why-torch-cuda-device-count-only-return-1/7245/4
    import torch
    import sys
    print('__Python VERSION:', sys.version)
    print('__pyTorch VERSION:', torch.__version__)
    print('__CUDA VERSION')
    from subprocess import call
    # call(["nvcc", "--version"]) does not work
    print('__CUDNN VERSION:', torch.backends.cudnn.version())
    print('__Number CUDA Devices:', torch.cuda.device_count())
    print('__Devices')
    call(["nvidia-smi", "--format=csv", "--query-gpu=index,name,driver_version,memory.total,memory.used,memory.free"])
    print('Active CUDA Device: GPU', torch.cuda.current_device())

    print ('Available devices ', torch.cuda.device_count())
    print ('Current cuda device ', torch.cuda.current_device())

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
        profiler="simple",
        # log_every_n_steps=1,
        # num_sanity_val_steps=0,
    )

    #fit
    # with torch.autograd.detect_anomaly():
    trainer.fit(trainmodule, datamodule)

    wandb.save(os.path.join(ckpt_dir_PATH, 'last.ckpt'))


def validate_args(args):
    if args.aml:
        assert args.data_path is None
        assert args.dataset_name is not None
    else:
        assert os.path.exists(args.data_path), 'data_path does not exist'
        assert args.dataset_name is None
    assert args.project_name, 'missing project name. e.g. ContrastiveLearning-cardgame-Scaling'

    if args.mode in ('train', 'param_summary'):
        config_path = get_full_config_path(args)
        assert os.path.exists(config_path), 'New config_path does not exist'
    if args.mode == 'train':
        if args.aml:
            assert args.checkpoint_dir is None, 'AML expects checkpoints to be written to ./outputs'
        else:
            assert os.path.exists(args.checkpoint_dir), f'New checkpoint_dir {args.checkpoint_dir} does not exist.'
    else:
        assert args.runID, 'missing runID, e.g. 1lygiuq3'
        assert os.path.exists(args.resume_checkpoint_dir), f'Resume checkpoint_dir {args.resume_checkpoint_dir} does not exist.'
        args.config_path = os.path.join(args.resume_checkpoint_dir, 'config.json')
        if args.mode in ('test', 'test_marginal'):
            assert args.ckpt_name, 'missing ckpt_name for testing. e.g. last.ckpt'


def main(args):
    validate_args(args)
    if args.aml:
        data_path = load_aml_dataset(args.dataset_name)
    else:
        data_path = args.data_path
    game_data = load_data(data_path)
    validate_data(game_data)
    hparams = load_hparams(args, game_data)

    # approve model summary before training
    if args.approve_before_training:
        val = input("Continue to build model? (y/n)")
        if not val.lower() in ('yes', 'y'):
            sys.exit(1)

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
    if hparams['model'] == 'contrastive':
        if hparams['contrastive_optimizer'] == 'adam':
            Opt_str = 'adam{}'.format(hparams['adam_lr'])
        elif hparams['contrastive_optimizer'] == 'sgd':
            Opt_str = 'sgd{}'.format(hparams['sgd_lr'])
        elif hparams['contrastive_optimizer'] == 'scheduled_adam':
            Opt_str = 'scheduledAdamW{}'.format(hparams['scheduled_adam_warmup_steps'])
        else:
            Opt_str = 'cosineAnnealingTmax{}'.format(hparams['cosine_annealing_T_max'])
        run_name = 'Con;Vec{};L{}H{}Lk{}Hk{};{};{}Kparams'.format(
            hparams['vec_repr'],
            hparams['N_enc'],
            hparams['num_heads'],
            hparams['N_enc_key'],
            hparams['num_heads_key'],
            Opt_str,
            round(max(model_summary.param_nums)/1000,2)
            )
    else:
        Opt_str = 'scheduledAdamW{}'.format(hparams['scheduled_adam_warmup_steps'])
        run_name = 'Gen;L{}H{};{};{}Kparams'.format(
            hparams['N_enc'],
            hparams['num_heads'],
            Opt_str,
            round(max(model_summary.param_nums)/1000,2)
        )

    project_name = args.project_name

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

    if args.aml:
        checkpoint_dir = './outputs'
    else:
        checkpoint_dir = args.checkpoint_dir

    # check point path
    if args.mode == 'train':
        # New training 
        ts = time.time()
        TIMESTAMP = st = datetime.datetime.fromtimestamp(ts).strftime('%Y%m%d-%H%M%S-')
        ckpt_dir_PATH = os.path.join(checkpoint_dir, project_name, TIMESTAMP+run_name)
        print('New Checkpoint Path:\n', ckpt_dir_PATH + '_<New wandb runID>')
        oldmask = os.umask(000)
        os.makedirs(ckpt_dir_PATH, 0o777)
        os.umask(oldmask)
    else:
        if args.aml:
            raise NotImplementedError('not sure what to do here')
        # Resume training
        ckpt_dir_PATH = args.resume_checkpoint_dir
        print('Resuming From and Saving to Checkpoint Path:\n', ckpt_dir_PATH)

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
    else: # test, test_marginal
        run_test(args, hparams, args.ckpt_name, trainmodule, game_datamodule, ckpt_dir_PATH, figsize=(10,15))

    return trainmodule, game_datamodule


if __name__ == '__main__':
    # parser
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)

    parser.add_argument('--project_name', type=str)
    parser.add_argument('--data_path', help='path to data json file')
    parser.add_argument('--mode', help='train, resume_train, test')
    parser.add_argument('--dataset_name', help='name of dataset')
    # new training
    parser.add_argument('--config_path', default=None, help='path to config.json, must provide if starting new training.  '
                        'If running in AML, path should be relative to this file.')
    parser.add_argument('--checkpoint_dir', default=None, help='path to save New checkpoints, must provide if starting new training.')
    # resume / testing
    parser.add_argument('--resume_max_epochs', default=None, help='must provide if resume training or testing')
    parser.add_argument('--resume_checkpoint_dir', default=None, help='path to resume config & checkpoint from and save to.')
    parser.add_argument('--ckpt_name', default=None, help='must provide if resume training or testing. Such as last.ckpt')
    parser.add_argument('--runID', default=None, help='wandb RunID must provide if resume training or testing')

    parser.add_argument('--gpu', help='gpu id', type=int)

    parser.add_argument(
        '--approve_before_training', help='Prompt for user to approve model configuration for training.', action='store_true'
    )
    parser.add_argument('--aml', help='add this flag if running via AML', action='store_true')

    # args and init
    args = parser.parse_args()

    wandb_login(args.aml)

    main(args)

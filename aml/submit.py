import argparse
import logging
import pathlib
import sys

import azureml.core

LOG = logging.getLogger(__name__)


def _get_ws():
    config_path = pathlib.Path(__file__).parent / 'config.json'
    return azureml.core.Workspace.from_config(path=config_path)


def _get_experiment_name(debug):
    if debug:
        return 'buns_exp_debug'
    else:
        return 'buns_exp'


def _get_environment(gpu, debug):
    conda = azureml.core.conda_dependencies.CondaDependencies()

    conda.add_conda_package('pip')

    conda.add_channel('pytorch')

    conda.add_conda_package('python=3.8')

    conda.add_channel('conda-forge')

    # The NC6 machines (with K80 GPUs) come with driver version 440.33.01, which supports
    # CUDA 10.2 max.  See https://docs.nvidia.com/deploy/cuda-compatibility/index.html.
    # But the 3090 GPU requires at least CUDA 11.  So we use CUDA 11 when doing local
    # "debug" runs and CUDA 10 in the cluster environment.
    if debug:
        conda.add_conda_package('cudatoolkit=11.0.221')
        conda.add_conda_package('pytorch=1.7.0=py3.8_cuda11.0.221_cudnn8.0.3_0')
    else:
        conda.add_conda_package('cudatoolkit=10.2.89')
        conda.add_conda_package('pytorch=1.7.1=py3.8_cuda10.2.89_cudnn7.6.5_0')

    conda.add_conda_package('torchvision')

    conda.add_pip_package('pytorch-lightning')
    conda.add_pip_package('wandb')
    conda.add_pip_package('matplotlib')
    conda.add_pip_package('tqdm')

    env = azureml.core.Environment('pytorch')
    env.python.conda_dependencies = conda

    env.environment_variables = {'CUDA_VISIBLE_DEVICES': str(gpu)}

    return env


def _request_signoff(config_path):
    with open(config_path) as f:
        LOG.info(f'---------------\nconfig:\n\n{f.read()}\n---------------')
    val = input("Continue training? (y/n)")
    if not val.lower() in ('yes', 'y'):
        raise RuntimeError('bun bun disapproved!')


def submit_job(debug):
    LOG.debug(f'debug mode: {debug}')

    LOG.debug('looking up workspace')
    ws = _get_ws()
    experiment_name = _get_experiment_name(debug)
    experiment = azureml.core.Experiment(workspace=ws, name=experiment_name)
    if debug:
        gpu = 0
    else:
        gpu = 0  # Always use GPU 0 if using AML
    environment = _get_environment(gpu, debug)

    if debug:
        compute_target = 'local'
    else:
        compute_target = azureml.core.ComputeTarget(ws, 'v100-cluster')

    source_directory=pathlib.Path.home() / 'Contrastive-Learning-Benchmarking'
    rel_script_path = 'SecondPass-CardGame-experiments/main.py'
    rel_config_path = 'config.json'

    absolute_config_path = (source_directory / rel_script_path).parent / rel_config_path
    _request_signoff(absolute_config_path)

    run_config = azureml.core.ScriptRunConfig(
        source_directory=source_directory,
        script=rel_script_path,
        arguments=[
            '--project_name', 'ContrastiveLearning-simple-SET-Wildcard-9',
            '--dataset_name', 'WildCardSETidx-2Attr-3Vals-0Train-5120Val-5120Test',
            '--config_path', rel_config_path,
            '--mode', 'train',
            '--gpu', str(gpu),
            '--aml'
        ],
        compute_target=compute_target,
        environment=environment,
    )

    LOG.debug('submitting job')
    return experiment.submit(config=run_config)


def _configure_logging():
    LOG.setLevel(logging.DEBUG)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    stream_handler.setFormatter(formatter)
    LOG.addHandler(stream_handler)


if __name__ == '__main__':
    _configure_logging()

    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        '--debug',
        action='store_true',
        help='Runs the job locally instead of in the cluster',
    )
    args = argparser.parse_args()

    run = submit_job(args.debug)
    LOG.info('waiting for job to complete...')
    run.wait_for_completion(show_output=True)

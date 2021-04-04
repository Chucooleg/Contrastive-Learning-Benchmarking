import argparse
import logging
import pathlib
import sys

import azureml.core

LOG = logging.getLogger(__name__)


def _get_ws():
    config_path = pathlib.Path(__file__).parent / 'config.json'
    return azureml.core.Workspace.from_config(path=config_path)


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


def submit_job(debug, cluster, tags):
    LOG.debug(f'debug mode: {debug}')

    LOG.debug('looking up workspace')
    ws = _get_ws()
    if debug:
        gpu = 0
    else:
        gpu = 0  # Always use GPU 0 if using AML
    environment = _get_environment(gpu, debug)

    if debug:
        compute_target = 'local'
    else:
        compute_target = azureml.core.ComputeTarget(ws, cluster)

    source_directory=pathlib.Path.home() / 'Contrastive-Learning-Benchmarking'
    rel_script_path = 'SecondPass-CardGame-experiments/main.py'
    rel_config_path = 'config.json'

    absolute_config_path = (source_directory / rel_script_path).parent / rel_config_path
    _request_signoff(absolute_config_path)

    # SET OPS
    # project_name = 'ContrastiveLearning-SET-Wildcard-Expand-SetOp-27' ## !! Update this
    # dataset_name = 'WildCardSETidxSetOps-3Attr-3Vals-8Pairs-0Train-5120Val-5120Test' ## !! Update this

    # SET UNION
    project_name = 'ContrastiveLearning-SET-Wildcard-Expand-Union-81' ## !! Update this
    dataset_name = 'WildCardSETidxUnion-4Attr-3Vals-8Pairs-0Train-5120Val-5120Test' ## !! Update this

    # 1 Wildcard
    # project_name = 'ContrastiveLearning-SET-Wildcard-27' ## !! Update this
    # dataset_name = 'WildCardSETidxUnion-3Attr-3Vals-1Pairs-0Train-5120Val-5120Test' ## !! Update this

    run_config = azureml.core.ScriptRunConfig(
        source_directory=source_directory,
        script=rel_script_path,
        arguments=[
            '--project_name', project_name,
            '--config_path', rel_config_path,
            '--dataset_name', dataset_name,
            '--mode', 'train',
            '--gpu', str(gpu),
            '--aml'
        ],
        compute_target=compute_target,
        environment=environment,
    )

    experiment = azureml.core.Experiment(workspace=ws, name=project_name)

    LOG.debug('submitting job')
    return experiment.submit(config=run_config, tags=tags)


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
    argparser.add_argument(
        '--cluster',
        default='v100-cluster',
        help='Name of the cluster, e.g. k80-cluster or v100-cluster',
    )
    argparser.add_argument(
        '--tags',
        help='tags to add to the run.  should be comma-separated, e.g. --tags tag1,tag2,tag3',
    )
    args = argparser.parse_args()

    if args.tags is None:
        tags = None
    else:
        tags = {tag: '' for tag in args.tags.split(',')}

    run = submit_job(args.debug, args.cluster, tags)
    LOG.info('waiting for job to complete...')
    run.wait_for_completion(show_output=True)






# # Steps to run training
# 1. `conda activate aml`
# 2. `cd ~/aml`
# 3. To do a local run: `python aml/submit.py --debug`
# 4. To do a cloud run: `python aml/submit.py --tag xyz`

# # Steps to add a new dataset
# 1. Go to the datasets section in AML Studio: https://ml.azure.com/data?wsid=/subscriptions/9f437c2f-8971-40f4-a021-cdba0cb08b80/resourcegroups/aml-rg/workspaces/legg-aml-ws&tid=959fafed-f244-49fc-823e-859ed39ca98d
# 2. Click "Create dataset" > "From local files"
# 3. Fill in dataset name.  Select "File" for dataset type.  Click "Next".
# 4. Select the "Currently selected datastore" bullet, and then click "Browse" and select the file(s) you want to use for your dataset.  Click "Next", then "Create".



# Check experiments
# https://ml.azure.com/experiments?wsid=/subscriptions/9f437c2f-8971-40f4-a021-cdba0cb08b80/resourcegroups/aml-rg/workspaces/legg-aml-ws&tid=959fafed-f244-49fc-823e-859ed39ca98d
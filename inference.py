import argparse
import importlib
import os
import torch
import librosa
import random
from models.SepReformer_Base_WSJ0.dataset import get_dataloaders
from models.SepReformer_Base_WSJ0.model import Model
from models.SepReformer_Base_WSJ0.engineLocal import Engine
from utils import util_system, util_implement
from utils.decorators import *


def load_wav(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Input file {file_path} does not exist!")

    samps_mix, _ = librosa.load(file_path, sr=8000)

    # Truncate samples as needed
    if len(samps_mix) % 8 != 0:
        remains = len(samps_mix) % 8
        samps_mix = samps_mix[:-remains]

    return samps_mix

# Parse args
parser = argparse.ArgumentParser(
    description="Command to start PIT training, configured by .yaml files")
parser.add_argument(
    "--model",
    type=str,
    default="IQformer_v6_0_0",
    dest="model",
    help="Insert model name")
parser.add_argument(
    "--engine_mode",
    choices=["train", "test", "test_wav"],
    default="train",
    help="This option is used to chooose the mode")
parser.add_argument(
    '-c',
    '--config',
    type=str,
    required=True,
    help="yaml file for configuration")
parser.add_argument(
    '-m',
    '--mixed_file',
    type=str,
    required=True,
    help='path of mixed wav file')
parser.add_argument(
    "--out_wav_dir",
    type=str,
    default=None,
    help="This option is used to specficy save directory for output wav file in test_wav mode")
args = parser.parse_args()

''' Build Setting '''
# Call configuration file (configs.yaml)
yaml_dict = util_system.parse_yaml(args.config)

# Run wandb and get configuration
wandb_run = util_system.wandb_setup(yaml_dict)
config = wandb_run.config if wandb_run else yaml_dict["wandb"]["init"]["config"]  # wandb login success or fail

# Call DataLoader [train / valid / test / etc...]
dataloaders = get_dataloaders(args, config["dataset"], config["dataloader"])
mixed_wav = load_wav(args.mixed_file)
# 转换为 PyTorch Tensor
mixed_wav = torch.tensor(mixed_wav, dtype=torch.float32)
# 添加额外的维度
mixed_wav = torch.unsqueeze(mixed_wav, 0)

file_name = os.path.basename(args.mixed_file)
file_name_ext = os.path.splitext(file_name)[0]  # 获取文件名（不含扩展名）

''' Build Model '''
# Call network model
model = Model(**config["model"])

''' Build Engine '''
# Call gpu id & device
gpuid = tuple(map(int, config["engine"]["gpuid"].split(',')))
# device = torch.device(f'cuda:{gpuid[0]}')
device = torch.device('cpu')

# Call Implement [criterion / optimizer / scheduler]
criterions = util_implement.CriterionFactory(config["criterion"], device).get_criterions()
optimizers = util_implement.OptimizerFactory(config["optimizer"], model.parameters()).get_optimizers()
schedulers = util_implement.SchedulerFactory(config["scheduler"], optimizers).get_schedulers()

# Call & Run Engine
engine = Engine(args, config, model, dataloaders, criterions, optimizers, schedulers, gpuid, device, wandb_run)
engine._inference(mixed_wav, file_name_ext, args.out_wav_dir)

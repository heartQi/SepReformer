import os
import torch
import csv
import time
import soundfile as sf
import numpy as np

from loguru import logger
from tqdm import tqdm
from utils import util_engine, functions
from utils.decorators import *
from torch.utils.tensorboard import SummaryWriter


@logger_wraps()
class Engine(object):
    def __init__(self, args, config, model, dataloaders, optimizers, gpuid, device):
        
        ''' Default setting '''
        self.engine_mode = args.engine_mode
        self.out_wav_dir = args.out_wav_dir
        self.config = config
        self.gpuid = gpuid
        self.device = device
        self.model = model.to(self.device)
        self.dataloaders = dataloaders # self.dataloaders['train'] or ['valid'] or ['test']
        self.main_optimizer = optimizers[0]

        self.pretrain_weights_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "log", "pretrain_weights")
        os.makedirs(self.pretrain_weights_path, exist_ok=True)
        self.scratch_weights_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "log", "scratch_weights")
        os.makedirs(self.scratch_weights_path, exist_ok=True)
        
        self.checkpoint_path = self.pretrain_weights_path if any(file.endswith(('.pt', '.pt', '.pkl')) for file in os.listdir(self.pretrain_weights_path)) else self.scratch_weights_path
        self.start_epoch = util_engine.load_last_checkpoint_n_get_epoch(self.checkpoint_path, self.model, self.main_optimizer, location=self.device)
        self.non_chunk = args.non_chunk
        self.chunk_size = args.chunk_size
        self.hop_len = args.hop_len
        
        # Logging 
        util_engine.model_params_mac_summary(
            model=self.model, 
            input=torch.randn(1, self.config['check_computations']['dummy_len']).to(self.device), 
            dummy_input=torch.rand(1, self.config['check_computations']['dummy_len']).to(self.device), 
            # metrics=['ptflops', 'thop', 'torchinfo']
            metrics=['ptflops']
        )
        
        logger.info(f"Clip gradient by 2-norm {self.config['engine']['clip_norm']}")

    @logger_wraps()
    def _inference(self, mixture, frames, mxiture_file, wav_dir=None):
        self.model.eval()
        with torch.inference_mode():
            nnet_input = torch.tensor(mixture, device=self.device)

            if self.non_chunk:
                on_test_start = time.time()
                estim_src, estim_src_bn = self.model(nnet_input)
                on_test_end = time.time()
                cost_time = on_test_end - on_test_start
                print("Total inference non_chunk------------:", cost_time)

            elif 1:
                estim_src_0 = torch.zeros(1, nnet_input.size(1))
                estim_src_1 = torch.zeros(1, nnet_input.size(1))

                for i in range(0, nnet_input.size(1), self.chunk_size):
                    # 获取当前 chunk_size 个元素的块，并保持第一个维度不变
                    chunk = nnet_input[:, i:i + self.chunk_size]
                    on_test_start = time.time()
                    estim_src_tmp, estim_src_bn = self.model(chunk)
                    on_test_end = time.time()
                    cost_time = on_test_end - on_test_start
                    print("Total inference chunk------------:", cost_time)

                    estim_src_0[0,i:i + self.chunk_size]= estim_src_tmp[0]
                    estim_src_1[0,i:i + self.chunk_size]= estim_src_tmp[1]
                estim_src = [estim_src_0, estim_src_1]
            else:
                window = np.hamming(frames.shape[0])
                frames_win = frames * window[:, np.newaxis]
                frames_win_ten = torch.tensor(frames_win)
                window_sum = np.zeros(nnet_input.size(1))

                estim_src_0 = torch.zeros(1, nnet_input.size(1))
                estim_src_1 = torch.zeros(1, nnet_input.size(1))
                for i in range(frames_win.shape[1]):
                    start = i * self.hop_len
                    chunk = frames_win_ten[:,i].unsqueeze(0).float()
                    on_test_start = time.time()
                    estim_src_tmp, _ = self.model(chunk)
                    #estim_src_tmp = [chunk, chunk]
                    on_test_end = time.time()
                    cost_time = on_test_end - on_test_start
                    print("inference chunk", cost_time)

                    estim_src_0[:, start:start + self.chunk_size] += estim_src_tmp[0]
                    estim_src_1[:, start:start + self.chunk_size] += estim_src_tmp[1]
                    window_sum[start:start + self.chunk_size] += window

                window_sum[window_sum == 0] = 1  # Avoid division by zero
                estim_src_0 = estim_src_0.numpy().squeeze()  # 去掉多余的维度，变为形状 (N,)
                estim_src_1 = estim_src_1.numpy().squeeze()  # 去掉多余的维度，变为形状 (N,)

                estim_src_0 /= window_sum
                estim_src_1 /= window_sum
                estim_src_0 = torch.tensor(estim_src_0).unsqueeze(0)
                estim_src_1 = torch.tensor(estim_src_1).unsqueeze(0)

                estim_src = [estim_src_0, estim_src_1]

            if self.engine_mode == "test_wav":
                if wav_dir == None: wav_dir = os.path.join(os.path.dirname(__file__), "wav_out")
                if wav_dir and not os.path.exists(wav_dir): os.makedirs(wav_dir)
                mixture = torch.squeeze(mixture).cpu().data.numpy()
                sf.write(os.path.join(wav_dir, mxiture_file + '.wav'),
                         0.5 * mixture / max(abs(mixture)), 8000)
                for i in range(self.config['model']['num_spks']):
                    src = torch.squeeze(estim_src[i]).cpu().data.numpy()
                    if self.non_chunk:
                        sf.write(os.path.join(wav_dir, mxiture_file + f'_T_Infer_Nonchunk_output_{i}.wav'),
                                 0.5 * src / max(abs(src)), 8000)
                    else:
                        sf.write(os.path.join(wav_dir, mxiture_file + f'_T_Infer_Chunk_output_{i}.wav'),
                                 0.5 * src / max(abs(src)), 8000)
        return
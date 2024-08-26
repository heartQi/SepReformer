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
        no_win = 0
        out_win = 0
        out_linear = 0
        win_in_win_out = 1
        stream = 1
        with torch.inference_mode():
            nnet_input = torch.tensor(mixture, device=self.device)
            if self.model.num_spks == 1:
                estim_src = [torch.zeros(1, nnet_input.size(1)).to(self.device)]

            else:
                estim_src = [torch.zeros(1, nnet_input.size(1)).to(self.device),
                             torch.zeros(1, nnet_input.size(1)).to(self.device)]

            if self.non_chunk:
                on_test_start = time.time()
                estim_src, estim_src_bn = self.model(nnet_input)
                on_test_end = time.time()
                cost_time = on_test_end - on_test_start
                print("Total inference non_chunk------------:", cost_time)
            elif no_win:
                for i in range(0, nnet_input.size(1), self.chunk_size):
                    if i + self.chunk_size > nnet_input.size(1):
                        break
                    # 获取当前 chunk_size 个元素的块，并保持第一个维度不变
                    chunk = nnet_input[:, i:i + self.chunk_size]
                    on_test_start = time.time()
                    estim_src_tmp, estim_src_bn = self.model(chunk)
                    on_test_end = time.time()
                    cost_time = on_test_end - on_test_start
                    print("Total inference chunk no win------------:", cost_time)

                    # 更新 estim_src
                    for idx in range(self.model.num_spks):
                            estim_src[idx][0, i:i + self.chunk_size] = estim_src_tmp[idx][0]
            elif out_win:
                estim_src_0 = torch.zeros(1, nnet_input.size(1))
                estim_src_1 = torch.zeros(1, nnet_input.size(1))
                pre_estim_src_0 = torch.zeros(self.chunk_size)
                pre_estim_src_1 = torch.zeros(self.chunk_size)
                transition_length = 80
                window = torch.from_numpy(np.hamming(2*transition_length)).float()
                fade_in = window[:transition_length]
                fade_out = window[transition_length:]

                for i in range(0, nnet_input.size(1), self.chunk_size):
                    if i + self.chunk_size > nnet_input.size(1):
                        break
                    # 获取当前 chunk_size 个元素的块，并保持第一个维度不变
                    chunk = nnet_input[:, i:i + self.chunk_size]
                    on_test_start = time.time()
                    estim_src_tmp, estim_src_bn = self.model(chunk)
                    on_test_end = time.time()
                    cost_time = on_test_end - on_test_start
                    print("Total inference chunk out win------------:", cost_time)

                    signal0_fadeout = torch.cat((pre_estim_src_0[:-transition_length], fade_out * pre_estim_src_0[-transition_length:]), dim=0)
                    signal0_fadein = torch.cat((fade_in * estim_src_tmp[0][0, :transition_length], estim_src_tmp[0][0, transition_length:]), dim=0)

                    signal1_fadeout = torch.cat((pre_estim_src_1[:-transition_length], fade_out * pre_estim_src_1[-transition_length:]), dim=0)
                    signal1_fadein = torch.cat((fade_in * estim_src_tmp[1][0, :transition_length], estim_src_tmp[1][0, transition_length:]), dim=0)

                    smoothed_signal0 = torch.cat((signal0_fadeout, signal0_fadein), dim=0)
                    smoothed_signal1 = torch.cat((signal1_fadeout, signal1_fadein), dim=0)

                    pre_estim_src_0 = signal0_fadein
                    pre_estim_src_1 = signal1_fadein
                    if i-self.chunk_size >= 0:
                        estim_src_0[0, i-self.chunk_size:i + self.chunk_size] = smoothed_signal0
                        estim_src_1[0, i-self.chunk_size:i + self.chunk_size] = smoothed_signal1
                    else:
                        estim_src_0[0, i:i + self.chunk_size] = estim_src_tmp[0]
                        estim_src_1[0, i:i + self.chunk_size] = estim_src_tmp[1]

                estim_src = [estim_src_0, estim_src_1]
            elif out_linear:
                estim_src_0 = torch.zeros(1, nnet_input.size(1))
                estim_src_1 = torch.zeros(1, nnet_input.size(1))
                pre_estim_src_0 = torch.zeros(self.chunk_size)
                pre_estim_src_1 = torch.zeros(self.chunk_size)
                transition_length = 5

                for i in range(0, nnet_input.size(1), self.chunk_size):
                    if i + self.chunk_size > nnet_input.size(1):
                        break
                    # 获取当前 chunk_size 个元素的块，并保持第一个维度不变
                    chunk = nnet_input[:, i:i + self.chunk_size]
                    on_test_start = time.time()
                    estim_src_tmp, estim_src_bn = self.model(chunk)
                    on_test_end = time.time()
                    cost_time = on_test_end - on_test_start
                    print("Total inference chunk out linear------------:", cost_time)

                    start_value0 = pre_estim_src_0[1-transition_length]
                    end_value0 = estim_src_tmp[0][0, transition_length-1]
                    interpolated_values = torch.from_numpy(np.linspace(start_value0, end_value0, 2*transition_length))
                    pre_signal0 = torch.cat((pre_estim_src_0[:-transition_length], interpolated_values[:transition_length]), dim=0)
                    signal0 = torch.cat((interpolated_values[transition_length:], estim_src_tmp[0][0, transition_length:]),dim=0)

                    start_value1 = pre_estim_src_1[1-transition_length]
                    end_value1 = estim_src_tmp[1][0, transition_length-1]
                    interpolated_values = torch.from_numpy(np.linspace(start_value1, end_value1, 2*transition_length))
                    pre_signal1= torch.cat((pre_estim_src_1[:-transition_length], interpolated_values[:transition_length]), dim=0)
                    signal1 = torch.cat((interpolated_values[transition_length:], estim_src_tmp[1][0, transition_length:]),dim=0)

                    smoothed_signal0 = torch.cat((pre_signal0, signal0), dim=0)
                    smoothed_signal1 = torch.cat((pre_signal1, signal1), dim=0)

                    pre_estim_src_0 = signal0
                    pre_estim_src_1 = signal1
                    if i - self.chunk_size >= 0:
                        estim_src_0[0, i - self.chunk_size:i + self.chunk_size] = smoothed_signal0
                        estim_src_1[0, i - self.chunk_size:i + self.chunk_size] = smoothed_signal1
                    else:
                        estim_src_0[0, i:i + self.chunk_size] = estim_src_tmp[0]
                        estim_src_1[0, i:i + self.chunk_size] = estim_src_tmp[1]

                estim_src = [estim_src_0, estim_src_1]
            elif win_in_win_out:
                window = torch.from_numpy(np.hamming(self.chunk_size))
                window_sum = torch.zeros(nnet_input.size(1))

                for i in range(0, nnet_input.size(1), self.hop_len):
                    if i + self.chunk_size > nnet_input.size(1):
                        break
                    # 获取当前 chunk_size 个元素的块，并保持第一个维度不变
                    chunk = nnet_input[:, i:i + self.chunk_size]
                    chunk = (chunk * window).float()
                    on_test_start = time.time()
                    estim_src_tmp, estim_src_bn = self.model(chunk)
                    #estim_src_tmp = [chunk, chunk]

                    on_test_end = time.time()
                    cost_time = on_test_end - on_test_start
                    print("Total inference chunk win in win out------------:", cost_time)
                    window_sum[i:i + self.chunk_size] += window

                    # 更新 estim_src
                    for idx in range(self.model.num_spks):
                            estim_src[idx][0, i:i + self.chunk_size] = estim_src[idx][0, i:i + self.chunk_size] + estim_src_tmp[idx][0]
                            estim_src[idx][0, i:i + self.hop_len] /= window_sum[i:i + self.hop_len]
            elif stream:
                # 遍历输入数据的块
                for i in range(0, nnet_input.size(1), self.chunk_size):
                    if i + self.chunk_size > nnet_input.size(1):
                        break
                    chunk = nnet_input[:, i:i + self.chunk_size]

                    on_test_start = time.time()
                    estim_src_tmp, estim_src_bn_tmp = self.model(chunk)
                    on_test_end = time.time()
                    cost_time = on_test_end - on_test_start
                    print("train chunk:", cost_time)

                    for idx in range(self.model.num_spks):
                        estim_src[idx][:, i:i + self.chunk_size].copy_(estim_src_tmp[idx][:, -self.chunk_size:])

            else:
                window = np.hamming(frames.shape[0])
                frames_win = frames * window[:, np.newaxis]
                frames_win_ten = torch.tensor(frames_win)
                window_sum = np.zeros(nnet_input.size(1))

                for i in range(frames_win.shape[1]):
                    start = i * self.hop_len
                    chunk = frames_win_ten[:,i].unsqueeze(0).float()
                    on_test_start = time.time()
                    #estim_src_tmp, _ = self.model(chunk)
                    estim_src_tmp = [chunk, chunk]
                    on_test_end = time.time()
                    cost_time = on_test_end - on_test_start
                    print("inference chunk", cost_time)
                    window_sum[start:start + self.chunk_size] += window

                    # 更新 estim_src
                    for idx in range(self.model.num_spks):
                            estim_src[idx][0, start:start + self.chunk_size] += estim_src_tmp[idx][0]
                            estim_src[idx][0, start:start + self.hop_len] /= window_sum[start:start + self.hop_len]

            if self.engine_mode == "test_wav":
                if wav_dir == None: wav_dir = os.path.join(os.path.dirname(__file__), "wav_out")
                if wav_dir and not os.path.exists(wav_dir): os.makedirs(wav_dir)
                mixture = torch.squeeze(mixture).cpu().data.numpy()
                sf.write(os.path.join(wav_dir, mxiture_file + '.wav'),
                         mixture, 8000)
                if self.model.num_spks > 1:
                    src0 = torch.squeeze(estim_src[0]).cpu().data.numpy()
                    src1 = torch.squeeze(estim_src[1]).cpu().data.numpy()
                    max_src = max(max(abs(src0)), max(abs(src1)))
                else:
                    src0 = torch.squeeze(estim_src[0]).cpu().data.numpy()
                    max_src = np.max(np.abs(src0))

                for i in range(self.config['model']['num_spks']):
                    src = torch.squeeze(estim_src[i]).cpu().data.numpy()
                    print(np.max(np.abs(src)))

                    if self.non_chunk:
                        sf.write(os.path.join(wav_dir, mxiture_file + f'_T_Infer_Nonchunk_output_{i}.wav'),
                                 src / max(max_src, 3000.0), 8000)
                    elif no_win:
                        sf.write(os.path.join(wav_dir, mxiture_file + f'_T_Infer_Chunk_output_{i}.wav'),
                                 src / max(max_src, 3000.0), 8000)
                    elif out_win:
                        sf.write(os.path.join(wav_dir, mxiture_file + f'_T_Infer_Chunk_outwin_{i}.wav'),
                                 src / max(max_src, 3000.0), 8000)
                    elif out_linear:
                        sf.write(os.path.join(wav_dir, mxiture_file + f'_T_Infer_Chunk_outlinear_{i}.wav'),
                                 src / max(max_src, 3000.0), 8000)
                    elif win_in_win_out:
                        sf.write(os.path.join(wav_dir, mxiture_file + f'_T_Infer_Chunk_winInwinOut_{i}.wav'),
                                 src / max(max_src, 3000.0), 8000)
                    else:
                        sf.write(os.path.join(wav_dir, mxiture_file + f'_T_Infer_Chunk_output_{i}.wav'),
                                 src, 8000)
        return
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
    def __init__(self, args, config, model, dataloaders, criterions, optimizers, schedulers, gpuid, device, wandb_run):
        
        ''' Default setting '''
        self.engine_mode = args.engine_mode
        self.out_wav_dir = args.out_wav_dir
        self.config = config
        self.gpuid = gpuid
        self.device = device
        self.model = model.to(self.device)
        self.dataloaders = dataloaders # self.dataloaders['train'] or ['valid'] or ['test']
        self.PIT_SISNR_mag_loss, self.PIT_SISNR_time_loss, self.PIT_SISNRi_loss, self.PIT_SDRi_loss = criterions
        self.main_optimizer = optimizers[0]
        self.main_scheduler, self.warmup_scheduler = schedulers
        
        self.pretrain_weights_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "log", "pretrain_weights")
        os.makedirs(self.pretrain_weights_path, exist_ok=True)
        self.scratch_weights_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "log", "scratch_weights")
        os.makedirs(self.scratch_weights_path, exist_ok=True)
        
        self.checkpoint_path = self.pretrain_weights_path if any(file.endswith(('.pt', '.pt', '.pkl')) for file in os.listdir(self.pretrain_weights_path)) else self.scratch_weights_path
        self.start_epoch = util_engine.load_last_checkpoint_n_get_epoch(self.checkpoint_path, self.model, self.main_optimizer, location=self.device)
        self.wandb_run = wandb_run
        self.non_chunk = args.non_chunk
        self.chunk_size = 400

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
    def _train(self, dataloader, epoch):
        self.model.train()
        tot_loss_freq = [0 for _ in range(self.model.num_stages)]
        tot_loss_time, num_batch = 0, 0
        pbar = tqdm(total=len(dataloader), unit='batches', bar_format='{l_bar}{bar:25}{r_bar}{bar:-10b}', colour="YELLOW", dynamic_ncols=True)
        for input_sizes, mixture, src, _ in dataloader:
            nnet_input = mixture
            nnet_input = functions.apply_cmvn(nnet_input) if self.config['engine']['mvn'] else nnet_input
            num_batch += 1
            pbar.update(1)
            # Scheduler learning rate for warm-up (Iteration-based update for transformers)
            if epoch == 1: self.warmup_scheduler.step()
            nnet_input = nnet_input.to(self.device)
            self.main_optimizer.zero_grad()

            if self.non_chunk:
                on_test_start = time.time()
                # estim_src, estim_src_bn = torch.nn.parallel.data_parallel(self.model, nnet_input, device_ids=self.gpuid)
                estim_src, estim_src_bn = self.model(nnet_input)
                on_test_end = time.time()
                cost_time = on_test_end - on_test_start
                print("train non_chunk:", cost_time)
            else:
                frame_len = nnet_input.size(1)
                # 初始化结果张量
                if self.model.num_spks == 1:
                    estim_src = [torch.zeros(2, frame_len).to(self.device)]
                    estim_src_bn = [[torch.zeros(2, frame_len, device=self.device)] for _ in range(self.model.num_stages)]
                else:
                    estim_src = [torch.zeros(2, frame_len).to(self.device), torch.zeros(2, frame_len).to(self.device)]
                    estim_src_bn = [[torch.zeros(2, frame_len, device=self.device), torch.zeros(2, frame_len, device=self.device)] for _ in range(self.model.num_stages)]
                input = torch.zeros(2, 20*80)

                # 遍历输入数据的块
                for i in range(0, frame_len, self.chunk_size):
                    if i + self.chunk_size > frame_len:
                        break
                    # 获取当前 chunk_size 个元素的块，并保持第一个维度不变
                    chunk = nnet_input[:, i:i + self.chunk_size]
                    on_test_start = time.time()
                    estim_src_tmp, estim_src_bn_tmp = self.model(chunk)
                    on_test_end = time.time()
                    cost_time = on_test_end - on_test_start
                    print("train chunk:",cost_time)
                    # 更新 estim_src
                    for idx in range(self.model.num_spks):
                        for idx_batch in range(dataloader.batch_size):
                            estim_src[idx][idx_batch, i:i + self.chunk_size] = estim_src_tmp[idx][idx_batch]

                    # 更新 estim_src_bn
                    for b in range(self.model.num_stages):
                        for r in range(self.model.num_spks):
                            for idx_batch in range(dataloader.batch_size):
                                estim_src_bn[b][r][idx_batch, i:i + self.chunk_size] = estim_src_bn_tmp[b][r][idx_batch]

            cur_loss_s_bn = []
            for idx, estim_src_value in enumerate(estim_src_bn):
                cur_loss_s_bn.append(self.PIT_SISNR_mag_loss(estims=estim_src_value, idx=idx, input_sizes=input_sizes, target_attr=src))
                tot_loss_freq[idx] += cur_loss_s_bn[idx].item() / (self.config['model']['num_spks'])
            cur_loss_s = self.PIT_SISNR_time_loss(estims=estim_src, input_sizes=input_sizes, target_attr=src)
            tot_loss_time += cur_loss_s.item() / self.config['model']['num_spks']
            alpha = 0.4 * 0.8**(1+(epoch-101)//5) if epoch > 100 else 0.4
            cur_loss = (1-alpha) * cur_loss_s + alpha * sum(cur_loss_s_bn) / len(cur_loss_s_bn)
            cur_loss = cur_loss / self.config['model']['num_spks']
            cur_loss.backward()
            if self.config['engine']['clip_norm']: torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config['engine']['clip_norm'])
            self.main_optimizer.step()
            dict_loss = {"T_Loss": tot_loss_time / num_batch}
            dict_loss.update({'F_Loss_' + str(idx): loss / num_batch for idx, loss in enumerate(tot_loss_freq)})
            pbar.set_postfix(dict_loss)
        pbar.close()
        tot_loss_freq = sum(tot_loss_freq) / len(tot_loss_freq)
        return tot_loss_time / num_batch, tot_loss_freq / num_batch, num_batch
    
    @logger_wraps()
    def _validate(self, dataloader):
        self.model.eval()
        tot_loss_freq = [0 for _ in range(self.model.num_stages)]
        tot_loss_time, num_batch = 0, 0
        pbar = tqdm(total=len(dataloader), unit='batches', bar_format='{l_bar}{bar:5}{r_bar}{bar:-10b}', colour="RED", dynamic_ncols=True)
        with torch.inference_mode():
            for input_sizes, mixture, src, _ in dataloader:
                nnet_input = mixture
                nnet_input = functions.apply_cmvn(nnet_input) if self.config['engine']['mvn'] else nnet_input
                nnet_input = nnet_input.to(self.device)
                num_batch += 1
                pbar.update(1)
                if self.non_chunk:
                    on_test_start = time.time()
                    #estim_src, estim_src_bn = torch.nn.parallel.data_parallel(self.model, nnet_input, device_ids=self.gpuid)
                    estim_src, estim_src_bn = self.model(nnet_input)
                    on_test_end = time.time()
                    cost_time = on_test_end - on_test_start
                    print("validate non_chunk:",cost_time)
                else:
                    frame_len = nnet_input.size(1)

                    # 初始化结果张量
                    if self.model.num_spks == 1:
                        estim_src = [torch.zeros(2, frame_len).to(self.device)]
                        estim_src_bn = [[torch.zeros(2, frame_len, device=self.device)] for _ in range(self.model.num_stages)]
                    else:
                        estim_src = [torch.zeros(2, frame_len).to(self.device), torch.zeros(2, frame_len).to(self.device)]
                        estim_src_bn = [[torch.zeros(2, frame_len, device=self.device),torch.zeros(2, frame_len, device=self.device)] for _ in range(self.model.num_stages)]

                    # 遍历输入数据的块
                    for i in range(0, frame_len, self.chunk_size):
                        # 获取当前 chunk_size 个元素的块，并保持第一个维度不变
                        chunk = nnet_input[:, i:i + self.chunk_size]

                        on_test_start = time.time()
                        estim_src_tmp, estim_src_bn_tmp = self.model(chunk)
                        on_test_end = time.time()
                        cost_time = on_test_end - on_test_start
                        print("validate chunk:", cost_time)

                        # 更新 estim_src
                        for idx in range(self.model.num_spks):
                            estim_src[idx][0, i:i + self.chunk_size] = estim_src_tmp[idx][0]
                            estim_src[idx][1, i:i + self.chunk_size] = estim_src_tmp[idx][1]

                        # 更新 estim_src_bn
                        for b in range(self.model.num_stages):
                            for r in range(self.model.num_spks):
                                estim_src_bn[b][r][0, i:i + self.chunk_size] = estim_src_bn_tmp[b][r][0]
                                estim_src_bn[b][r][1, i:i + self.chunk_size] = estim_src_bn_tmp[b][r][1]

                cur_loss_s_bn = []
                for idx, estim_src_value in enumerate(estim_src_bn):
                    cur_loss_s_bn.append(self.PIT_SISNR_mag_loss(estims=estim_src_value, idx=idx, input_sizes=input_sizes, target_attr=src))
                    tot_loss_freq[idx] += cur_loss_s_bn[idx].item() / (self.config['model']['num_spks'])
                cur_loss_s_SDR = self.PIT_SISNR_time_loss(estims=estim_src, input_sizes=input_sizes, target_attr=src)
                tot_loss_time += cur_loss_s_SDR.item() / self.config['model']['num_spks']
                dict_loss = {"T_Loss":tot_loss_time / num_batch}
                dict_loss.update({'F_Loss_' + str(idx): loss / num_batch for idx, loss in enumerate(tot_loss_freq)})
                pbar.set_postfix(dict_loss)
        pbar.close()
        tot_loss_freq = sum(tot_loss_freq) / len(tot_loss_freq)
        return tot_loss_time / num_batch, tot_loss_freq / num_batch, num_batch
    
    #TODO: .wav 저장 모드 따로 설정하기.
    @logger_wraps()
    def _test(self, dataloader, wav_dir=None):
        self.model.eval()
        total_loss_SISNRi, total_loss_SDRi, num_batch = 0, 0, 0
        pbar = tqdm(total=len(dataloader), unit='batches', bar_format='{l_bar}{bar:5}{r_bar}{bar:-10b}', colour="grey", dynamic_ncols=True)
        with torch.inference_mode():
            csv_file_name_sisnr = os.path.join(os.path.dirname(__file__),'test_SISNRi_value.csv')
            csv_file_name_sdr = os.path.join(os.path.dirname(__file__),'test_SDRi_value.csv')
            with open(csv_file_name_sisnr, 'w', newline='') as csvfile_sisnr, open(csv_file_name_sdr, 'w', newline='') as csvfile_sdr:
                idx = 0
                writer_sisnr = csv.writer(csvfile_sisnr, quotechar='|', quoting=csv.QUOTE_MINIMAL)
                writer_sdr = csv.writer(csvfile_sdr, quotechar='|', quoting=csv.QUOTE_MINIMAL)
                for input_sizes, mixture, src, key in dataloader:
                    if len(key) > 1:
                        raise("batch size is not one!!")
                    #nnet_input = mixture.to(self.device)
                    nnet_input = mixture.to('cpu')
                    num_batch += 1
                    pbar.update(1)
                    if self.non_chunk:
                        on_test_start = time.time()
                        # estim_src, _ = torch.nn.parallel.data_parallel(self.model, nnet_input, device_ids=self.gpuid)
                        estim_src, _ = self.model(nnet_input)
                        on_test_end = time.time()
                        cost_time = on_test_end - on_test_start
                        print("test non_chunk", cost_time)
                    else:
                        frame_len = nnet_input.size(1)
                        # 初始化结果张量
                        if self.model.num_spks == 1:
                            estim_src = [torch.zeros(1, frame_len).to(self.device)]
                        else:
                            estim_src = [torch.zeros(1, frame_len).to(self.device),
                                         torch.zeros(1, frame_len).to(self.device)]

                        for i in range(0, frame_len, self.chunk_size):
                            # 获取当前 chunk_size 个元素的块，并保持第一个维度不变
                            chunk = nnet_input[:, i:i + self.chunk_size]
                            on_test_start = time.time()
                            estim_src_tmp, _ = self.model(chunk)
                            on_test_end = time.time()
                            cost_time = on_test_end - on_test_start
                            print("test chunk", cost_time)

                            for idx in range(self.model.num_spks):
                                estim_src[idx][0, i:i + self.chunk_size] += estim_src_tmp[idx][0]

                    cur_loss_SISNRi, cur_loss_SISNRi_src = self.PIT_SISNRi_loss(estims=estim_src, mixture=mixture, input_sizes=input_sizes, target_attr=src, eps=1.0e-15)
                    total_loss_SISNRi += cur_loss_SISNRi.item() / self.config['model']['num_spks']
                    cur_loss_SDRi, cur_loss_SDRi_src = self.PIT_SDRi_loss(estims=estim_src, mixture=mixture, input_sizes=input_sizes, target_attr=src)
                    total_loss_SDRi += cur_loss_SDRi.item() / self.config['model']['num_spks']
                    writer_sisnr.writerow([key[0][:-4]] + [cur_loss_SISNRi_src[i].item() for i in range(self.config['model']['num_spks'])])
                    writer_sdr.writerow([key[0][:-4]] + [cur_loss_SDRi_src[i].item() for i in range(self.config['model']['num_spks'])])
                    if self.engine_mode == "test_wav":
                        if wav_dir == None: wav_dir = os.path.join(os.path.dirname(__file__),"wav_out")
                        if wav_dir and not os.path.exists(wav_dir): os.makedirs(wav_dir)
                        mixture = torch.squeeze(mixture).cpu().data.numpy()
                        sf.write(os.path.join(wav_dir,key[0][:-4]+str(idx)+'_mixture.wav'), 0.5*mixture/max(abs(mixture)), 8000)
                        for i in range(self.config['model']['num_spks']):
                            src = torch.squeeze(estim_src[i]).cpu().data.numpy()
                            if self.non_chunk:
                                sf.write(os.path.join(wav_dir, key[0][:-4] + str(idx) + '_T_Test_Nonchunk_out_' + str(i) + '.wav'),
                                         0.5 * src / max(abs(src)), 8000)
                            else:
                                sf.write(os.path.join(wav_dir, key[0][:-4] + str(idx) + '_T_Test_Chunk_out_' + str(i) + '.wav'),
                                         0.5 * src / max(abs(src)), 8000)
                    idx += 1
                    dict_loss = {"SiSNRi": total_loss_SISNRi/num_batch, "SDRi": total_loss_SDRi/num_batch}
                    pbar.set_postfix(dict_loss)
        pbar.close()
        return total_loss_SISNRi/num_batch, total_loss_SDRi/num_batch, num_batch
    
    @logger_wraps()
    def run(self):
        #with torch.cuda.device(self.device):
        with torch.device(self.device):
            if self.wandb_run: self.wandb_run.watch(self.model, log="all")
            writer_src = SummaryWriter(os.path.join(os.path.dirname(os.path.abspath(__file__)), "log/tensorboard"))
            if "test" in self.engine_mode:
                on_test_start = time.time()
                test_loss_src_time_1, test_loss_src_time_2, test_num_batch = self._test(self.dataloaders['test'], self.out_wav_dir)
                on_test_end = time.time()
                test_speed = (on_test_end - on_test_start) / test_num_batch
                logger.info(f"[TEST] Loss(time/mini-batch) \n - Epoch {self.start_epoch:2d}: SISNRi = {test_loss_src_time_1:.4f} dB | SDRi = {test_loss_src_time_2:.4f} dB | Speed = ({on_test_end - on_test_start:.2f}s/{test_num_batch:d})")
                results = {
                        'Test SISNRi Loss': test_loss_src_time_1,
                        'Test SDRi Loss': test_loss_src_time_2, 
                        'Test Speed': test_speed}
                if self.wandb_run: self.wandb_run.log(results)
                logger.info(f"Testing done!")
            else:
                start_time = time.time()
                if self.start_epoch > 1:
                    init_loss_time, init_loss_freq, valid_num_batch = self._validate(self.dataloaders['valid'])
                else:
                    init_loss_time, init_loss_freq = 0, 0
                end_time = time.time()
                logger.info(f"[INIT] Loss(time/mini-batch) \n - Epoch {self.start_epoch:2d}: Loss_t = {init_loss_time:.4f} dB | Loss_f = {init_loss_freq:.4f} dB | Speed = ({end_time-start_time:.2f}s)")
                for epoch in range(self.start_epoch, self.config['engine']['max_epoch']):
                    valid_loss_best = init_loss_time
                    train_start_time = time.time()
                    train_loss_src_time, train_loss_src_freq, train_num_batch = self._train(self.dataloaders['train'], epoch)
                    train_end_time = time.time()
                    valid_start_time = time.time()
                    valid_loss_src_time, valid_loss_src_freq, valid_num_batch = self._validate(self.dataloaders['valid'])
                    valid_end_time = time.time()
                    if epoch > self.config['engine']['start_scheduling']: self.main_scheduler.step(valid_loss_src_time)
                    logger.info(f"[TRAIN] Loss(time/mini-batch) \n - Epoch {epoch:2d}: Loss_t = {train_loss_src_time:.4f} dB | Loss_f = {train_loss_src_freq:.4f} dB | Speed = ({train_end_time - train_start_time:.2f}s/{train_num_batch:d})")
                    logger.info(f"[VALID] Loss(time/mini-batch) \n - Epoch {epoch:2d}: Loss_t = {valid_loss_src_time:.4f} dB | Loss_f = {valid_loss_src_freq:.4f} dB | Speed = ({valid_end_time - valid_start_time:.2f}s/{valid_num_batch:d})")
                    if epoch in self.config['engine']['test_epochs']:
                        on_test_start = time.time()
                        test_loss_src_time_1, test_loss_src_time_2, test_num_batch = self._test(self.dataloaders['test'])
                        on_test_end = time.time()
                        logger.info(f"[TEST] Loss(time/mini-batch) \n - Epoch {epoch:2d}: SISNRi = {test_loss_src_time_1:.4f} dB | SDRi = {test_loss_src_time_2:.4f} dB | Speed = ({on_test_end - on_test_start:.2f}s/{test_num_batch:d})")
                    test_sisnri_loss = locals().get('test_loss_src_time_1', 0)
                    test_sdri_loss = locals().get('test_loss_src_time_2', 0)
                    test_speed = (on_test_end - on_test_start) / test_num_batch if 'on_test_end' in locals() and on_test_end else 0
                    results = {
                        'Learning Rate': self.main_optimizer.param_groups[0]['lr'],
                        'Train Loss': train_loss_src_time, 
                        'Train Speed': (train_end_time - train_start_time) / train_num_batch,
                        'Valid Loss': valid_loss_src_time, 
                        'Valid Speed': (valid_end_time - valid_start_time) / valid_num_batch,
                        'Test SISNRi Loss': test_sisnri_loss,
                        'Test SDRi Loss': test_sdri_loss, 
                        'Test Speed': test_speed}
                    valid_loss_best = util_engine.save_checkpoint_per_best(valid_loss_best, valid_loss_src_time, train_loss_src_time, epoch, self.model, self.main_optimizer, self.checkpoint_path, self.wandb_run)
                    # Logging to monitoring tools (Tensorboard && Wandb)
                    writer_src.add_scalars("Metrics", {
                        'Learning Rate': self.main_optimizer.param_groups[0]['lr'],
                        'Loss_train_time': train_loss_src_time, 
                        'Loss_valid_time': valid_loss_src_time}, epoch)
                    writer_src.flush()
                    if self.wandb_run: self.wandb_run.log(results)
                logger.info(f"Training for {self.config['engine']['max_epoch']} epoches done!")

    @logger_wraps()
    def _inference(self, mixture, frames, mxiture_file, wav_dir=None):
        self.model.eval()
        with torch.inference_mode():
            nnet_input = torch.tensor(mixture, device=self.device)

            if self.non_chunk:
                on_test_start = time.time()
                estim_src, _ = self.model(nnet_input)
                on_test_end = time.time()
                cost_time = on_test_end - on_test_start
                print("inference non_chunk", cost_time)

            elif 1:
                estim_src_0 = torch.zeros(1, nnet_input.size(1))
                estim_src_1 = torch.zeros(1, nnet_input.size(1))

                for i in range(0, nnet_input.size(1), self.chunk_size):
                    # 获取当前 chunk_size 个元素的块，并保持第一个维度不变
                    chunk = nnet_input[:, i:i + self.chunk_size]
                    on_test_start = time.time()
                    estim_src_tmp, _ = self.model(chunk)
                    on_test_end = time.time()
                    cost_time = on_test_end - on_test_start
                    print("inference chunk", cost_time)

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
                    #estim_src_tmp, _ = self.model(chunk)
                    estim_src_tmp = [chunk, chunk]
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
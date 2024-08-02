import numpy as np
import librosa
import soundfile as sf

class SELMIC:
    def __init__(self):
        self.mic_exclude = [False, False]
        self.mms_pows = np.zeros(len(self.mic_exclude))
        self.mms_powl = np.zeros(len(self.mic_exclude))
        self.mms_short_term = 30
        self.mms_long_term = 50
        self.mms_dt_thr = 1280
        self.mms_st_thr = 2 #768
        self.vad_upper_thr = 0.5
        self.vad_lower_thr = 0.1
        self.mic_select = 0
        self.far_status = False
        self.mms_vad = [None] * len(self.mic_exclude)  # Placeholder for VAD objects

def DSP_MUL_16_16_RSFT(a, b, c):
    return (a * b) >> c

def detect_active_microphone(signal1, signal2, selmic):
    rms1 = np.sqrt(np.mean(signal1**2))
    rms2 = np.sqrt(np.mean(signal2**2))

    rms_values = [rms1, rms2]

    for i in range(2):
        rms = rms_values[i]

        selmic.mms_pows[i] += (rms - selmic.mms_pows[i]) / selmic.mms_short_term
        selmic.mms_powl[i] += (rms - selmic.mms_powl[i]) / selmic.mms_long_term

    pows = np.array([selmic.mms_pows[0], selmic.mms_pows[1]])
    powl = np.array([selmic.mms_powl[0], selmic.mms_powl[1]])

    vad_logratio = [100,100]
    rms_st_idx = np.argmax(pows)

    change_thr = selmic.mms_dt_thr if selmic.far_status else selmic.mms_st_thr
    eng_diff = pows[rms_st_idx] / (powl[selmic.mic_select] + 1)

    if (eng_diff > change_thr and vad_logratio[rms_st_idx] > selmic.vad_upper_thr) or \
            (eng_diff < change_thr and vad_logratio[rms_st_idx] > selmic.vad_upper_thr and vad_logratio[
                selmic.mic_select] < selmic.vad_lower_thr):
        selmic.mic_select = rms_st_idx
    else:
        selmic.mic_select = selmic.mic_select
    ##print(pows[0], pows[1])
    print(selmic.mic_select)
    return selmic.mic_select

def main():
    # 加载两个麦克风采集的WAV文件
    dir0 = "/Users/mervin.qi/Desktop/PSE/Workspace/SepReformer/test/mervin_roman-mix_2_output_0.wav"
    dir1 = "/Users/mervin.qi/Desktop/PSE/Workspace/SepReformer/test/mervin_roman-mix_2_output_1.wav"
    signal1, sr1 = librosa.load(dir0, sr=8000)
    signal2, sr2 = librosa.load(dir1, sr=8000)

    # 反归一化到原始的采样值范围
    min_value = -32768  # 假设原始的最小采样值
    max_value = 32767  # 假设原始的最大采样值

    signal1_ = signal1 * (max_value - min_value) / 2 + (max_value + min_value) / 2
    signal2_ = signal2 * (max_value - min_value) / 2 + (max_value + min_value) / 2

    # AEF参数设置
    selmic = SELMIC()

    # 检测激活状态的麦克风
    framesize = 80  # 设置帧大小
    framenr = min(len(signal1), len(signal2)) // framesize

    output_signal = []

    for i in range(framenr):
        frame1_ = signal1_[i * framesize:(i + 1) * framesize]
        frame1 = signal1[i * framesize:(i + 1) * framesize]
        frame2_ = signal2_[i * framesize:(i + 1) * framesize]
        frame2 = signal2[i * framesize:(i + 1) * framesize]

        active_mic_index = detect_active_microphone(frame1_, frame2_, selmic)
        if active_mic_index == 0:
            output_signal.extend(frame1)
        else:
            output_signal.extend(frame2)

    # 保存输出信号到新的WAV文件
    output_signal = np.array(output_signal)
    output_path = "/Users/mervin.qi/Desktop/PSE/Workspace/SepReformer/test/active_mic_output.wav"
    sf.write(output_path, output_signal, sr1)
    print(f"激活麦克风的输出已保存到: {output_path}")

if __name__ == "__main__":
    main()

import numpy as np
import librosa
import soundfile as sf


def read_wav(filename):
    y, sr = librosa.load(filename, sr=None)
    return y, sr


def write_wav(filename, y, sr):
    sf.write(filename, y, sr)


def frame_signal(signal, frame_length, hop_length):
    frames = librosa.util.frame(signal, frame_length=frame_length, hop_length=hop_length)
    return frames


def apply_hamming_window(frames):
    window = np.hamming(frames.shape[0])
    return frames * window[:, np.newaxis]


def overlap_add(frames, hop_length):
    signal_length = hop_length * (frames.shape[1] - 1) + frames.shape[0]
    signal = np.zeros(signal_length)
    window_sum = np.zeros(signal_length)
    window = np.hamming(frames.shape[0])

    for i in range(frames.shape[1]):
        start = i * hop_length
        signal[start:start + frames.shape[0]] += frames[:, i]
        window_sum[start:start + frames.shape[0]] += window

    # Normalize signal
    window_sum[window_sum == 0] = 1  # Avoid division by zero
    signal /= window_sum
    return signal


def main(input_wav, output_wav, frame_length=160, hop_length=None):
    # 读取输入wav文件
    y, sr = read_wav(input_wav)

    # 设置帧长和帧移
    if hop_length is None:
        hop_length = frame_length // 2

    # 分帧
    frames = frame_signal(y, frame_length, hop_length)

    # 加汉明窗
    frames_windowed = apply_hamming_window(frames)

    # 重构信号
    output_signal = overlap_add(frames_windowed, hop_length)

    # 写入输出wav文件
    write_wav(output_wav, output_signal, sr)

if __name__ == "__main__":
    input_wav = "/Users/mervin.qi/Desktop/PSE/Workspace/SepReformer/test/mervin_roman-mix_3.wav"
    output_wav = "/Users/mervin.qi/Desktop/PSE/Workspace/SepReformer/test/win_pro_wav3.wav"
    frame_length = 1600
    hop_length = 400 #帧移
    main(input_wav, output_wav, frame_length, hop_length)
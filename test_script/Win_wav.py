import soundfile as sf
import numpy as np



def apply_window(signal, window_type='hamming'):
    if window_type == 'hamming':
        window = np.hamming(len(signal))
    elif window_type == 'hanning':
        window = np.hanning(len(signal))
    elif window_type == 'blackman':
        window = np.blackman(len(signal))
    else:
        raise ValueError("Unsupported window type.")

    return signal * window


def remove_window(windowed_signal, window_type='hamming'):
    if window_type == 'hamming':
        window = np.hamming(len(windowed_signal))
    elif window_type == 'hanning':
        window = np.hanning(len(windowed_signal))
    elif window_type == 'blackman':
        window = np.blackman(len(windowed_signal))
    else:
        raise ValueError("Unsupported window type.")

    return windowed_signal / window


if __name__ == "__main__":
    input_wav = "/Users/mervin.qi/Desktop/PSE/Workspace/SepReformer/test/mervin_roman-mix_3.wav"
    output_wav = "/Users/mervin.qi/Desktop/PSE/Workspace/SepReformer/test/win_pro_wav.wav"
    chunk_size = 1440
    window_length = 1600
    signal, sample_rate = sf.read(input_wav)
    win_chunk = np.zeros(window_length)
    output = np.zeros(len(signal))
    for i in range(0, len(signal), chunk_size):
        if i+chunk_size>len(signal):
            break
        chunk = signal[i:i + chunk_size]
        win_chunk = np.concatenate((win_chunk[1440:], chunk))
        apply_window(win_chunk, window_type='hamming')
        win_chunk /=2
        remove_window(win_chunk, window_type='hamming')
        print("i",i)
        print("i+chunk_size",i+chunk_size)
        output[i:i+chunk_size] = win_chunk[0:1440]

    # Write the processed signal to a new WAV file
    sf.write(output_wav, output, sample_rate)



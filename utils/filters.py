from scipy.signal import butter, lfilter, iirnotch
import numpy as np

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, axis=-1, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data, axis=axis)
    return y


def butter_highpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    return b, a

def butter_highpass_filter(data, cutoff, fs, axis=-1, order=5):
    b, a = butter_highpass(cutoff, fs, order=order)
    y = lfilter(b, a, data, axis=axis)
    return y    


def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def butter_lowpass_filter(data, cutoff, fs, axis=-1, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data, axis=axis)
    return y      

def notch_filter(wav, fs=500, w0=50, Q=20, axis=-1):    
    b, a = iirnotch(2 * w0/fs, Q)
    wav = lfilter(b, a, wav, axis=axis)
    return wav

def notch_filter_multi_harmonics(wav, fs=500, w0=np.arange(50,201,50), Q=20, axis=-1):    
    for w0_ in w0:
        b, a = iirnotch(2 * w0_/fs, Q)
        wav = lfilter(b, a, wav, axis=axis)
    return wav


def window_rms(a, window_size):
    a2 = np.power(a,2)
    window = np.ones(window_size)/float(window_size)
    return np.sqrt(np.convolve(a2, window, 'valid'))      
import numpy as np
import math
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr
from scipy.fftpack import fft

num_fft = 400

def SNR(y1, y2):
    N = np.sum(np.square(y1), axis=1)
    D = np.sum(np.square(y2 - y1), axis=1)

    SNR = 10 * np.log10(N / D)

    return np.mean(SNR)

def SNR_improvement(y_in, y_out, y_clean):
    return SNR(y_clean, y_out)-SNR(y_clean, y_in)

def CC(x, y):
    x = x.squeeze()
    y = y.squeeze()
    cc_total = 0
    num = x.shape[0]
    for i in range(num):
        cc, p_value = pearsonr(x[i], y[i])
        cc_total = cc_total + cc

    return cc_total / num

def Pvalue(x, y):
    x = x.squeeze()
    y = y.squeeze()
    p_total = 0
    num = x.shape[0]
    for i in range(num):
        cc, p_value = pearsonr(x[i], y[i])
        p_total = p_total + p_value

    return p_total / num

def RRMSE(denoise, clean):
    clean = clean.squeeze()
    denoise = denoise.squeeze()
    rmse1 = np.sqrt(mean_squared_error(denoise, clean))
    rmse2 = np.sqrt(mean_squared_error(clean, np.zeros(clean.shape, dtype=float)))

    return rmse1 / rmse2

def get_PSD(records):
    x_fft = fft(records, num_fft)
    x_fft = np.abs(x_fft)
    psd = x_fft ** 2 / num_fft
    return psd

def RRMSE_s(denoise, clean):
    clean = clean.squeeze()
    denoise = denoise.squeeze()
    rmse1 = np.sqrt(mean_squared_error(get_PSD(denoise), get_PSD(clean)))
    rmse2 = np.sqrt(mean_squared_error(get_PSD(clean), np.zeros(clean.shape, dtype=float)))

    return rmse1/rmse2
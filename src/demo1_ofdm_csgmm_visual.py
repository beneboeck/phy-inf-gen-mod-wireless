# Copyright 2025 Rohde & Schwarz

import sys
import os
# add the src directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
from src.modules import csgmm_modules as mod
from src.training import csgmm_training as tr
import matplotlib.pyplot as plt

def main():

    np.random.seed(123456)

    miter = 1500 # maximal number of iterations for training
    ydim = 30 # number of pilots for OFDM
    n_subc = 24 # number of subcarriers in the OFDM grid (predefined by dataset) # see Appendix D.2 in https://openreview.net/forum?id=FFJFT93oa7
    n_times = 14 # number of OFDM symbols (predefined by dataset) # see Appendix D.2 in https://openreview.net/forum?id=FFJFT93oa7
    subc_spacing = 15 * 1e3  # subcarrier spacing in Hz (predefined by dataset) # see Appendix D.2 in https://openreview.net/forum?id=FFJFT93oa7
    time_duration = 1e-3/n_times  # OFDM symbol duration in seconds (predefined by dataset) # see Appendix D.2 in https://openreview.net/forum?id=FFJFT93oa7
    hdim = n_subc * n_times  # dimension of the OFDM channel (predefined by dataset) # see Appendix D.2 in https://openreview.net/forum?id=FFJFT93oa7
    delay_max = 1.5e-5 # e.g., 6e-6 for urban, 1.5e-5 for rural [in seconds] # maximal delay considered in the delay doppler domain
    doppler_max = 700 # e.g., 250 for urban, 700 for rural [in Hz] # maximal doppler considered in the delay dopppler domain
    sdim = 32 * 32 # 32 * 32 # resolution in the delay doppler domain
    nt = 500 # number of training samples
    snr_db = 20 # snr for the noise in the training data
    K = 10 # number of GMM components
    pmax = sdim # maximal number of allowed paths

    # load and reshape QuaDRiGa data (either urban or rural)
    str_load = '../data/QuaDRiGa/ofdm5grural_12000.npy' # rural
    #str_load = '../data/QuaDRiGa/ofdm5gurban_12000.npy' # urban

    X_2d = np.load(str_load)
    n_x = X_2d.shape[0]
    X = X_2d.reshape((n_x,-1))
    X = X[:nt,:]

    # construct selection matrix
    Eys = np.eye(hdim)
    idx = np.arange(hdim)
    np.random.shuffle(idx)
    Eys_sh = Eys[idx, :]
    A = Eys_sh[:ydim, :]

    # compute the compressed and noisy data
    Y_cplx = np.einsum('ij,kj->ki', A, X, optimize='greedy')
    SNR_db = snr_db
    SNR = 10 ** (snr_db / 10)
    noise_var = np.mean(np.sum(np.abs(Y_cplx) ** 2, axis=1), axis=0) / (SNR * ydim)
    Y_cplx = Y_cplx + np.sqrt(noise_var) / np.sqrt(2) * (np.random.randn(Y_cplx.shape[0], Y_cplx.shape[1]) + 1j * np.random.randn(Y_cplx.shape[0],Y_cplx.shape[1]))

    # create the corresponding 2D OFDM selection matrix for visualization purposes
    A_2D_mask = np.zeros((n_subc, n_times))
    for i in range(ydim):
        mask_i = np.real(np.squeeze(A[i, :]))
        A_2D_mask += mask_i.reshape((n_subc, n_times))

    # visualize training data
    fig, axes = plt.subplots(2, 4, figsize=(11, 7))
    fig.suptitle(f'Absolute Value of Exemplary Channels (first row) and Training Data (second row)', fontsize=13, fontweight='bold')
    axes = axes.flatten()
    x_ticks = np.array([0, n_times-1])  # Show every 2nd tick
    y_ticks = np.array([0, n_subc-1])
    for n, ax in enumerate(axes):
        if n < 4:
            im = ax.imshow(np.abs(X_2d[n, :, :]), cmap='viridis', aspect='auto')
            ax.set_title(f"ground truth {n + 1}")
            ax.set_xticks(x_ticks)
            ax.set_yticks(y_ticks)
            ax.set_xticklabels(['0',str(round(1e3 * n_times * time_duration,4))])
            ax.set_yticklabels(['0', str(round(1e-3 * n_subc * subc_spacing,4))])
            ax.set_xlabel("time in ms")
            ax.set_ylabel("bandwidth in kHz")
            fig.colorbar(im, ax=axes[n])  # Add colorbar to each plot
        else:
            X_2d_noise = X_2d[n-4, :, :] + np.sqrt(noise_var) / np.sqrt(2) * (np.random.randn(X_2d.shape[1], X_2d.shape[2]) + 1j * np.random.randn(X_2d.shape[1],X_2d.shape[2]))
            im = ax.imshow(np.abs(X_2d_noise[:, :] * A_2D_mask), cmap='viridis', aspect='auto')
            ax.set_title(f"observation {n + 1 - 4}")
            ax.set_xticks(x_ticks)
            ax.set_yticks(y_ticks)
            fig.colorbar(im, ax=axes[n])  # Add colorbar to each plot

    # Adjust layout
    plt.tight_layout()
    plt.show()

    # create dictionaries
    # dictionary for the delay domain
    D_delay = np.zeros((n_subc,int(np.sqrt(sdim))), dtype=np.complex64)
    delays_dic = np.linspace(0, delay_max, int(np.sqrt(sdim)), endpoint=False)
    for i in range(int(np.sqrt(sdim))):
        D_delay[:, i] = np.exp(-1j * 2 * np.pi * delays_dic[i] * subc_spacing * np.arange(n_subc))
    #
    # # dictionary for the doppler domain
    D_doppler = np.zeros((n_times, int(np.sqrt(sdim))), dtype=np.complex64)
    doppler_dic = np.linspace(-doppler_max, doppler_max, int(np.sqrt(sdim)), endpoint=False)
    for i in range(int(np.sqrt(sdim))):
        D_doppler[:, i] = np.exp(1j * 2 * np.pi * doppler_dic[i] * time_duration * np.arange(n_times))

    D = np.kron(D_delay, D_doppler) # dictionary

    # create the CSGMM
    csgmm = mod.CSGMM(K, ydim, sdim, hdim, A, D, fix_zeta=noise_var)
    # train the CSGMM
    csgmm, logl_track = tr.train_csgmm_np(csgmm, miter, Y_cplx, pmax, [delays_dic, doppler_dic], sdim)

    plt.plot(logl_track)
    plt.title('Training Log Likelihood over iterations')
    plt.show()

if __name__ == '__main__':
    main()
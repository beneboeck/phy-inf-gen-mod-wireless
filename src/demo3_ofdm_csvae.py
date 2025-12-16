# Copyright 2025 Rohde & Schwarz

import sys
import os
# add the src directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
import torch
from src.utils import dataset as ds
from src.modules import csvae_modules as mod
from src.training import csvae_training as tr
import matplotlib.pyplot as plt

def main():

    np.random.seed(123456)

    miter = 700 # maximal number of iterations for training
    ydim = 250 # number of pilots for OFDM
    n_subc = 24 # number of subcarriers in the OFDM grid (predefined by dataset) # see data/documentary
    n_times = 14 # number of OFDM symbols (predefined by dataset) # see data/documentary
    subc_spacing = 15 * 1e3  # subcarrier spacing in Hz (predefined by dataset) # see data/documentary
    time_duration = 1e-3/n_times  # OFDM symbol duration in seconds (predefined by dataset) # see data/documentary
    hdim = n_subc * n_times  # dimension of the OFDM channel (predefined by dataset) # see data/documentary
    delay_max = 1.5e-5  # e.g., 6e-6 for urban, 1.5e-5 for rural [in seconds] # maximal delay considered in the delay doppler domain
    doppler_max = 700  # e.g., 250 for urban, 700 for rural [in Hz] # maximal doppler considered in the delay dopppler domain
    sdim = 40 * 40 # 32 * 32 # resolution in the delay doppler domain
    nt = 3000 # number of training samples
    nval = 1000  # number of validation samples
    ntest = 1  # not needed here, but just there for the dataset decompositions
    snr_db_range = [10,15] # snr range for the noise in the training data
    # parameters defining the CSVAE architecture and training
    ld = 32 # latent dimension
    dec_chf = 2 # factor for the decoder channels
    n_enc = 2 # number of encoder layers
    end_width = 100 # maximal width of the encoder layers
    lr = 0.0006 # learning rate
    bs = 128  # batchsize
    pmax = sdim # maximal number of allowed paths
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')  # device on which you want to run the simulation
    directory = './results/demos/demo3/'

    # load and reshape QuaDRiGa data (for this demo we only consider the training dataset in small)
    #str_load = '../data/QuaDRiGa/OFDM_14T_24S/H_10000.npy'
    str_load = '../data/QuaDRiGa/ofdm5grural_12000.npy'
    X_2d = np.load(str_load)
    n_x = X_2d.shape[0]
    X = X_2d.reshape((n_x,-1))
    X_torch = torch.from_numpy(X).cfloat()

    # compute the noisy data
    SNR_list_dB = (snr_db_range[1] - snr_db_range[0]) * np.random.rand((X.shape[0])) + snr_db_range[0] # draw random SNR values
    SNR_list = 10 ** (SNR_list_dB / 10)
    signal_energy = np.mean(np.sum(np.abs(X) ** 2, axis=1), axis=0)
    noise_var = signal_energy / (SNR_list * ydim)  # compute the corresponding noise variances

    # construct selection matrix
    Eys = np.eye(hdim)
    idx = np.arange(hdim)
    np.random.shuffle(idx)
    Eys_sh = Eys[idx, :]
    A = Eys_sh[:ydim, :]
    A = torch.from_numpy(A).cfloat()

    Y_cplx = torch.from_numpy(X + np.sqrt(noise_var[:, None]) / np.sqrt(2) * (np.random.randn(X.shape[0], X.shape[1]) + 1j * np.random.randn(X.shape[0], X.shape[1]))).cfloat().to(device)
    Y_cplx = torch.einsum('ij,kj->ki', A, X_torch)
    # split into train, val and test data
    Y_cplx_train = Y_cplx[:nt, :]
    Y_cplx_val = Y_cplx[nt:nt + nval, :]
    Y_cplx_test = Y_cplx[nt + nval:nt + nval + ntest, :]

    noise_var = torch.from_numpy(noise_var).float().to(device)

    noise_var_train = noise_var[:nt]
    noise_var_val = noise_var[nt:nt + nval]
    noise_var_test = noise_var[nt + nval:nt + nval + ntest]

    # stack standard input (N x 2*24*14) - real valued
    Y_train_stack = torch.cat((torch.real(Y_cplx_train), torch.imag(Y_cplx_train)), dim=1)
    Y_val_stack = torch.cat((torch.real(Y_cplx_val), torch.imag(Y_cplx_val)), dim=1)
    Y_test_stack = torch.cat((torch.real(Y_cplx_test), torch.imag(Y_cplx_test)), dim=1)

    _, _, _, loader_train, loader_val, _ = ds.default_ds_dl_split_w_sigma(Y_train_stack, Y_val_stack, Y_test_stack, noise_var_train, noise_var_val, noise_var_test,bs)

    del _

    # create the corresponding 2D OFDM selection matrix for visualization purposes
    A_2D_mask = np.zeros((n_subc, n_times))
    for i in range(ydim):
        mask_i = np.real(np.squeeze(A[i, :].numpy()))
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
            X_2d_noise = X_2d[n-4, :, :] + np.sqrt(noise_var[n].cpu().numpy()) / np.sqrt(2) * (np.random.randn(X_2d.shape[1], X_2d.shape[2]) + 1j * np.random.randn(X_2d.shape[1],X_2d.shape[2]))
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
    D = torch.from_numpy(D).to(device)

    # create the CSVAE
    csvae = mod.CSVAE_2D(ydim, sdim, ld, A, D, n_enc, dec_chf, end_width, device, varying_zeta='True')
    # train the CSVAE
    csvae, loss, _, _, _, _, _, _, _ = tr.train_csvae_torch(csvae, miter, lr, loader_train, loader_val, device, pmax, [delays_dic, doppler_dic], sdim, directory = directory, tracking = True)

    plt.plot(loss)
    plt.title('Training loss over iterations')
    plt.show()

if __name__ == '__main__':
    main()
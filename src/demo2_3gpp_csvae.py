# Copyright 2025 Rohde & Schwarz

import sys
import os
# add the src directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
from src.modules import csvae_modules as mod
from src.training import csvae_training as tr
from src.utils import dataset as ds
import matplotlib.pyplot as plt
import torch

def main():

    np.random.seed(123456)
    torch.manual_seed(123456)

    miter = 1500 # maximal number of iterations for training
    ydim = 16 # dimension of the observations
    hdim = 16  # dimension of the channels
    sdim = 256 # 256 # resolution in the angular domain
    nt = 3000 # number of training samples
    nval = 1000 # number of validation samples
    ntest = 1 # not needed here, but just there for the dataset decompositions
    ngen = 10000 # number of samples to be generated
    snr_db_range = [10,15] # snr range for the noise in the training data
    pmax = sdim # 16 * 16 # maximal number of allowed paths
    device = device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu') # device on which you want to run the simulation (GPU is recommended)
    directory = './results/demos/demo2/'
    # hyperparameters
    ld = 32 # latent dimension
    dec_chf = 5 # dec channel factor
    end_width = 100 # encoder endwidth
    n_enc = 3 # depth encoder
    lr = 4e-5 # learning rate
    bs = 128 # batchsize

    # load 3GPP data
    str_load = '../data/modified_3GPP/h16_1paths_12000.npy'
    X = np.load(str_load)

    # compute the noisy data
    SNR_list_dB = (snr_db_range[1] - snr_db_range[0]) * np.random.rand((X.shape[0])) + snr_db_range[0] # draw random SNR values
    SNR_list = 10 ** (SNR_list_dB / 10)
    signal_energy = np.mean(np.sum(np.abs(X) ** 2, axis=1), axis=0)
    noise_var = signal_energy / (SNR_list * ydim) # compute the corresponding noise variances

    Y_cplx = torch.from_numpy(X + np.sqrt(noise_var[:,None]) / np.sqrt(2) * (np.random.randn(X.shape[0],X.shape[1]) + 1j * np.random.randn(X.shape[0],X.shape[1]))).cfloat().to(device)
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
    # create dictionaries
    params_pi = np.linspace(-np.pi / 2, np.pi / 2, sdim, endpoint=False)
    D = np.zeros((ydim, sdim), dtype=np.complex64)
    for i in range(ydim):
        for j in range(sdim):
            D[i, j] = 1 / np.sqrt(sdim) * np.exp(-1j * np.pi * (i * np.sin(params_pi[j])))

    D = torch.from_numpy(D).cfloat().to(device)
    A = torch.eye(ydim).to(device)

    # create the CSVAE
    csvae = mod.CSVAE_1D(ydim, sdim, ld, A, D, n_enc, dec_chf, end_width, device, varying_zeta='True').to(device)
    # train the CSVAE
    csvae, risk_train, kl1_train, kl2_train, rec_train, risk_val, kl1_val, kl2_val, rec_val = tr.train_csvae_torch(csvae, miter, lr, loader_train, loader_val, device, pmax, [params_pi], sdim, directory=directory, iter_vis=20)

    csvae.eval()
    s_samples = csvae.sample(ngen, pmax)
    power_samples = np.abs(s_samples) ** 2
    mean_power = np.mean(power_samples, axis=0)
    mean_power = mean_power / np.sum(mean_power)

    plt.plot(params_pi,mean_power)
    plt.title('power angular profile')
    plt.savefig(directory + 'PAP.png')
    plt.close()

if __name__ == '__main__':
    main()
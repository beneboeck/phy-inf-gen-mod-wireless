# Copyright 2025 Rohde & Schwarz

import sys
import os
# add the src directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pywt
import numpy as np
from scipy import linalg as scilinalg
import math
import matplotlib.pyplot as plt
import datetime
import os
import csv
import torch
from src.utils import torch_utils as tu
from scipy.stats import entropy
from sklearn import linear_model
from sklearn.linear_model import orthogonal_mp_gram
from sklearn.linear_model import OrthogonalMatchingPursuit


def compute_inv_cholesky(A): # n_components, n_dim, n_dim
    [n_components,n_dim,_] = np.shape(A)
    inv_chol = np.empty((n_components, n_dim, n_dim),dtype=np.complex64)

    for k, A_matrix in enumerate(A):
        try:
            A_chol = scilinalg.cholesky(A_matrix, lower=True)
        except scilinalg.LinAlgError:
            try:
                A_chol = scilinalg.cholesky(A_matrix + 0.01 * np.eye(A_matrix.shape[0],A_matrix.shape[0]), lower=True)
            except TypeError:
                print('problem')
        inv_chol[k] = np.conj(scilinalg.solve_triangular(A_chol, np.eye(n_dim), lower=True).T)

    return inv_chol


def create_dictionary(obs_dim,sp_dim,dic):
    D = np.zeros((obs_dim, sp_dim), dtype=complex)

    if dic == 'ANG':
        angles = np.linspace(-np.pi / 2, np.pi / 2, sp_dim + 2)[1:-1]
        #sin_angles = np.linspace(-1 / 2, 1 / 2, sp_dim + 2)[1:-1]
        params = angles
        for i in range(sp_dim):
            D[:, i] = steering_vector(angles[i], obs_dim)
        #D = 1/np.sqrt(obs_dim) * D

    # v_max = 40 m/s, f_c = 2.7 GHz -> doppler_max = 360 Hz
    if dic == 'DOP':
        dopplers = np.linspace(-360, 360, sp_dim)
        params = dopplers
        for i in range(sp_dim):
            D[:, i] = steering_vector_doppler(dopplers[i], obs_dim)

    if dic == 'RND':
        D = 1 / np.sqrt(2 * obs_dim) * (
                    np.random.randn(obs_dim, sp_dim) + 1j * np.random.randn(obs_dim, sp_dim))

    if dic == 'DEL':
        delays_dic = np.linspace(0, 1.6 * 10 ** (-5), sp_dim)
        params = delays_dic
        for i in range(sp_dim):
            D[:, i] = steering_vector_delay(delays_dic[i], obs_dim,)

    return D,params

def steering_vector(angle,dim):
    return np.exp(1j * np.pi * np.sin(angle) * np.arange(dim))
    #return np.exp(1j * np.pi * angle * np.arange(dim))

def steering_vector_doppler(doppler,dim):
    return np.exp(1j * 2 * np.pi * doppler * 0.5 * np.arange(dim))

def steering_vector_delay(delay,dim,spacing = 100*10**6/1024):
    return np.exp(-1j * 2 * np.pi * delay * spacing * np.arange(dim))


def create_log(global_dic_path, global_dic, overall_path):
    # CREATING FILES AND DIRECTORY
    now = datetime.datetime.now()
    date = str(now)[:10]
    time = str(now)[11:16]
    time = time[:2] + '_' + time[3:]

    key = date + '_' + time + '_'
    key2 = key
    for n in global_dic_path:
        key = key + n + str(global_dic_path[n])


    dir_path = overall_path + key
    os.mkdir(dir_path)

    glob_file = open(dir_path + '/glob_var_file.txt', 'w')  # only the important results and the framework
    log_file = open(dir_path + '/log_file.txt', 'w')  # log_file which keeps track of the training and such stuff

    for n in global_dic:
        glob_file.write(n + ': ' + str(global_dic[n]) + '\n')
    return dir_path,key2,log_file,glob_file

def plots_training_tracking_and_saving(arrays,dir_path,comments):
    for i,array in enumerate(arrays):
        plt.plot(array)
        plt.title(comments[i])
        plt.grid(True)
        plt.savefig(dir_path + '/' + comments[i] + '.png')
        plt.close()

def save_array_to_csv(array, filename, header):
    if array.shape[1] != 2:
        raise ValueError("Array must have exactly two columns")

    # Save the array to a CSV file
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([header[0],header[1]])  # Add header (optional)
        writer.writerows(array)


def para_spread(s,paras):
    powers = np.abs(s)**2
    mu = (np.sum(paras * powers))/(np.sum(powers))
    spread = np.sqrt((np.sum((paras - mu)**2 * powers))/(np.sum(powers)))
    return spread

def compute_cdf(values,grid):
    cdf = np.zeros(len(grid))
    for i in range(len(grid)):
        cdf[i] = len(values[values <= grid[i]])/len(values)
    return cdf


def apply_sbl_torch(y,AD,max_iter,device,no_zeta=False,zeta_in=0,log_file = []):
    # applys SBL and returns the CME estimate of x
    # y: n_obs_dim | AD : n_obs_dim x n_latent_dim
    AD = torch.tensor(AD).float().to(device)
    y = torch.tensor(y).float().to(device)
    [odim,sdim] = AD.shape

    cov_diag = torch.rand(sdim).to(device)
    zeta = zeta_in
    posterior_mean_old = 3000 * torch.ones(AD.shape[1]).to(device)
    for iter in range(max_iter):

        # e-step
        Covsy = cov_diag[:, None] * AD.T  # K x S x M
        Eys = torch.eye(odim).to(device)  # [K,N,N]
        CovY = (AD * cov_diag[None, :]) @ AD.T + zeta * Eys
        L_PreY = torch.squeeze(tu.compute_inv_cholesky_torch(CovY[None, :, :], device,log_file))
        PreY = L_PreY @ torch.transpose(L_PreY,dim0=0,dim1=1)
        CovsyPy = Covsy @ PreY  # K x S x M
        diagCs_yk = cov_diag - torch.sum(CovsyPy * Covsy, dim=1)  # K x S
        postMeans = CovsyPy @ y  # NS x K x M
        error = torch.linalg.norm(postMeans - posterior_mean_old)**2
        if (error < 1 * 1e-4) & (iter > 5):
            break
        posterior_mean_old = postMeans
        cov_diag = torch.real(torch.abs(postMeans)**2 + diagCs_yk)

    Covsy = cov_diag[:, None] * AD.T  # K x S x M
    Eys = torch.eye(odim).to(device)  # [K,N,N]
    CovY = (AD * cov_diag[None, :]) @ AD.T + zeta * Eys
    L_PreY = torch.squeeze(tu.compute_inv_cholesky_torch(CovY[None, :, :], device, log_file))
    PreY = L_PreY @ torch.transpose(L_PreY, dim0=0, dim1=1)
    CovsyPy = Covsy @ PreY  # K x S x M
    diagCs_yk = cov_diag - torch.sum(CovsyPy * Covsy, dim=1)  # K x S
    postMeans = CovsyPy @ y  # NS x K x M
    print(f'gap: {error:.4f}, iter: {iter}\n')
    log_file.write(f'gap: {error:.4f}, iter: {iter}\n')
    return postMeans.to('cpu').numpy()

def apply_sbl(y,AD,max_iter,zeta_in=0):
    # applys SBL and returns the CME estimate of x
    # y: n_obs_dim | AD : n_obs_dim x n_latent_dim
    [odim,sdim] = AD.shape

    cov_diag = np.random.rand(sdim)
    zeta = zeta_in
    posterior_mean_old = 3000 * np.ones((AD.shape[1]))
    for iter in range(max_iter):
        # e-step
        Covsy = cov_diag[:, None] * np.conj(AD).T  # K x S x M
        Eys = np.eye(odim,dtype=np.complex64)  # [K,N,N]
        CovY = (AD * cov_diag[None, :]) @ np.conj(AD).T + zeta * Eys
        L_PreY = np.squeeze(compute_inv_cholesky(CovY[None, :, :]))
        PreY = L_PreY @ np.transpose(np.conj(L_PreY),(1,0))
        CovsyPy = Covsy @ PreY  # K x S x M
        diagCs_yk = cov_diag - np.sum(CovsyPy * np.conj(Covsy), axis=1)  # K x S
        postMeans = CovsyPy @ y  # NS x K x M
        error = np.linalg.norm(postMeans - posterior_mean_old)**2
        if (error < 1 * 1e-4) & (iter > 5):
            break
        posterior_mean_old = postMeans
        cov_diag = np.real(np.abs(postMeans)**2 + diagCs_yk)

    Covsy = cov_diag[:, None] * np.conj(AD).T  # K x S x M
    Eys = np.eye(odim,dtype=np.complex64)  # [K,N,N]
    CovY = (AD * cov_diag[None, :]) @ np.conj(AD).T + zeta * Eys
    L_PreY = np.squeeze(compute_inv_cholesky(CovY[None, :, :]))
    PreY = L_PreY @ np.transpose(np.conj(L_PreY),(1,0))
    CovsyPy = Covsy @ PreY  # K x S x M
    diagCs_yk = cov_diag - np.sum(CovsyPy * np.conj(Covsy), axis=1)  # K x S
    postMeans = CovsyPy @ y  # NS x K x M
    print(f'gap: {error:.4f}, iter: {iter}\n')
    return postMeans
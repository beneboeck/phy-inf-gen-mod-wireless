# Copyright 2025 Rohde & Schwarz

import os
import sys
# add the src directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import matplotlib.pyplot as plt
import numpy as np
import torch
from src.utils import visualization as vis

def train_csgmm_np(csgmm, miter, Y, pmax, grid, sdim, directory=None, init=True, init_core=None, tracking=True, iter_vis=100):
    """
    trains the csgmm in numpy (with fixed noise variance) with the option of tracking the training and visualizing randomly drawn samples
    (either by saving it to a directory, or directly displaying them)

    Parameters:
    - csgmm: The CSGMM module that is to be trained
    - miter: maximal number of iterations
    - Y: training data with dimensions nt x ydim (cplx valued)
    - pmax: only important of tracking=True, because then we only plot the pmax most important paths
    - grid: list(!) of arrays describing the gridpoints used for the dictionary
    - directory: here we save intermediate samples that we think might be interesting to save (if None, then we directly display them)
    - init: how to initialize the CSGMM for training. Typically we use random, but we could also initialize with an already trained csgmm_core
    - init_core: this would be the csgmm_core for initialization
    - iter_vis: after how many iterations, the newly random samples should be plotted again
    """

    if init == True:
        csgmm.init_paras() # initialize

    ref_logl = -10 ** 5  # first reference log likelihood for stopping criterion
    logl_track = []  # track the log likelihood

    for iter in range(miter):
        # visualize data every iter_vis iterations
        if tracking == True:
            if (iter % iter_vis == 0) & (directory == None):
                samples = csgmm.sample(9, pmax)
                vis.display_intermediate_samples(samples, grid, iter, sdim)
            elif (iter % iter_vis == 0) & (directory != None):
                samples = csgmm.sample(5, pmax)
                vis.save_intermediate_samples(samples, grid, iter, sdim, directory)

        # e-step storing the log responsibilities, the sample-wise log likelihood, the posterior means and the diagonals of the posterior covariances
        log_respos, log_likeli, posterior_means, diag_post_covs = csgmm.e_step(Y)
        # compute the overall log likelihood
        logl = np.mean(log_likeli)
        logl_track.append(logl)
        # check termination criterion
        gap = np.real(logl - ref_logl)
        if iter % 10 == 0:
            print(f'\riteration: {iter}, log-l update-gap: {gap:.4f}', end='')
        if gap < 1e-3:
            break
        ref_logl = logl
        # m-step
        csgmm.m_step(log_respos, posterior_means, diag_post_covs)

    return csgmm, logl_track


def train_csgmm_torch(csgmm, miter, Y, pmax, grid, sdim, directory=None, noise_var=None, init=True, init_core=None, tracking=True, iter_vis=100):
    """
    trains the csgmm in torch (with potentially varying noise variances) with the option of tracking the training and visualizing randomly drawn samples
    (either by saving it to a directory, or directly displaying them)

    Parameters:
    - csgmm: The CSGMM module that is to be trained
    - miter: maximal number of iterations
    - Y: training data with dimensions nt x ydim (cplx valued)
    - pmax: only important of tracking=True, because then we only plot the pmax most important paths
    - grid: list(!) of arrays describing the gridpoints used for the dictionary
    - directory: here we save intermediate samples that we think might be interesting to save (if None, then we directly display them)
    - noise_var: noise variances, a tensor on gpu of length nt
    - init: how to initialize the CSGMM for training. Typically we use random, but we could also initialize with an already trained csgmm_core
    - init_core: this would be the csgmm_core for initialization
    - iter_vis: after how many iterations, the newly random samples should be plotted again
    """

    if init == True:
        L_PreY = csgmm.init_paras(Y, noise_var) # initialize

    ref_logl = -10 ** 5  # first reference log likelihood for stopping criterion
    logl_track = []  # track the log likelihood

    with torch.no_grad():
        for iter in range(miter):
            # visualize data every iter_vis iterations
            if tracking == True:
                if (iter % iter_vis == 0) & (directory == None):
                    samples = csgmm.sample(9, pmax)
                    vis.display_intermediate_samples(samples, grid, iter, sdim)
                elif (iter % iter_vis == 0) & (directory != None):
                    samples = csgmm.sample(5, pmax)
                    vis.save_intermediate_samples(samples, grid, iter, sdim, directory)

            # e-step storing the log responsibilities, the sample-wise log likelihood, the posterior means and the diagonals of the posterior covariances
            log_respos, log_likeli, posterior_means, diag_post_covs = csgmm.e_step(Y, L_PreY)
            # compute the overall log likelihood
            logl = torch.mean(log_likeli)
            logl_track.append(logl.to('cpu').numpy())
            # check termination criterion
            gap = torch.real(logl - ref_logl)
            if iter % 10 == 0:
                print(f'\riteration: {iter}, log-l update-gap: {gap:.4f}', end='')
            if gap < 1e-3:
                break
            ref_logl = logl
            # m-step
            L_PreY = csgmm.m_step(log_respos, posterior_means, diag_post_covs, noise_var)
            del log_respos, posterior_means, diag_post_covs
            torch.cuda.empty_cache()

    return csgmm, logl_track

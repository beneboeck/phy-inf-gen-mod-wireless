# Copyright 2025 Rohde & Schwarz

import os
import sys
# add the src directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import matplotlib.pyplot as plt
import numpy as np
import torch
from src.utils import visualization as vis



def train_csvae_torch(csvae, miter, lr, loader_train, loader_eval, device, pmax, grid, sdim, directory=None, tracking=True, iter_vis=20):
    """
    trains the csvae in torch (with potentially varying noise variances) with the option of tracking the training and visualizing randomly drawn samples
    (either by saving it to a directory, or directly displaying them)

    Parameters:
    - csvae: The CSVAE module that is to be trained
    - miter: maximal number of iterations
    - lr: the learning rate (Adam Optimiter is fixed)
    - loader_train: the dataloader for training (outputs two tensors: channel observations (bs x ydim), noise variances (bs))
    - loader_eval: same as loader_train just for evaluation
    - device: device on which everything is computed
    - pmax: only important of tracking=True, because then we only plot the pmax most important paths
    - grid: list(!) of arrays describing the gridpoints used for the dictionary
    - directory: here we save intermediate samples that we think might be interesting to save (if None, then we directly display them)
    - iter_vis: after how many iterations, the newly random samples should be plotted again
    """

    # initiliaze reconstruction and kl1 and kl2 lists for training and evaluation
    rec_train, kl1_train, kl2_train, rec_val, kl1_val, kl2_val, risk_val = (torch.zeros(miter) for _ in range(7))
    # initialize parameters for early stopping criterion and adapting the learning rate
    slope, adapt_lr = -1., False
    # optimizer
    optimizer = torch.optim.Adam(lr=lr, params=csvae.CSVAE_core.parameters())
    iter_tr = 0
    for i in range(miter):
        n_iter = 0
        # visualize data every iter_vis iterations
        if tracking == True:
            if (i % iter_vis == 0) & (directory == None):
                samples = csvae.sample(9, pmax)
                vis.display_intermediate_samples(samples, grid, i, sdim)
            elif (i % iter_vis == 0) & (directory != None):
                samples = csvae.sample(5, pmax)
                vis.save_intermediate_samples(samples, grid, i, sdim, directory)
        for ind, samples in enumerate(loader_train):
            # in the first argument, we have the noisy and compressed channel observations
            samples_in = samples[0].to(device)
            # in the second argument, we have the noise variance (is neglected if self.varying_zeta == False)
            zeta_in = samples[1].to(device)
            # compute objective
            kl1, kl2, rec, risk = csvae.compute_objective(samples_in, zeta_in)
            # track training
            kl1_train[i] += kl1.detach().to('cpu')
            kl2_train[i] += kl2.detach().to('cpu')
            rec_train[i] += rec.detach().to('cpu')
            # apply update step
            optimizer.zero_grad()
            risk.backward()
            optimizer.step()
            n_iter += 1
        # average over number of iterations
        kl1_train[i] = kl1_train[i] / n_iter
        kl2_train[i] = kl2_train[i] / n_iter
        rec_train[i] = rec_train[i] / n_iter

        print(f'\repoch: {i}, kl1: {kl1_train[i].item():.4f}, kl2: {kl2_train[i].item():.4f}, rec: {rec_train[i].item():.4f}, total: {- (rec_train[i] - kl2_train[i] - kl1_train[i]).item():4f}', end='')

        # every 5 iterations, we track the validation performance
        with torch.no_grad():
            if i % 5 == 0:
                i5 = int(i / 5)
                # take care that gradients are not computed
                csvae.CSVAE_core.eval()
                n_iter = 0
                for ind, samples in enumerate(loader_eval):
                    # in the first argument, we have the noisy and compressed channel observations
                    samples_in = samples[0].to(device)
                    # in the second argument, we have the noise variance (is neglected if self.varying_zeta == False)
                    zeta_in = samples[1].to(device)
                    # compute objective
                    kl1, kl2, rec, risk = csvae.compute_objective(samples_in, zeta_in)
                    # track validation performance
                    kl1_val[i5] += kl1.detach().to('cpu')
                    kl2_val[i5] += kl2.detach().to('cpu')
                    rec_val[i5] += rec.detach().to('cpu')
                    risk_val[i5] += risk.detach().to('cpu')
                    n_iter += 1
                # average over number of iterations
                kl1_val[i5] = kl1_val[i5] / n_iter
                kl2_val[i5] = kl2_val[i5] / n_iter
                rec_val[i5] = rec_val[i5] / n_iter
                risk_val[i5] = risk_val[i5] / n_iter
                csvae.CSVAE_core.train()
                print(f'\reval: kl1: {kl1_val[i5]:.4f}, kl2: {kl2_val[i5]:.4f}, rec: {rec_val[i5]:.4f}, total: {risk_val[i5]:.4f}', end='')
                # adapting the learning rate once and otherwise we early stop if the validation performance does not improve over the previous 5 or 10 iterations
                if ((i > 40) & (adapt_lr == False)) | ((i > 60) & (adapt_lr == True)):
                    steps = 5 if adapt_lr == False else 10
                    x_range_lr = torch.arange(steps)
                    x_lr = torch.ones(steps, 2)
                    x_lr[:, 0] = x_range_lr
                    beta_lr = torch.linalg.inv(x_lr.T @ x_lr) @ x_lr.T @ risk_val[i5 - steps + 1:i5 + 1].clone()[:,
                                                                         None]
                    slope_lr = beta_lr[0].detach().to('cpu').numpy()[0]
                    print(f'\rslope risk val: {slope_lr:.6f}', end='')
                    if (slope_lr > 0) & (adapt_lr == False):
                        print('\radapting learning rate', end='')
                        optimizer.param_groups[0]['lr'] = optimizer.param_groups[0]['lr'] / 5
                        adapt_lr = True
                    elif (slope_lr > 0) & (adapt_lr == True):
                        break
    kl1_train, kl2_train, rec_train = [x[:i + 1].numpy() for x in (kl1_train, kl2_train, rec_train)]
    risk_val, kl1_val, kl2_val, rec_val = [x[:i5 + 1].numpy() for x in (risk_val, kl1_val, kl2_val, rec_val)]
    return csvae, - (rec_train - kl2_train - kl1_train), kl1_train, kl2_train, rec_train, risk_val, kl1_val, kl2_val, rec_val
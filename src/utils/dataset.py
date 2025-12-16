# Copyright 2025 Rohde & Schwarz

import torch
from torch.utils.data import DataLoader, Dataset

# This file contains standard dataset and dataloader definitions
# Many of these datasets and loaders are not used for the simulations in our paper

class default_dataset(Dataset):
    """Torch dataset that only considers one array of training samples """
    def __init__(self,y):
        super().__init__()
        self.y = y.float()

    def __len__(self):
        return self.y.size(0)

    def __getitem__(self,idx):
        return self.y[idx,:]

class default_dataset_w_zeta(Dataset):
    """Torch dataset that considers one array of training samples and one array of noise varainces"""
    def __init__(self,y,zeta):
        super().__init__()
        self.y = y.float()
        self.zeta = zeta.float()

    def __len__(self):
        return self.y.size(0)

    def __getitem__(self,idx):
        return self.y[idx,:], self.zeta[idx]

class dataset_with_gt(Dataset):
    """Torch dataset that considers one array of training samples and one array of ground truth channels"""
    def __init__(self,x,y):
        super().__init__()
        self.x = x.cfloat()
        self.y = y.float()

    def __len__(self):
        return self.x.size(0)

    def __getitem__(self,idx):
        return self.x[idx,:],self.y[idx,:]

class dataset_with_gt_w_sigma(Dataset):
    """Torch dataset that considers one array of training samples and one array of ground truth channels"""
    def __init__(self,x,y,zeta):
        super().__init__()
        self.x = x.cfloat()
        self.y = y.float()
        self.zeta = zeta.float()

    def __len__(self):
        return self.x.size(0)

    def __getitem__(self,idx):
        return self.x[idx,:],self.y[idx,:],self.zeta[idx]


def default_ds_dl_split(X_train,X_val,X_test,bs_train):
    """takes training, validation and test data and generates the corresponding loaders using the default dataset with only one array"""
    ds_train = default_dataset(X_train)
    ds_val = default_dataset(X_val)
    ds_test = default_dataset(X_test)
    dl_train = DataLoader(ds_train, shuffle=True, batch_size=bs_train)
    dl_val = DataLoader(ds_val, shuffle=True, batch_size=bs_train)
    dl_test = DataLoader(ds_test, shuffle=True, batch_size=bs_train)
    return ds_train,ds_val,ds_test,dl_train,dl_val,dl_test

def ds_dl_split_with_gt(X_train,X_val,X_test,Y_train,Y_val,Y_test,bs_train):
    """takes training, validation and test data and generates the corresponding loaders using the dataset with training and ground truth array (not used in the paper)"""
    ds_train = dataset_with_gt(X_train,Y_train)
    ds_val = dataset_with_gt(X_val,Y_val)
    ds_test = dataset_with_gt(X_test,Y_test)
    dl_train = DataLoader(ds_train, shuffle=True, batch_size=bs_train)
    dl_val = DataLoader(ds_val, shuffle=True, batch_size=bs_train)
    dl_test = DataLoader(ds_test, shuffle=True, batch_size=bs_train)
    return ds_train,ds_val,ds_test,dl_train,dl_val,dl_test

def default_ds_dl_split_w_sigma(X_train,X_val,X_test,zeta_train, zeta_val, zeta_test,bs_train):
    """takes training, validation and test data and generates the corresponding loaders using the dataset with training and noise variance array"""
    ds_train = default_dataset_w_zeta(X_train, zeta_train)
    ds_val = default_dataset_w_zeta(X_val, zeta_val)
    ds_test = default_dataset_w_zeta(X_test, zeta_test)
    dl_train = DataLoader(ds_train, shuffle=True, batch_size=bs_train)
    dl_val = DataLoader(ds_val, shuffle=True, batch_size=bs_train)
    dl_test = DataLoader(ds_test, shuffle=True, batch_size=bs_train)
    return ds_train,ds_val,ds_test,dl_train,dl_val,dl_test

def ds_dl_split_with_gt_w_sigma(X_train,X_val,X_test,Y_train,Y_val,Y_test,bs_train,sigma_train, sigma_val, sigma_test):
    """takes training, validation and test data and generates the corresponding loaders using the dataset with training, noise variance and ground truth array (not used in the paper)"""
    ds_train = dataset_with_gt_w_sigma(X_train,Y_train,sigma_train)
    ds_val = dataset_with_gt_w_sigma(X_val,Y_val,sigma_val)
    ds_test = dataset_with_gt_w_sigma(X_test,Y_test,sigma_test)
    dl_train = DataLoader(ds_train, shuffle=True, batch_size=bs_train)
    dl_val = DataLoader(ds_val, shuffle=True, batch_size=bs_train)
    dl_test = DataLoader(ds_test, shuffle=True, batch_size=bs_train)
    return ds_train,ds_val,ds_test,dl_train,dl_val,dl_test

def ds_dl_split_with_A(A_train,A_val,A_test,Y_train,Y_val,Y_test,bs_train):
    """takes training, validation and test data and generates the corresponding loaders using the dataset with training array and varying measurement matrices (not used in the paper)"""
    ds_train = dataset_with_gt(A_train,Y_train)
    ds_val = dataset_with_gt(A_val,Y_val)
    ds_test = dataset_with_gt(A_test,Y_test)
    dl_train = DataLoader(ds_train, shuffle=True, batch_size=bs_train)
    dl_val = DataLoader(ds_val, shuffle=True, batch_size=len(ds_val))
    dl_test = DataLoader(ds_test, shuffle=True, batch_size=len(ds_test))
    return ds_train,ds_val,ds_test,dl_train,dl_val,dl_test


def ds_dl_split_with_A_and_noiseVar(A_train,A_val,A_test,Y_train,Y_val,Y_test, noiseVar_train, noiseVar_val, noiseVar_test, bs_train):
    """takes training, validation and test data and generates the corresponding loaders using the dataset with training array, nosie variance array, and varying measurement matrices (not used in the paper)"""
    ds_train = dataset_with_gt_w_sigma(A_train,Y_train,noiseVar_train)
    ds_val = dataset_with_gt_w_sigma(A_val,Y_val,noiseVar_val)
    ds_test = dataset_with_gt_w_sigma(A_test,Y_test,noiseVar_test)
    dl_train = DataLoader(ds_train, shuffle=True, batch_size=bs_train)
    dl_val = DataLoader(ds_val, shuffle=True, batch_size=len(ds_val))
    dl_test = DataLoader(ds_test, shuffle=True, batch_size=len(ds_test))
    return ds_train,ds_val,ds_test,dl_train,dl_val,dl_test



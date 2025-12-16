# Copyright 2025 Rohde & Schwarz

import torch
def compute_inv_cholesky_torch2(A,device): # n_components, n_dim, n_dim
    [n_components,n_dim,_] = A.size()
    A_chol = torch.linalg.cholesky(A)
    inv_chol = torch.linalg.solve_triangular(A_chol, torch.eye(n_dim)[None,:,:].to(device), upper=False)
    inv_chol = torch.transpose(inv_chol,1,2).conj()
    return inv_chol
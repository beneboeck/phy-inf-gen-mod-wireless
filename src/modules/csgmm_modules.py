# Copyright 2025 Rohde & Schwarz

import sys
import os
# add the src directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from scipy.special import logsumexp
from src.utils import utils_general as ug
from src.utils import torch_utils as t_utils
import torch
import math

# This file includes six different classes
# - CSGMM_core:
#           This class is implemented in numpy and, thus, can be used for CPU simulations.
#           It contains the CSGMM parameters (only), and is included as an attribute in CSGMM.
# - CSGMM:
#           This class is implemented in numpy and, thus, can be used for CPU simulations. This class only!! works when
#           having a constant noise variance (stored in self.zeta) in the training dataset
#           (should be set to an increment if no noise is included in the data)
#           This class contains the logic behind the CSMM (i.e., computing the moments of s|k,y and the m- and e-step)
# - CSGMM_core_torch:
#           This class is implemented in torch and, thus, can be used for GPU simulations.
# #         It contains the CSGMM parameters (only), and is included as an attribute in CSGMM_varZ_torch.
# - CSGMM_varZ_torch:
#           This class is implemented in torch and, thus, can be used for GPU simulations. This class allows to have
#           varying noise variances throughout the training dataset (in the Zeta array).
#           This class contains the logic behind the CSMM (i.e., computing the moments of s|k,y and the m- and e-step)
# - CSGMM_core_kron_torch:
#           This class is implemented in torch and, thus, can be used for GPU simulations.
#           It contains the CSGMM kronecker parameters (only), and is included as an attribute in CSGMM_kron_varZ_torch.
# - CSGMM_kron_varZ_torch:
#           This class is implemented in torch and, thus, can be used for GPU simulations. This class allows to have
#           varying noise variances throughout the training dataset (in the Zeta array).
#           This class contains the logic behind the CSMM kronecker (i.e., computing the moments of s|k,y and the m- and e-step)
##################


###
# the core module of the CSGMM in numpy
###
class CSGMM_core():
    """the class that stores the parameters of the zero mean diagonal covariance GMM"""
    def __init__(self,n_components,sdim):
        self.n_components = n_components # number of GMM components
        self.sdim = sdim # dimension of the sparse representation
        self.gamma = np.empty((self.n_components, self.sdim)) # variance parameters for each component
        self.weights = np.empty((self.n_components)) # prior weights for each component

    def init_paras(self):
        """intializes the GMM parameters randomly"""
        self.weights = np.ones(self.n_components) / self.n_components
        self.gamma = np.random.rand(self.n_components, self.sdim)
        self.gamma = self.gamma / np.sum(self.gamma, axis=1)[:, None] * self.sdim

###
# standard complex-valued CSGMM with fixed measurement matrix and fixed noise variance in numpy
###
class CSGMM():
    """the class that contains all methods for training as well as the core GMM as attribute"""
    def __init__(self, n_components, odim, sdim, hdim, A, D, fix_zeta=0):
        self.n_components = n_components # number of GMM components
        self.odim = odim # observation dimension
        self.sdim = sdim # dimension of the sparse representation
        self.hdim = hdim # dimension of the channels
        self.A = A # measurement matrix (can be complex valued)
        self.D = D # dictionary (can be complex valued)
        self.AD = self.A @ self.D # multiplication of A and D as we typically only require A \cdot D
        self.zeta = fix_zeta # fixed noise variance for training as well as evaluation

        self.sGMM_core = CSGMM_core(self.n_components, self.sdim) # the core module
        self.CovY = np.empty((self.n_components, self.odim, self.odim),dtype=np.complex64) # the covariances of the GMM implicity defined for y
        self.PreY= np.empty((self.n_components, self.odim, self.odim),dtype=np.complex64) # the inverse covariances of the GMM implicity defined for y
        self.L_PreY = np.empty((self.n_components, self.odim, self.odim),dtype=np.complex64) # the cholesky matrices of self.PreY

    def init_paras(self):
        """initializes the GMM parameters as well as the second moments for y"""
        self.sGMM_core.init_paras()
        Eys = np.eye(self.odim,self.odim)
        self.CovY = (self.AD[None,:,:] * self.sGMM_core.gamma[:,None,:]) @ np.conj(self.AD).T[None,:,:] + self.zeta * Eys[None,:,:]
        self.L_PreY = ug.compute_inv_cholesky(self.CovY)
        self.PreY = self.L_PreY @ np.conj(np.transpose(self.L_PreY, (0, 2, 1)))

    def e_step(self, Y):  # n_samples, odim
        """e-step"""
        # compute the log responsibilities
        log_respos, log_likeli = self.compute_log_respos(Y)
        # compute the (posterior) means and diagonals of the (posterior) covariances of s|y,k
        posterior_means, posterior_covs = self.compute_sparse_posterior(Y) # n_samples,n_components,sdim (both)
        return log_respos, log_likeli, posterior_means, posterior_covs

    def compute_log_gaussian_prob(self, Y):
        """computes the logarithm of the Gaussian probabilities p(y_i|k) per component of each training sample using the implicit GMM for y (cf. Eq. (31) & (32) in https://openreview.net/forum?id=FFJFT93oa7)"""
        # compute the log-det term of the Gaussian
        log_det = np.real(2 * np.sum(np.log(np.diagonal(self.L_PreY, axis1=1, axis2=2)), axis=1))  # n_components
        # compute the remaining part of the log Gaussian
        Ly_shift = np.einsum('kji,lj->lki', np.conj(self.L_PreY), Y,optimize='greedy')  # n_samples,n_components,n_dim
        log_prob = -np.sum(np.abs(Ly_shift) ** 2, axis=2)  # n_samples, n_components
        return log_det[None, :] + log_prob  # n_samples, n_components ## REALVALUED

    def compute_log_weights(self):
        """compute the log weights"""
        return np.log(self.sGMM_core.weights)  # n_components

    def compute_log_respos(self, Y):
        """computes the logarithm of the responsibilities per component for each training sample using Bayes (cf. Eq. (31) in https://openreview.net/forum?id=FFJFT93oa7) and the per-sample log likelihood"""
        log_gauss = self.compute_log_gaussian_prob(Y)  # n_samples, n_components
        log_weights = self.compute_log_weights()  # n_components
        log_likeli = logsumexp(log_gauss + log_weights[None, :], axis=1)  # n_samples
        log_respos = (log_gauss + log_weights[None, :] - log_likeli[:, None])
        log_gauss = None
        del log_gauss
        return log_respos, log_likeli  # n_samples, n_compnents || n_samples

    def compute_sparse_posterior(self, Y):
        """ computes the (posterior) means and the diagonals of the (posterior) covariances of s|y,k (cf. Eq (24) & (25) in https://openreview.net/forum?id=FFJFT93oa7)"""
        Covsy = self.sGMM_core.gamma[:,:,None] * np.conj(self.AD).T[None,:,:] # n_components, sdim, odim
        CovsyPy = Covsy @ self.PreY # n_components, sdim, odim
        diagCs_yk = self.sGMM_core.gamma - np.sum(CovsyPy * np.conj(Covsy), axis=2)  # n_components, sdim
        postMeans = np.einsum('kil,nl->nki', CovsyPy, Y, optimize='optimal') # n_samples, n_components, odim
        return postMeans, diagCs_yk

    def m_step(self, log_respos,posterior_means,diag_post_covs):
        """m-step"""
        # compute the responsibilities and delete the log responsibilities
        respos = np.exp(log_respos)
        log_respos = None
        del log_respos

        # compute the component-wise sum over all training samples
        nk = np.real(respos).sum(axis=0) + 10 * np.finfo(respos.dtype).eps
        # update the covariance parameters of the core module and delete all arrays that are not needed anymore to save memory (cf. Eq. (34) in https://openreview.net/forum?id=FFJFT93oa7)
        self.sGMM_core.gamma = np.sum(respos[:, :, None] * np.abs(posterior_means) ** 2, axis=0) / nk[:,None] + np.real(diag_post_covs)  # [K,dimA]
        posterior_means = None
        diag_post_covs = None
        del diag_post_covs
        del posterior_means
        # regularize the covariances
        self.sGMM_core.gamma[self.sGMM_core.gamma < 1e-7] = 1e-7
        # update the weights (cf. Eq. (35) in https://openreview.net/forum?id=FFJFT93oa7)
        self.sGMM_core.weights = nk / respos.shape[0]
        # update the statistics of the implicit GMM for y (cf. Appendix E-step in Appendix A.5 and (32) in https://openreview.net/forum?id=FFJFT93oa7)
        Eys = np.eye(self.odim)
        self.CovY = (self.AD[None,:,:] * self.sGMM_core.gamma[:,None,:]) @ np.conj(self.AD).T + self.zeta * Eys
        self.L_PreY = ug.compute_inv_cholesky(self.CovY)
        self.PreY = self.L_PreY @ np.conj(np.transpose(self.L_PreY,(0,2,1)))

    def sample(self, n_samples, pmax):
        """sample from the trained GMM"""
        # compute the number of samples per component
        n_samples_comp = np.random.multinomial(n_samples, self.sGMM_core.weights)
        samples = np.zeros((n_samples, self.sdim), dtype=np.complex64)
        # iterate through the components to sample from the covariances
        s_samples_new = np.zeros((n_samples, self.sdim), dtype=np.complex64)
        for i in range(self.n_components):
            samples[np.sum(n_samples_comp[:i]): np.sum(n_samples_comp[:i]) + n_samples_comp[i], :] = np.sqrt(
                self.sGMM_core.gamma[i]) / (np.sqrt(2)) * (np.random.randn(n_samples_comp[i],self.sdim) + 1j * np.random.randn(n_samples_comp[i], self.sdim))
        for i in range(n_samples):
            sorted_indices = np.argsort(np.abs(samples[i, :]) ** 2)[::-1]
            top_five_values = samples[i, sorted_indices[:pmax]]  # Get the top 5 values
            top_five_indices = sorted_indices[:pmax]
            s_samples_new[i, top_five_indices] = top_five_values
            s_samples_new[i, :] = np.sqrt(np.sum(np.abs(samples[i, :]) ** 2) / np.sum(np.abs(s_samples_new[i, :]) ** 2)) * s_samples_new[i, :]
        samples = s_samples_new
        # shuffle the sampled the data to not have them sorted according to the GMM components
        idx_samples = np.arange(n_samples)
        np.random.shuffle(idx_samples)
        samples = samples[idx_samples, :]
        return samples
###
# the core module of the CSGMM in torch
###

class CSGMM_core_torch():
    """the class that stores the parameters of the zero mean diagonal covariance GMM"""
    def __init__(self,n_components,sdim,device):
        self.n_components = n_components # number of GMM components
        self.sdim = sdim # dimension of the sparse representation
        self.device = device # device for cuda
        self.gamma = torch.empty((self.n_components, self.sdim)).to(device) # variance parameters for each component
        self.weights = torch.empty((self.n_components)).to(device) # prior weights for each component

    def init_paras(self):
        """initializes the GMM parameters randomly"""
        self.weights = torch.ones(self.n_components).to(self.device) / self.n_components
        self.gamma = torch.rand(self.n_components, self.sdim).to(self.device)
        self.gamma = self.gamma / torch.sum(self.gamma, dim=1)[:, None] * self.sdim


###
# complex-valued CSGMM with fixed measurement matrix and varying noise variances for training in torch
###

class CSGMM_varZ_torch():
    """the class that contains all methods for training as well as the core GMM as attribute"""
    def __init__(self, n_components, odim, sdim, hdim, A, D, device):
        self.n_components = n_components # number of GMM components
        self.odim = odim # observation dimension
        self.sdim = sdim # dimension of the sparse representation
        self.hdim = hdim # dimension of the channels
        self.A = A # measurement matrix (can be complex valued)
        self.D = D # dictionary (can be complex valued)
        self.AD = self.A @ self.D # multiplication of A and D as we typically only require A \cdot D
        self.device = device # device for cuda

        self.sGMM_core = CSGMM_core_torch(self.n_components, self.sdim, device) # the core module

    def init_paras(self,Y, Zetas):
        """initializes the GMM parameters as well as the second moments for y"""
        self.sGMM_core.init_paras()
        Eys = torch.eye(self.odim,self.odim).to(self.device)
        L_PreY = torch.empty((Y.shape[0], self.n_components, self.odim, self.odim), dtype = torch.cfloat).to(self.device)
        for k in range(self.n_components):
            CovYk = (self.AD[None, :, :] * self.sGMM_core.gamma[None, k, None, :]) @ torch.conj(self.AD).T[None,:, :] + Zetas[:, None,None] * Eys[None, :, :]
            L_PreY[:,k,:,:] = t_utils.compute_inv_cholesky_torch2(CovYk, self.device)
            del CovYk
        return L_PreY

    def e_step(self, Y, L_PreY):  # n_samples, n_obs_dim // n_samples, n_components, obs_dim, obs_dim
        """e-step"""
        # compute the log responsibilities
        log_respos, log_likeli = self.compute_log_respos(Y, L_PreY)
        # compute the (posterior) means and diagonals of the (posterior) covariances of s|y,k
        posterior_means,posterior_covs = self.compute_sparse_posterior(Y, L_PreY) # n_samples,n_components,sdim (both)
        return log_respos, log_likeli,posterior_means,posterior_covs

    def compute_log_gaussian_prob(self, Y, L_PreY):
        """computes the logarithm of the Gaussian probabilities p(y_i|k) per component of each training sample using the implicit GMM for y (cf. Eq. (31) & (32) in https://openreview.net/forum?id=FFJFT93oa7)"""
        # compute the log-det term of the Gaussian
        log_det = torch.real(2 * torch.sum(torch.log(torch.diagonal(L_PreY, dim1=2, dim2=3)), dim=2))  # n_samples, n_components
        # compute the remaining part of the log Gaussian and delete arrays that are not needed anymore
        Ly_shift = torch.einsum('lkji,lj->lki', torch.conj(L_PreY), Y)  # n_samples,n_components,n_dim
        log_prob = -torch.sum(torch.abs(Ly_shift) ** 2, axis=2)  # n_samples, n_components
        del Ly_shift
        torch.cuda.empty_cache()
        return log_det + log_prob  # n_samples, n_components

    def compute_log_weights(self):
        """compute the log weights"""
        return torch.log(self.sGMM_core.weights)  # n_components

    def compute_log_respos(self, Y, L_PreY):
        """computes the logarithm of the responsibilities per component for each training sample using Bayes (cf. Eq. (31) in https://openreview.net/forum?id=FFJFT93oa7) and the per-sample log likelihood"""
        log_gauss = self.compute_log_gaussian_prob(Y, L_PreY)  # n_samples, n_components
        log_weights = self.compute_log_weights()  # n_components
        log_likeli = torch.logsumexp(log_gauss + log_weights[None, :], axis=1)  # n_samples
        log_respos = (log_gauss + log_weights[None, :] - log_likeli[:, None])
        log_gauss = None
        del log_gauss
        torch.cuda.empty_cache()
        return log_respos, log_likeli  # n_samples, n_compnents || n_samples

    def compute_sparse_posterior(self, Y, L_PreY):
        """ computes the (posterior) means and the diagonals of the (posterior) covariances of s|y,k (cf. Eq (24) & (25) in https://openreview.net/forum?id=FFJFT93oa7)"""
        diagCs_yk = torch.empty((L_PreY.shape[0],self.n_components,self.sdim),dtype=torch.cfloat).to(self.device)
        postMeans = torch.empty((L_PreY.shape[0],self.n_components,self.sdim),dtype=torch.cfloat).to(self.device)
        # compute both per component and delete arrays immediately that are not needed to save memory (this is critical as we use all data samples in the training dataset)
        for k in range(self.n_components):
            Covsyk = self.sGMM_core.gamma[k,:,None] * torch.conj(self.AD).T # S x M
            PreYk = L_PreY[:,k,:,:] @ L_PreY[:,k,:,:].conj_physical().transpose_(2, 1) # N,odim,odim (inplace operations!)
            CovsyPyk = Covsyk[None, :, :] @ PreYk  # N x S x M
            del PreYk
            torch.cuda.empty_cache()
            diag_term = torch.sum(CovsyPyk * Covsyk[None,:,:].conj_physical(), dim=2) # (inplace operations!)
            del Covsyk
            torch.cuda.empty_cache()
            diagCs_yk[:,k,:] = self.sGMM_core.gamma[None,k,:] - diag_term
            del diag_term
            torch.cuda.empty_cache()
            postMeans[:,k,:] = torch.einsum('nil,nl->ni', CovsyPyk, Y)  # NS x K x M
            del CovsyPyk
            torch.cuda.empty_cache()
        return postMeans, diagCs_yk

    def m_step(self, log_respos, posterior_means, diag_post_covs, Zetas):
        """m-step"""
        # compute the responsibilities and delete the log responsibilities
        respos = torch.exp(log_respos)
        torch.cuda.empty_cache()
        nk = torch.real(respos).sum(axis=0) + torch.tensor(1e-8).to(self.device)
        # update the covariance parameters of the core module and delete all arrays that are not needed anymore to save memory (cf. Eq. (34) in https://openreview.net/forum?id=FFJFT93oa7)
        self.sGMM_core.gamma = torch.sum(respos[:, :, None] * (torch.abs(posterior_means) ** 2 + torch.real(diag_post_covs)), dim=0) / nk[:,None]  # [K,dimA]
        del diag_post_covs, posterior_means, log_respos
        torch.cuda.empty_cache()
        self.sGMM_core.gamma[self.sGMM_core.gamma < 1e-7] = 1e-7
        # update the weights (cf. Eq. (35) in https://openreview.net/forum?id=FFJFT93oa7)
        self.sGMM_core.weights = nk / respos.shape[0]
        Eys = torch.eye(self.odim).to(self.device)
        L_PreY = torch.empty((respos.shape[0], self.n_components, self.odim, self.odim), dtype=torch.cfloat).to(self.device)
        for k in range(self.n_components):
            CovYk = (self.AD[None, :, :] * self.sGMM_core.gamma[None, k, None, :]) @ torch.conj(self.AD).T[None, :,:] + Zetas[:, None, None] * Eys[None,:, :]
            L_PreY[:, k, :, :] = t_utils.compute_inv_cholesky_torch2(CovYk, self.device)
            del CovYk
        del nk, Eys, respos
        torch.cuda.empty_cache()
        return L_PreY

    def sample(self,n_samples, pmax):
        prior = self.sGMM_core.weights.to('cpu').numpy().astype(np.float64)
        # regularize if numerical issues led to prior being no proper distribution (only happens for very small datasets)
        prior = prior / prior.sum()
        n_samples_comp = np.random.multinomial(n_samples, prior)
        samples = np.zeros((n_samples,self.sdim),dtype=np.complex64)
        # iterate through the components to sample from the covariances
        s_samples_new = np.zeros((n_samples, self.sdim), dtype=np.complex64)
        for i in range(self.n_components):
            samples[np.sum(n_samples_comp[:i]) : np.sum(n_samples_comp[:i]) + n_samples_comp[i],:] = np.sqrt(self.sGMM_core.gamma[i,:].to('cpu').numpy())/(np.sqrt(2)) * (np.random.randn(n_samples_comp[i],self.sdim) + 1j * np.random.randn(n_samples_comp[i],self.sdim))
        for i in range(n_samples):
            sorted_indices = np.argsort(np.abs(samples[i, :]) ** 2)[::-1]
            top_five_values = samples[i, sorted_indices[:pmax]]  # Get the top 5 values
            top_five_indices = sorted_indices[:pmax]
            s_samples_new[i, top_five_indices] = top_five_values
            s_samples_new[i, :] = np.sqrt(np.sum(np.abs(samples[i, :]) ** 2) / np.sum(np.abs(s_samples_new[i, :]) ** 2)) * s_samples_new[i, :]
        samples = s_samples_new
        idx_samples = np.arange(n_samples)
        np.random.shuffle(idx_samples)
        samples = samples[idx_samples, :]
        return samples

###
# the core module of the CSGMM in torch with kronecker approximation
###
class CSGMM_core_kron_torch():
    def __init__(self,n_components,sdim,device):
        self.n_components = n_components # number of GMM components
        self.sdim = int(np.sqrt(sdim)) # dimension of the sparse representation in one (!) dimension
        self.device = device # device for cuda
        self.gamma1 = torch.empty((self.n_components, self.sdim)).to(device) # variance parameters in one dimension
        self.gamma2 = torch.empty((self.n_components, self.sdim)).to(device) # variance parameters in the other dimension
        self.weights = torch.empty((self.n_components)).to(device) # prior weights for each component

    def init_paras(self):
        """intializes the GMM parameters randomly"""
        self.weights = torch.ones(self.n_components).to(self.device) / self.n_components
        self.gamma1 = torch.rand(self.n_components, self.sdim).to(self.device)
        self.gamma1 = self.gamma1 / torch.sum(self.gamma1, dim=1)[:, None] * self.sdim
        self.gamma2 = torch.rand(self.n_components, self.sdim).to(self.device)
        self.gamma2 = self.gamma2 / torch.sum(self.gamma2, dim=1)[:, None] * self.sdim

###
# complex-valued CSGMM with kronecker approximation, fixed measurement matrix and varying noise variances  in torch
###

class CSGMM_kron_varZ_torch():
    """the class that contains all methods for training as well as the core GMM as attribute"""
    def __init__(self, n_components, odim, sdim, hdim, A, D, device):
        self.n_components = n_components # number of GMM components
        self.odim = odim # observation dimension
        self.sdim = sdim # dimension of the sparse representation
        self.hdim = hdim # dimension of the channels
        self.A = A # measurement matrix (can be complex valued)
        self.D = D # dictionary (can be complex valued)
        self.AD = self.A @ self.D  # multiplication of A and D as we typically only require A \cdot D
        self.device = device # device for cuda

        self.sGMM_core = CSGMM_core_kron_torch(self.n_components, self.sdim, device) # core GMM

    def init_paras(self,Y, Zetas):
        """initializes the GMM parameters as well as the second moments for y"""
        self.sGMM_core.init_paras()
        Eys = torch.eye(self.odim,self.odim).to(self.device)
        L_PreY = torch.empty((Y.shape[0], self.n_components, self.odim, self.odim), dtype = torch.cfloat).to(self.device)
        for k in range(self.n_components):
            gammak = torch.kron(self.sGMM_core.gamma1[k,:],self.sGMM_core.gamma2[k,:])
            CovYk = (self.AD[None, :, :] * gammak[None,None,:]) @ torch.conj(self.AD).T[None,:, :] + Zetas[:, None,None] * Eys[None, :, :]
            L_PreY[:,k,:,:] = t_utils.compute_inv_cholesky_torch2(CovYk, self.device)
            del CovYk, gammak
        torch.cuda.empty_cache()
        return L_PreY

    def e_step(self, Y, L_PreY):  # n_samples, n_obs_dim // n_samples, n_components, obs_dim, obs_dim
        """e-step"""
        # compute the log responsibilities
        log_respos, log_likeli = self.compute_log_respos(Y, L_PreY)
        # compute the (posterior) means and diagonals of the (posterior) covariances of s|y,k
        posterior_means,posterior_covs = self.compute_sparse_posterior(Y, L_PreY)
        return log_respos, log_likeli,posterior_means,posterior_covs

    def compute_log_gaussian_prob(self, Y, L_PreY):
        """computes the logarithm of the Gaussian probabilities p(y_i|k) per component of each training sample using the implicit GMM for y (cf. Eq. (31) & (32) in https://openreview.net/forum?id=FFJFT93oa7)"""
        # compute the log-det term of the Gaussian
        log_det = torch.real(2 * torch.sum(torch.log(torch.diagonal(L_PreY, dim1=2, dim2=3)), dim=2))  # n_samples, n_components
        # compute the remaining part of the log Gaussian
        Ly_shift = torch.einsum('lkji,lj->lki', torch.conj(L_PreY), Y)  # n_samples,n_components,n_dim
        log_prob = -torch.sum(torch.abs(Ly_shift) ** 2, axis=2)  # n_samples, n_components
        del Ly_shift
        torch.cuda.empty_cache()
        return log_det + log_prob  # n_samples, n_components
    def compute_log_weights(self):
        """compute the log weights"""
        return torch.log(self.sGMM_core.weights)  # n_components

    def compute_log_respos(self, Y, L_PreY):
        """computes the logarithm of the responsibilities per component for each training sample using Bayes (cf. Eq. (31) in https://openreview.net/forum?id=FFJFT93oa7) and the per-sample log likelihood"""
        log_gauss = self.compute_log_gaussian_prob(Y, L_PreY)  # n_samples, n_components
        log_weights = self.compute_log_weights()  # n_components
        log_likeli =torch.logsumexp(log_gauss + log_weights[None, :], axis=1)  # n_samples
        log_respos = (log_gauss + log_weights[None, :] - log_likeli[:, None])
        log_gauss = None
        del log_gauss
        torch.cuda.empty_cache()
        return log_respos, log_likeli  # n_samples, n_compnents || n_samples

    def compute_sparse_posterior(self, Y, L_PreY):
        """ computes the (posterior) means and the diagonals of the (posterior) covariances of s|y,k (cf. Eq (24) & (25) in https://openreview.net/forum?id=FFJFT93oa7)"""
        diagCs_yk = torch.empty((L_PreY.shape[0],self.n_components,self.sdim),dtype=torch.cfloat).to(self.device)
        postMeans = torch.empty((L_PreY.shape[0],self.n_components,self.sdim),dtype=torch.cfloat).to(self.device)
        for k in range(self.n_components):
            gammak = torch.kron(self.sGMM_core.gamma1[k, :], self.sGMM_core.gamma2[k, :])
            Covsyk = gammak[:,None] * torch.conj(self.AD).T # S x M
            PreYk = L_PreY[:,k,:,:] @ L_PreY[:,k,:,:].conj_physical().transpose_(2, 1) # N,odim,odim
            CovsyPyk = Covsyk[None, :, :] @ PreYk  # N x S x M
            del PreYk
            torch.cuda.empty_cache()
            diag_term = torch.sum(CovsyPyk * Covsyk[None,:,:].conj_physical(), dim=2)
            del Covsyk
            torch.cuda.empty_cache()
            diagCs_yk[:,k,:] = gammak[None,:] - diag_term
            del diag_term, gammak
            torch.cuda.empty_cache()
            postMeans[:,k,:] = torch.einsum('nil,nl->ni', CovsyPyk, Y)  # NS x K x S
            del CovsyPyk
            torch.cuda.empty_cache()
        return postMeans, diagCs_yk

    def m_step(self, log_respos, posterior_means, diag_post_covs, Zetas): # posterior_means: (N_t, K, S),
        """m-step"""
        # compute the responsibilities and delete the log responsibilities
        respos = torch.exp(log_respos)
        torch.cuda.empty_cache()
        nk = torch.real(respos).sum(axis=0) + torch.tensor(1e-8).to(self.device)
        posterior_means = posterior_means.view(respos.shape[0],self.n_components,int(math.sqrt(self.sdim)),int(math.sqrt(self.sdim))) # N_t, K, sqrt(S), sqrt(S)
        diag_post_covs = diag_post_covs.view(respos.shape[0],self.n_components,int(math.sqrt(self.sdim)),int(math.sqrt(self.sdim)))

        gamma1 = self.sGMM_core.gamma1
        gamma2 = self.sGMM_core.gamma2
        gamma1_old = 1000 * torch.ones(self.sGMM_core.gamma1.shape).to(self.device)
        gamma2_old = 1000 * torch.ones(self.sGMM_core.gamma2.shape).to(self.device)

        # iterate for coordinate search
        for iter in range(100):
            nominator = torch.sum(respos[:, :, None, None] * (torch.abs(posterior_means) ** 2 + torch.real(diag_post_covs)), dim=0) # K, sqrt(S) (1), sqrt(S) (2)
            denominator1 = int(math.sqrt(self.sdim)) * gamma2 * nk[:,None] # K, sqrt(S) (2)
            gamma1 = torch.sum(nominator/denominator1[:,None,:],dim=2) # K, sqrt(S) (1)
            gamma1[gamma1 < 1e-5] = 1e-5
            denominator2 = int(math.sqrt(self.sdim)) * gamma1 * nk[:, None]  # K, sqrt(S) (1)
            gamma2 = torch.sum(nominator / denominator2[:, :, None], dim=1)  # K, sqrt(S) (2)
            gamma2[gamma2 < 1e-5] = 1e-5
            if (torch.mean(torch.sum(torch.abs(gamma1 - gamma1_old)**2,dim=1)) < 1e-3) & (torch.mean(torch.sum(torch.abs(gamma2 - gamma2_old)**2,dim=1)) < 1e-3):
                break
            gamma1_old = gamma1
            gamma2_old = gamma2
        # update the covariance parameters of the core module and delete all arrays that are not needed anymore to save memory (cf. Eq. (44) & (45) in https://openreview.net/forum?id=FFJFT93oa7)
        self.sGMM_core.gamma1 = gamma1
        self.sGMM_core.gamma2 = gamma2

        del diag_post_covs, posterior_means, log_respos, nominator, denominator1, gamma1, denominator2, gamma2, gamma1_old, gamma2_old
        torch.cuda.empty_cache()
        self.sGMM_core.gamma1[self.sGMM_core.gamma1 < 1e-5] = 1e-5
        self.sGMM_core.gamma2[self.sGMM_core.gamma2 < 1e-5] = 1e-5
        # update the weights (cf. Eq. (35) in https://openreview.net/forum?id=FFJFT93oa7)
        self.sGMM_core.weights = nk / respos.shape[0]
        # update the statistics of the implicit GMM for y (cf. Appendix E-step in Appendix A.5 and (32) in https://openreview.net/forum?id=FFJFT93oa7)
        Eys = torch.eye(self.odim).to(self.device)
        L_PreY = torch.empty((respos.shape[0], self.n_components, self.odim, self.odim), dtype=torch.cfloat).to(self.device)
        for k in range(self.n_components):
            gammak = torch.kron(self.sGMM_core.gamma1[k,:], self.sGMM_core.gamma2[k,:])
            CovYk = (self.AD[None, :, :] * gammak[None, None, :]) @ torch.conj(self.AD).T[None, :,:] + Zetas[:, None, None] * Eys[None,:, :]
            L_PreY[:, k, :, :] = t_utils.compute_inv_cholesky_torch2(CovYk, self.device)
            del CovYk, gammak
        del nk, Eys, respos
        torch.cuda.empty_cache()
        return L_PreY

    def sample(self,n_samples, pmax):
        prior = self.sGMM_core.weights.to('cpu').numpy().astype(np.float64)
        # regularize if numerical issues led to the prior being no proper distribution
        prior = prior / prior.sum()
        n_samples_comp = np.random.multinomial(n_samples, prior)
        samples = np.zeros((n_samples,self.sdim),dtype=np.complex64)
        k_max = torch.argmax(self.sGMM_core.weights)
        # iterate through the components to sample from the covariances
        s_samples_new = np.zeros((n_samples, self.sdim), dtype=np.complex64)
        for i in range(self.n_components):
            gammai = torch.kron(self.sGMM_core.gamma1[i,:], self.sGMM_core.gamma2[i,:])
            samples[np.sum(n_samples_comp[:i]) : np.sum(n_samples_comp[:i]) + n_samples_comp[i],:] = np.sqrt(gammai.to('cpu').numpy())/(np.sqrt(2)) * (np.random.randn(n_samples_comp[i],self.sdim) + 1j * np.random.randn(n_samples_comp[i],self.sdim))
        for i in range(n_samples):
            sorted_indices = np.argsort(np.abs(samples[i, :]) ** 2)[::-1]
            top_five_values = samples[i, sorted_indices[:pmax]]  # Get the top 5 values
            top_five_indices = sorted_indices[:pmax]
            s_samples_new[i, top_five_indices] = top_five_values
            s_samples_new[i, :] = np.sqrt(np.sum(np.abs(samples[i, :]) ** 2) / np.sum(np.abs(s_samples_new[i, :]) ** 2)) * s_samples_new[i, :]
        samples = s_samples_new
        idx_samples = np.arange(n_samples)
        np.random.shuffle(idx_samples)
        samples = samples[idx_samples,:]
        return samples

# Copyright 2025 Rohde & Schwarz

from utils import torch_utils as tu
import torch
import torch.nn as nn
import math
import numpy as np

###
# the core module of the 1D-CSVAE with deep decoder motivated decoder
###

class CSVAE_core_1D(nn.Module):
    """core module 1D-CSVAE"""
    def __init__(self, input_dim, latent_dim, output_dim, n_enc, dec_chf, end_width, device):
        super().__init__()
        self.device = device # device fpr cuda
        self.input_dim = input_dim # input dimension for the decoder (i.e., observation dimension)
        self.latent_dim = latent_dim # latent dimension of the CSVAE
        self.output_dim = output_dim # output dimension (i.e., the dimension of the sparse representation)
        self.n_enc = n_enc # number of layer in the fully connected encoder
        self.end_width = end_width  # maximal width of the fully connected encoder neural network
        self.dec_chf = dec_chf # scaling factor of the number of channels in the decoder architecture

        # encoder construction
        # compute the layer-to-layer steps for the linearly increasing width of the layers in the encoder neural network
        if (self.n_enc - 1) != 0:
            if self.end_width > self.input_dim:
                steps_enc = (self.end_width - self.input_dim) // (self.n_enc - 1)
            else:
                steps_enc = - ((self.input_dim - self.end_width) // (self.n_enc - 1))

            encoder = []
            layer_dim = self.input_dim

            # create the layers
            for n in range(n_enc - 1):
                encoder.append(nn.Linear(layer_dim, layer_dim + steps_enc))
                encoder.append(nn.ReLU())
                layer_dim = layer_dim + steps_enc
            encoder.append(nn.Linear(layer_dim, self.end_width))
            encoder.append(nn.ReLU())
        else:
            encoder = []
            layer_dim = self.input_dim
            encoder.append(nn.Linear(layer_dim, self.end_width))
            encoder.append(nn.ReLU())
        self.encoder = nn.Sequential(*encoder)

        # final linear layer that maps to the latent dimension
        self.fc_mu = nn.Linear(self.end_width, self.latent_dim)
        self.fc_var = nn.Linear(self.end_width, self.latent_dim)

        # decoder construction (specific for an output dimension that equals any power of 2 (greater 8)!!)
        if (self.output_dim/8) % 1 != 0:
            print('decoder architecture does not work, change sparse dimension to be dividable by self.output_dim/8')
        decoder = []
        decoder.append(nn.Linear(self.latent_dim, self.output_dim//8 * self.dec_chf))
        decoder.append(nn.ReLU())
        decoder.append(nn.Linear(self.output_dim//8 * self.dec_chf, self.output_dim//8 * self.dec_chf * 4))
        decoder.append(nn.ReLU())
        decoder.append(nn.Unflatten(1, (self.dec_chf * 4, self.output_dim//8)))
        decoder.append(nn.Conv1d(self.dec_chf * 4, self.dec_chf * 16, 1, bias=False))
        decoder.append(nn.ReLU())
        decoder.append(nn.Upsample(scale_factor=2, mode='linear'))
        decoder.append(nn.Conv1d(self.dec_chf * 16, self.dec_chf * 48, 1, bias=False))
        decoder.append(nn.ReLU())
        decoder.append(nn.Upsample(scale_factor=2, mode='linear'))
        decoder.append(nn.Conv1d(self.dec_chf * 48, self.dec_chf * 48 * 2, 1, bias=False))
        decoder.append(nn.ReLU())
        decoder.append(nn.Upsample(scale_factor=2, mode='linear'))
        decoder.append(nn.Conv1d(self.dec_chf * 48 * 2, 1, 1, bias=False))
        self.decoder = nn.Sequential(*decoder)

    def encode(self, x):
        """encodes x """
        out = self.encoder(x)
        mu, log_var = self.fc_mu(out), self.fc_var(out)
        return mu, log_var

    def reparameterize(self, log_var, mu):
        """VAE reparameterization trick"""
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(log_var)
        return mu + eps * std

    def decode(self, z):
        "decodes z with regularizing log gamma"
        log_gamma = torch.squeeze(self.decoder(z),dim=1)
        log_gamma[log_gamma < -10] = -10
        return log_gamma

    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(log_var, mu)
        log_gamma = self.decode(z)
        return log_gamma, mu, log_var

###
# complex valued 1D CSVAE with fixed measurement matrix and fixed or varying noise variances for training
###

class CSVAE_1D(nn.Module):
    """main class for the CSVAE when having only one dimension"""
    def __init__(self, odim, sdim, ldim, A, D, n_enc, dec_chf, end_width, device, varying_zeta, fix_zeta=0.0):
        super().__init__()
        self.sdim = sdim # dimension of the sparse representation
        self.odim = odim # dimension of the observations
        self.A = A.cfloat().to(device) # measurement matrix
        self.D = D.cfloat().to(device) # dictionary
        self.AD = self.A @ self.D # A \cdot D
        self.device = device # device for cuda
        self.varying_zeta = varying_zeta # boolean whether we use varying or fixed noise variances for training
        self.zeta = torch.from_numpy(np.array(fix_zeta)).float().to(device) # fixed noise variance if we use fixed one
        self.CSVAE_core = CSVAE_core_1D(2 * odim, ldim, sdim, n_enc, dec_chf, end_width, device).to(device) # core module

    def compute_objective(self, samples_in, noise_var):
        # processing chain through the CSVAE core module
        log_gamma, mu, log_var = self.CSVAE_core(samples_in)
        # compute the posterior mean and the covariances of y
        posterior_means, diagLCovY = self.compute_sparse_posterior(samples_in, log_gamma, noise_var)
        # compute the first term in the objective (standard kl from VAEs)
        kl1 = self.compute_kl1_divergence(log_var, mu)
        # compute the second term (new kl divergence) with already applied reformulation (see Appendix)
        kl2 = self.compute_kl2_divergence(log_gamma, posterior_means, diagLCovY, noise_var)
        # compute the reconstruction loss with already applied reformulation (see Appendix)
        rec = self.reconstruction_loss(samples_in, posterior_means, noise_var)
        # save memory
        del posterior_means, diagLCovY, log_gamma, mu, log_var
        torch.cuda.empty_cache()
        # compute the objective
        risk = - (rec - kl2 - kl1)
        return kl1, kl2, rec, risk

    def compute_sparse_posterior(self, Y, log_gamma, zeta_in):
        """
        computes the posterior means (i.e., E[s|z_i,y_i] for all i in the batch and z_i ~ q(z|y_i)) and Cov[y_i|z_i]
        see Equation (24) and (26) in Appendix A.3 in (https://openreview.net/forum?id=FFJFT93oa7)
        """
        # make the data complex valued again
        Y_cplx = Y[:, :self.odim] + 1j * Y[:, self.odim:]
        # compute the gammas
        gamma_full = log_gamma.exp()
        bs = gamma_full.shape[0]
        Covsy = gamma_full[:, :, None] * torch.conj(self.AD).T[None, :, :]
        Eys = torch.eye(self.odim, dtype=torch.complex64)[None, :, :].repeat(bs, 1, 1).to(self.device)  # [K,N,N]
        if self.varying_zeta == 'False':
            CovY = (self.AD[None, :, :] * gamma_full[:, None, :]) @ torch.conj(self.AD).T[None, :, :] + self.zeta * Eys
        elif self.varying_zeta == 'True':
            CovY = (self.AD[None, :, :] * gamma_full[:, None, :]) @ torch.conj(self.AD).T[None, :, :] + zeta_in[:, None,None] * Eys
        LCovY = torch.linalg.cholesky(CovY)
        pmeans_partial = torch.cholesky_solve(Y_cplx.unsqueeze(2), LCovY).squeeze(2)
        diagLCovY = torch.real(torch.diagonal(LCovY, dim1=-2, dim2=-1))
        postMeans = torch.bmm(Covsy, pmeans_partial.unsqueeze(2)).squeeze(2)  # computes einsum('zij,zj->zi', Covsy, pmeans_partial)

        # delete all arrays that are not needed anymore
        del Eys, Covsy, pmeans_partial, CovY
        torch.cuda.empty_cache()
        return postMeans, diagLCovY

    def compute_kl1_divergence(self, log_var, mu):
        """
        computes the standard KL diverence from VAEs
        see Equation (27) in Appendix A.4 in (https://openreview.net/forum?id=FFJFT93oa7)
        """
        return torch.mean(torch.sum(-0.5 * (1 + log_var - mu ** 2 - (log_var).exp()), dim=1))  # [1]
    def compute_kl2_divergence(self, log_gamma, posterior_means, diagLCovY, zeta_in):
        """
        computes the second KL divergence in CSVAEs but already reformulated (either for fixed or varying noise variances)
        see second "bracket" term in Equation (30) in Appendix A.4 in (https://openreview.net/forum?id=FFJFT93oa7)
        """
        if self.varying_zeta == 'False':
            kl = - self.odim * torch.log(self.zeta) + 2 * torch.sum(torch.log(diagLCovY),dim=1) + torch.sum(torch.abs(posterior_means) ** 2 / log_gamma.exp(), dim=1)
        elif self.varying_zeta == 'True':
            kl = - self.odim * torch.log(zeta_in) + 2 * torch.sum(torch.log(diagLCovY),dim=1) + torch.sum(torch.abs(posterior_means) ** 2 / log_gamma.exp(), dim=1)
        return torch.mean(kl)

    def reconstruction_loss(self, Y, posterior_mean, zeta_in):  # M x N | n_b x 2 x N | n_b x M | n_b x M x M
        """
        computes the reconstruction loss (already reformulated)
        see first "bracket" term in Equation (30) in Appendix A.4 in (https://openreview.net/forum?id=FFJFT93oa7)
        """
        # make the data complex valued again
        Y_cplx = Y[:, :self.odim] + 1j * Y[:, self.odim:]
        s_error = torch.linalg.norm(Y_cplx - posterior_mean @ self.AD.T, axis=1) ** 2  # n_b, computes norm(Y_cplx - einsum('ij,hj->hi', self.AD, posterior_mean), axis=1) ** 2  # n_b
        if self.varying_zeta == 'False':
            return torch.mean(- (Y_cplx.shape[1] * torch.log(torch.pi * self.zeta) + (s_error) / self.zeta))
        elif self.varying_zeta == 'True':
            return torch.mean(- (Y_cplx.shape[1] * torch.log(torch.pi * zeta_in) + (s_error) / zeta_in))

    def sample(self, n_samples, pmax):
        """sample generation method"""
        s_samples = np.zeros((n_samples, self.sdim), dtype=np.complex64)
        n_50 = n_samples // 50 # this leads to generating 50 samples at a time and is just for not exploding your GPU memory resources
        for n in range(n_50):
            z_samples = torch.randn(50, self.CSVAE_core.latent_dim).to(self.device)
            log_gamma = self.CSVAE_core.decode(z_samples).detach()
            gamma_full = log_gamma.exp()
            sqrt_gammas = torch.sqrt(gamma_full)
            s_samples50 = sqrt_gammas / torch.sqrt(torch.tensor(2).to(self.device)) * (torch.randn(50, self.sdim).to(self.device) + 1j * torch.randn(50, self.sdim).to(self.device))
            s_samples50 = s_samples50.detach().to('cpu').numpy()
            s_samples_new = np.zeros((50, self.sdim), dtype=np.complex64)
            for i in range(50):
                sorted_indices = np.argsort(np.abs(s_samples50[i, :]) ** 2)[::-1]
                top_five_values = s_samples50[i, sorted_indices[:pmax]]  # Get the top pmax values
                top_five_indices = sorted_indices[:pmax]
                s_samples_new[i, top_five_indices] = top_five_values
                s_samples_new[i, :] = np.sqrt(np.sum(np.abs(s_samples50[i, :]) ** 2) / np.sum(np.abs(s_samples_new[i, :]) ** 2)) * s_samples_new[ i, :]
            s_samples[n * 50:(n + 1) * 50, :] = s_samples_new
            del log_gamma, gamma_full, sqrt_gammas
            torch.cuda.empty_cache()
        if n_samples % 50 != 0:
            z_samples = torch.randn(n_samples - n_50 * 50, self.CSVAE_core.latent_dim).to(self.device)
            log_gamma = self.CSVAE_core.decode(z_samples)
            gamma_full = log_gamma.exp()
            sqrt_gammas = torch.sqrt(gamma_full)
            s_samples50 = sqrt_gammas / torch.sqrt(torch.tensor(2).to(self.device)) * (torch.randn(n_samples - n_50 * 50, self.sdim).to(self.device) + 1j * torch.randn(
                    n_samples - n_50 * 50, self.sdim).to(self.device))
            s_samples50 = s_samples50.detach().to('cpu').numpy()
            s_samples_new = np.zeros((n_samples - n_50 * 50, self.sdim), dtype=np.complex64)
            for i in range(n_samples - n_50 * 50):
                sorted_indices = np.argsort(np.abs(s_samples50[i, :]) ** 2)[::-1]
                top_five_values = s_samples50[i, sorted_indices[:pmax]]  # Get the top pmax values
                top_five_indices = sorted_indices[:pmax]
                s_samples_new[i, top_five_indices] = top_five_values
                s_samples_new[i, :] = np.sqrt(np.sum(np.abs(s_samples50[i, :]) ** 2) / np.sum(np.abs(s_samples_new[i, :]) ** 2)) * s_samples_new[i, :]
            s_samples[n_50 * 50:, :] = s_samples_new
        return s_samples


###
# the core module of the 2D-CSVAE with deep decoder motivated decoder, kronecker approximation and fully connected encoder
###
class CSVAE_core_kron_2D(nn.Module):
    """core module 2D-CSVAE with kronecker approximation"""
    def __init__(self, input_dim, latent_dim, output_dim, n_enc, end_width, dec_chf, device):
        super().__init__()
        self.device = device # device for cuda
        self.input_dim = input_dim # input dimension for the decoder (i.e., observation dimension)
        self.latent_dim = latent_dim # latent dimension of the CSVAE
        self.output_dim = output_dim # output dimension (i.e., the dimension of the sparse representation of BOTH DIMS CONCATENATED!!)
        self.n_enc = n_enc # number of layers in the fully connected encoder
        self.end_width = end_width # maximal width of the fully connected encoder neural network
        self.dec_chf = dec_chf # scaling factor of the number of channels in the decoder architecture

        # encoder construction
        # compute the layer-to-layer steps for the linearly increasing width of the layers in the encoder neural network
        if (self.n_enc - 1) != 0:
            if self.end_width > self.input_dim:
                steps_enc = (self.end_width - self.input_dim) // (self.n_enc - 1)
            else:
                steps_enc = - ((self.input_dim - self.end_width) // (self.n_enc - 1))
            encoder = []
            layer_dim = self.input_dim
            # create the layers
            for n in range(n_enc - 1):
                encoder.append(nn.Linear(layer_dim, layer_dim + steps_enc))
                encoder.append(nn.ReLU())
                layer_dim = layer_dim + steps_enc
            encoder.append(nn.Linear(layer_dim, self.end_width))
            encoder.append(nn.ReLU())
        else:
            encoder = []
            layer_dim = self.input_dim
            encoder.append(nn.Linear(layer_dim, self.end_width))
            encoder.append(nn.ReLU())
        self.encoder = nn.Sequential(*encoder)
        # final linear layer that maps to the latent dimension
        self.fc_mu = nn.Linear(self.end_width, self.latent_dim)
        self.fc_var = nn.Linear(self.end_width, self.latent_dim)

        # decoder construction, we have three networks, one which is a fully connected one mapping from the latent dimension to an intermediate stage and then two seperate deep decoder motivated ones for each dimension respectively
        decoder_lin = []
        decoder_d1 = []
        decoder_d2 = []

        # pre-stage fully connected one
        decoder_lin.append(nn.Linear(self.latent_dim,self.dec_chf * 2 * int(math.sqrt(self.output_dim))//8))
        decoder_lin.append(nn.ReLU())

        # deep decoder motivated decoder for one dimension
        if (int(math.sqrt(self.output_dim))/8 % 1) != 0:
            print('decoder architecture does not work, change sparse dimension to be dividable by sqrt(self.output_dim)//8')
        decoder_d1.append(nn.Linear(self.dec_chf * 2 * int(math.sqrt(self.output_dim))//8, self.dec_chf * 4 * int(math.sqrt(self.output_dim))//8))
        decoder_d1.append(nn.ReLU())
        decoder_d1.append(nn.Unflatten(1,(self.dec_chf * 4, int(math.sqrt(self.output_dim))//8)))
        decoder_d1.append(nn.Conv1d(self.dec_chf * 4,self.dec_chf * 16,1))
        decoder_d1.append(nn.ReLU())
        decoder_d1.append(nn.Upsample(scale_factor=2,mode='linear'))
        decoder_d1.append(nn.Conv1d(self.dec_chf * 16, self.dec_chf * 64, 1))
        decoder_d1.append(nn.ReLU())
        decoder_d1.append(nn.Upsample(scale_factor=2, mode='linear'))
        decoder_d1.append(nn.Conv1d(self.dec_chf * 64, self.dec_chf * 128, 1))
        decoder_d1.append(nn.ReLU())
        decoder_d1.append(nn.Upsample(scale_factor=2, mode='linear'))
        decoder_d1.append(nn.Conv1d(self.dec_chf * 128, 1, 1))

        # deep decoder motivated decoder for the other dimension
        decoder_d2.append(nn.Linear(self.dec_chf * 2 * int(math.sqrt(self.output_dim))//8, self.dec_chf * 4 * int(math.sqrt(self.output_dim))//8))
        decoder_d2.append(nn.ReLU())
        decoder_d2.append(nn.Unflatten(1, (self.dec_chf * 4, int(math.sqrt(self.output_dim))//8)))
        decoder_d2.append(nn.Conv1d(self.dec_chf * 4,self.dec_chf * 16, 1))
        decoder_d2.append(nn.ReLU())
        decoder_d2.append(nn.Upsample(scale_factor=2, mode='linear'))
        decoder_d2.append(nn.Conv1d(self.dec_chf * 16, self.dec_chf * 64, 1))
        decoder_d2.append(nn.ReLU())
        decoder_d2.append(nn.Upsample(scale_factor=2, mode='linear'))
        decoder_d2.append(nn.Conv1d(self.dec_chf * 64, self.dec_chf * 128, 1))
        decoder_d2.append(nn.ReLU())
        decoder_d2.append(nn.Upsample(scale_factor=2, mode='linear'))
        decoder_d2.append(nn.Conv1d(self.dec_chf * 128, 1, 1))

        self.decoder_lin = nn.Sequential(*decoder_lin)
        self.decoder_d1 = nn.Sequential(*decoder_d1)
        self.decoder_d2 = nn.Sequential(*decoder_d2)

    def encode(self, x):
        """encodes x """
        x_in = nn.Flatten()(x)
        out = self.encoder(x_in)
        mu, log_var = self.fc_mu(out), self.fc_var(out)
        return mu, log_var

    def reparameterize(self, log_var, mu):
        """VAE reparameterization trick"""
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(log_var)
        return mu + eps * std

    def decode(self, z):
        """decodes z with regularizing log gamma"""
        x1 = self.decoder_lin(z)
        log_gamma_1 = torch.squeeze(self.decoder_d1(x1),dim=1)
        log_gamma_2 = torch.squeeze(self.decoder_d2(x1), dim=1)
        log_gamma_1[log_gamma_1 < -10] = -10
        log_gamma_2[log_gamma_2 < -10] = -10
        return log_gamma_1,log_gamma_2

    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(log_var, mu)
        log_gamma_1,log_gamma_2 = self.decode(z)
        return log_gamma_1,log_gamma_2, mu, log_var

###
# complex valued 2D CSVAE with fixed measurement matrix, kronecker approximation and fixed or varying noise variances for training
###

class CSVAE_kron_2D(nn.Module):
    def __init__(self, odim, sdim, ldim, A, D, n_enc, dec_chf, end_width, device, fix_zeta=0.0, varying_zeta = 'False'):
        super().__init__()
        self.sdim = sdim # dimension of the sparse representation of ALL DIMENSION STACKED
        self.odim = odim # dimension of the observations
        self.A = A.cfloat().to(device) # measurement matrix
        self.D = D.cfloat().to(device) # dictionary
        self.AD = self.A @ self.D # A \cdot D
        self.device = device # device for cuda
        self.zeta = torch.from_numpy(np.array(fix_zeta)).float().to(device) # fixed noise variance if we use fixed one
        self.varying_zeta = varying_zeta # boolean whether we use varying or fixed noise variances for training
        self.CSVAE_core = CSVAE_core_kron_2D(2 * odim, ldim, sdim, n_enc, end_width, dec_chf, device).to(device) # core module

    def compute_objective(self, samples_in, noise_var):
        # processing chain through the CSVAE core module
        log_gamma_1, log_gamma_2, mu, log_var = self.CSVAE_core(samples_in)
        # compute the posterior mean and the covarinces of y
        posterior_means, diagLCovY, gamma_full = self.compute_sparse_posterior(samples_in, log_gamma_1, log_gamma_2, noise_var)
        # compute the first term in the objective (standard kl from VAEs)
        kl1 = self.compute_kl1_divergence(log_var, mu)
        # compute the second term (new kl divergence) with already applied reformulation (see Appendix)
        kl2 = self.compute_kl2_divergence(gamma_full, posterior_means, diagLCovY, noise_var)
        # compute the reconstruction loss with already applied reformulation (see Appendix)
        rec = self.reconstruction_loss(samples_in, posterior_means, noise_var)
        # save memory
        del posterior_means, diagLCovY, gamma_full, mu, log_var
        torch.cuda.empty_cache()
        # compute the objective
        risk = - (rec - kl2 - kl1)
        return kl1, kl2, rec, risk

    def compute_sparse_posterior(self, Y, log_gamma_1, log_gamma_2, zeta_in):
        """
        computes the posterior means (i.e., E[s|z_i,y_i] for all i in the batch and z_i ~ q(z|y_i)) and Cov[y_i|z_i]
        see Equation (24) and (26) in Appendix A.3 in (https://openreview.net/forum?id=FFJFT93oa7)
        """
        # make the data complex valued again
        Y_cplx = Y[:,:self.odim] + 1j * Y[:,self.odim:]
        gamma_full = torch.einsum('ia,ic->iac', log_gamma_1.exp(), log_gamma_2.exp()).reshape(-1, log_gamma_1.shape[1] * log_gamma_2.shape[1])
        bs = gamma_full.shape[0]
        Covsy = gamma_full[:, :, None] * torch.conj(self.AD).T[None, :, :]
        Eys = torch.eye(self.odim,dtype=torch.complex64)[None, :, :].repeat(bs, 1, 1).to(self.device)  # [K,N,N]
        if self.varying_zeta == 'False':
            CovY = (self.AD[None, :, :] * gamma_full[:, None, :]) @ torch.conj(self.AD).T[None, :, :] + self.zeta * Eys
        elif self.varying_zeta == 'True':
            CovY = (self.AD[None, :, :] * gamma_full[:, None, :]) @ torch.conj(self.AD).T[None, :, :] + zeta_in[:,None,None] * Eys
        LCovY = torch.linalg.cholesky(CovY)
        pmeans_partial = torch.cholesky_solve(Y_cplx.unsqueeze(2), LCovY).squeeze(2)
        diagLCovY = torch.real(torch.diagonal(LCovY, dim1=-2, dim2=-1))
        postMeans = torch.bmm(Covsy, pmeans_partial.unsqueeze(2)).squeeze(2)  # computes einsum('zij,zj->zi', Covsy, pmeans_partial)

        # delete all arrays that are not needed anymore
        del Eys, Covsy, pmeans_partial, CovY
        torch.cuda.empty_cache()
        return postMeans, diagLCovY, gamma_full

    def compute_kl1_divergence(self, log_var, mu):
        """
        computes the standard KL diverence from VAEs
        see Equation (27) in Appendix A.4 in (https://openreview.net/forum?id=FFJFT93oa7)
        """
        return torch.mean(torch.sum(-0.5 * (1 + log_var - mu ** 2 - (log_var).exp()), dim=1))  # [1]
    def compute_kl2_divergence(self, gamma_full, posterior_means, diagLCovY, zeta_in):
        """
        computes the second KL divergence in CSVAEs but already reformulated (either for fixed or varying noise variances)
        see second "bracket" term without preceding minus (in the second line) in Equation (30) in Appendix A.4 in (https://openreview.net/forum?id=FFJFT93oa7)
        """
        if self.varying_zeta == 'False':
            kl = - self.odim * torch.log(self.zeta) + 2 * torch.sum(torch.log(diagLCovY),dim=1) + torch.sum(torch.abs(posterior_means) ** 2 / gamma_full, dim=1)
        elif self.varying_zeta == 'True':
            kl = - self.odim * torch.log(zeta_in) + 2 * torch.sum(torch.log(diagLCovY),dim=1) + torch.sum(torch.abs(posterior_means) ** 2 / gamma_full, dim=1)
        return torch.mean(kl)

    def reconstruction_loss(self, Y, posterior_mean, zeta_in):  # M x N | n_b x 2 x N | n_b x M | n_b x M x M
        """
        computes the reconstruction loss (already reformulated)
        see first "bracket" term with preceding minus (in the first line) in Equation (30) in Appendix A.4 in (https://openreview.net/forum?id=FFJFT93oa7)
        """
        # make the data complex valued again
        Y_cplx = Y[:, :self.odim] + 1j * Y[:, self.odim:]
        s_error = torch.linalg.norm(Y_cplx - posterior_mean @ self.AD.T, axis=1) ** 2  # n_b, computes norm(Y_cplx - einsum('ij,hj->hi', self.AD, posterior_mean), axis=1) ** 2  # n_b
        if self.varying_zeta == 'False':
            return torch.mean(- (Y_cplx.shape[1] * torch.log(torch.pi * self.zeta) + (s_error) / self.zeta))
        elif self.varying_zeta == 'True':
            return torch.mean(- (Y_cplx.shape[1] * torch.log(torch.pi * zeta_in) + (s_error) / zeta_in))

    def sample(self,n_samples, pmax):
        """sample generation method"""
        z_samples = torch.randn(n_samples,self.CSVAE_core.latent_dim).to(self.device)
        log_gamma_1, log_gamma_2 = self.CSVAE_core.decode(z_samples)
        gamma_full = torch.einsum('ia,ic->iac', log_gamma_1.exp(), log_gamma_2.exp()).reshape(-1, log_gamma_1.shape[1] *log_gamma_2.shape[1])
        sqrt_gammas = torch.sqrt(gamma_full)
        s_samples = sqrt_gammas/torch.sqrt(torch.tensor(2).to(self.device)) * (torch.randn(n_samples,self.sdim).to(self.device) + 1j * torch.randn(n_samples,self.sdim).to(self.device))
        s_samples = s_samples.detach().to('cpu').numpy()
        # iterate through the components to sample from the covariances
        s_samples_new = np.zeros((n_samples, self.sdim), dtype=np.complex64)
        for i in range(n_samples):
            sorted_indices = np.argsort(np.abs(s_samples[i, :]) ** 2)[::-1]
            top_five_values = s_samples[i, sorted_indices[:pmax]]  # Get the top 5 values
            top_five_indices = sorted_indices[:pmax]
            s_samples_new[i, top_five_indices] = top_five_values
            s_samples_new[i, :] = np.sqrt(np.sum(np.abs(s_samples[i, :]) ** 2) / np.sum(np.abs(s_samples_new[i, :]) ** 2)) * s_samples_new[i, :]
        s_samples = s_samples_new
        return s_samples

###
# the core module of the 2D-CSVAE with deep decoder motivated decoder and fully connected encoder
###

class CSVAE_core_2D(nn.Module):
    """core module 2D-CSVAE"""
    def __init__(self, input_dim, latent_dim, output_dim, n_enc,end_width, dec_chf2d, device):
        super().__init__()
        self.device = device # device for cuda
        self.input_dim = input_dim # input dimension for the decoder (i.e., observation dimension)
        self.latent_dim = latent_dim # latent dimension of the CSVAE
        self.output_dim = output_dim # output dimension (i.e., the dimension of the sparse representation of BOTH DIMS CONCATENATED!!)
        self.end_width = end_width # maximal width of the fully connected encoder neural network
        self.n_enc = n_enc # number of layers in the fully connected encoder
        self.dec_chf2d = dec_chf2d # scaling factor of the number of channels in the decoder architecture

        # encoder construction
        # compute the layer-to-layer steps for the linearly increasing width of the layers in the encoder neural network
        if (self.n_enc - 1) != 0:
            if self.end_width > self.input_dim:
                steps_enc = (self.end_width - self.input_dim) // (self.n_enc - 1)
            else:
                steps_enc = - ((self.input_dim - self.end_width) // (self.n_enc - 1))
            encoder = []
            layer_dim = self.input_dim
            for n in range(n_enc - 1):
                encoder.append(nn.Linear(layer_dim, layer_dim + steps_enc))
                encoder.append(nn.ReLU())
                layer_dim = layer_dim + steps_enc
            encoder.append(nn.Linear(layer_dim, self.end_width))
            encoder.append(nn.ReLU())
        else:
            encoder = []
            layer_dim = self.input_dim
            encoder.append(nn.Linear(layer_dim, self.end_width))
            encoder.append(nn.ReLU())
        self.encoder = nn.Sequential(*encoder)
        # final linear layer that maps to the latent dimension
        self.fc_mu = nn.Linear(self.end_width, self.latent_dim)
        self.fc_var = nn.Linear(self.end_width, self.latent_dim)

        # decoder construction
        if (int(math.sqrt(self.output_dim))/8 % 1) != 0:
            print('decoder architecture does not work, change sparse dimension to be dividable by sqrt(self.output_dim)//8')
        decoder = []
        decoder.append(nn.Linear(self.latent_dim,self.dec_chf2d * 4 * int(math.sqrt(self.output_dim))//8))
        decoder.append(nn.ReLU())
        decoder.append(nn.Linear(self.dec_chf2d * 4 * int(math.sqrt(self.output_dim))//8, self.dec_chf2d * 2 * int(math.sqrt(self.output_dim))//8 * int(math.sqrt(self.output_dim))//8))
        decoder.append(nn.ReLU())
        decoder.append(nn.Unflatten(1,(self.dec_chf2d * 2,int(math.sqrt(self.output_dim))//8,int(math.sqrt(self.output_dim))//8)))
        decoder.append(nn.Conv2d(self.dec_chf2d * 2,self.dec_chf2d * 8,1))
        decoder.append(nn.ReLU())
        decoder.append(nn.Upsample(scale_factor=2, mode='bilinear'))
        decoder.append(nn.Conv2d(self.dec_chf2d * 8, self.dec_chf2d * 16, 1))
        decoder.append(nn.ReLU())
        decoder.append(nn.Upsample(scale_factor=2, mode='bilinear'))
        decoder.append(nn.Conv2d(self.dec_chf2d * 16, self.dec_chf2d * 64, 1))
        decoder.append(nn.ReLU())
        decoder.append(nn.Upsample(scale_factor=2, mode='bilinear'))
        decoder.append(nn.Conv2d(self.dec_chf2d * 64,1,1))
        decoder.append(nn.Flatten())
        self.decoder = nn.Sequential(*decoder)

    def encode(self, x):
        """encodes x """
        x_in = nn.Flatten()(x)
        out = self.encoder(x_in)
        mu, log_var = self.fc_mu(out), self.fc_var(out)
        return mu, log_var

    def reparameterize(self, log_var, mu):
        """VAE reparameterization trick"""
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(log_var)
        return mu + eps * std

    def decode(self, z):
        """decodes z with regularizing log gamma"""
        log_gamma = self.decoder(z)
        log_gamma[log_gamma < -10] = -10
        return log_gamma

    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(log_var, mu)
        log_gamma = self.decode(z)
        return log_gamma, mu, log_var

###
# complex valued 2D CSVAE with fixed measurement matrix, kronecker approximation and fixed or varying noise variances for training
###

class CSVAE_2D(nn.Module):
    def __init__(self, odim, sdim, ldim, A, D, n_enc, dec_chf2d, end_width, device, fix_zeta=0.0, varying_zeta = 'False'):
        super().__init__()
        self.sdim = sdim # dimension of the sparse representation of ALL DIMENSION STACKED
        self.odim = odim # dimension of the observations
        self.A = A.cfloat().to(device) # measurement matrix
        self.D = D.cfloat().to(device) # dictionary
        self.AD = self.A @ self.D # A \cdot D
        self.device = device # device for cuda
        self.zeta = torch.from_numpy(np.array(fix_zeta)).float().to(device) # fixed noise variance if we use fixed one
        self.varying_zeta = varying_zeta # boolean whether we use varying or fixed noise variances for training
        self.CSVAE_core = CSVAE_core_2D(2 * odim, ldim, sdim, n_enc, end_width, dec_chf2d, device).to(device) # core module

    def compute_objective(self, samples_in, noise_var):
        # processing chain through the CSVAE core module
        log_gamma, mu, log_var = self.CSVAE_core(samples_in)
        # compute the posterior mean and the covarinces of y
        posterior_means, diagLCovY, gamma_full = self.compute_sparse_posterior(samples_in, log_gamma, noise_var)
        # compute the first term in the objective (standard kl from VAEs)
        kl1 = self.compute_kl1_divergence(log_var, mu)
        # compute the second term (new kl divergence) with already applied reformulation (see Appendix)
        kl2 = self.compute_kl2_divergence(log_gamma, posterior_means, diagLCovY, noise_var)
        # compute the reconstruction loss with already applied reformulation (see Appendix)
        rec = self.reconstruction_loss(samples_in, posterior_means, noise_var)
        # save memory
        del posterior_means, diagLCovY, log_gamma, mu, log_var
        torch.cuda.empty_cache()
        # compute the objective
        risk = - (rec - kl2 - kl1)
        return kl1, kl2, rec, risk

    def compute_sparse_posterior(self, Y, log_gamma, zeta_in):
        """
        computes the posterior means (i.e., E[s|z_i,y_i] for all i in the batch and z_i ~ q(z|y_i)) and Cov[y_i|z_i]
        see Equation (24) and (26) in Appendix A.3 in (https://openreview.net/forum?id=FFJFT93oa7)
        """
        # make the data complex valued again
        Y_cplx = Y[:,:self.odim] + 1j * Y[:,self.odim:]
        gamma_full = log_gamma.exp()
        bs = gamma_full.shape[0]
        Covsy = gamma_full[:, :, None] * torch.conj(self.AD).T[None, :, :]
        Eys = torch.eye(self.odim,dtype=torch.complex64)[None, :, :].repeat(bs, 1, 1).to(self.device)  # [K,N,N]
        if self.varying_zeta == 'False':
            CovY = (self.AD[None, :, :] * gamma_full[:, None, :]) @ torch.conj(self.AD).T[None, :, :] + self.zeta * Eys
        elif self.varying_zeta == 'True':
            CovY = (self.AD[None, :, :] * gamma_full[:, None, :]) @ torch.conj(self.AD).T[None, :, :] + zeta_in[:,None,None] * Eys

        LCovY = torch.linalg.cholesky(CovY)
        pmeans_partial = torch.cholesky_solve(Y_cplx.unsqueeze(2), LCovY).squeeze(2)
        diagLCovY = torch.real(torch.diagonal(LCovY, dim1=-2, dim2=-1))
        postMeans = torch.bmm(Covsy, pmeans_partial.unsqueeze(2)).squeeze(2) # computes einsum('zij,zj->zi', Covsy, pmeans_partial)

        # delete all arrays that are not needed anymore
        del Eys, Covsy, pmeans_partial, CovY, LCovY
        torch.cuda.empty_cache()
        return postMeans, diagLCovY, gamma_full

    def compute_kl1_divergence(self, log_var, mu):
        """
        computes the standard KL diverence from VAEs
        see Equation (27) in Appendix A.4 in (https://openreview.net/forum?id=FFJFT93oa7)
        """
        return torch.mean(torch.sum(-0.5 * (1 + log_var - mu ** 2 - (log_var).exp()), dim=1))  # [1]

    def compute_kl2_divergence(self, log_gamma, posterior_means, diagLCovY, zeta_in):
        """
        computes the second KL divergence in CSVAEs but already reformulated (either for fixed or varying noise variances)
        see second "bracket" term without preceding minus (in the second line) in Equation (30) in Appendix A.4 in (https://openreview.net/forum?id=FFJFT93oa7)
        """
        if self.varying_zeta == 'False':
            kl = - self.odim * torch.log(self.zeta) + 2 * torch.sum(torch.log(diagLCovY),dim=1) + torch.sum(torch.abs(posterior_means) ** 2 / log_gamma.exp(), dim=1)
        elif self.varying_zeta == 'True':
            kl = - self.odim * torch.log(zeta_in) + 2 * torch.sum(torch.log(diagLCovY),dim=1) + torch.sum(torch.abs(posterior_means) ** 2 / log_gamma.exp(), dim=1)
        return torch.mean(kl)

    def reconstruction_loss(self, Y, posterior_mean, zeta_in):  # M x N | n_b x 2 x N | n_b x M | n_b x M x M
        """
        computes the reconstruction loss (already reformulated)
        see first "bracket" term with preceding minus (in the first line) in Equation (30) in Appendix A.4 in [2] (https://openreview.net/forum?id=FFJFT93oa7)
        """
        # make the data complex valued again
        Y_cplx = Y[:, :self.odim] + 1j * Y[:, self.odim:]
        s_error = torch.linalg.norm(Y_cplx - posterior_mean @ self.AD.T, axis=1) ** 2  # n_b, computes norm(Y_cplx - einsum('ij,hj->hi', self.AD, posterior_mean), axis=1) ** 2  # n_b
        if self.varying_zeta == 'False':
            return torch.mean(- (Y_cplx.shape[1] * torch.log(torch.pi * self.zeta) + (s_error) / self.zeta))
        elif self.varying_zeta == 'True':
            return torch.mean(- (Y_cplx.shape[1] * torch.log(torch.pi * zeta_in) + (s_error) / zeta_in))

    def sample(self,n_samples, pmax):
        """sample generation method"""
        s_samples = np.zeros((n_samples,self.sdim),dtype=np.complex64)
        n_50 = n_samples//50 # this leads to generating 50 samples at a time and is just for not exploding your GPU memory resources
        for n in range(n_50):
            z_samples = torch.randn(50,self.CSVAE_core.latent_dim).to(self.device)
            log_gamma = self.CSVAE_core.decode(z_samples).detach()
            gamma_full = log_gamma.exp()
            sqrt_gammas = torch.sqrt(gamma_full)
            s_samples50 = sqrt_gammas/torch.sqrt(torch.tensor(2).to(self.device)) * (torch.randn(50,self.sdim).to(self.device) + 1j * torch.randn(50,self.sdim).to(self.device))
            s_samples50 = s_samples50.detach().to('cpu').numpy()
            s_samples_new = np.zeros((50, self.sdim), dtype=np.complex64)
            for i in range(50):
                sorted_indices = np.argsort(np.abs(s_samples50[i, :]) ** 2)[::-1]
                top_five_values = s_samples50[i, sorted_indices[:pmax]]  # Get the top pmax values
                top_five_indices = sorted_indices[:pmax]
                s_samples_new[i, top_five_indices] = top_five_values
                s_samples_new[i, :] = np.sqrt(np.sum(np.abs(s_samples50[i, :]) ** 2) / np.sum(np.abs(s_samples_new[i, :]) ** 2)) * s_samples_new[i, :]
            s_samples[n * 50:(n + 1) * 50, :] = s_samples_new
            del log_gamma, gamma_full, sqrt_gammas
            torch.cuda.empty_cache()
        if n_samples % 50 != 0:
            z_samples = torch.randn(n_samples - n_50 * 50, self.CSVAE_core.latent_dim).to(self.device)
            log_gamma = self.CSVAE_core.decode(z_samples)
            gamma_full = log_gamma.exp()
            sqrt_gammas = torch.sqrt(gamma_full)
            s_samples50 = sqrt_gammas / torch.sqrt(torch.tensor(2).to(self.device)) * (torch.randn(n_samples - n_50 * 50, self.sdim).to(self.device) + 1j * torch.randn(n_samples - n_50 * 50, self.sdim).to(self.device))
            s_samples50 = s_samples50.detach().to('cpu').numpy()
            s_samples_new = np.zeros((n_samples - n_50 * 50, self.sdim), dtype=np.complex64)
            for i in range(n_samples - n_50 * 50):
                sorted_indices = np.argsort(np.abs(s_samples50[i, :]) ** 2)[::-1]
                top_five_values = s_samples50[i, sorted_indices[:pmax]]  # Get the top pmax values
                top_five_indices = sorted_indices[:pmax]
                s_samples_new[i, top_five_indices] = top_five_values
                s_samples_new[i, :] = np.sqrt(np.sum(np.abs(s_samples50[i, :]) ** 2) / np.sum(np.abs(s_samples_new[i, :]) ** 2)) * s_samples_new[i, :]
            s_samples[n_50 * 50:, :] = s_samples_new
        return s_samples

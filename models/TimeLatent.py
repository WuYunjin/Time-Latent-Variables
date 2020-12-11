import logging
import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Normal
from torch.distributions.relaxed_bernoulli import RelaxedBernoulli

import os,sys
sys.path.append(os.getcwd())
from helpers.torch_utils import set_seed


class TimeLatent(object):
    _logger = logging.getLogger(__name__)

    def __init__(self, num_X, max_lag, num_samples, device, prior_rho_A, prior_sigma_W, temperature, sigma_Z, sigma_X):

        self.num_X = num_X
        self.max_lag = max_lag
        self.num_samples = num_samples
        self.device = device

        self.prior_rho_A = prior_rho_A
        self.temperature = temperature
        self.prior_sigma_W = prior_sigma_W
        self.prior_sigma_Z = sigma_Z*torch.ones(size=[num_X],device=device)
        
        self.likelihood_sigma_X = sigma_X*torch.ones(size=[num_X],device=device)

        self.posterior_sigma_Z = sigma_Z*torch.ones(size=[num_X],device=device)

        self.init_prior()
        self.init_posterior()
        self._logger.debug('Finished building model')

    def init_prior(self):
        # Set up priors
        K = self.max_lag
        m = self.num_X
        rho = self.prior_rho_A
        sigma = self.prior_sigma_W

        temperature = torch.tensor([self.temperature],device=self.device)
        prior_A = rho*torch.ones( size=(K,2*m,2*m),device=self.device )
        prior_A[:,m:,:] = torch.tensor([0],device=self.device)
        
        # Set the diagonal
        for i in range(m):
            prior_A[:,m+i,m+i] = torch.tensor([1],device=self.device)

        self.prior_A = RelaxedBernoulli( temperature= temperature, probs=prior_A )


        prior_W_scale = sigma*torch.ones( size=(K,2*m,2*m),device=self.device )
        prior_W_scale[:,m:,:] = torch.tensor([0],device=self.device)
        # Set the diagonal
        for i in range(m):
            prior_W_scale[:,m+i,m+i] = torch.tensor([sigma],device=self.device)

        self.prior_W = Normal(loc=torch.zeros_like(prior_W_scale,device=self.device),scale=prior_W_scale)

        # the proir over Z depends on the sample of A and W, so we don't set up Z here.

    def init_posterior(self):
        # Set up posteriors
        K = self.max_lag
        m = self.num_X

        temperature = torch.tensor([self.temperature], device=self.device)
        estimate_A = torch.rand(size=(K,2*m,2*m), device=self.device)
        estimate_A[:,m:,:] = torch.tensor([0],device=self.device)
        # Set the diagonal
        for i in range(m):
            estimate_A[:,m+i,m+i] = torch.rand(size=[1],device=self.device)

        estimate_A = estimate_A.requires_grad_(True)
        
        self.posterior_A = RelaxedBernoulli(temperature=temperature, probs=estimate_A)

        estimate_W_scale = torch.rand(size=(K,2*m,2*m),device=self.device)
        estimate_W_scale[:,m:,:] = torch.tensor([0],device=self.device)
        # Set the diagonal
        for i in range(m):
            estimate_W_scale[:,m+i,m+i] = torch.rand(size=[1],device=self.device)
        
        estimate_W_scale  = estimate_W_scale.requires_grad_(True)

        estimate_W_loc = torch.rand(size=(K,2*m,2*m),device=self.device)
        estimate_W_loc[:,m:,:] = torch.tensor([0],device=self.device)
        # Set the diagonal
        for i in range(m):
            estimate_W_loc[:,m+i,m+i] = torch.rand(size=[1],device=self.device)
        
        estimate_W_loc  = estimate_W_loc.requires_grad_(True)
        self.posterior_W = Normal(loc= estimate_W_loc, scale=estimate_W_scale ) 

    def ln_p_AWZ(self,A,W,Z):
        
        K = self.max_lag
        m = self.num_X
        ln_p_A =  self.prior_A.log_prob(A)[:,:m,:].sum() + sum([torch.diagonal((self.prior_A.log_prob(A)[i,m:,m:]),0) for i in range(K)]).sum() 
        ln_p_W =  self.prior_W.log_prob(W)[:,:m,:].sum() + sum([torch.diagonal((self.prior_W.log_prob(W)[i,m:,m:]),0) for i in range(K)]).sum() 

        # store the distributions from pZ(1),pZ(2)....pZ(T)
        # p_Z = []
        
        sigma_Z = self.prior_sigma_Z
        p_Z1 = Normal(loc=torch.zeros_like(sigma_Z,device=self.device),scale=sigma_Z)
        # p_Z.append(p_Z1)
        ln_p_Z1 = p_Z1.log_prob(Z[0])

        ln_p_ZK = torch.zeros(size=[m],device=self.device)
        
        for t in range(2,K+1):
            A_22 = A[:t-1,m:,m:]
            W_22 = W[:t-1,m:,m:]

            mean_t = []
            for i in range(1,t):
                A_22_i = torch.diagonal(A_22[i-1])
                W_22_i = torch.diagonal(W_22[i-1])
                mean_t.append(Z[t-1-i]*A_22_i *W_22_i)


            p_Zt = Normal(loc=sum(mean_t),scale=sigma_Z)
            # p_Z.append(p_Zt)
            ln_p_ZK += p_Zt.log_prob(Z[t-1])
        
        
        ln_p_ZT = torch.zeros(size=[m],device=self.device)
        T = self.num_samples
        for t in range(K+1,T+1):
            A_22 = A[:,m:,m:]
            W_22 = W[:,m:,m:]

            mean_t = []
            for i in range(1,K+1):
                A_22_i = torch.diagonal(A_22[i-1])
                W_22_i = torch.diagonal(W_22[i-1])
                mean_t.append(Z[t-1-i]*A_22_i *W_22_i)

            p_Zt = Normal(loc=sum(mean_t),scale=sigma_Z)
            # p_Z.append(p_Zt)
            ln_p_ZT += p_Zt.log_prob(Z[t-1])

        # self.p_Z = p_Z
        return  (ln_p_ZT.sum() + ln_p_ZK.sum() + ln_p_Z1.sum()) + ln_p_A + ln_p_W
        

    
    def ln_p_X_AWZ(self,X,A,W,Z):

        sigma = self.likelihood_sigma_X
        T = self.num_samples
        
        K = self.max_lag
        m = self.num_X

        Sum_X_mu = torch.tensor([0.0],device=self.device)
        
        # t=1
        Sum_X_mu += (X[0]**2).sum()

        # 2<=t<=K
        for t in range(2,K+1):
            A_11 = A[:,:m,:m]
            W_11 = W[:,:m,:m]

            A_12 = A[:,:m,m:]
            W_12 = A[:,:m,m:]

            mu = torch.zeros(size=[m],device=self.device)
            for i in range(1,t):
                A_11_i = A_11[i-1]
                W_11_i = W_11[i-1]
                A_12_i = A_12[i-1]
                W_12_i = W_12[i-1]

                mu += torch.matmul(X[t-1-i], (A_11_i *W_11_i).t() )+torch.matmul(Z[t-1-i], (A_12_i *W_12_i).t() )

            Sum_X_mu += ((X[t-1]-mu)**2).sum()

        # K+1 <= t <= T
        for t in range(K+1,T+1):
            A_11 = A[:,:m,:m]
            W_11 = W[:,:m,:m]

            A_12 = A[:,:m,m:]
            W_12 = A[:,:m,m:]

            mu = torch.zeros(size=[m],device=self.device)
            for i in range(1,K+1):
                A_11_i = A_11[i-1]
                W_11_i = W_11[i-1]
                A_12_i = A_12[i-1]
                W_12_i = W_12[i-1]

                mu += torch.matmul(X[t-1-i], (A_11_i *W_11_i).t() )+torch.matmul(Z[t-1-i], (A_12_i *W_12_i).t() )

            Sum_X_mu += ((X[t-1]-mu)**2).sum()


        return  -T/2* torch.log(2* torch.tensor([np.math.pi],device=self.device)) - T/2* torch.log( sigma*sigma ).sum() - 1/( 2*(sigma*sigma).sum()) *Sum_X_mu


    
    def sample_Z(self,A,W):
        # we sample Z from q(Z) 
        
        # store the distributions from qZ(1),qZ(2)....qZ(T)
        # q_Z = []
        # sample Z(1)
        m = self.num_X
        sigma_Z = self.posterior_sigma_Z
        q_Z1 = Normal(loc=torch.zeros_like(sigma_Z,device=self.device),scale=sigma_Z)
        # q_Z.append(q_Z1)
        Z1 = q_Z1.rsample()
        
        ln_q_Z = torch.tensor([q_Z1.log_prob(Z1).sum()],device=self.device)

        # store the sample Z(1:T)
        T = self.num_samples
        Z = []
        Z.append(Z1)
        
        # sample Z(2:K)
        K = self.max_lag
        for t in range(2,K+1):
            A_22 = A[:t-1,m:,m:]
            W_22 = W[:t-1,m:,m:]

            mean_t = []
            for i in range(1,t):
                A_22_i = torch.diagonal(A_22[i-1])
                W_22_i = torch.diagonal(W_22[i-1])
                mean_t.append(Z[t-1-i]*A_22_i *W_22_i)

            # # Normalize mean_t, otherwise it will too large and leads to large Z(t) even NAN.
            # mean_t = F.normalize(sum(mean_t),dim=0)
            
            # q_Zt = Normal(loc=mean_t,scale=sigma_Z)
            # q_Z.append(q_Zt)
            # Z.append(q_Zt.rsample())
            # ln_q_Z.append(q_Zt.log_prob(Z[t-1]))

            
            q_Zt = Normal(loc=sum(mean_t),scale=sigma_Z)
            # q_Z.append(q_Zt)
            Z_t = q_Zt.rsample()
            ln_q_Z += q_Zt.log_prob(Z_t).sum()

            # Normalize Z_t, otherwise it will too large and leads to large Z(t) even NAN.
            Z_t = F.normalize(Z_t,dim=0)
            Z.append(Z_t)
        
             
        # sample Z(K+1:T)
        for t in range(K+1,T+1):
            A_22 = A[:,m:,m:]
            W_22 = W[:,m:,m:]
            mean_t = []

            for i in range(1,K+1):
                A_22_i = torch.diagonal(A_22[i-1])
                W_22_i = torch.diagonal(W_22[i-1])
                mean_t.append(Z[t-1-i]*A_22_i *W_22_i)

            # # Normalize mean_t, otherwise it will too large and leads to large Z(t) even NAN.
            # mean_t = F.normalize(sum(mean_t),dim=0)
            # q_Zt = Normal(loc=mean_t,scale=sigma_Z)
            # q_Z.append(q_Zt)
            # Z.append(q_Zt.rsample())
            # ln_q_Z.append(q_Zt.log_prob(Z[t-1]))

            q_Zt = Normal(loc=sum(mean_t),scale=sigma_Z)
            # q_Z.append(q_Zt)
            Z_t = q_Zt.rsample()
            ln_q_Z += q_Zt.log_prob(Z_t).sum()

            # Normalize Z_t, otherwise it will too large and leads to large Z(t) even NAN.
            Z_t = F.normalize(Z_t,dim=0)
            Z.append(Z_t)

        # self.q_Z = q_Z
        return Z, ln_q_Z

    def ln_q_Z(self,Z):
        
        loss = torch.tensor([0.0],device=self.device)
        T = self.num_samples 
        for i in range(T):
            Zt = Z[i]
            log_prob = self.q_Z[i].log_prob(Zt)
            loss += log_prob.sum()

        return loss

    def loss(self,X):
        """
        return: the negative ELBO
        """
        # sample
        A = self.posterior_A.rsample()
        W = self.posterior_W.rsample()
        # ln_q_Z = self.ln_q_Z(Z)
        Z,ln_q_Z = self.sample_Z(A,W) # We calculate ln_q_Z when sample Z so that we don't need to save the q_Z, which reduces the running memory.

        # Because we assume the X won't cause Z and Zi and mutually independent, A_22 is always the diagonal matrix and A_21 is always a zero matrix.
        m = self.num_X
        K = self.max_lag
        ln_q_A = self.posterior_A.log_prob(A)[:,:m,:].sum() + sum([torch.diagonal((self.posterior_A.log_prob(A)[i,m:,m:]),0) for i in range(K)]).sum()
        ln_q_W = self.posterior_W.log_prob(W)[:,:m,:].sum() + sum([torch.diagonal((self.posterior_W.log_prob(W)[i,m:,m:]),0) for i in range(K)]).sum()

        

        ln_q_AWZ =  ln_q_Z + ln_q_A + ln_q_W
                    

        # Calculating L_kl
        L_kl = - (ln_q_AWZ - self.ln_p_AWZ(A,W,Z)) 

        # Calculating L_ell
        L_ell = self.ln_p_X_AWZ(X,A,W,Z) 

        ELBO =  L_kl + L_ell
        # ELBO =  L_ell 
        # self._logger.info("ln_q_Z:{}, ln_q_A: {}, ln_q_W: {}, ln_p_AWZ(A,W,Z):{}, L_ell:{} ".format(ln_q_Z.item(), ln_q_A.item(), ln_q_W.item() ,self.ln_p_AWZ(A,W,Z).item(), L_ell.item()))
        loss = -ELBO
        return loss
            

    @property
    def logger(self):
        try:
            return self._logger
        except:
            raise NotImplementedError('self._logger does not exist!')
    


if __name__ == '__main__':
    set_seed(2020)
    num_X = 3
    num_samples = 1000
    X = torch.randn(size=(num_samples,num_X))

    model = TimeLatent(num_X, 3, num_samples, 'cpu', 0.5, 1.0, 2.0, 1.0, 1.0)
    
    loss = model.loss(X)
    print(loss)
    loss.backward(retain_graph=True)
    # print(model.posterior_A.probs.grad)

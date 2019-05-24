import torch, numpy as np
from .settings import CTX, RG
from .SSM import Observation, Latent
from scipy.stats import norm
from abc import ABC

class GaussianObservation(Observation, ABC):
    
    def __init__(self, noise_std):

        super(GaussianObservation, self).__init__()

        self.D = len(noise_std)
        self.ps["noise_std"] = torch.tensor(np.log(noise_std), requires_grad=RG, **CTX)
        self._dist = torch.distributions.normal.Normal(torch.zeros(self.D, **CTX), 
                                                      torch.ones(self.D, **CTX))
        
    def sample(self, y):
        mu = self.conditional_param(y)
        sig = self.noise_std
        return (torch.randn(*mu.shape, **CTX) * torch.exp(sig) + mu).detach()
    
    def logp(self, y, x):

        # during filtering: 
        # for a batch of y in [batchsize, nsample, ndim]
        #                x in [batchsize, ndim]

        assert x.shape[-1] == self.D, "observation dimension on the last axis should be %d" % self.D
        if len(y.shape) == 3 and len(x.shape) == 2:
            x = torch.unsqueeze(x,1)
        
        mu = self.conditional_param(y)
        std = self.noise_std
        return (self._dist.log_prob((x-mu)/std.exp()) - std).sum(-1)

    def sample_n(self, y, n):
        mu = self.conditional_param(y)
        noise_std = self.noise_std
        return torch.randn( *((n,) + mu.shape), **CTX) * torch.exp(noise_std) + mu[None,...]

    def conditional_param(self, y):
        raise NotImplementedError

    def dlogp(self, y, x):
        
        n = y.shape[0]

        noise_std = self.noise_std
        sigma_2 = torch.exp(-2*noise_std)
        
        m = self.conditional_param(y)
        d = (x - m)
        dl = (d * sigma_2)
        dmu = self.dconditional_param(dl, y)
        dstd= -1. + d ** 2 * sigma_2

        return torch.cat([dstd, dmu], -1)

    def dnatsuff(self, y, x):
        
        n = y.shape[0]
        noise_std = self.noise_std
        sigma_2 = torch.exp(-2*noise_std)

        m = self.conditional_param(y)
        dl = (sigma_2 * x)
        dmu = self.dconditional_param(dl, y)
        dstd = (x**2 - 2 * x * m) * sigma_2

        return torch.cat([dstd, dmu], -1)

    def dnorm(self, y):
            
        noise_std = self.noise_std
        sigma_2   = torch.exp(-2*noise_std)

        m = self.conditional_param(y)

        dstd = (-m**2 * sigma_2 + 1)
        dl = (m * sigma_2)
        dldp = self.dconditional_param(dl, y)
        dmean = dldp
        return torch.cat([dstd, dmean], -1)


    def suff(self, x):
        return torch.cat([x,x**2],-1)

    def dnat(self, y):
        n = y.shape[0]

        noise_std = self.noise_std
        sigma_2 = torch.exp(-2*noise_std)
        n_mean_param = self.ps.nparam-self.D
        
        m    = self.conditional_param(y)
        dl   = sigma_2.expand([n,-1])
        dldp = self.dconditional_param(dl, y)

        dstd1 = - 2 *  m * sigma_2
        dstd2 = sigma_2.expand([n,-1])
        dstd  = torch.cat([dstd1, dstd2], -1)
        
        dmu = (dldp).view(n,(n_mean_param))

        g = torch.cat([dstd, dmu], -1)
        return g

    def dnatsuff_from_dnatsuff(self, nat, suff):
        
        n = nat.shape[0]
        assert n == suff.shape[0]

        s1, s2     = torch.split(suff, [self.D, self.D], -1)
        n1, n2, n3 = torch.split(nat,  [self.D, self.D, nat.shape[-1]-self.D*2], -1)
        
        dstd = s1 * n1 + s2 * n2

        s1 = torch.unsqueeze(s1, 2)
        n3 = n3.view(s1.shape[:2]+(-1,))
        dmu  = (s1*n3).view((n, self.ps.nparam-self.D))

        return torch.cat([dstd, dmu], -1)

        
class GaussianLatent(Latent):
    
    def __init__(self, noise_std):

        super(GaussianLatent, self).__init__()
        self.D = len(noise_std)
        self.ps["noise_std"] = torch.tensor(np.log(noise_std), requires_grad=RG, **CTX)
        self._dist = torch.distributions.normal.Normal(torch.zeros(self.D, **CTX), 
                                                       torch.ones(self.D, **CTX) 
                                                       )
        
    def apply_constraints(self):
        self.noise_std.data.clamp_(min=-5,max=np.log(1))

    def step(self, zt, t):
        
        mu = self.conditional_param(zt, t)
        sig = self.noise_std
        return (torch.randn(*mu.shape, **CTX) * torch.exp(sig) + mu).detach()
    
    def step_n(self, zt, t, n):
        mu = self.conditional_param(zt, t)
        sig = self.noise_std
        return (torch.randn( *((n,) + mu.shape), **CTX) * torch.exp(sig) + mu[None,...]).detach()

    def conditional_param(self, zt, t):
        raise NotImplementedError

    def natsuff(self, ztm1, zt, t):

        assert zt.shape[-1] == ztm1.shape[-1] == self.D

        std = self.noise_std
        mu = self.conditional_param(ztm1, t)
        ns  = -0.5 * torch.sum((zt**2 - 2*zt*mu) /(torch.exp(2*std)), -1)
        return ns

    def norm(self, ztm1, t):
        std = self.noise_std
        mu  = self.conditional_param(ztm1, t)
        ns  = 0.5 * torch.sum((mu**2) / torch.exp(2*std) + torch.log(2*np.pi*torch.exp(2*std)), -1)
        return ns

    def logp(self, ztm1, zt, t):

        assert zt.shape[-1] == ztm1.shape[-1] == self.D

        mu = self.conditional_param(ztm1, t)
        std = self.noise_std
        return (self._dist.log_prob((zt-mu)/std.exp()) - std).sum(-1)

    def dlogp(self, ztm1, zt, t):
        
        n = ztm1.shape[0]

        noise_std = self.noise_std
        sigma_2 = torch.exp(-2*noise_std)

        m = self.conditional_param(ztm1, t)
        d = (zt - m)
        dl = (d * sigma_2)
        dmean = self.dconditional_param(dl, ztm1,t)
        dstd = -1. + d ** 2 * sigma_2
        return torch.cat([dstd, dmean], -1)

    def dnatsuff(self, ztm1, zt, t):
        
        noise_std = self.noise_std
        sigma_2 = torch.exp(-2*noise_std)
        n = ztm1.shape[0]

        m = self.conditional_param(ztm1, t)
        dstd = (zt**2 - 2 * zt * m) * sigma_2
        dl = (sigma_2 * zt)
        dmean = self.dconditional_param(dl, ztm1,t)

        return torch.cat([dstd, dmean], -1)

    def dnorm(self, ztm1, t):
            
        noise_std = self.noise_std
        sigma_2 = torch.exp(-2*noise_std)

        n = ztm1.shape[0]

        m = self.conditional_param(ztm1, t)
        dstd = (-m**2 * sigma_2 + 1)
        dl = (m * sigma_2)
        dmean = self.dconditional_param(dl, ztm1,t)
        return torch.cat([dstd, dmean], -1)

    def suff(self, x):
        return torch.cat([x, x**2], -1)

    def dnat(self, ztm1, t):
        n = ztm1.shape[0]

        noise_std = self.noise_std
        sigma_2 = torch.exp(-2*noise_std)
        n_mean_param = self.ps.nparam-self.D
        
        # d(1/2sig)(dlogsig), d(mu/sig)/dlogsig, d(mu/sig)/mu_param
        m = self.conditional_param(ztm1, t)

        g1 = - 2 *  m * sigma_2
        g2 = sigma_2.expand([n,-1])

        dl   = (sigma_2)[None,:]
        dldp = self.dconditional_param(dl, ztm1, t)
        g3  = (dldp).reshape(n,(n_mean_param))

        g = torch.cat([g1,g2,g3], -1)
        return g

    def dnatsuff_from_dnatsuff(self, nat, suff):

        n = nat.shape[0]
        assert n == suff.shape[0]

        s1, s2     = torch.split(suff, [self.D, self.D], -1)
        n1, n2, n3 = torch.split(nat,  [self.D, self.D, nat.shape[-1]-self.D*2], -1)
        
        dstd = s1 * n1 + s2 * n2
        
        s1 = torch.unsqueeze(s1, 2)
        n3 = n3.view(s1.shape[:2]+(-1,))
        dmu  = (s1*n3).view((n, self.ps.nparam-self.D))

        return torch.cat([dstd, dmu], -1)

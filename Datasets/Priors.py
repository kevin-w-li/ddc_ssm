import torch
import numpy as np
from autograd.scipy.stats import norm
from .settings import CTX, RG
from .SSM import *

class GaussianPrior(Prior):

    def __init__(self, mu, std):

        super(GaussianPrior, self).__init__()
        
        assert len(mu) == len(std)
        
        self.D = len(std)
        self.ps["noise_std"] = torch.tensor(np.log(std), requires_grad=RG, **CTX)
        self.ps["mean"]      = torch.tensor(mu, requires_grad=RG, **CTX)
        self.dist = torch.distributions.normal.Normal(torch.zeros(self.D, **CTX), 
                                                      torch.ones(self.D, **CTX))

    def sample(self, n):
        return self.dist.sample((n,)) * self.mean + self.noise_std

    def logp(self, x):
        x = x - self.mean
        x = x / torch.exp(self.noise_std)
        return torch.sum(self.dist.log_prob(x) - self.noise_std, -1)

    def dlogp(self, x):
        
        D = self.D
        d = x - self.mean
        dstd = -1. + d ** 2 * torch.exp(-2*self.noise_std)
        dmu  = d * torch.exp(-2*self.noise_std)
        return torch.cat([dstd, dmu], -1)

    def suff(self, x):
        return torch.cat([x,x**2], -1)

    def dnat(self, n=1):

        noise_std = self.noise_std
        m = self.mean
        sigma_2   = torch.exp(-2*noise_std)

        g =  torch.zeros(n, 2 * self.D, **CTX)
        g[:, :self.D] = - 2 * m * sigma_2
        g[:, self.D:2*self.D] = sigma_2
        return g

    def dnatsuff(self, x):

        n = x.shape[0]
        
        noise_std = self.noise_std
        m = self.mean
        sigma_2 = torch.exp(-2*noise_std)

        dstd= (x**2 - 2 * x * m) * sigma_2
        dmu = sigma_2 * x

        return torch.cat([dstd, dmu], -1)

    def dnatsuff_from_dnatsuff(self, nat, suff):
        
        n = suff.shape[0]
        g = torch.zeros((n, self.D + self.D), requires_grad=RG, **CTX)
        
        n1, n2 = torch.split(nat, [self.D, self.D], -1)
        s1, s2 = torch.split(suff, [self.D, self.D], -1)

        g[:, :self.D] = n1*s1 + n2*s2

        g[:,self.D:] = (s1 * n2)
        return g


    def dnorm(self, n=1):
            
        noise_std = self.noise_std
        m = self.mean

        sigma_2 = torch.exp(-2*noise_std)

        dstd = (-m**2 * sigma_2 + 1)
        dmu  = m * sigma_2
        return torch.cat([dstd, dmu])

class SymmetricGaussianMixturePrior(Prior):
    
    def __init__(self, D, m, std):

        super(SymmetricGaussianMixturePrior, self).__init__()
        self.D = D
        
        self._ps = np.append(self.ps, [m])
        self.pp.add("mean", (1,))
        self._ps = np.append(self.ps, [std])
        self.pp.add("std", (1,))

        self.ps = np.array(self.ps)
        self.set_attr_from_ps()

    def sample(self, n):
        
        std  = self.pp.get(self.ps, "std")
        mean = self.pp.get(self.ps, "mean")

        m = mean * (np.random.randint(2, size=(n,1)) * 2 - 1) * np.ones((1, self.D))
        return m + np.random.randn(n,self.D)*np.exp(std)
    
    def _logp(self, x, ps):

        std = self.pp.get(ps, "std")
        m  = self.pp.get(ps, "mean")

        n  = x.shape[0]

        logp_1 = np.sum(norm.logpdf(x, -m, std), axis=-1) + np.log(0.5)
        logp_2 = np.sum(norm.logpdf(x, +m, std), axis=-1) + np.log(0.5)
        
        logp = logsumexp([logp_1, logp_2], axis=0)

        return logp

    def dlogp(self, x):
        # no gradients
        return np.zeros((x.shape[0], len(self.ps) ))

class BinaryPrior(Prior):

    def __init__(self, D, std=0.3):

        super(BinaryPrior, self).__init__()
        self.D = D
        self.ps["prop"] = torch.tensor(np.random.randn(D)*std, requires_grad = RG, **CTX)

    def sample(self, n):
        
        p = self.prop
        p = 1.0/(1.0+torch.exp(-p))
        x = torch.rand(n, self.D, **CTX) + p
        x = torch.floor(x)
        return x

    def logp(self, x):

        p = self.prop
        p = 1.0/(1.0+torch.exp(-p))
        logp = torch.sum(x*p.log() + (1-x) * (1-p).log(), -1)

        return logp

    def dlogp(self, x):
        
        p = self.prop
        p = 1.0/(1.0+torch.exp(-p))

        return x - p

    def suff(self, x):
        return x

    def dnat(self, n=1):
        return torch.ones([]).expand([n, self.D])

    def dnorm(self, n=1):

        p = self.prop
        p = 1.0/(1.0+torch.exp(-p))
        p = p.expand( [n,-1] )

        return p

    def dnatsuff_from_dnatsuff(self, dnat, suff):
        return suff

    def dnatsuff(self, x):
        return self.suff(x)


import torch, numpy as np
from .settings import CTX, RG

def xavier_uniform(std, *shape):
    a = np.sqrt(6. / sum(shape))
    w = - a + np.random.rand(*shape) * 2 * a
    w = torch.tensor(w, requires_grad=RG, **CTX)
    return w
    
def xavier_normal(std, D_o, D_i):
    a = 1/np.sqrt(D_i)
    w = np.random.randn(D_o, D_i) * a * std
    w = torch.tensor(w, requires_grad=RG, **CTX)
    return w


def randgn(mu, beta, rho, size=[]):
    '''
    Random genralized Gaussian sample
    '''
    assert mu.shape[-1] == beta.shape[-1] == rho.shape[-1]
    rho = rho.expand_as(mu)
    g = torch.distributions.Gamma(1/rho,1).sample(size) ** (1./rho)
    s = g * (torch.floor(torch.rand(*g.shape, **CTX) + 0.5) * 2 - 1)
    x = mu + s * beta
    return x

def gnlogpdf(x, mu, beta, rho):
    return - torch.log( beta * 2.0 ) - torch.lgamma(1.+1./rho) - ((x-mu)/beta).abs()**rho

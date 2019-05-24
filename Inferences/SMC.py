import torch
from scipy.stats import norm
from tqdm import trange
from settings import CTX
from copy import deepcopy

def particle_EP(model, xt, nsample=1000, bar=False, pairwise=False):

    T, nchain, D = xt.shape
    
    samples    = torch.empty(T,nchain, nsample,model.latent.D, **CTX)
    samples[0,...] = torch.randn(nchain, nsample,model.latent.D, **CTX)  * model.init_noise_std
    samples_m1 = torch.empty(T,nchain, nsample,model.latent.D, **CTX)
    weights = torch.ones(T,nchain, nsample, **CTX)/nsample
    
    s = samples[0,...]
    w = weights[0,...]
    
    iterable = trange(T, ncols=100) if bar else range(T)
    for t in iterable:

        samples_m1[t] = s
        s = model.latent.step(s,t)
        l  = model.observation.logp(s, xt[t])
        
        logp = l-l.max()
        w = torch.exp(logp)
        assert torch.all(torch.isfinite(w))
        w /=torch.sum(w,1,keepdim=True)
        
        s1 = torch.sum(s*w[...,None], (1,))
        s2 = torch.sum(s[:,:,:,None]*s[:,:,None,:]*w[...,None,None], (1,))
        Sigma = s2 - s1[:,:,None] * s1[:,None,:]
        L = torch.cholesky(Sigma)
        s = torch.randn(*s.shape, **CTX)
        s = torch.einsum("ijk,ikl->ijl",s, L) + s1[:,None,:]
        samples[t,...] = s 
        
    if not pairwise: 
        return samples, weights
    else:
        return samples, samples_m1, weights


def filtering_bootstrap(model, xt, nsample=1000, bar=False, pairwise=False):

    T, nchain, D = xt.shape
    
    samples    = torch.empty(T,nchain, nsample,model.latent.D, **CTX)
    samples[0,...] = torch.randn(nchain, nsample,model.latent.D, **CTX)  * model.init_noise_std
    samples_m1 = torch.empty(T,nchain, nsample,model.latent.D, **CTX)
    weights = torch.ones(T,nchain, nsample, **CTX)/nsample
    
    s = samples[0,...]
    w = weights[0,...]
    
    iterable = trange(T, ncols=100) if bar else range(T)
    for t in iterable:

        for j in range(nchain):
            idx = torch.multinomial(w[j,:], nsample, replacement=True)
            s[j,:,:] = s[j, idx,:]

        samples_m1[t] = s
        s = model.latent.step(s,t)
        l = model.observation.logp(s, xt[t])
        samples[t,...] = s 
        
        logp = l-l.max()
        w = torch.exp(logp)
        assert torch.all(torch.isfinite(w))
        w /=torch.sum(w,(1,),keepdim=True)
        weights[t,...] = w
        
    if not pairwise: 
        return samples, weights
    else:
        return samples, samples_m1, weights

class ParticleLearner(object):
    
    def __init__(self, model, opt, nsample, nwake):
        
        self.model = model
        self.nsample = nsample
        self.nwake = nwake
        self.opt = [deepcopy(opt), deepcopy(opt)]

    def initialise(self):
        
        model = self.model
        nchain = self.nwake
        nsample = self.nsample
        nchain = self.nwake

        self.s = torch.randn(nchain, nsample,model.latent.D, **CTX)  * model.init_noise_std
        self.w = torch.ones((nchain, nsample), **CTX)/nsample
        self.it = 0

    def step(self, xt):

        s = self.s
        w = self.w
        model = self.model
        nchain = self.nwake
        nsample = self.nsample
        nchain = self.nwake

        for j in range(nchain):
            idx = torch.multinomial(w[j,:], nsample, replacement=True)
            s[j,:,:] = s[j, idx,:]

        s_m1 = s
        s = model.latent.step(s,0)
        l  = model.observation.logp(s, xt)
        
        logp = l-l.max()
        w = torch.exp(logp)
        assert torch.all(torch.isfinite(w))
        w /=torch.sum(w,1,keepdim=True)

        self.w = w
        self.s = s
        
        dlogpZ = model.latent.dlogp_tensor(s_m1, s, 0)
        dlogpX = model.observation.dlogp_tensor(s, xt[:,None,:].expand(-1,nsample,-1))
        dlogpZ = torch.mean(dlogpZ * w[...,None], -2)
        dlogpX = torch.mean(dlogpX * w[...,None], -2)
        dlogpZ = dlogpZ.mean(0)
        dlogpX = dlogpX.mean(0)

        self.opt[0].step(self.model.latent.ps, dlogpZ, self.it)
        self.opt[1].step(self.model.observation.ps, dlogpX, self.it)
        self.it += 1




def filtering_bootstrap_deep(model, xt, nsample=1000, bar=False):
        
    assert hasattr(model.observation, "depth")
    depth = model.observation.depth

    T, nchain, D = xt.shape
    
    samples = np.random.randn(T,nchain, nsample,model.latent.D) * model.init_noise_std
    weights = np.ones((T,nchain, nsample))/nsample
    
    s = samples[0,...]
    w = weights[0,...]
    xs = dict()
    for i in range(depth):
        xs["xt%d"%(i+1)] = np.zeros((T, nchain, nsample, model.observation.Ds[i]))

    iterable = trange(T, ncols=100) if bar else range(T)
    for t in iterable:

        for j in range(nchain):
            idx = np.random.choice(nsample, nsample, p=w[j,:])
            s[j,:,:] = s[j, idx,:]
        s = model.latent.step(s,t)
        x = model.observation.sample(s)
        x[-1] = xt[t]
        logp  = model.observation.logp(*x)
        
        logp -= logp.max()+10
        w = np.exp(logp)
        assert np.all(np.isfinite(w))
        w /=np.sum(w,1,keepdims=True)
        
        weights[t,...] = w
        samples[t,...] = s 
        for i in range(depth-1):
            xs["xt%d"%(i+1)][t,...]  = x[i+1]
        
    return samples, weights, xs

def filtering_mean(model, xt, nsample=1000, bar=False):

    T, nchain, D = xt.shape
    
    samples = np.random.randn(T,nchain, nsample,model.latent.D) * model.init_noise_std
    weights = np.ones((T,nchain, nsample))/nsample
    
    s = samples[0,...]
    w = weights[0,...]
    
    iterable = trange(T) if bar else range(T)
    for t in iterable:

        s = model.latent.step(s,t)
        logp  = model.observation.logp(s, xt[t])
        
        logp -= logp.max()+10
        w = np.exp(logp)
        assert np.all(np.isfinite(w))
        w /=np.sum(w,1,keepdims=True)
        
        weights[t,...] = w
        samples[t,...] = s 
        m = np.sum(w[...,None] * s, axis=-2, keepdims=True)
        s = m + np.random.randn(*(s.shape[:-2]+(nsample,s.shape[-1])))*5
        
    return samples, weights

def EES(weights):
    return 1/torch.sum(weights**2,-1)

def LML(weights):
    assert weights.ndim==3
    return np.sum(np.log(np.mean(weights, -1)),0)

def importance_sampling(model, x, nsample=1000):
    
    assert model.depth==2
    samples = model.dists[0].sample(nsample)
    
    logp  = model.dists[1].logp(samples[None,...], x)
    
    logp -= logp.max()+10
    w = np.exp(logp)
    assert np.all(np.isfinite(w))
    w /=np.sum(w,1,keepdims=True)
    
    return samples, w

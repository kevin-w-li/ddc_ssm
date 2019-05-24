import torch, sys, os
import numpy as np
from settings import CTX

class FeatureMap(object):

    def __init__(self, M, D, nl, lam, lim):
        self.M = M
        self.D = D
        self.lam = lam
        self.alphas = {}
        #self.nl = lambda x: torch.log(1+torch.exp(x))
        if nl is None:
            self.nl = torch.nn.Tanh()
        else:
            self.nl = nl
        self.nl_final = torch.nn.Tanh()

        self.lim = torch.tensor(lim, requires_grad=False, **CTX)

    def __call__(self, x):
        return self.map_data(x)
        
    def map_mean(self, x, w = None):
        
        f = self.map_data(x)
        if w is None:
            return torch.mean(f.reshape(-1, f.shape[-1]),-2)
        else:
            dim = f.shape[:-1]
            assert dim == w.shape
            return torch.sum(f * w[...,None],-2)
    
    def map_cov(self, x):
        
        s = self.map_data(x)
        s = s.t().mm(s) / s.shape[0]
        return s

    def learn_weights(self, f, y):
        
        #y = y.reshape(-1, y.shape[-1])
        a = torch.gesv(f.t().matmul(y), f.t().matmul(f)+torch.eye(self.M, **CTX)*self.lam, )[0]
        
        return a
        
    def learn_fun(self, key, x, y, nvalid=0):
            
        f = self.map_data(x)
        if nvalid:
            f_te = f[-nvalid:]
            y_te = y[-nvalid:]
            f_tr = f[:-nvalid]
            y_tr = y[:-nvalid]
        else:
            f_tr = f
            y_tr = y

        a = self.learn_weights(f_tr, y_tr)
        self.alphas[key] = a
        
        if nvalid:
            y_p = f_te.matmul(a)
            r2  = 1 - torch.mean((y_p - y_te)**2)/(y_te.var() + 1e-3)
            print("%20s r2: %.5f" % (key, r2))
        return a

    def batch_learn_fun(self, key, fun, batch_size, nbatch, nvalid=0):
            
        x,y = fun(100)
        ftf = torch.zeros((self.M, self.M))
        fty = torch.zeros((self.M, y.shape[-1]))
        for n in range(nbatch):
            x,y = fun(batch_size)
            f = self.map_data(x)
            ftf += f.T.matmul(f)
            fty += f.T.matmul(y)

        a = torch.linalg.solve(ftf+torch.eye(self.M)*self.lam, fty)
        
        if nvalid:
            x_te,y_te = fun(nvalid)
            f_te   = self.map_data(x_te)
            y_p = f_te.matmul(a)
            r2  = 1 - torch.mean(torch.mean((y_p - y_te, -1)**2)/y_te.var(-1))
            print ("%20s r2: %.5f" % (key, r2))

        self.alphas[key] = a
        return a
        

    def eval_fun(self, key, x):

        f = self.map_data(x)
        return f.matmul(self.alphas[key])
        
    def approx_E(self, key, m):
        
        assert m.shape[-1] == self.M
        
        return m.matmul(self.alphas[key])

    def constrain_params(self):
    
        for p in self.params:
            p.data = torch.clamp(p, -self.lim, self.lim)
        
        return         

class ContinuousFeature(FeatureMap):
    
    def __init__(self, M, D, std, r, nl=None, lam=1e-3, adaptive=False, lim=np.inf):
        
        self.W = (np.random.randn(M, D)*std / np.sqrt(D)).T
        if isinstance(r, np.ndarray):
            assert r.shape[0] == M
            assert r.shape[1] == D
            self.b = r.T
        else:
            self.b = (r[0] + np.random.rand(M, D)*(r[1]-r[0])).T
        self.b = -np.sum(self.b * self.W, 0)[None,...]
        self.W = torch.tensor(self.W, requires_grad=adaptive, **CTX)
        self.b = torch.tensor(self.b, requires_grad=adaptive, **CTX)
        super(ContinuousFeature, self).__init__(M, D, nl, lam, lim)
        self.adaptive=adaptive
        self.t = 0

    def map_data(self, x):
        self.t+=1 
        assert x.shape[-1]  == self.D
        return self.nl(torch.matmul(x, self.W) + self.b)
    
    @property
    def params(self):
        return self.W, self.b
        

class DeepFeature(FeatureMap):
    
    def __init__(self, Ms, std, r, nl=None, nl_final=None, lam=1e-3, adaptive=False, lim=np.inf):
        
        self.Ws = {}
        self.Ms = Ms
        self.adaptive=adaptive
        
        for i in range(len(Ms)-1):
            W = np.random.randn(Ms[i+1], Ms[i]).T*(std/np.sqrt(Ms[i]))
            if i == 0:
                b = (r[0] + np.random.rand(Ms[i+1], Ms[i])*(r[1]-r[0])).T
                b = -np.sum(b * W, 0)
            else:
                b = np.zeros(Ms[i+1])
            W = torch.tensor(W, requires_grad=adaptive, **CTX)
            b = torch.tensor(b, requires_grad=adaptive, **CTX)
            self.Ws["W"+str(i)] = W
            self.Ws["b"+str(i)] = b
        super().__init__(Ms[-1], Ms[0], nl, lam, lim)
        if nl_final is not None:
            self.nl_final = nl_final

    def map_data(self, x):
    
        assert x.shape[-1]  == self.D
        for i in range(len(self.Ms)-1):
            W = self.Ws["W"+str(i)]
            b = self.Ws["b"+str(i)]
            nl= self.nl if i != len(self.Ms)-2 else self.nl_final
            x = nl(torch.matmul(x, W) + b)
        return x
    
    @property
    def params(self):
        return self.Ws.values()

class GaussianFeature(FeatureMap):

    def __init__(self, M, D, std, r, lam=1e-3, lim=np.inf, adaptive=False):
        
        self.adaptive=False
        if D == 1:
            self.mus = torch.linspace(r[0], r[1], M, **CTX)[:,None]
        else:
            self.mus = torch.random.uniform(r[0], r[1], size=(M,D), **CTX)
        self.sigmas  = std
        super().__init__(M, D, torch.exp, lam, lim)

    def map_data(self, x):
        assert x.shape[-1] == self.D
        shape = x.shape[:-1]
        x = x.view(-1, self.D)
        f = torch.exp(-torch.sum(x[:,None,:] - self.mus[None,:,:],-1)**2/self.sigmas**2)
        f = f.reshape(shape+(self.M,))
        return f

class RecurrentFeature(FeatureMap):
    
    def __init__(self, f, gammas, lam=1e-3, lim=np.inf, adaptive=False):
        
        self.f = f
        self.gammas = torch.tensor(gammas, **CTX)
        self.h = torch.zeros(f.M, len(gammas), **CTX)
        M = f.M * len(gammas)
        super().__init__(M, f.D, torch.exp, lam, lim)

    def map_data(self, x):
        if self.h.dim()==2:
            self.h = self.h.expand((x.shape[0],)+ self.h.shape)
        self.h = self.h * self.gammas + self.f(x)[...,None].expand_as(self.h)

        return self.h.view(x.shape[:-1] + (self.M,))

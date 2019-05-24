import torch, numpy as np
from scipy.stats import norm
from .SSM import Observation
from .GaussianNoise import GaussianObservation
from .settings import RG, CTX
from .Utils import xavier_uniform, xavier_normal, randgn, gnlogpdf
from torch.distributions import Laplace

class LinearGaussianObservation(GaussianObservation):
    
    name = "LG"
        
    def __init__(self, A, noise_std, ignore_dim=False):
        
        super(LinearGaussianObservation, self).__init__(noise_std)
        self.Ds = [self.D]
        self.ignore_dim=ignore_dim
        self.D_in= A.shape[1]
        if ignore_dim:
            self.mask = torch.zeros(self.D_in, **CTX)
            self.mask[-1] = 1.0
        self.ps["A"] = torch.tensor(A, requires_grad=RG, **CTX)
        self.string = self.name + "_[%d,%d]_%.1f" % (A.shape[0], A.shape[1], noise_std[0])
        
    def __str__(self):
        return self.string

    def conditional_param(self, y):
        if self.ignore_dim:
            y = y * self.mask
        return torch.matmul(y, self.A.t())

    def dconditional_param(self, dl, y):
        if self.ignore_dim:
            y = y * self.mask
        d = (dl[...,:,None] * y[...,None,:]).reshape((y.shape[:-1])+(-1,))
        return d

class DeepBinaryObservation(Observation):
    
    name="DB"
    
    def __init__(self, Ds, nl, dnl, weight_std=1.0):
        super().__init__()
        
        self.D_in = Ds[0]
        self.D= Ds[-1]
        self.Ds = Ds
        self._depth = len(Ds)-1

        self.nl = nl
        self.dnl =dnl
        self.nl_final = lambda x: 1.0/(1+torch.exp(-x))
        self.dnl_final = lambda x: self.nl_final(x) * (1-self.nl_final(x))

        for i in range(self._depth):
            D_in  = Ds[i] 
            D_out = Ds[i+1]

            std = weight_std/np.sqrt(D_in)
            
            W = xavier_normal(weight_std, D_out,D_in)
            self.ps["W%d"%i] = W

            b = torch.zeros(D_out, requires_grad=RG, **CTX)
            self.ps["b%d"%i] = b
            
        self.string = self.name + "_%s_%.1f" % (str(Ds).replace(" ",""), weight_std)
    
    def __str__(self):
        return self.string
            
    def conditional_param(self, z):
            
        m = z
        ms = [z]
        for i in range(self._depth):
            W = self.ps["W"+str(i)]
            b = self.ps["b"+str(i)]
            m = torch.matmul(m, W.t()) + b
            ms.append(m)
            if i < self._depth-1:
                m = self.nl(m)
            else:
                m = self.nl_final(m)
                
            ms.append(m)
        return ms
        
    def sample(self, z):
        
        m = self.conditional_param(z)[-1]
        x = torch.rand(*m.shape, **CTX) + m
        x = torch.floor(x)
        return x

    def logp(self, z, x):
        
        if len(z.shape) == 3 and len(x.shape) == 2:
            x = x[:,None,:]
        m = self.conditional_param(z)[-1]
        logp = torch.sum(x * torch.log(m) + (1-x) * torch.log(1-m), -1)

        return logp

    def _backward(self, above, ms):
        
        n = above.shape[:-1]
        pi = self.ps.nparam
        dl = torch.zeros(n+(self.ps.nparam,), **CTX)

        for i in range(self._depth-1,-1,-1):

            W = self.ps["W"+str(i)]

            D_in  = self.Ds[i]
            D_out = self.Ds[i+1]
            
            if i == self._depth-1:
                dadlin = above
            else:
                dadlin = (above * self.dnl(ms[2*(i+1)-1]))
            d = dadlin[...,:,None] * ms[2*i][...,None,:]

            dl[:,pi-D_out:pi] = dadlin
            pi -= D_out

            dl[:,pi-D_in*D_out:pi] = d.reshape(n + (-1,))
            pi -= D_in*D_out

            above = torch.matmul(dadlin, W)
        return dl

    def dlogp(self, z, x):
        
        ms = self.conditional_param(z)
        m  = ms[-1]
        above = x - m
        dlogp = self._backward(above, ms)
        return dlogp

    def suff(self, x):
        return x

    def dnat(self, z):

        n = z.shape[0] 
        ms = self.conditional_param(z)
        m  = ms[-1]
        Ds = self.Ds
        nd = Ds[-1] * Ds[-2] + Ds[-1]
        for i in range(self._depth-2,-1,-1):
            nd += (1+Ds[i]) * Ds[i+1] * Ds[-1]
        dl = torch.zeros(n, nd, requires_grad=RG, **CTX)
        pi = nd

        for i in range(self._depth-1,-1,-1):

            W = self.ps["W"+str(i)]

            D_in  = self.Ds[i]
            D_out = self.Ds[i+1]
            
            if i == self._depth-1:
                dadlin = torch.ones_like(m)
                dl[:,pi-D_out:pi] = dadlin
                pi -= D_out

                d = dadlin[:,:,None] * ms[2*i][:,None,:]
                dl[:,pi-D_in*D_out:pi] = d.reshape(n, -1)
                pi -= D_in*D_out
                above = dadlin[:,:,None] * W[None,:,:]
            else:
                dadlin = (above * self.dnl(ms[2*(i+1)-1])[:,None,:])
                dl[:,pi-D_out*Ds[-1]:pi] = dadlin.reshape(n,-1)
                pi -= D_out*Ds[-1]

                d = dadlin[:,:,:,None] * ms[2*i][:,None,None,:]
                dl[:,pi-D_in*D_out*Ds[-1]:pi] = d.reshape(n, -1)
                pi -= D_in*D_out*Ds[-1]
                
                above = torch.matmul(dadlin, W)
        assert pi == 0
        return dl

    def dnatsuff(self, z, x):

        n = z.shape[0] 
        ms = self.conditional_param(z)
        m  = ms[-1]
        above = x
        d = self._backward(above, ms) 

        return d

    def dnatsuff_from_dnatsuff(self, dnat, suff):
        
        Ds = self.Ds
        n = suff.shape[0] 
        nd = Ds[-1] * Ds[-2] + Ds[-1]
        for i in range(self._depth-2,-1,-1):
            nd += (1+Ds[i]) * Ds[i+1] * Ds[-1]
        dl = torch.zeros(n, self.ps.nparam, **CTX)
        pi = self.ps.nparam

        for i in range(self._depth-1,-1,-1):

            D_in  = self.Ds[i]
            D_out = self.Ds[i+1]
            
            if i == self._depth-1:
                dnat2 = dnat[:,nd-D_out:nd]
                dl[:,pi-D_out:pi] = dnat2 * suff
                pi -= D_out
                nd -= D_out

                dnat1 = dnat[:,nd-D_in*D_out:nd].reshape(n, D_out,D_in)
                dl[:,pi-D_out*D_in:pi] = (dnat1*suff[:,:,None]).view(n, -1)
                pi -= D_in*D_out
                nd -= D_in*D_out

            else:
                dnat2 = dnat[:,nd-D_out*Ds[-1]:nd].reshape(n, Ds[-1], D_out)
                dl[:,pi-D_out:pi] = torch.einsum("ijk,ij->ik", (dnat2, suff))
                pi -= D_out
                nd -= D_out * Ds[-1]

                dnat1 = dnat[:,nd-D_in*D_out*Ds[-1]:nd].reshape(n, Ds[-1], D_out, D_in)
                dl[:,pi-D_out*D_in:pi] = torch.einsum("ijkl,ij->ikl", (dnat1, suff)).reshape(n,-1)
                pi -= D_in*D_out
                nd -= D_in*D_out*Ds[-1]
                
        assert pi == 0
        assert nd == 0
        return dl

    def dnorm(self, z):

        ms = self.conditional_param(z)
        m  = ms[-1]
        above = m
        dnorm = self._backward(above, ms)

        return dnorm 
        

class SpikeObservation(Observation):
    
    name = "S"
    
    def __init__(self, D_in, D_out, weight_std=1.0, g=0.1):
        
        self.D_in = D_in
        self.D_out = D_out
        self.A = torch.tensor(np.random.randn(D_in, D_out) * weight_std / np.sqrt(D_in), **CTX)
        self.g = g
        
    def conditional_param(self, z):
        
        m = self.g * torch.exp(z @ self.A )
        m = 1-torch.exp(-m)
        return m
        
    def sample(self, z):
         
        m = self.conditional_param(z)
        return torch.floor(torch.rand(*m.shape, **CTX) + m)
    

class DeepGaussianObservation(GaussianObservation):
    
    name = "DG"
    def __init__(self, Ds, noise_std, nl, dnl, weight_std=1.0, ignore_dim=False):

        
        self.D_in = Ds[0]
        self.D= Ds[-1]
        self.Ds = Ds
        self._depth = len(Ds)-1
        self.ignore_dim=ignore_dim

        super().__init__([noise_std]*self.D)

        self.nl  = nl
        self.dnl = dnl
        self.nl_final = lambda x: 1/(1+torch.exp(-x))
        self.dnl_final = lambda x: self.nl_final(x)*(1-self.nl_final(x))

        for i in range(self._depth):
            D_in  = Ds[i] 
            D_out = Ds[i+1]

            W = xavier_normal(weight_std, D_out,D_in)
            self.ps["W%d"%i] = W

            b = torch.zeros(D_out, requires_grad=RG, **CTX)
            self.ps["b%d"%i] = b
            
        self.string = self.name + "_%s_%.1f_%.1f" % (str(Ds).replace(" ",""), noise_std, weight_std)
    
    def __str__(self):
        return self.string
            
    def apply_constraints(self):
        return 
        for i in range(self._depth):
            self.ps["W"+str(i)].clamp_(-1,1)
            '''
            W = self.ps["W"+str(i)].data
            n = torch.norm(W)
            if n > 10:
                self.ps["W"+str(i)].data = W / n * 10
            '''
    
    def conditional_param(self, z, keep=False):
        
        if self.ignore_dim:
            z = z * torch.tensor([1,0], **CTX)
            
        m = z
        ms = [z]
        for i in range(self._depth):
            W = self.ps["W"+str(i)]
            b = self.ps["b"+str(i)]
            m = torch.matmul(m, W.t()) + b
            ms.append(m)
            if i < self._depth-1:
                m = self.nl(m)
            else:
                m = self.nl_final(m)
                
            ms.append(m)
        if not keep:
            ms = ms[-1]
        return ms

    def dconditional_param(self, dl, z):
        if self.ignore_dim:
            z = z * torch.tensor([1,0], **CTX)
        return self._backward(dl, z)
        
    def _backward(self, above, z): 
    
        if self.ignore_dim:
            z = z * torch.tensor([1,0], **CTX)
            
        n = above.shape[:-1]
        pi = self.ps.nparam - self.D
        dl = torch.zeros(n+(pi,), **CTX)
        ms = self.conditional_param(z, keep=True)

        for i in range(self._depth-1,-1,-1):

            W = self.ps["W"+str(i)]

            D_in  = self.Ds[i]
            D_out = self.Ds[i+1]
            
            if i == self._depth-1:
                dadlin = above * self.dnl_final(ms[2*(i+1)-1])
            else:
                dadlin = above * self.dnl(ms[2*(i+1)-1])
            d = dadlin[...,:,None] * ms[2*i][...,None,:]

            dl[:,pi-D_out:pi] = dadlin
            pi -= D_out

            dl[:,pi-D_in*D_out:pi] = d.reshape(n + (-1,))
            pi -= D_in*D_out

            above = torch.matmul(dadlin, W)
        return dl

    def dnat(self, z):
        if self.ignore_dim:
            z = z * torch.tensor([1,0], **CTX)
        n = z.shape[0]
        Ds = self.Ds

        noise_std = self.ps["noise_std"]
        sigma_2 = torch.exp(-2*noise_std)
        
        ms = self.conditional_param(z, keep=True)
        m  = ms[-1]

        # d(1/2sig)(dlogsig), d(mu/sig)/dlogsig, d(mu/sig)/mu_param
        nd = 2 * self.D + self.Ds[-1]*(self.Ds[-2]+1)
        for i in range(self._depth-2,-1,-1):
            nd += (1+Ds[i]) * Ds[i+1] * Ds[-1]

        dl = torch.zeros(n, nd, **CTX)

        dl[:, :self.D] = - 2 *  m * sigma_2
        dl[:, self.D:2*self.D] = sigma_2

        pi = nd

        for i in range(self._depth-1,-1,-1):

            W = self.ps["W"+str(i)]

            D_in  = self.Ds[i]
            D_out = self.Ds[i+1]
            
            if i == self._depth-1:
                dadlin = self.dnl_final(ms[2*(i+1)-1]) * sigma_2
                dl[:,pi-D_out:pi] = dadlin
                pi -= D_out

                d = dadlin[:,:,None] * ms[2*i][:,None,:]
                dl[:,pi-D_in*D_out:pi] = d.reshape(n, -1)
                pi -= D_in*D_out
                above = dadlin[:,:,None] * W[None,:,:]
            else:
                dadlin = (above * self.dnl(ms[2*(i+1)-1])[:,None,:])
                dl[:,pi-D_out*Ds[-1]:pi] = dadlin.reshape(n,-1)
                pi -= D_out*Ds[-1]

                d = dadlin[:,:,:,None] * ms[2*i][:,None,None,:]
                dl[:,pi-D_in*D_out*Ds[-1]:pi] = d.reshape(n, -1)
                pi -= D_in*D_out*Ds[-1]
                
                above = torch.tensordot(dadlin, W, 1)

        assert pi == 2*self.D
        return dl

    def dnatsuff_from_dnatsuff(self, dnat, suff):
        
        Ds = self.Ds
        n = suff.shape[0] 
        nd = 2 * self.D + self.Ds[-1]*(self.Ds[-2]+1)
        for i in range(self._depth-2,-1,-1):
            nd += (1+Ds[i]) * Ds[i+1] * Ds[-1]
        dl = torch.zeros(n, self.ps.nparam, **CTX)
        pi = self.ps.nparam
        s1, s2     = torch.split(suff, [self.D, suff.shape[1]-self.D], dim=1)
        n1, n2, n3 = torch.split(dnat, [self.D, self.D, dnat.shape[1]-self.D*2], dim=1)
        
        dl[:, :self.D] = s1 * n1 + s2 * n2

        for i in range(self._depth-1,-1,-1):

            D_in  = self.Ds[i]
            D_out = self.Ds[i+1]
            
            if i == self._depth-1:
                dnat2 = dnat[:,nd-D_out:nd]
                dl[:,pi-D_out:pi] = dnat2 * s1
                pi -= D_out
                nd -= D_out

                dnat1 = dnat[:,nd-D_in*D_out:nd].reshape(n, D_out,D_in)
                dl[:,pi-D_out*D_in:pi] = torch.reshape(dnat1*s1[:,:,None], (n, -1))
                pi -= D_in*D_out
                nd -= D_in*D_out

            else:
                dnat2 = dnat[:,nd-D_out*Ds[-1]:nd].reshape(n, Ds[-1], D_out)
                dl[:,pi-D_out:pi] = torch.einsum("ijk,ij->ik", dnat2, s1)
                pi -= D_out
                nd -= D_out * Ds[-1]

                dnat1 = dnat[:,nd-D_in*D_out*Ds[-1]:nd].reshape(n, Ds[-1], D_out, D_in)
                dl[:,pi-D_out*D_in:pi] = torch.einsum("ijkl,ij->ikl", dnat1, s1).reshape(n,-1)
                pi -= D_in*D_out
                nd -= D_in*D_out*Ds[-1]
                
        assert pi == self.D
        assert nd == self.D*2
        return dl

                
class BumpObservation(GaussianObservation):

    name="BUMP"

    def __init__(self, pix, range, bar_width=0.1, noise_std=0.0):

        super().__init__([noise_std]*pix)
        self.D_in = 2
        self.pix = pix
        self.range = range
        self.bar_width = bar_width * range * 2
        self.grid = torch.linspace(-range,range,pix, **CTX)
        self.string = self.name + "_%d_%.1f_%.1f_%.1f" % (pix, range, bar_width, noise_std)
    
    def __str__(self):
        return self.string

    def conditional_param(self, z):

        z = z[...,0]
        #z.clamp_(-self.range, self.range)
        im = torch.exp(-0.5*(z[...,None]-self.grid)**2/self.bar_width**2)
     
        return im
        
class ImageObservation(GaussianObservation):
        
    name="IMG"

    def __init__(self, pix, range, margin=0, noise_std=0.0, bar_width=1):

        super().__init__([noise_std]*pix)
        self.pix = pix
        self.range = range
        self.margin = margin
        self.bar_width = bar_width

    def conditional_param(self, z):

        z = z[...,0]
        shape = z.shape
        z = z.reshape(-1)
        im = torch.zeros(np.prod(shape), self.pix, **CTX)
        z.clamp_(-self.range, self.range)
        z = z / self.range * (self.pix/2 - self.margin-(self.bar_width-1)/2)
        z += self.pix/2.0
        z.clamp_((self.bar_width-1)//2, self.pix-1-(self.bar_width-1)/2)
        z = z.long()
        
        for i in range(self.bar_width):
            im[range(im.shape[0]), z+i-(self.bar_width-1)//2] = 1.0

        im = im.reshape(*shape, self.pix)
        return im

class PoissonImageObservation(Observation):
    
    name = "POISIMG"
    def __init__(self, pix, range, bar_width=0.1, min_rate = 0.1, max_rate = 5):

        super().__init__()
        self.D_in = 2
        self.D = pix
        self.pix = pix
        self.range = range
        self.bar_width = bar_width * range * 2
        self.grid = torch.linspace(-range,range,pix, **CTX)
        self.max_rate = max_rate
        self.min_rate = min_rate
        self.string = self.name + "_%d_%.1f_%.1f" % (pix, range, bar_width)
    
    def __str__(self):
        return self.string

    def conditional_param(self, z):
        
        z = z[...,0]
        im = torch.exp(-0.5*(z[...,None]-self.grid)**2/self.bar_width**2)*self.max_rate+self.min_rate
        return im

    def sample(self, z):
        im = self.conditional_param(z)
        im = torch.poisson(im)
        return im

    def logp(self, z, x):
        mu = self.conditional_param(z) 
        assert x.shape[-1] == self.D, "observation dimension on the last axis should be %d" % self.D
        if len(z.shape) == 3 and len(x.shape) == 2:
            x = torch.unsqueeze(x,1)
        return torch.sum( - mu + x * mu.log() - (x + 1).lgamma(), -1)
     
class PoissonObservation(Observation):

    name = "Pois"
    def __init__(self, A, g):
        
        self.D_in = A.shape[1]
        self.D = A.shape[0]
        self.A = torch.tensor(A.T, **CTX)
        self.g = g
        
    def conditional_param(self, z):
        return self.g * torch.exp(z @ self.A)
        
    def sample(self, z):
        
        mu = self.conditional_param(z)
        return torch.poisson(mu)
        
    def logp(self, z, x):
        
        assert z.shape[-1] == self.D_in
        assert x.shape[-1] == self.D
        
        mu = self.conditional_param(z)
        assert x.shape[-1] == self.D, "observation dimension on the last axis should be %d" % self.D
        if len(z.shape) == 3 and len(x.shape) == 2:
            x = torch.unsqueeze(x,1)
        
        return torch.sum( - mu + x * mu.log() - (x + 1).lgamma(), -1)
        
class DeepPoissonObservation(Observation):

    name="DP"
    
    def __init__(self, Ds, nl, dnl, weight_std=1.0):
        super().__init__()
        
        self.D_in = Ds[0]
        self.D= Ds[-1]
        self.Ds = Ds
        self._depth = len(Ds)-1

        self.nl = nl
        self.dnl =dnl
        self.nl_final = torch.nn.Softplus()
        self.dnl_final = torch.nn.Sigmoid()

        for i in range(self._depth):
            D_in  = Ds[i] 
            D_out = Ds[i+1]

            W = xavier_normal(weight_std, D_out,D_in)
            self.ps["W%d"%i] = W

            b = torch.zeros(D_out, requires_grad=RG, **CTX)
            self.ps["b%d"%i] = b
            
        self.string = self.name + "_%s_%.1f" % (str(Ds).replace(" ",""), weight_std)
    
    def __str__(self):
        return self.string
            
    def conditional_param(self, z):
            
        m = z
        ms = [z]
        for i in range(self._depth):
            W = self.ps["W"+str(i)]
            b = self.ps["b"+str(i)]
            m = torch.matmul(m, W.t()) + b
            ms.append(m)
            if i < self._depth-1:
                m = self.nl(m)
            else:
                m = self.nl_final(m)
                
            ms.append(m)
        return ms
        
    def sample(self, z):
        
        m = self.conditional_param(z)[-1]
        x = torch.poisson(m)
        return x

    def logp(self, z, x):
        
        if len(z.shape) == 3 and len(x.shape) == 2:
            x = x[:,None,:]
        m = self.conditional_param(z)[-1]
        return torch.sum( - m + x * m.log() , -1)

    def _backward(self, above, ms):
        
        n = above.shape[:-1]
        pi = self.ps.nparam
        dl = torch.zeros(n+(self.ps.nparam,), **CTX)

        for i in range(self._depth-1,-1,-1):

            W = self.ps["W"+str(i)]

            D_in  = self.Ds[i]
            D_out = self.Ds[i+1]
            
            if i == self._depth-1:
                dadlin = above * self.dnl_final(ms[2*(i+1)-1])
            else:
                dadlin = (above * self.dnl(ms[2*(i+1)-1]))
            d = dadlin[...,:,None] * ms[2*i][...,None,:]

            dl[:,pi-D_out:pi] = dadlin
            pi -= D_out

            dl[:,pi-D_in*D_out:pi] = d.reshape(n + (-1,))
            pi -= D_in*D_out

            above = torch.matmul(dadlin, W)
        return dl

    def dlogp(self, z, x):
        
        ms = self.conditional_param(z)
        m  = ms[-1]
        above = -1 + x / m
        dlogp = self._backward(above, ms)
        return dlogp

    def suff(self, x):
        return x

    def dnat(self, z):

        n = z.shape[0] 
        ms = self.conditional_param(z)
        m  = ms[-1]
        Ds = self.Ds
        nd = Ds[-1] * Ds[-2] + Ds[-1]
        for i in range(self._depth-2,-1,-1):
            nd += (1+Ds[i]) * Ds[i+1] * Ds[-1]
        dl = torch.zeros(n, nd, requires_grad=RG, **CTX)
        pi = nd

        for i in range(self._depth-1,-1,-1):

            W = self.ps["W"+str(i)]

            D_in  = self.Ds[i]
            D_out = self.Ds[i+1]
            
            if i == self._depth-1:
                dadlin = 1.0 / m * self.dnl_final(ms[2*(i+1)-1])
                dl[:,pi-D_out:pi] = dadlin
                pi -= D_out

                d = dadlin[:,:,None] * ms[2*i][:,None,:]
                dl[:,pi-D_in*D_out:pi] = d.reshape(n, -1)
                pi -= D_in*D_out
                above = dadlin[:,:,None] * W[None,:,:]
            else:
                dadlin = (above * self.dnl(ms[2*(i+1)-1])[:,None,:])
                dl[:,pi-D_out*Ds[-1]:pi] = dadlin.reshape(n,-1)
                pi -= D_out*Ds[-1]

                d = dadlin[:,:,:,None] * ms[2*i][:,None,None,:]
                dl[:,pi-D_in*D_out*Ds[-1]:pi] = d.reshape(n, -1)
                pi -= D_in*D_out*Ds[-1]
                
                above = torch.matmul(dadlin, W)
        assert pi == 0
        return dl

    def dnatsuff(self, z, x):

        n = z.shape[0] 
        ms = self.conditional_param(z)
        m  = ms[-1]
        above = x / m
        d = self._backward(above, ms) 

        return d

    def dnatsuff_from_dnatsuff(self, dnat, suff):
        
        Ds = self.Ds
        n = suff.shape[0] 
        nd = Ds[-1] * Ds[-2] + Ds[-1]
        for i in range(self._depth-2,-1,-1):
            nd += (1+Ds[i]) * Ds[i+1] * Ds[-1]
        dl = torch.zeros(n, self.ps.nparam, **CTX)
        pi = self.ps.nparam

        for i in range(self._depth-1,-1,-1):

            D_in  = self.Ds[i]
            D_out = self.Ds[i+1]
            
            if i == self._depth-1:
                dnat2 = dnat[:,nd-D_out:nd]
                dl[:,pi-D_out:pi] = dnat2 * suff
                pi -= D_out
                nd -= D_out

                dnat1 = dnat[:,nd-D_in*D_out:nd].reshape(n, D_out,D_in)
                dl[:,pi-D_out*D_in:pi] = (dnat1*suff[:,:,None]).view(n, -1)
                pi -= D_in*D_out
                nd -= D_in*D_out

            else:
                dnat2 = dnat[:,nd-D_out*Ds[-1]:nd].reshape(n, Ds[-1], D_out)
                dl[:,pi-D_out:pi] = torch.einsum("ijk,ij->ik", (dnat2, suff))
                pi -= D_out
                nd -= D_out * Ds[-1]

                dnat1 = dnat[:,nd-D_in*D_out*Ds[-1]:nd].reshape(n, Ds[-1], D_out, D_in)
                dl[:,pi-D_out*D_in:pi] = torch.einsum("ijkl,ij->ikl", (dnat1, suff)).reshape(n,-1)
                pi -= D_in*D_out
                nd -= D_in*D_out*Ds[-1]
                
        assert pi == 0
        assert nd == 0
        return dl

    def dnorm(self, z):

        ms = self.conditional_param(z)
        m  = ms[-1]
        above = torch.ones_like(m)
        dnorm = self._backward(above, ms)

        return dnorm 
        
class LinearLaplaceObservation(Observation):
    
    name = "Lap"

    def __init__(self, A, noise_std):
        
        super().__init__()
        self.D = len(noise_std)
        self.ps["noise_std"] = torch.tensor(np.log(noise_std), requires_grad=RG, **CTX)
        self.ps["A"]   = torch.tensor(A, requires_grad=RG, **CTX)
        self.string = self.name + "_[%d,%d]_%.1f" % (A.shape[0], A.shape[1], noise_std[0])

    def __str__(self):
        return self.string

    def logp(self, z, x, t):
        mu = self.conditional_param(z)
        l  = Laplace(mu, self.noise_std.exp()).log_prob(x).sum(-1)
        return l

    def natsuff(self, z, x, t):
        mu = self.conditional_param(z)
        return (- ((x-mu)/self.noise_std.exp()).abs()).sum(-1)

    def norm(self, z, t):
        n = z.shape[0]
        return (self.noise_std + torch.log(2) ).sum(-1).expand(n)
    def conditional_param(self, z, keep=False):
        
        tz = z
        tz = z.clamp(-5,5)
        #tz = 5.0 * torch.tanh(0.2 * z)

        m = torch.matmul(tz, self.A.t())
        if not keep:
            return m
        else:
            return tz, m

    def apply_constraints(self):
        return

    def sample(self, z):
        
        mu = self.conditional_param(z)
        return Laplace(mu, self.noise_std.exp()).sample()
         
    def dlogp(self, z, x, t):

        n = z.shape[0]

        noise_std = self.noise_std.exp()

        ty, m = self.conditional_param(z, keep=True)
        d = (x - m) / noise_std
        dstd = -1. + d.abs()
        dA   = (d.sign() / noise_std)[...,None] * ty[...,None,:]
        dA   = dA.reshape(n, -1)

        dl = torch.cat([dstd, dA], -1)
        return dl

    def dnatsuff(self, z, x, t):
        n = z.shape[0]

        noise_std = self.noise_std.exp()

        ty, m = self.conditional_param(z, True)
        d = (x - m) / noise_std
        dstd = d.abs()
        dA   = (d.sign() / noise_std)[...,None] * ty[...,None,:]
        dA   = dA.reshape(n, -1)
        
        dl = torch.cat([dstd, dA], -1)
        return dl

    def dnorm(self, z, t):

        n = z.shape[0]

        noise_std = self.noise_std.exp()
        
        ones = torch.ones(n, self.D, **CTX)
        m    = self.conditional_param(z)
        dstd = 1. * ones
        dA   = torch.zeros(n, torch.prod(self.A.shape), **CTX)

        dl = torch.cat([dstd, dA], -1)
        return dl

class AuditoryVisualObservation(GaussianObservation):

    def __init__(self, b, c, noise_std):
        assert len(noise_std) == 2
        super().__init__(noise_std)
        self.D_in= 1
        self.ps["b"] = torch.tensor([b], **CTX)
        self.ps["c"] = torch.tensor([c], **CTX)

    def conditional_param(self, y):
        b = self.b
        c = self.c
        out = torch.cat([b*y, torch.tanh(c * y)], -1)
        return out

    def dconditional_param(self, dl, y):
        n = y.shape[0]
        c = self.c
        dp = torch.cat([y, y * (1-torch.tanh(c*y)**2)], -1)
        return (dl * dp[:,:]).view( (n, 2))

        
class OccludedToneObservation(GaussianObservation):
    
    def __init__(self, noise_std):
         
        super().__init__(noise_std)

    def conditional_param(self, zt):
        assert zt.shape[-1] == self.D+2
        mask_values = (zt[...,-2:-1]-0.5).clamp(0,2)
        xt = torch.max(zt[...,:-2], mask_values)
        '''
        mask = torch.cat([zt[...,-2:-1]]*self.D, -1)
        wings = torch.where(zt[...,-1]==2, mask_values[...,0], torch.zeros_like(xt[...,1]))
        xt[...,0] = wings
        xt[...,2] = wings
        '''

        return xt

class OccludedObservation(GaussianObservation):
    
    def __init__(self, dist, noise_std):
         
        self.dist = dist
        super().__init__([noise_std]*self.dist.D)
        self.D_in = self.dist.D_in+1


    def conditional_param(self, zt):
        assert zt.shape[-1] == self.D_in
        xt = self.dist.conditional_param(zt[...,:-1])
        xt = torch.max(xt, zt[...,-1:])

        return xt

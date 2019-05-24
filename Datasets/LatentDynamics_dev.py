import torch, numpy as np
from .GaussianNoise import GaussianLatent
from .settings import RG, CTX
from .Utils import xavier_uniform, xavier_normal, randgn, gnlogpdf
from .SSM import Latent
from torch.distributions import Laplace

class LocalLinearLatent(GaussianLatent):
        
    name="LLL"

    def __init__(self, Ds, noise_std, weight_std = 1.0, nl=None, dnl=None, out_scale = np.inf):
    
        
        self.D = Ds[0]
        self.Ds = Ds
        self.M = Ds[-1]
        self._depth = len(Ds)
        noise_std = [noise_std] * self.D
        self.out_scale = out_scale
        
        
        if nl is None:
            self.nl = lambda x: torch.tanh(x)
            self.dnl = lambda x: 1-torch.tanh(x)**2
        else:
            self.nl = nl
            self.dnl = dnl
        super(LocalLinearLatent, self).__init__(noise_std)
            
        # nonlinear weights
        for di in range(self._depth-1):
            D_in  = Ds[di]
            D_out = Ds[di+1]
            W = xavier_normal(weight_std, D_out, D_in)
            self.ps["W%d"%di] = W

            # bias
            b = torch.zeros(D_out, requires_grad=RG, **CTX)
            self.ps["b%d"%di] = b

        # linear weights
        #V = torch.randn(self.D,self.M,self.D, requires_grad=RG, **CTX) * weight_std
        V = torch.tensor( np.random.randn(self.D,self.M,self.D)*0.1/np.sqrt(self.D), requires_grad=RG, **CTX )
        self.ps["V"] = V
        
        self.string = self.name + "_%s_%.1f_%.1f" % (str(self.Ds), noise_std[0], weight_std)
        
    def __str__(self):
        return self.string
        
    def apply_constraints(self):
        #self.ps["noise_std"].clamp_(np.log(0.01), np.log(1))
        return
        
    def conditional_param(self, zt, t, keep=False):
        
        zt = 5 * torch.tanh(0.2 * zt)
        m  = zt
        ms = [zt]
        for i in range(self._depth-1):
            W = self.ps["W"+str(i)]
            b = self.ps["b"+str(i)]
            m = torch.matmul(m, W.t()) + b
            ms.append(m)
            if i < self._depth-2:
                m = self.nl(m)
            else:
                m = m - m.max(-1, keepdim=True)[0]
                m = torch.exp(m)/torch.exp(m).sum(-1, keepdim=True)
            ms.append(m)
    
        V = self.ps["V"]
        mu = torch.einsum("jkl,...k,...l->...j", V, m, zt)
        ms.append(mu)
        
        if np.isfinite(self.out_scale):
            mu = self.out_scale * torch.tanh(mu/self.out_scale)
            
        if keep:
            return mu, ms
        else:
            return mu

    def dconditional_param(self, dl, zt, t):

        n = zt.shape[0]

        mu, ms = self.conditional_param(zt, t, True)
        zt = ms[0]
        V = self.ps["V"]

        grad = torch.zeros((n, self.ps.nparam-self.D), **CTX)
        
        pi = self.ps.nparam-self.D

        D_out = self.Ds[0]
        if np.isfinite(self.out_scale):
            dl = dl * (1-torch.tanh(ms[-1] / self.out_scale )**2)
        dW = dl[:,:,None,None] * ms[-2][:,None,:,None] * zt[:,None,None,:]

        grad[:,pi-D_out*self.M*D_out:pi] = dW.reshape(n,-1)
        pi -= D_out*self.M*D_out

        above = torch.einsum("ij,jlk,ik->il", dl, V, zt)

        for i in range(self._depth-2, -1, -1):
            D_in  = self.Ds[i]
            D_out = self.Ds[i+1]
            W = self.ps["W"+str(i)]
            
            if i == self._depth-2:
                above = above.unsqueeze(-2).matmul(ms[-2].unsqueeze(-1) * ( torch.eye(ms[-2].shape[-1], **CTX) - ms[-2].unsqueeze(-2))).squeeze(-2)
            else:
                above = above * self.dnl(ms[2*i+1])
            dW = above[:,:,None] * ms[2*i][:,None,:]
            dc = above
            above = torch.matmul(above, W)

            grad[:,pi-D_out:pi] = dc.reshape(n,-1)
            pi -= D_out
            grad[:,pi-D_in*D_out:pi] = dW.reshape(n,-1)
            pi -= D_in*D_out
        assert pi == 0
        
        return grad

class RotationLatent(GaussianLatent):

    name = "ROT"
    
    def __init__(self, angle, slope, noise_std):

        super(RotationLatent, self).__init__(noise_std)
        
        self.ps["angle"] = torch.tensor([angle], **CTX)
        self.ps["slope"] = torch.tensor([slope], **CTX)
        self.string = self.name + "_%d/pi_%d_%.1f" % (angle*np.pi, slope, noise_std[0])
    
    def __str__(self):
        return self.string

    def conditional_param(self, zt, t):
            
        slope = self.slope[0]
        angle = self.angle[0]

        a = torch.tensor([[  torch.cos(angle), -torch.sin(angle)],
                        [torch.sin(angle),  torch.cos(angle)]], **CTX)
        
        r        = torch.sqrt(torch.sum(zt**2,-1))
        r_ratio  = 1.0/(torch.exp(-slope*4*(r-0.3)) + 1) / r

        ztp1  = torch.matmul(zt, a)
        ztp1 *= r_ratio[...,None]

        return ztp1

class LinearGaussianLatent(GaussianLatent):
    
    name = "LG"
    
    def __init__(self, A, noise_std):
        
        super(LinearGaussianLatent, self).__init__(noise_std)
        self.ps["A"] = torch.tensor(A, **CTX)
        self.string = self.name + "_[%d,%d]_%.1f" % (A.shape[0], A.shape[1], noise_std[0])
        
    def __str__(self):
        return self.string

    def conditional_param(self, zt, t):
        return torch.matmul(zt, self.A.t())

    def dconditional_param(self, dl, zt, t):
        n = zt.shape[0]
        A = self.A
        return ((dl)[:,:,None] * zt[:,None,:]).reshape((n, -1))

class DeepLatent(GaussianLatent):

    name = "DEEP"

    def __init__(self, Ds, noise_std, weight_std = 1.0, nl=None, dnl=None):
    
    
        assert Ds[0] == Ds[-1]
        self.D = Ds[0]
        self.Ds = Ds
        self.M = Ds[-1]
        self._depth = len(Ds)
        noise_std = [noise_std] * self.D
        self.out_scale = 1.0
        if nl is None:
            self.nl = lambda x: torch.tanh(x)
            self.dnl = lambda x: 1-torch.tanh(x)**2
        else:
            self.nl = nl
            self.dnl = dnl
        super().__init__(noise_std)
            
        # nonlinear weights
        for di in range(self._depth-1):
            D_in  = Ds[di]
            D_out = Ds[di+1]
            W = xavier_normal(weight_std, D_out, D_in)
            self.ps["W%d"%di] = W

            # bias
            b = torch.zeros(D_out, requires_grad=RG, **CTX)
            self.ps["b%d"%di] = b
    
        self.string = self.name + "_%s_%.1f_%.1f" % (str(self.Ds), noise_std[0], weight_std)
        
    def __str__(self):
        return self.string

    def conditional_param(self, zt, t):
        
        m  = zt
        for i in range(self._depth-1):
            W = self.ps["W"+str(i)]
            b = self.ps["b"+str(i)]
            m = torch.matmul(m, W.t()) + b
            if i != self._depth-2:
                m = self.nl(m)
            else:
                m = self.out_scale * torch.tanh(m/self.out_scale)
        
        return m

    def dconditional_param(self, dl, zt, t):

        n = zt.shape[0]

        #forward pass
        m  = zt
        ms = [zt]
        for i in range(self._depth-1):
            W = self.ps["W"+str(i)]
            b = self.ps["b"+str(i)]
            m = torch.matmul(m, W.t()) + b
            ms.append(m)
            if i != self._depth-2:
                m = self.nl(m)
            else:
                m = self.out_scale * torch.tanh(m/self.out_scale)
            ms.append(m)
                
        grad = torch.zeros((n, self.ps.nparam-self.D), **CTX)
        
        pi = self.ps.nparam-self.D

        above = dl

        for i in range(self._depth-2, -1, -1):
            D_in  = self.Ds[i]
            D_out = self.Ds[i+1]
            W = self.ps["W"+str(i)]
            if i == self._depth-2:
                above = above * (1-torch.tanh(ms[-2] / self.out_scale )**2)
            else:
                above = above * self.dnl(ms[2*i+1])

            dW = above[:,:,None] * ms[2*i][:,None,:]
            dc = above
            above = torch.matmul(above, W)

            grad[:,pi-D_out:pi] = dc.reshape(n,-1)
            pi -= D_out
            grad[:,pi-D_in*D_out:pi] = dW.reshape(n,-1)
            pi -= D_in*D_out
        assert pi == 0
        
        return grad

class PPCLatent(GaussianLatent):

    def __init__(self, fs, weight_std, noise_std):
    
        self.D = fs.D
        D = fs.D
        M = fs.M
        self.fs = fs
        noise_std = [noise_std] * fs.D
        super(PPCLatent, self).__init__(noise_std)
        
        #W = np.random.randn(D,M,D) * weight_std
        self.ps['W'] = torch.tensor(np.random.randn(D,M) * weight_std / np.sqrt(M), requires_grad=RG, **CTX)

    def conditional_param(self, zt, t):
        
        W = self.W
        f = self.fs(zt)
        mu = torch.einsum("jk,...k->...j", W, f)
        
        return mu

    def dconditional_param(self, dl, zt, t):

        n = zt.shape[0]
        W = self.W

        f = self.fs(zt)

        dW= torch.reshape(dl[:,:,None] * f[:,None,:], (n, -1))
        db= dl

        return torch.cat([dW], -1)
        
class CoxLatent(Latent):
    '''
    https://www.tandfonline.com/doi/pdf/10.1080/07350015.2015.1092977?needAccess=true
    '''
    def __init__(self, v, c, phi):
        
        self.D = 1
        self.v = v
        self.c = c
        self.phi = phi
        
    def step(self, htm, t):
        
        zt = torch.poisson(htm * self.phi / self.c)
        ht = torch.distributions.Gamma(zt+self.v, 1/self.c).sample()
        
        return ht


class DoubleWellLatent(GaussianLatent):
    
    def __init__(self, b, c, noise_std):
        super(DoubleWellLatent, self).__init__(noise_std)
        self.b = b
        self.c = c
        self.s = torch.tensor(1.5, **CTX)

    def conditional_param(self, zt, t):
        c = self.c
        b = self.b
        '''
        zt = torch.where(zt>2, self.s, zt)
        zt = torch.where(zt<-2, -self.s, zt)
        '''
        zt.clamp_(-2,2)
        mu = b * zt + c * zt * ( 2 - zt ** 2)
        return mu

    def dconditional_param(self, dl, zt, t):
        '''
        zt = torch.where(zt>2, self.s, zt)
        zt = torch.where(zt<-2, -self.s, zt)
        '''
        n = zt.shape[0]
        zt.clamp_(-2,2)
        dzdp = torch.stack([zt, zt*(1-zt**2)], axis=-1)
        return torch.sum(dl[:,:,None] * dzdp, 1)

class LinearGeneralizedGaussianLatent(Latent):
    
    name = "GLG"

    # https://sccn.ucsd.edu/wiki/Generalized_Gaussian_Probability_Density_Function#Generating_Random_Samples
    
    def __init__(self, A, rho, noise_std):
        
        super().__init__()
        self.D = len(noise_std)
        self.ps["noise_std"] = torch.tensor(np.log(noise_std), requires_grad=RG, **CTX)
        self.ps["rho"] = torch.tensor(np.log(rho), requires_grad=RG, **CTX)
        self.ps["A"]   = torch.tensor(A, requires_grad=RG, **CTX)
        self.string = self.name + "_[%d,%d]_%.1f" % (A.shape[0], A.shape[1], noise_std[0])
        
    def __str__(self):
        return self.string

    def conditional_param(self, ztm, keep=False):
        
        tztm = ztm
        tztm = ztm.clamp(-10,10)
        #tztm = 5.0 * torch.tanh(0.2 * ztm)

        zt = torch.matmul(tztm, self.A.t())
        if not keep:
            return zt
        else:
            return tztm, zt

    def logp(self, ztm, zt, t):
        mu = self.conditional_param(ztm)
        l  = gnlogpdf(zt, mu, self.noise_std.exp(), self.rho.exp())
        return l.sum(-1)

    def natsuff(self, ztm, zt, t):
        mu = self.conditional_param(ztm)
        return (- ((zt-mu)/self.noise_std.exp()).abs()**self.rho.exp()).sum(-1)

    def norm(self, ztm, t):
        n = ztm.shape[0]
        return (self.noise_std + np.log(2) + torch.lgamma(1.+1./self.rho.exp())).sum(-1).expand(n)

    def apply_constraints(self):
        #self.ps["rho"] = torch.log(1*torch.ones(self.D, requires_grad=RG,**CTX))
        return

    def step(self, zt, t):
        
        invalid = True
        while invalid:
            mu = self.conditional_param(zt)
            zt  = randgn(mu, self.noise_std.exp(), self.rho.exp())
            d  = (zt - mu)
            invalid = ((d==0).any())
        return zt

    def step_n(self, zt, t, n):
                                    
        mu = self.conditional_param(zt)
        return randgn(mu, self.noise_std.exp(), self.rho.exp(), size=[n])
         
    def dlogp(self, ztm, zt, t):
        
        n = ztm.shape[0]

        rho = self.rho.exp()
        noise_std = self.noise_std.exp()

        ty, m = self.conditional_param(ztm, keep=True)
        d = (zt - m) / noise_std
        dstd = -1. + rho * d.abs()**rho
        drho = (1. + 1./ rho).digamma() / rho - d.abs()**rho * d.abs().log() * rho
        dA   = (rho * d.abs()**(rho-1) * d.sign() / noise_std)[...,None] * ty[...,None,:]
        dA   = dA.reshape(n, -1)

        dl = torch.cat([dstd, drho, dA], -1)
        return dl

    def dnatsuff(self, ztm, zt, t):
        n = ztm.shape[0]

        rho = self.rho.exp()
        noise_std = self.noise_std.exp()

        ty, m = self.conditional_param(ztm, True)
        d = (zt - m) / noise_std
        dstd = rho * d.abs()**rho
        drho = - d.abs()**rho * d.abs().log() * rho
        dA   = (rho * d.abs()**(rho-1) * d.sign() / noise_std)[...,None] * ty[...,None,:]
        dA   = dA.reshape(n, -1)
        
        dl = torch.cat([dstd, drho, dA], -1)
        return dl

    def dnorm(self, ztm, t):

        n = ztm.shape[0]

        rho = self.rho.exp()
        noise_std = self.noise_std.exp()
        
        ones = torch.ones(n, self.D, **CTX)
        m    = self.conditional_param(ztm)
        dstd = 1. * ones
        drho = (- (1. + 1./ rho).digamma() / rho) * ones
        dA   = torch.zeros(n, np.prod(self.A.shape), **CTX)

        dl = torch.cat([dstd, drho, dA], -1)
        return dl

class LinearLaplaceLatent(Latent):
    
    name = "Lap"

    def __init__(self, A, noise_std):
        
        super().__init__()
        self.D = len(noise_std)
        self.ps["noise_std"] = torch.tensor(np.log(noise_std), requires_grad=RG, **CTX)
        self.ps["A"]   = torch.tensor(A, requires_grad=RG, **CTX)
        self.string = self.name + "_[%d,%d]_%.1f" % (A.shape[0], A.shape[1], noise_std[0])

    def __str__(self):
        return self.string

    def logp(self, ztm, zt, t):
        mu = self.conditional_param(ztm)
        l  = Laplace(mu, self.noise_std.exp()).log_prob(zt).sum(-1)
        return l

    def natsuff(self, ztm, zt, t):
        mu = self.conditional_param(ztm)
        return (- ((zt-mu)/self.noise_std.exp()).abs()).sum(-1)

    def norm(self, ztm, t):
        n = ztm.shape[0]
        return (self.noise_std + np.log(2) ).sum(-1).expand(n)
    def conditional_param(self, ztm, keep=False):
        
        tztm = ztm
        tztm = ztm.clamp(-10,10)
        tztm = 5.0 * torch.tanh(0.2 * ztm)

        zt = torch.matmul(tztm, self.A.t())
        if not keep:
            return zt
        else:
            return tztm, zt

    def apply_constraints(self):
        self.noise_std.data.clamp_(min=-5,max=np.log(1))
        return

    def step(self, zt, t):
        
        mu = self.conditional_param(zt)
        return Laplace(mu, self.noise_std.exp()).sample()

    def step_n(self, zt, t, n):
                                    
        mu = self.conditional_param(zt)
        return Laplace(mu, self.noise_std.exp()).sample([n])
         
    def dlogp(self, ztm, zt, t):
        
        n = ztm.shape[0]

        noise_std = self.noise_std.exp()

        ty, m = self.conditional_param(ztm, keep=True)
        d = (zt - m) / noise_std
        dstd = -1. + d.abs()
        dA   = (d.sign() / noise_std)[...,None] * ty[...,None,:]
        dA   = dA.reshape(n, -1)

        dl = torch.cat([dstd, dA], -1)
        return dl

    def dnatsuff(self, ztm, zt, t):
        n = ztm.shape[0]

        noise_std = self.noise_std.exp()

        ty, m = self.conditional_param(ztm, True)
        d = (zt - m) / noise_std
        dstd = d.abs()
        dA   = (d.sign() / noise_std)[...,None] * ty[...,None,:]
        dA   = dA.reshape(n, -1)
        
        dl = torch.cat([dstd, dA], -1)
        return dl

    def dnorm(self, ztm, t):

        n = ztm.shape[0]

        noise_std = self.noise_std.exp()
        
        ones = torch.ones(n, self.D, **CTX)
        m    = self.conditional_param(ztm)
        dstd = 1. * ones
        dA   = torch.zeros(n, np.prod(self.A.shape), **CTX)

        dl = torch.cat([dstd, dA], -1)
        return dl

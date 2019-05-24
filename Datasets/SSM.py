import torch, numpy as np
from collections import OrderedDict
from .settings import RG, CTX

class Parameters(OrderedDict):
    """A helper class to index into a parameter vector."""
    
    @property
    def shapes(self):
        return [v.shape for v in self.values()]

    @property
    def nparam(self):
        return sum([v.numel() for v in self.values()])

    def __iadd__(self, g):
        
        assert g.numel() == self.nparam
        i = 0
        for k, v in self.items():
            nel = v.numel()
            shape = v.shape
            self[k] += g[i:i+nel].view(*shape)
            i += nel
        return self

    def __imul__(self, a):
        for k, v in self.items():
            self[k] *= a
        return self
        

    def vec(self):
        if len(self.values()) == 0:
            return torch.tensor([], *CTX)
        else:
            return torch.cat([t.flatten() for t in self.values()], -1)
            
    def load(self, ps):
        
        assert len(ps) == self.nparam 
        
        idx = 0
        for k, v in self.items():
            shape = v.shape
            length= np.prod(shape)
            self[k].data = torch.tensor(ps[idx:idx+length].reshape(shape), **CTX)
            idx += length


class Distribution(object):

    def __init__(self):
        self.ps = Parameters()

    def sample(self, *args):
        raise NotImplementedError

    def logp(self, *args):
        return self._logp(*args)

    def dlogp(self, *args):
        return self._dlogp(*args)

    def dnatsuff(self, *args):
        return self._dnatsuff(*args)

    def apply_constraints(self):
        return
        
    def dlogp_tensor(self, *args):
        
        z, x = args[:2]
        other_args = args[2:]
        if z.ndimension() -x.ndimension() == 1:
            x = x[...,None,:]
            x = x.expand( [-1]*(x.ndimension()-2) + [z.shape[-2]] + [-1])
            assert x.shape[:-1] == z.shape[:-1]

        shape = z.shape[:-1]
        x = x.reshape((np.prod(shape),) + x.shape[-1:])
        z = z.reshape((np.prod(shape),) + z.shape[-1:])
        g = self.dlogp(z, x, *other_args)
        return g.reshape(shape+g.shape[-1:])
        
    def dnatsuff_tensor(self, *args):
        
        z, x = args[:2]
        other_args = args[2:]
        if z.ndimension() -x.ndimension() == 1:
            x = x[...,None,:]
            x = x.expand( [-1]*(x.ndimension()-2) + [z.shape[-2]] + [-1])
            assert x.shape[:-1] == z.shape[:-1]

        shape = z.shape[:-1]
        x = x.reshape((np.prod(shape),) + x.shape[-1:])
        z = z.reshape((np.prod(shape),) + z.shape[-1:])
        g = self.dnatsuff(z, x, *other_args)
        return g.reshape(shape+g.shape[-1:])

    def dnorm(self, *args):
        return self._dnorm(*args)

    def __getattr__(self, name):
        if name.startswith('_'):
            raise AttributeError(name)
        if name in self.ps:
            return self.ps[name]
        else:
            raise AttributeError(name)

class Latent(Distribution):

    def sample(self, T, N):
        raise NotImplementedError

    def sample(self, z0, T, N):
        zs = torch.zeros(T,N,self.D, **CTX)
        zs[0,...] = self.step(z0, 0)
        for i in range(1,T):
            zs[i,...] = self.step(zs[i-1,...], i)
        return zs

    def logp(self, ztm1, zt, t=None):
        raise NotImplementedError

    def dlogp(self, ztm1, zt, t=None):
        raise NotImplementedError

    def dnorm(self, ztm1, t=None):
        raise NotImplementedError

    def dnatsuff(self, ztm1, zt, t=None):
        raise NotImplementedError
        

class Observation(Distribution):

    def logp(self, z, x):
        raise NotImplementedError

    def dlogp(self, z, x):
        raise NotImplementedError

class Prior(Distribution):

    def logp(self, x):
        raise NotImplementedError

    def dlogp(self, x):
        raise NotImplementedError

class StateSpaceModel(object):
    
    def __init__(self, latent, observation, init_noise_std, init_fun = None):

        self.latent=latent
        self.observation=observation
        self.init_noise_std=init_noise_std
        if init_fun is None:
            self.sample_initial = lambda N: torch.randn(N, self.latent.D, **CTX) * self.init_noise_std
        else:
            self.sample_initial = init_fun

    def _insert_xt(self, s, xt):
        
        if isinstance(xt, list):
            for xi, x in enumerate(xt):
                s["xt%d"%xi] = x
            s["xt"] = x
        else:
            s["xt"] = xt
        return s
        
    def sample(self, T, N):
        z0 = self.sample_initial(N)
        zt = self.latent.sample(z0, T, N)
        #zt = torch.r_[z0[None,...],zt]       
        xt = self.observation.sample(zt)
        ztm1 = torch.cat((z0[None,...],zt[:-1 ,...]), 0)
        s = {"zt": zt, "zt-1": ztm1}
        
        return self._insert_xt(s, xt)
    
    def step(self, ztm1, t, d=None):
        
        if d is None:
            d = {}
            
        d["zt-1"] = ztm1
        
        zt = self.latent.step(ztm1,t)
        xt  = self.observation.sample(zt)
        d["zt"] = zt

        return self._insert_xt(d, xt)
    
    @property
    def ps(self):
        return torch.cat([self.latent.ps.vec(), self.observation.ps.vec()])
        
    def filename(self, disc=None):
         
        fn = "SSM_%s_%s" % (str(self.latent), str(self.observation))
        if disc is not None:
            fn = "_".join([fn, str(disc)])
        return fn
        
    def save(self, disc=None):
        
        fn = "ckpts/snapshots/" + self.filename(disc)
        ps = self.ps.detach().cpu().numpy()
        
        np.savez(fn,ps=ps)
        
    def load(self, ps=None, disc=None):
    
        if ps is None:
            fn = "ckpts/snapshots/" + self.filename(disc) + ".npz"
            ps = np.load(fn)["ps"]
        elif isinstance(ps, np.ndarray):
            assert (len(ps) == len(self.ps))
            Z_length = len(self.latent.ps.vec())
            self.latent.ps.load(ps[:Z_length])
            self.observation.ps.load(ps[Z_length:])
        else:
            raise(NameError("input argument type wrong"))


# TODO 
class DeepModel(object):
    
    def __init__(self, dists, name, seed=None):
        
        self.name  = name
        self.dists = dists
        self.depth = len(dists)
        self.seed  = seed

        self.ps = torch.concatenate([c.ps for c in self.dists])
        pps     = OrderedDict([("p"+str(i), self.dists[i].pp) for i in range(self.depth)])
        self.pp = ParameterParser.from_pps(pps)
        self.pidxs = OrderedDict([(k, v[0]) for (k,v) in self.pp.idxs_and_shapes.items()])

    def logp(self, *args):

        assert len(args)  == self.depth

        logp =  self.dists[0].logp(args[0])

        for i in range(1,self.depth):
            logp += self.dists[i].logp(args[i-1], args[i])
        return logp

    def dlogp(self, *args):
        
        assert len(args) == self.depth
        
        dlogps = []
        dlogps.append(self.dists[0].dlogp(args[0]))

        for i in range(1,self.depth):
            dlogps.append( self.dists[i].dlogp(args[i-1], args[i]) )
        dlogp = torch.concatenate(dlogps, axis=1)
        return dlogp

    def sample(self, N, no_latent=False):
        
        data = OrderedDict()
        d = self.dists[0].sample(N)

        if not no_latent:
            data["x0"] = d

        for i in range(1, self.depth):

            d = self.dists[i].sample(d)  

            if i < self.depth-1:
                if not no_latent:
                    data["x"+str(i)]=d
            else:
                data["x"+str(i)]=d
                

        return data
    
    def sync_ps(self):
        for i in range(self.depth):
            self.dists[i].ps = self.pp.get(self.ps, "p"+str(i))
            self.dists[i].sync_ps()

    def default_model_name(self):
        
        fn = "model_params/"+self.name + "/"
        if hasattr(self.dists[0], "depth"):
            Ds_obs = self.dists[-1].Ds
            fn += "_".join(map(str,Ds_obs))
        else:
            fn += "_".join(["%d"%d.D for d in self.dists])
        if self.seed is not None:
            fn += "_%02d"%self.seed
        
        return fn
   
    def save(self, desc=""): 
        
        fn = self.default_model_name()
        if len(desc) != 0:
            fn += "_"+desc
        torch.savez(fn, gen=self.ps)
        return fn

    def load(self, desc=""):

        fn = self.default_model_name()
        if len(desc) != 0:
            fn += "_"+desc
        self.ps = torch.load(fn+".torchy")["gen"]
        self.sync_ps()


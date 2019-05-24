from __future__ import division
import numpy as np
import torch
from collections import OrderedDict, defaultdict
from abc import ABCMeta, abstractmethod
from settings import CTX
from copy import deepcopy

fun_approx_lam = 1e-3

def array_to_tensor(*args):
    return (torch.tensor(a, **CTX) for a in args)

class History(object):
    
    def __init__(self):
        self.data = defaultdict(lambda: [])
    
    def __getitem__(self, k):
        return self.data[k]

    def __setitem__(self, k, v):
        self.data[k] = v

    def retain(self, n):
        if n is None:
            if len(list(self.data.items())[0][1]) > 300:
                raise NameError("storing too much history, set nretain to a number")
            return
        for k, v in self.data.items():
            self.data[k] = self.data[k][-n:]
    
    def log(self, d, keys=[]):

        if len(keys) == 0:
            keys = list(d.keys())
        for k, v in d.items():
            if k in keys:
                if isinstance(v, torch.Tensor):
                    v = v.cpu().detach().numpy()
                self.data[k].append(v)   

    def finalise(self):
        data = {}
        for k, v in self.data.items():
            data[k] = np.asarray(v)
        self.data = data

class DDCSSM(object):

    def log(self, keys=[]):
        
        self.wake_history.log(self.w_data, keys=keys)
        self.sleep_history.log(self.s_data, keys=keys)

        if not self.smoothing:
            self.wake_history.retain(self.nretain)
            self.sleep_history.retain(self.nretain)
    
    @property
    def sleep_data(self):
        return self.s_data 

    @property
    def wake_data(self):
        return self.w_data 

    def finalise(self):

        self.wake_history.finalise()
        self.sleep_history.finalise()
        self.param_history.finalise()

    def approx_E(self, var_name, post):
        
        out = {}
        for fn in self.reg.fs[var_name].alphas.keys():
            if fn[0] == "E":
                    out[fn]= self.reg.fs[var_name].approx_E(fn, post)
        return out

    def compute_forward_statistics(self):
        
        Es = self.approx_E("zt", self.w_data["mzt_x1:t"])
        for fn in Es.keys():
            self.w_data[fn+"t_x1:t"] = Es[fn]
            
        Es = self.approx_E("zt-1", self.w_data["mzt-1_x1:t"])
        for fn in Es.keys():
            self.w_data[fn+"t-1_x1:t"] = Es[fn]

    def gradient_step(self):
        

        if not any(self.plastic):
            return 
        
        ps = self.model.ps.cpu()
        
        if self.plastic[0]:
            gs = self.w_data["dlogpZ"]
            gs = torch.mean(gs,tuple(range(gs.ndimension()-1))).detach()
            self.opt[0].step(self.model.latent.ps, gs, self.it)
            self.model.latent.apply_constraints()

        if self.plastic[1]:
            gs = self.w_data["dlogpX"]
            gs = torch.mean(gs,tuple(range(gs.ndimension()-1))).detach()
            self.opt[1].step(self.model.observation.ps, gs, self.it)
            self.model.observation.apply_constraints()
        
        pd = dict(ps=ps)
        self.param_history.log(pd)

        self.it += 1

    def initialise(self, x0, weights_std=0.0, init_n=1000, init_range=None):

        self.t = 0
        self.it = 0
        self.niter = 0

        # sleep data
        
        z0 = self.model.sample_initial(self.nsleep)

        if init_range is not None:
            z0.clamp_(*init_range)

        self.w_data  = {}
        self.wake_history = History()
        self.sleep_history = History()
        self.param_history = History()
        self.s_data = self.model.step(z0, self.t)
        psi = self.reg.transform_data(self.s_data, ["mzt"])["mzt"]
        self.s_data["mzt_x1:t"]     = psi[None,...].expand(self.nsleep,-1)
        self.w_data["mzt_x1:t"]     = psi[None,...].expand(self.nwake ,-1)
        
        psi = self.reg.transform_data(self.s_data, ["mzt-1"])["mzt-1"]
        self.s_data["mzt-1_x1:t-1"] = psi[None,...].expand(self.nsleep,-1)
        self.w_data["mzt-1_x1:t-1"] = psi[None,...].expand(self.nwake ,-1)

        self.w_data[ "xt"] = x0

    def filter(self):
        return self.sleep_train_test()

    def learn(self, **kwargs):
        return self.wake_filter(**kwargs)

class GenericDDCSSM(DDCSSM):
    
    __metaclass__ = ABCMeta
    
    @abstractmethod
    def __init__(self, model, reg, opt=None, nsleep=1000, nwake=100, backup_nsleep=1000, 
                g_type="exp", s_type="sample",
                smoothing=False, plastic=[False, False], stationary=True, factorise=True, 
                nretain = None):
        
        self.model = model
        self.reg = reg
        self.nsleep = nsleep
        self.nwake  = nwake
        self.backup_nsleep = backup_nsleep
        self.s_data = None
        self.w_data  = None
        self.plastic=plastic
        if not isinstance(opt, list):
            if opt is not None:
                self.opt = [deepcopy(opt), deepcopy(opt)]
            else:
                self.opt = None
        else:
            self.opt = opt
        self.smoothing=smoothing
        self.stationary = stationary
        self.factorise=factorise
        self.g_type = g_type
        self.s_type = s_type
        self.nretain = nretain

        if opt is None:
            self.plastic = [False, False]
        
        self.wake_history = History()
        self.sleep_history = History()
        # number of gradient iterations
        self.it = 0
        # time step of the chain
        self.t = 0

    def propagate(self, x=None):
        
        self.t += 1
        
        s_data = self.s_data.copy()
        self.s_data = {}
        if "mzt_x1:t" in s_data: 
            self.s_data["mzt-1_x1:t-1"] = s_data["mzt_x1:t"]
        self.s_data['zt'] = s_data['zt']
        self.model.step(self.s_data["zt"], self.t, d=self.s_data)
        
        if x is not None:
            self.w_data = {"mzt-1_x1:t-1": self.w_data["mzt_x1:t"]}
            self.w_data["xt"]=x

    def smooth(self, nvalid=0):

        if not "mzt-1_x1:t-1,zt->zt-1" in self.reg.Ws and self.stationary:
            T = len(self.sleep_history["mzt-1_x1:t-1"])-1
            self.s_data["zt"]   = torch.tensor(self.sleep_history["zt"][T], **CTX)
            self.s_data["zt-1"] = torch.tensor(self.sleep_history["zt-1"][T], **CTX)
            self.s_data["mzt-1_x1:t-1"] = torch.tensor(self.sleep_history["mzt-1_x1:t-1"][T], **CTX)
            self.reg.train_weights(self.s_data, "mzt-1_x1:t-1,zt->zt-1", nvalid=nvalid, clip=True)
        
        self.w_data["mzt_x1:T"] = torch.tensor(self.wake_history["mzt_x1:t"][-1], **CTX)
        self.s_data["mzt_x1:T"] = torch.tensor(self.sleep_history["mzt_x1:t"][-1], **CTX)
        
        self.w_data["mz"] = 0.0
        self.w_data["mz-1"] = 0.0

        for t in range(self.t, -1, -1):

            if not self.stationary:
                for v in ["zt-1","zt","mzt-1_x1:t-1"]:
                    self.s_data[v]   = torch.tensor(self.sleep_history[v][t], **CTX)
                self.reg.train_weights(self.s_data, "mzt-1_x1:t-1,zt->zt-1", nvalid=nvalid, clip=True)

            self.w_data["mzt-1_x1:t-1"] = torch.tensor(self.wake_history["mzt-1_x1:t-1"][t], **CTX)
            self.w_data["mzt-1_x1:T"]   = self.reg.predict(self.w_data, "mzt-1_x1:t-1,zt->zt-1",
                                                              inputs="mzt-1_x1:t-1,mzt_x1:T", clip=True)
            
            self.compute_backward_statistics()
            self.log(["mzt_x1:T", "mzt-1_x1:T", "Ezt_x1:T", "Ezt-1_x1:T"])
            self.w_data["mzt_x1:T"]   = self.w_data["mzt-1_x1:T"]
        
        for v in ["mzt_x1:T", "mzt-1_x1:T", "Ezt-1_x1:T", "Ezt_x1:T"]:
            self.wake_history[v] = self.wake_history[v][::-1]
            if 'm' ==  v[0]:
                self.wake_history[v] = [v for v in self.wake_history[v]]
        
    def compute_backward_statistics(self):
        
        Es = self.approx_E("zt", self.w_data["mzt_x1:T"])
        for fn in Es.keys():
            self.w_data[fn+"t_x1:T"] = Es[fn]

        Es = self.approx_E("zt-1", self.w_data["mzt-1_x1:T"])
        for fn in Es.keys():
            self.w_data[fn+"t-1_x1:T"] = Es[fn]

    
    def wake_filter(self, nvalid=0, log=True):
        
        if not any(self.plastic):
            return
        
        # if number of sleep sample is too small, use fake sample or feature grid

        if self.nsleep<self.backup_nsleep:
            ori_s_data = {}  
            ztm1 = self.s_data["zt-1"]
            ztm1 = ztm1[np.random.choice(self.nsleep, self.backup_nsleep)]
            d  = self.model.step(ztm1, self.t)
            for k in ["zt-1", "zt", "xt"]:
                ori_s_data[k]   = self.s_data[k]
                self.s_data[k]  = d[k]

        ztm1 = self.s_data["zt-1"]
        zt   = self.s_data["zt"]
        xt   = self.s_data["xt"]

        if self.plastic[0]:
            
            if self.g_type=="exp":
                self.s_data["dnormZ"]    = self.model.latent.dnorm(ztm1,self.t)
                if self.s_type == "sample":
                    self.reg.train_weights_closed(self.s_data, "zt-1->dnormZ", nvalid=nvalid)
                    self.w_data["dnormZ"]    = self.reg.predict(self.w_data, "zt-1->dnormZ", inputs="mzt-1_x1:t")
                elif self.s_type == "mean":
                    self.reg.train_weights_closed(self.s_data, "mzt-1_x1:t->dnormZ", nvalid=nvalid)
                    self.w_data["dnormZ"]    = self.reg.predict(self.w_data, "mzt-1_x1:t->dnormZ")

                
            if self.g_type=="exp":

                self.s_data["dnatsuffZ"] = self.model.latent.dnatsuff(ztm1, zt, self.t)

                if self.s_type=="sample":
                    self.reg.train_weights_closed(self.s_data, "zt-1,xt->dnatsuffZ", nvalid=nvalid, clip=True)
                    self.w_data["dnatsuffZ"] = self.reg.predict(self.w_data, "zt-1,xt->dnatsuffZ", 
                                                                inputs="mzt-1_x1:t,xt", clip=True)          
                    '''
                    self.s_data["fzz"] = self.fs["zt-1zt"](tensor.cat([ztm1, zt], 0))
                    self.reg.train_weights_closed(self.s_data, "mzt-1_x1:t-1,xt->fzz", nvalid=nvalid, clip=True)
                    self.reg.train_weights_closed(self.s_data, "fzz->dnatsuffZ", nvalid=nvalid, clip=True)
                    self.w_data["mzz"] = self.reg.predict(self.w_data, "zt-1,xt->fzz", clip=True, inputs="mzt-1_x1:t-1,xt")
                    self.w_data["dnatsuffZ"] = self.reg.predict(self.w_data, "fzz->dnatsuffZ", inputs="mzz", clip=True)          
                    '''

                elif self.s_type=="mean":
                    self.reg.train_weights_closed(self.s_data, "mzt-1_x1:t-1,xt->dnatsuffZ", nvalid=nvalid, clip=True)
                    self.w_data["dnatsuffZ"] = self.reg.predict(self.w_data, "mzt-1_x1:t-1,xt->dnatsuffZ", clip=True)

                self.w_data["dlogpZ"]    = self.w_data["dnatsuffZ"] - self.w_data["dnormZ"]
                if log:
                    self.wake_history.log(self.w_data, keys=["dlogpZ", "dnatsuffZ", "dnormZ"])

            elif self.g_type=="logp":
                
                    
                self.s_data["dlogpZ"] = self.model.latent.dlogp(ztm1, zt, self.t)

                if self.s_type=="sample":
                    self.reg.train_weights_closed(self.s_data, "zt-1,xt->dlogpZ", nvalid=nvalid, clip=True)
                    self.w_data["dlogpZ"] = self.reg.predict(self.w_data, "zt-1,xt->dlogpZ", inputs="mzt-1_x1:t,xt", clip=True)

                elif self.s_type=="mean":
                    self.reg.train_weights_closed(self.s_data, "mzt-1_x1:t-1,xt->dlogpZ", nvalid=nvalid, clip=True)
                    self.w_data["dlogpZ"] = self.reg.predict(self.w_data, "mzt-1_x1:t-1,xt->dlogpZ", clip=True)
                if log:
                    self.wake_history.log(self.w_data, keys=["dlogpZ"])


        if self.plastic[1]:
            
            reg = self.reg
            if self.g_type == "exp":

                self.s_data["dnatX"]     = self.model.observation.dnat(zt)
                self.s_data["dnormX"]    = self.model.observation.dnorm(zt)
                
                if self.s_type == "sample":
                    self.reg.train_weights_closed(self.s_data, "zt->dnatX", nvalid=nvalid)
                    self.reg.train_weights_closed(self.s_data, "zt->dnormX", nvalid=nvalid)
                    self.w_data["dnatX"]     = self.reg.predict(self.w_data, "zt->dnatX", inputs="mzt_x1:t")
                    self.w_data["dnormX"]    = self.reg.predict(self.w_data, "zt->dnormX", inputs="mzt_x1:t")

                elif self.s_type == "mean":
                    self.reg.train_weights_closed(self.s_data, "mzt_x1:t->dnatX", nvalid=nvalid)
                    self.reg.train_weights_closed(self.s_data, "mzt_x1:t->dnormX", nvalid=nvalid)
                    self.w_data["dnatX"]     = self.reg.predict(self.w_data, "mzt_x1:t->dnatX")
                    self.w_data["dnormX"]    = self.reg.predict(self.w_data, "mzt_x1:t->dnormX")
                    
                suff = self.model.observation.suff(self.w_data["xt"])
                self.w_data["dnatsuffX"] = self.model.observation.dnatsuff_from_dnatsuff(
                                                    self.w_data["dnatX"], 
                                                    suff)
                                                    
                self.w_data["dlogpX"]    = self.w_data["dnatsuffX"] - self.w_data["dnormX"]
                if log:
                    self.wake_history.log(self.w_data, keys=["dlogpX", "dnatX", "dnormX", "dnatsuffX"])

            elif self.g_type == "logp":

                self.s_data["dlogpX"]     = self.model.observation.dlogp(zt, xt)
                self.reg.train_weights_closed(self.s_data, "zt,xt->dlogpX", nvalid=nvalid)
                self.w_data["dlogpX"]     = self.reg.predict(self.w_data, "zt,xt->dlogpX", inputs="mzt_x1:t,xt")

                if log:
                    self.wake_history.log(self.w_data, keys=["dlogpX"])
        
        if self.nsleep<self.backup_nsleep:
            for k,v in ori_s_data.items():
                self.s_data[k] = v

    def wake_smooth(self, nvalid=0):
        
        if not np.any(self.plastic):
            return
        
        self.s_data = {}
        self.w_data = {}
        if "dlogpZ" in self.wake_history.data:
            self.wake_history.data.pop("dlogpZ")
        if "dlogpX" in self.wake_history.data:
            self.wake_history.data.pop("dlogpX")
        

        '''
        if self.plastic[0]:
            self.w_data["dlogpZ"] = []
        if self.plastic[1]:
            self.w_data["dlogpX"] = []
        '''

        lat = self.model.latent
        obs = self.model.observation

        if self.stationary:
            T = len(self.sleep_history["mzt_x1:t"])
            for k in ["zt", "xt", "zt-1", "mzt-1_x1:t-1"]:
                self.s_data[k] = torch.tensor(self.sleep_history[k][T-1], **CTX)
            self.s_data["dnatsuffZ"] = lat.dnatsuff(self.s_data["zt-1"], self.s_data["zt"], 0)
            self.s_data["dnormZ"] = lat.dnorm(self.s_data["zt-1"], 0)

            self.reg.train_weights_closed(self.s_data, "mzt-1_x1:t-1,zt->dnatsuffZ", nvalid=nvalid)
            self.reg.train_weights_closed(self.s_data, "zt-1->dnormZ", nvalid=nvalid, lam = fun_approx_lam)

            self.s_data["dnatX"]     = obs.dnat(self.s_data["zt"])
            self.s_data["dnormX"]    = obs.dnorm(self.s_data["zt"])

            self.reg.train_weights_closed(self.s_data, "zt->dnatX", lam=fun_approx_lam)
            self.reg.train_weights_closed(self.s_data, "zt->dnormX", lam=fun_approx_lam)

        for t in range(self.t+1):

            self.w_data["xt"]           = torch.tensor(self.wake_history["xt"][t], **CTX)
            self.w_data["mzt_x1:T"]     = torch.tensor(self.wake_history["mzt_x1:T"][t], **CTX).clamp(-1.5,1.5)
            self.w_data["mzt_x1:t"]     = torch.tensor(self.wake_history["mzt_x1:t"][t], **CTX).clamp(-1.5,1.5)
            self.w_data["mzt-1_x1:t-1"] = torch.tensor(self.wake_history["mzt-1_x1:t-1"][t], **CTX).clamp(-1.5,1.5)
            self.w_data["mzt-1_x1:T"]   = torch.tensor(self.wake_history["mzt-1_x1:T"][t], **CTX).clamp(-1.5,1.5)

            if self.plastic[0]:
                
                if not self.stationary:
                    for k in ["zt", "xt", "zt-1", "mzt-1_x1:t-1"]:
                        self.s_data[k] = torch.tensor(self.sleep_history[k][t], **CTX)
                    ztm1, zt, xt = self.s_data["zt-1"], self.s_data["zt"], self.s_data["xt"]
                    if self.g_type=="exp":
                        self.s_data["dnormZ"] = lat.dnorm(ztm1, t)
                        self.s_data["dnatsuffZ"] = lat.dnatsuff(ztm1, zt, t)
                        self.reg.train_weights_closed(self.s_data, "mzt-1_x1:t-1,zt->dnatsuffZ", nvalid=nvalid, clip=True)
                        self.reg.train_weights_closed(self.s_data, "zt-1->dnormZ", nvalid=nvalid, lam=fun_approx_lam)
                    elif self.g_type=="logp":
                        self.s_data["dlogp"] = lat.dlogp(ztm1, zt, t)
                        self.reg.train_weights_closed(self.s_data, "mzt-1_x1:t-1,zt->dlogp", nvalid=nvalid, clip=True)
                
                if self.g_type == "exp":
                    self.w_data["dnatsuffZ"] = self.reg.predict(self.w_data, "mzt-1_x1:t-1,zt->dnatsuffZ", inputs="mzt-1_x1:t-1,mzt_x1:T", clip=True)
                    self.w_data["dnormZ"]    = self.reg.predict(self.w_data, "zt-1->dnormZ", inputs="mzt-1_x1:T")

                    self.w_data["dlogpZ"]   = self.w_data["dnatsuffZ"] - self.w_data["dnormZ"]
                else:
                    self.w_data["dlogpZ"] = self.reg.predict(self.w_data, "mzt-1_x1:t-1,zt->dlogp", inputs="mzt-1_x1:t-1,mzt_x1:T", clip=True)

                self.wake_history.log(self.w_data, keys=["dlogpZ"])

            if self.plastic[1]:
                
                if not self.stationary:

                    for k in ["zt", "xt"]:
                        self.s_data[k] = torch.tensor(self.sleep_history[k][t], **CTX)
                    zt, xt = self.s_data["zt"], self.s_data["xt"]
                    self.s_data["xt"]        = xt
                    self.s_data["dnatX"]     = self.model.observation.dnat(zt)
                    self.s_data["dnormX"]    = self.model.observation.dnorm(zt)

                    self.reg.train_weights_closed(self.s_data, "zt->dnatX", lam=fun_approx_lam, nvalid=nvalid)
                    self.reg.train_weights_closed(self.s_data, "zt->dnormX", lam=fun_approx_lam, nvalid=nvalid)

                self.w_data["dnatX"]     = self.reg.predict(self.w_data, "zt->dnatX",  inputs="mzt_x1:T")
                self.w_data["dnormX"]    = self.reg.predict(self.w_data, "zt->dnormX", inputs="mzt_x1:T")
                suff = self.model.observation.suff(torch.tensor(self.wake_history["xt"][t], **CTX))
                self.w_data["dnatsuffX"] = self.model.observation.dnatsuff_from_dnatsuff(
                                                    self.w_data["dnatX"], 
                                                    suff)
                                                    
                self.w_data["dlogpX"]   = self.w_data["dnatsuffX"] - self.w_data["dnormX"]
                self.wake_history.log(self.w_data, keys=["dlogpX"])
                
        if self.plastic[0]:
            self.w_data["dlogpZ"] = torch.tensor(np.mean(self.wake_history["dlogpZ"], 0), **CTX)
        if self.plastic[1]:
            self.w_data["dlogpX"] = torch.tensor(np.mean(self.wake_history["dlogpX"], 0), **CTX)
    
        
    def sleep_train_test(self):
        raise(NotImplementedError())    

    def sleep_train(self):
        raise(NotImplementedError())    

    def sleep_test(self):
        raise(NotImplementedError())    

    def compute_approximate_gradients(self):
        pass
    

class DistributionRegression(GenericDDCSSM):
    
    name="distreg"

    def __init__(self, *args, **kwargs):
        super(DistributionRegression, self).__init__(*args, **kwargs)
    
    def sleep_train_test(self, nvalid=0):

        self.reg.train_weights(self.s_data, "mzt-1_x1:t-1,xt->zt", nvalid=nvalid, clip=True)
        self.reg.train_weights(self.s_data, "mzt-1_x1:t-1,xt->zt-1", nvalid=nvalid, clip=True)
        self.s_data["mzt_x1:t"] = self.reg.predict(self.s_data, "mzt-1_x1:t-1,xt->zt", clip=True)
        self.s_data["mzt-1_x1:t"] = self.reg.predict(self.s_data, "mzt-1_x1:t-1,xt->zt-1", clip=True)
        self.w_data[ "mzt_x1:t"] = self.reg.predict(self.w_data,  "mzt-1_x1:t-1,xt->zt", clip=True)
        self.w_data[ "mzt-1_x1:t"] = self.reg.predict(self.w_data,  "mzt-1_x1:t-1,xt->zt-1", clip=True)

        self.compute_forward_statistics()
        self.log()

    def sleep_test(self, w_data=None):
        if w_data is None:
            w_data = self.w_data
        
        self.s_data[ "mzt_x1:t"] = self.reg.predict(self.s_data,  "mzt-1_x1:t-1,xt->zt", clip=True)   
        self.s_data[ "mzt-1_x1:t"] = self.reg.predict(self.s_data,  "mzt-1_x1:t-1,xt->zt-1", clip=True)   

        self.w_data[ "mzt_x1:t"] = self.reg.predict(w_data,  "mzt-1_x1:t-1,xt->zt", clip=True)   
        self.w_data[ "mzt-1_x1:t"] = self.reg.predict(w_data,  "mzt-1_x1:t-1,xt->zt-1", clip=True)   
        self.compute_forward_statistics()
        self.log()

class SampleRegression(GenericDDCSSM):
    
    name="samreg"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def sleep_train_test(self, nvalid=0):

        self.reg.train_weights(self.s_data, "mzt-1_x1:t-1,xt->zt-1", nvalid=nvalid, clip=True)
        self.reg.train_weights(self.s_data, "zt-1,xt->zt", nvalid=nvalid, clip=True)
        self.s_data["mzt-1_x1:t"] = self.reg.predict(self.s_data, "mzt-1_x1:t-1,xt->zt-1", clip=True)
        self.s_data["mzt_x1:t"] = self.reg.predict(self.s_data, "zt-1,xt->zt", clip=True, 
                                                                inputs="mzt-1_x1:t,xt")
        self.w_data["mzt-1_x1:t"] = self.reg.predict(self.w_data, "mzt-1_x1:t-1,xt->zt-1", clip=True)
        self.w_data["mzt_x1:t"] = self.reg.predict(self.w_data, "zt-1,xt->zt", clip=True, 
                                                                inputs="mzt-1_x1:t,xt")

        self.compute_forward_statistics()
        self.log()


class PairDistributionRegression(GenericDDCSSM):
    
    name="pairdistreg"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def initialise(self, x0, weights_std=0.0, init_n=1000, init_range=None):

        self.t = 0
        self.it = 0
        self.niter = 0

        # sleep data
        
        z0 = self.model.sample_initial(self.nsleep)

        if init_range is not None:
            z0.clamp_(*init_range)

        self.w_data  = {}
        self.wake_history = History()
        self.sleep_history = History()
        self.param_history = History()
        self.s_data = self.model.step(z0, self.t)
        self.create_pair()
        psi = self.s_data["fzzt"].mean(0)
        self.s_data["mzzt-1_x1:t-1"]     = psi[None,...].expand(self.nsleep,-1)
        self.w_data["mzzt-1_x1:t-1"]     = psi[None,...].expand(self.nwake ,-1)
        
        self.w_data[ "xt"] = x0

    def compute_forward_statistics(self):
        
        Es = self.approx_E("zzt", self.w_data["mzzt_x1:t"])
        for fn in Es.keys():
            self.w_data[fn+"t_x1:t"] = Es[fn]

    def compute_forward_statistics_test(self):
        return compute_forward_statistics()        

    def create_pair(self):
        zzt = torch.cat([self.s_data["zt-1"], self.s_data["zt"]], -1)
        self.s_data["fzzt"] = self.reg.fs["zzt"](zzt)
    
    def sleep_train_test(self, nvalid=0):
        
        self.reg.train_weights(self.s_data, "mzzt-1_x1:t-1,xt->fzzt", nvalid=nvalid, clip=True)
        self.s_data["mzzt_x1:t"] = self.reg.predict(self.s_data,  "mzzt-1_x1:t-1,xt->fzzt", clip=True)
        self.w_data["mzzt_x1:t"] = self.reg.predict(self.w_data,  "mzzt-1_x1:t-1,xt->fzzt", clip=True)

        self.compute_forward_statistics()
        self.log()

    def propagate(self, x=None):
        
        assert 'zt-1' in self.s_data
        self.t += 1
        
        s_data = self.s_data.copy()
        self.s_data = {}
        if "mzzt_x1:t" in s_data: 
            self.s_data["mzzt-1_x1:t-1"] = s_data["mzzt_x1:t"]
        self.s_data['zt'] = s_data['zt']
        self.model.step(self.s_data["zt"], self.t, d=self.s_data)
        self.create_pair()
        
        if x is not None:
            self.w_data = {"mzzt-1_x1:t-1": self.w_data["mzzt_x1:t"]}
            self.w_data["xt"]=x


    def sleep_test(self, w_data=None):

        self.w_data["mzzt_x1:t"] = self.reg.predict(self.w_data,  "mzzt-1_x1:t-1,xt->fzzt", clip=True)

        self.compute_forward_statistics_test()
        self.log()

    def wake_filter(self, nvalid=0, log=True, lag = 0):
        
        if not any(self.plastic):
            return
        
        # if number of sleep sample is too small, use fake sample or feature grid

        if self.nsleep<self.backup_nsleep:
            ori_s_data = {}  
            if lag == 0:
                ztm1 = self.s_data["zt-1"]
            else:
                ztm1 = torch.tensor(self.sleep_history["zt-1"][-lag], **CTX)
            ztm1 = ztm1[np.random.choice(self.nsleep, self.backup_nsleep)]
            d  = self.model.step(ztm1, self.t)
            for k in ["zt-1", "zt", "xt", "fzzt"]:
                ori_s_data[k]   = self.s_data[k]
                self.s_data[k]  = d[k]
            self.create_pair()
        
        if lag == 0:
            ztm1 = self.s_data["zt-1"]
            zt   = self.s_data["zt"]
            xt   = self.s_data["xt"]
        else:   
            ztm1, zt, xt = array_to_tensor(*(self.sleep_history[k][-lag] for k in ["zt-1", "zt", "xt"]))


        if self.plastic[0]:
             
            self.s_data["dnatsuffZ"] = self.model.latent.dnatsuff(ztm1, zt, self.t-1)
            self.reg.train_weights_closed(self.s_data, "fzzt->dnatsuffZ", nvalid=nvalid)
            dnatsuff = self.reg.predict(self.w_data, "fzzt->dnatsuffZ", inputs="mzzt_x1:t")

            self.s_data["dnormZ"] = self.model.latent.dnorm(ztm1, self.t-1)
            self.reg.train_weights_closed(self.s_data, "fzzt->dnormZ", nvalid=nvalid)
            dnorm = self.reg.predict(self.w_data, "fzzt->dnormZ", inputs="mzzt_x1:t")
            self.w_data["dlogpZ"] = dnatsuff - dnorm
            if log:
                self.wake_history.log(self.w_data, keys=["dlogpZ"])


        if self.plastic[1]:
            
            reg = self.reg
            self.s_data["dnatX"]     = self.model.observation.dnat(zt)
            self.s_data["dnormX"]    = self.model.observation.dnorm(zt)
            
            self.reg.train_weights_closed(self.s_data, "fzzt->dnatX", nvalid=nvalid)
            self.reg.train_weights_closed(self.s_data, "fzzt->dnormX", nvalid=nvalid)
            self.w_data["dnatX"]     = self.reg.predict(self.w_data, "fzzt->dnatX", inputs="mzzt_x1:t")
            self.w_data["dnormX"]    = self.reg.predict(self.w_data, "fzzt->dnormX", inputs="mzzt_x1:t")

            suff = self.model.observation.suff(self.w_data["xt"])
            self.w_data["dnatsuffX"] = self.model.observation.dnatsuff_from_dnatsuff(
                                                self.w_data["dnatX"], 
                                                suff)
            self.w_data["dlogpX"]    = self.w_data["dnatsuffX"] - self.w_data["dnormX"]

            if log:
                self.wake_history.log(self.w_data, keys=["dlogpX"])
        
        if self.nsleep<self.backup_nsleep:
            for k,v in ori_s_data.items():
                self.s_data[k] = v

class DecayDistributionRegression(PairDistributionRegression):

    name="decaydistreg"

    def __init__(self, gammas, *args, look_back=0,**kwargs):
        super().__init__(*args, **kwargs)
        
        self.look_back=look_back
        self.gammas = torch.tensor(gammas, **CTX)
        self.ngamma = len(gammas)
        self.h = torch.zeros(self.reg.fs["zt"].M * self.ngamma, **CTX)
        '''
        self.Ws = []
        for i in range(self.ngamma):
            W = torch.randn(self.reg.fs["zt"].M, self.reg.fs["zt"].M, **CTX)
            W /= torch.svd(W, compute_uv=False)[1].max()
            self.Ws += W,
        self.Ws= torch.stack(self.Ws, -1)
        '''        
        self.W = torch.randn(self.reg.fs["zt"].M*self.ngamma, self.reg.fs["zt"].M*self.ngamma, **CTX)
        self.W /= torch.svd(self.W, compute_uv=False)[1].max()*1.01

    def create_pair(self, overwrite=True):

        z = self.s_data["zt"] 
        phi = self.phi(z, overwrite)
        if overwrite:
            self.s_data["fzzt"] = phi
            
    def phi(self, z, overwrite):
        
        n = z.shape[0]
        self.old_h = self.h + 0
        if self.h.dim()==1:
            self.h = self.h.expand((n,)+ self.h.shape)

        gamma = self.reg.fs["zt"](z)

        if overwrite:
            self.h = self.h @ self.W 
            self.h[...,:self.reg.fs["zt"].M] += gamma
            new_h = self.h
        else:
            new_h = self.old_h + 0
            new_h = new_h @ self.W 
            new_h[..., :self.reg.fs["zt"].M] = gamma
            
        return new_h

    def compute_forward_statistics(self):
        
        zt = self.s_data["zt"]
        s_data = {"zt": zt, "fzzt": self.s_data["fzzt"]}
        w  = self.reg.train_weights_closed(s_data, "fzzt->fzt", nvalid=100)
        self.w_data["Ezt_x1:t"] = self.reg.predict(self.w_data, "fzzt->fzt", inputs="mzzt_x1:t")
        
        look_back = min(self.look_back, self.t)
        for t in range(0,look_back):
            zt = torch.tensor(self.sleep_history["zt"][-(t+1)], **CTX)
            s_data = {"zt-%d"%(t+1): zt, "fzzt": self.s_data["fzzt"]}
            w  = self.reg.train_weights_closed(s_data, "fzzt->fzt-%d"%(t+1), nvalid=100)
            self.w_data["Ezt-%d_x1:t"%(t+1)] = self.reg.predict(self.w_data, "fzzt->fzt-%d"%(t+1), inputs="mzzt_x1:t")

    def compute_forward_statistics_test(self):
        
        self.w_data["Ezt_x1:t"] = self.reg.predict(self.w_data, "fzzt->fzt", inputs="mzzt_x1:t")
        
        look_back = min(self.look_back, self.t)
        for t in range(0,look_back):
            self.w_data["Ezt-%d_x1:t"%(t+1)] = self.reg.predict(self.w_data, "fzzt->fzt-%d"%(t+1), inputs="mzzt_x1:t")

class KBR(GenericDDCSSM):
    
    name = "KBR"
    
    def __init__(self, model, reg, *args, **kwargs):
        reg.tensor_op = '*'
        super(KBR, self).__init__(model, reg, *args, **kwargs)
    
    def sleep_train_test(self, nvalid=0):
        
        self.reg.train_kbr(self.s_data, "zt-1->zt,xt")
        self.reg.train_kbr(self.s_data, "zt-1->zt-1,xt")

        self.sleep_test()

    def sleep_test(self):

        mzt_te = self.reg.predict_kbr(self.w_data, "zt-1->zt,xt", inputs="mzt-1_x1:t-1,xt")
        self.w_data["mzt_x1:t"]  = mzt_te
        mzt_te = self.reg.predict_kbr(self.w_data, "zt-1->zt-1,xt", inputs="mzt-1_x1:t-1,xt")
        self.w_data["mzt-1_x1:t"]  = mzt_te
        
        self.compute_forward_statistics()
        self.log()

    def smooth(self, nvalid=0):
        
        if self.stationary:

            T = len(self.sleep_history["mzt-1_x1:t-1"])-1
            self.s_data["zt"] = self.sleep_history["zt"][T]
            self.s_data["zt-1"] = self.sleep_history["zt-1"][T]
            self.reg.train_kbr(self.s_data, "zt-1->zt-1,zt")
        
        self.w_data["mzt_x1:T"] = self.wake_history["mzt_x1:t"][-1].copy()
        self.s_data["mzt_x1:T"] = self.sleep_history["mzt_x1:t"][-1].copy()

        for t in range(self.t, -1, -1):

            if not self.stationary:

                self.s_data["zt"] = self.sleep_history["zt"][t]
                self.s_data["zt-1"] = self.sleep_history["zt-1"][t]
                self.reg.train_kbr(self.s_data, "zt-1->zt-1,zt")

            self.w_data["mzt-1_x1:t-1"] = self.wake_history["mzt-1_x1:t-1"][t]
            self.w_data["mzt-1_x1:T"]   = self.reg.predict_kbr(self.w_data, "zt-1->zt-1,zt", inputs="mzt-1_x1:t-1,mzt_x1:T")

            self.compute_backward_statistics()
            self.log(["mzt_x1:T", "mzt-1_x1:T", "Ezt_x1:T", "Ezt-1_x1:T"])
            self.w_data["mzt_x1:T"]   = self.w_data["mzt-1_x1:T"].copy()
        
        self.wake_history["mzt_x1:T"] = self.wake_history["mzt_x1:T"][::-1]
        self.wake_history["mzt-1_x1:T"] = self.wake_history["mzt-1_x1:T"][::-1]
        self.wake_history["Ezt-1_x1:T"] = self.wake_history["Ezt-1_x1:T"][::-1]
        self.wake_history["Ezt_x1:T"] = self.wake_history["Ezt_x1:T"][::-1]

from .Regressors import LinearRegressor, LinearRegressorStep
import autograd.numpy as np
from tqdm import tqdm
from collections import OrderedDict

def shuffle_dict(d):
    N = list(d.values())[0].shape[0]
    idx = np.random.permutation(N)
    for k, v in d.items():
        d[k] = v[idx]
    return d

def index_dict(d, idx):
    dout = type(d)() 
    for k, v in d.items():
        dout[k] = v[idx]
    return d

def sleep_sample(self, nvalid=0, save_means=False):
    
    ddc = self.reg        
    self.sleep_data = self.model.sample(self.nsleep)

    nl_obs = self.nlayer-1
    
    for i in range(self.nlayer-2,-1,-1):
        ddc.train_weights(self.sleep_data, "x%d->x%d" % (i+1,i), nvalid=nvalid)
        if i == self.nlayer-2:
            if save_means:
                self.sleep_data["mx%d_x%d"%(i, i+1)]=\
                    ddc.predict(self.sleep_data, "x%d->x%d" % (i+1,i))
        if i < self.nlayer-2:
            ddc.copy_weights("x%d->x%d" % (i+1,i), "mx%d_x%d->x%d" % (i+1,nl_obs,i) )
            if save_means:
                self.sleep_data["mx%d_x%d"%(i, nl_obs)]=\
                    ddc.predict(self.sleep_data, "mx%d_x%d->x%d" % (i+1,nl_obs,i))
            
    self.compute_approximate_gradients(nvalid=nvalid)

def sleep_mean(self, nvalid=0):
    
    ddc = self.reg        
    self.sleep_data = self.model.sample(self.nsleep)
    nl_obs = self.nlayer-1
    
    for i in range(self.nlayer-2,-1,-1):
        if i == self.nlayer-2:
            ddc.train_weights(self.sleep_data, "x%d->x%d" % (i+1,i), nvalid=nvalid)
            self.sleep_data["mx%d_x%d"%(i, i+1)]=\
                ddc.predict(self.sleep_data, "x%d->x%d" % (i+1,i))
        else:
            ddc.train_weights(self.sleep_data, "mx%d_x%d->x%d" % (i+1,nl_obs,i), nvalid=nvalid)
            self.sleep_data["mx%d_x%d"%(i, nl_obs)]=\
                ddc.predict(self.sleep_data, "mx%d_x%d->x%d" % (i+1,nl_obs))
            
    self.compute_approximate_gradients(nvalid=nvalid)



class DDCHM(object):
    
    def __init__(self, model, fs, opt, s_type, lam=1e-8, nsleep=1000, 
                nwake=1000, seed=None, layer_plastic=None, niter=None):
        
        self.reg = LinearRegressor(fs, lam=lam)
        self.nlayer = model.depth
        #assert self.nlayer == model.depth
        self.model = model
        self.nsleep=nsleep
        self.nwake=nwake
        self.seed=seed
        
        self.opt = opt
        
        if layer_plastic is None:
            self.layer_plastic = [True] * self.nlayer
        elif len(layer_plastic) == self.nlayer:
            self.layer_plastic = layer_plastic
        else:
            raise NameError("layer_plastic not valid")
        
        self.s_type=s_type 
        self.niter = niter
        self.sleep_data = None
        self.wake_data  = None


    def sleep(self, nvalid=0):
        if self.s_type=="mean":
            sleep_mean(self, nvalid)
        elif self.s_type == "sample":
            sleep_sample(self, nvalid)
        
    def approx_E(self, mean, fun):
        n = self.wake_data[mean].shape[0]
        return np.dot(self.wake_data[mean].reshape(n,-1), self.reg.Ws[fun])
    
    def gradient_step(self, g, it):
 
        self.opt.step(self.model.ps, g, it)
        self.model.sync_ps()

    def train(self, niter, wake_data, bar=True, snap=False):

        batch_data = OrderedDict()

        if bar:
            r = tqdm(range(niter), leave=True, desc="training", ncols=100)
        else:
            r = range(niter)

        N = list(wake_data.values())[0].shape[0]
        epoch = 0

        for i in r:
             
            for k, v in wake_data.items():
                batch_data[k] = np.take(v, range(self.nwake*i, self.nwake*(i+1)), axis=0, mode="wrap")
                if i * self.nwake >= N*(epoch+1):
                    shuffle_dict(wake_data)
                    epoch+=1 

            self.sleep()
            self.wake(batch_data, i)
            if snap and i % snap == 0:
                self.save()

        self.niter = niter
        self.save()

    def default_file_name(self):
        
        fn = self.model.default_model_name()
        fn += "_M%03d" % list(self.reg.fs.values())[-1].M
        fn += "_ns%03d" % self.nsleep
        fn += "_nw%03d" % self.nwake
        fn += "_s%s" % self.s_type
        fn += "_g%s" % self.g_type
        fn += "_lr%d" % (int(np.log10(self.opt.lr)))

        if self.niter is not None:
            fn += "_ni%d"%(int(np.log10(self.niter)))

        if self.seed is not None:
            fn += "_%02d"%self.seed
        
        return fn
    
    def save(self, desc=""):
        
        fn = self.default_file_name()
        if len(desc) != 0:
            fn += "_%s" % desc
        np.savez(fn, gen=self.model.ps)
        return fn
    
    def load(self, desc=""):

        fn = self.default_file_name()
        if len(desc) != 0:
            fn += "_%s" % desc
        p  = np.load(fn + ".npz")
        self.model.ps = p["gen"]
        self.model.sync_ps()
        
class LogpHM(DDCHM):

    g_type = "logp"
    
    def wake(self, wake_data, it):
        
        ddc = self.reg
        self.wake_data = wake_data.copy()

        gs = []

        nl_obs = self.nlayer-1

        if self.layer_plastic[-1]:            

            grad_name = "x%d->dlogp%d" % (nl_obs, nl_obs)
            g = self.reg.predict(self.wake_data, grad_name)
            self.wake_data["dlogp%d"%(nl_obs)] = g
            gs.insert(0, g.mean(0))

            # NOT NECESSARY
            grad_name = "x%d->dnorm%d" % (nl_obs, nl_obs)
            g = self.reg.predict(self.wake_data, grad_name)
            self.wake_data["dnorm%d"%(nl_obs)] = g

            grad_name = "x%d->dnatsuff%d" % (nl_obs, nl_obs)
            g = self.reg.predict(self.wake_data, grad_name)
            self.wake_data["dnatsuff%d"%(nl_obs)] = g


        else:
            gs.insert(0, np.zeros_like(self.model.dists[-1].ps))

        if self.nlayer > 1:

            for i in range(self.nlayer-2,-1,-1):
                mean_name = "mx%d_x%d" % (i,nl_obs)
                if i == self.nlayer-2:
                    fun_name  = "x%d->x%d" % (nl_obs, self.nlayer-2)
                else:
                    fun_name  = "mx%d_x%d->x%d" % (i+1,nl_obs,i)
                self.wake_data[mean_name] = ddc.predict(self.wake_data, fun_name)
                
                if self.layer_plastic[i]:

                    grad_name = "x%d->dlogp%d"%(i,i)
                    g = self.approx_E(mean_name, grad_name)
                    self.wake_data["dlogp%d"%i] = g
                    gs.insert(0,g.mean(0))

                    # NOT NECESSARY
                    grad_name = "x%d->dnorm%d"%(i, i)
                    g = self.approx_E(mean_name, grad_name)
                    self.wake_data["dnorm%d"%(i)] = g
                    
                    grad_name = "x%d->dnatsuff%d"%(i,i)
                    g = self.approx_E(mean_name, grad_name)
                    self.wake_data["dnatsuff%d"%i] = g

                else:
                    gs.insert(0,np.zeros_like(self.model.dists[i].ps))
        gs = np.concatenate(gs)

        self.gradient_step(gs, it)

    def compute_approximate_gradients(self, nvalid=0):

        if self.layer_plastic[0]:
            self.sleep_data["dlogp0"] = self.model.dists[0].dlogp(self.sleep_data["x0"])
            self.reg.train_weights(self.sleep_data, "x0->dlogp0",  nvalid=nvalid)

            # not necessary
            self.sleep_data["dnorm0"] = self.model.dists[0].dnorm(n=self.nsleep)
            self.reg.train_weights(self.sleep_data, "x0->dnorm0",  nvalid=nvalid)
            self.sleep_data["dnatsuff0"] = self.model.dists[0].dnatsuff(self.sleep_data["x0"])
            self.reg.train_weights(self.sleep_data, "x0->dnatsuff0",  nvalid=nvalid)

        for i in range(1, self.model.depth):
            if not self.layer_plastic[i]:
                continue
            self.sleep_data["dlogp%d" % i] = self.model.dists[i].dlogp(self.sleep_data["x%d"%(i-1)], self.sleep_data["x%d"%(i)])
            self.reg.train_weights(self.sleep_data, "x%d->dlogp%d"%(i,i),  nvalid=nvalid)

            # not necessary
            self.sleep_data["dnorm%d" % i] = self.model.dists[i].dnorm(self.sleep_data["x%d"%(i-1)])
            self.sleep_data["dnatsuff%d" % i] = self.model.dists[i].dnatsuff(self.sleep_data["x%d"%(i-1)], self.sleep_data["x%d"%(i)])
            self.reg.train_weights(self.sleep_data, "x%d->dnorm%d"%(i,i),  nvalid=nvalid)
            self.reg.train_weights(self.sleep_data, "x%d->dnatsuff%d"%(i,i),  nvalid=nvalid)
            
class ExpFamHM(DDCHM):
    
    g_type = "exp"

    def compute_approximate_gradients(self, nvalid=0):

        if self.layer_plastic[0]:

            self.sleep_data["fsuff0"] = self.model.dists[0].suff(self.sleep_data["x0"])
            self.reg.train_weights(self.sleep_data, "x0->fsuff0",  nvalid=nvalid)

        for i in range(1, self.model.depth):

            if not self.layer_plastic[i]:
                continue
            n = self.nsleep
            self.sleep_data["dnorm%d" % i] = self.model.dists[i].dnorm(self.sleep_data["x%d"%(i-1)])
            self.reg.train_weights(self.sleep_data, "x%d->dnorm%d"%(i-1,i),  nvalid=nvalid)

            if i == (self.model.depth-1):
                self.sleep_data["dnat%d" % i] = self.model.dists[i].dnat(self.sleep_data["x%d"%(i-1)])
                self.reg.train_weights(self.sleep_data, "x%d->dnat%d"%(i-1,i),  nvalid=nvalid)
            else:
                self.sleep_data["dnatsuff%d" % i] = self.model.dists[i].dnatsuff(self.sleep_data["x%d"%(i-1)], self.sleep_data["x%d"%(i)])
                self.reg.train_weights(self.sleep_data, "x%d->dnatsuff%d"%(i,i),  nvalid=nvalid)
                
    def wake(self, wake_data, it):
        
        ddc = self.reg
        self.wake_data = wake_data.copy()

        gs = []

        nl_obs = self.nlayer-1

        mean_name_higher= "mx%d_x%d"%(self.nlayer-2, nl_obs)
        fun_name = "x%d->x%d"%(nl_obs, self.nlayer-2)
        self.wake_data[mean_name_higher] = ddc.predict(self.wake_data, fun_name)

        if self.layer_plastic[-1]:            


            grad_name = "x%d->dnat%d"%(self.nlayer-2, nl_obs)
            dnat  = self.approx_E(mean_name_higher, grad_name)
            self.wake_data["dnat%d"%(nl_obs)] = dnat
            #dnat  = self.model.dists[-1].dnat(self.wake_data["x%d"%(self.nlayer-2)])

            grad_name = "x%d->dnorm%d"%(self.nlayer-2, nl_obs)
            dnorm  = self.approx_E(mean_name_higher, grad_name)
            self.wake_data["dnorm%d"%(nl_obs)] = dnorm
            #dnorm  = self.model.dists[-1].dnorm(self.wake_data["x%d"%(self.nlayer-2)])

            suff = self.model.dists[-1].suff(self.wake_data["x%d"%(nl_obs)])
            dnatsuff = self.model.dists[-1].dnatsuff_from_dnatsuff(dnat, suff)
            self.wake_data["dnatsuff%d"%(nl_obs)] = dnatsuff

            g = (dnatsuff - dnorm)
            self.wake_data["dlogp%d"%(nl_obs)] = g
            gs.insert(0, g.mean(0))
            

        else:
            gs.insert(0, np.zeros_like(self.model.dists[-1].ps))
        
        if self.nlayer > 1:
            
            for i in range(self.nlayer-2,0,-1):
            
                mean_name_lower = mean_name_higher
                mean_name_higher= "mx%d_x%d" % (i-1,nl_obs)
                fun_name  = "mx%d_x%d->x%d" % (i,nl_obs,i-1)
                self.wake_data[mean_name_higher] = ddc.predict(self.wake_data, fun_name)

                if self.layer_plastic[i]:

                    grad_name = "x%d->dnatsuff%d"%(i,i)
                    dnatsuff = self.approx_E(mean_name_lower, grad_name)
                    self.wake_data["dnatsuff%d"%i] = dnatsuff
                    #dnatsuff = self.model.dists[i].dnatsuff(self.wake_data["x%d"%(i-1)], self.wake_data["x%d"%(i)])

                    grad_name = "x%d->dnorm%d"%(i-1,i)
                    dnorm    = self.approx_E(mean_name_higher, grad_name)
                    self.wake_data["dnorm%d"%i] = dnorm
                    #dnorm    = self.model.dists[i].dnorm(self.wake_data["x%d"%(i-1)])

                    g = (dnatsuff - dnorm)
                    #g2 = (self.model.dists[i].dlogp(self.wake_data["x%d"%(i-1)], self.wake_data["x%d"%(i)])).mean(0)
                    #assert np.allclose(g, g2)
                    self.wake_data["dlogp%d"%i] = g
                    gs.insert(0,g.mean(0))

                else:
                    gs.insert(0,np.zeros_like(self.model.dists[i].ps))

            if self.layer_plastic[0]:

                grad_name = "x0->fsuff0"
                suff = self.approx_E(mean_name_higher, grad_name)
                #suff = self.model.dists[0].suff(self.wake_data["x0"])

                dnat = self.model.dists[0].dnat()
                dnatsuff = self.model.dists[0].dnatsuff_from_dnatsuff(dnat, suff)
                self.wake_data["dnatsuff0"] = dnatsuff

                dnorm = self.model.dists[0].dnorm(n=dnatsuff.shape[0])
                self.wake_data["dnorm0"] = dnorm
                g    = (dnatsuff - dnorm)
                #g    = self.model.dists[0].dlogp(self.wake_data["x0"]).mean(0)
                #assert np.allclose(g_exp,g)
                self.wake_data["dlogp0"] = g
                gs.insert(0, g.mean(0))

            else:
                gs.insert(0,np.zeros_like(self.model.dists[0].ps))

        gs = np.concatenate(gs)

        self.gradient_step(gs, it)

class ExpFamHM2(DDCHM):
    
    g_type = "exp2"

    def compute_approximate_gradients(self, nvalid=0):

        sleep_data_1 = index_dict(self.sleep_data, range(0,self.nsleep/2))
        sleep_data_2 = index_dict(self.sleep_data, range(self.nsleep/2,self.nsleep))


        if self.layer_plastic[0]:

            self.sleep_data["fsuff0"] = self.model.dists[0].suff(self.sleep_data["x0"])
            self.reg.train_weights(self.sleep_data, "x0->fsuff0",  nvalid=nvalid)

        for i in range(1, self.model.depth):

            if not self.layer_plastic[i]:
                continue

            sleep_data_1["dnorm%d" % i] = self.model.dists[i].dnorm(sleep_data_1["x%d"%(i-1)])
            self.reg.train_weights(sleep_data_1, "x%d->dnorm%d"%(i-1,i),  nvalid=nvalid)

            if i == (self.model.depth-1):
                sleep_data_2["dnat%d" % i] = self.model.dists[i].dnat(sleep_data_2["x%d"%(i-1)])
                self.reg.train_weights(sleep_data_2, "x%d->dnat%d"%(i-1,i),  nvalid=nvalid)
            else:
                sleep_data_2["dnatsuff%d" % i] = self.model.dists[i].dnatsuff(sleep_data_2["x%d"%(i-1)], sleep_data_2["x%d"%(i)])
                self.reg.train_weights(sleep_data_2, "x%d->dnatsuff%d"%(i,i),  nvalid=nvalid)
                
    def wake(self, wake_data, it):
        
        ddc = self.reg
        self.wake_data = wake_data.copy()

        gs = []

        nl_obs = self.nlayer-1

        mean_name_higher= "mx%d_x%d"%(self.nlayer-2, nl_obs)
        fun_name = "x%d->x%d"%(nl_obs, self.nlayer-2)
        self.wake_data[mean_name_higher] = ddc.predict(self.wake_data, fun_name)

        if self.layer_plastic[-1]:            


            grad_name = "x%d->dnat%d"%(self.nlayer-2, nl_obs)
            dnat  = self.approx_E(mean_name_higher, grad_name)
            #dnat  = self.model.dists[-1].dnat(self.wake_data["x%d"%(self.nlayer-2)])

            grad_name = "x%d->dnorm%d"%(self.nlayer-2, nl_obs)
            dnorm  = self.approx_E(mean_name_higher, grad_name)
            #dnorm  = self.model.dists[-1].dnorm(self.wake_data["x%d"%(self.nlayer-2)])

            suff = self.model.dists[-1].suff(self.wake_data["x%d"%(nl_obs)])
            g = (self.model.dists[-1].dnatsuff_from_dnatsuff(dnat, suff) - dnorm)
            self.wake_data["dlogp%d"%(nl_obs)] = g
            gs.insert(0, g.mean(0))
        else:
            gs.insert(0, np.zeros_like(self.model.dists[-1].ps))
        
        if self.nlayer > 1:
            
            for i in range(self.nlayer-2,0,-1):
            
                mean_name_lower = mean_name_higher
                mean_name_higher= "mx%d_x%d" % (i-1,nl_obs)
                fun_name  = "mx%d_x%d->x%d" % (i,nl_obs,i-1)
                self.wake_data[mean_name_higher] = ddc.predict(self.wake_data, fun_name)

                if self.layer_plastic[i]:

                    grad_name = "x%d->dnatsuff%d"%(i,i)
                    dnatsuff = self.approx_E(mean_name_lower, grad_name)
                    #dnatsuff = self.model.dists[i].dnatsuff(self.wake_data["x%d"%(i-1)], self.wake_data["x%d"%(i)])

                    grad_name = "x%d->dnorm%d"%(i-1,i)
                    dnorm    = self.approx_E(mean_name_higher, grad_name)
                    #dnorm    = self.model.dists[i].dnorm(self.wake_data["x%d"%(i-1)])

                    g = (dnatsuff - dnorm)
                    #g2 = (self.model.dists[i].dlogp(self.wake_data["x%d"%(i-1)], self.wake_data["x%d"%(i)])).mean(0)
                    #assert np.allclose(g, g2)
                    self.wake_data["dlogp%d"%i] = g
                    gs.insert(0,g.mean(0))

                else:
                    gs.insert(0,np.zeros_like(self.model.dists[i].ps))

            if self.layer_plastic[0]:

                grad_name = "x0->fsuff0"
                suff = self.approx_E(mean_name_higher, grad_name)
                #suff = self.model.dists[0].suff(self.wake_data["x0"])

                dnat = self.model.dists[0].dnat()
                dnatsuff = self.model.dists[0].dnatsuff_from_dnatsuff(dnat, suff)
                dnorm = self.model.dists[0].dnorm()
                g    = (dnatsuff - dnorm)
                #g    = self.model.dists[0].dlogp(self.wake_data["x0"]).mean(0)
                #assert np.allclose(g_exp,g)
                self.wake_data["dlogp0"] = g
                gs.insert(0, g.mean(0))

            else:
                gs.insert(0,np.zeros_like(self.model.dists[0].ps))

        gs = np.concatenate(gs)

        self.gradient_step(gs, it)


class LogpExpFamHM(DDCHM):
    

    g_type = "logpexp"

    def compute_approximate_gradients(self, nvalid=0):

        if self.layer_plastic[0]:
            self.sleep_data["dlogp0"] = self.model.dists[0].dlogp(self.sleep_data["x0"])
            self.reg.train_weights(self.sleep_data, "x0->dlogp0",  nvalid=nvalid)

        for i in range(1, self.model.depth-1):
            if not self.layer_plastic[i]:
                continue
            self.sleep_data["dlogp%d" % i] = self.model.dists[i].dlogp(self.sleep_data["x%d"%(i-1)], self.sleep_data["x%d"%(i)])
            self.reg.train_weights(self.sleep_data, "x%d->dlogp%d"%(i,i),  nvalid=nvalid)

        i = self.model.depth-1
        self.sleep_data["dnorm%d" % i] = self.model.dists[i].dnorm(self.sleep_data["x%d"%(i-1)])
        self.reg.train_weights(self.sleep_data, "x%d->dnorm%d"%(i-1,i),  nvalid=nvalid)

        self.sleep_data["dnat%d" % i] = self.model.dists[i].dnat(self.sleep_data["x%d"%(i-1)])
        self.reg.train_weights(self.sleep_data, "x%d->dnat%d"%(i-1,i),  nvalid=nvalid)
                
    def wake(self, wake_data, it):
        
        ddc = self.reg
        self.wake_data = wake_data.copy()

        gs = []

        nl_obs = self.nlayer-1

        mean_name_higher= "mx%d_x%d"%(self.nlayer-2, nl_obs)
        fun_name = "x%d->x%d"%(nl_obs, self.nlayer-2)
        self.wake_data[mean_name_higher] = ddc.predict(self.wake_data, fun_name)

        if self.layer_plastic[-1]:            


            grad_name = "x%d->dnat%d"%(self.nlayer-2, nl_obs)
            dnat  = self.approx_E(mean_name_higher, grad_name)
            #dnat  = self.model.dists[-1].dnat(self.wake_data["x%d"%(self.nlayer-2)])

            grad_name = "x%d->dnorm%d"%(self.nlayer-2, nl_obs)
            dnorm  = self.approx_E(mean_name_higher, grad_name)
            #dnorm  = self.model.dists[-1].dnorm(self.wake_data["x%d"%(self.nlayer-2)])

            suff = self.model.dists[-1].suff(self.wake_data["x%d"%(nl_obs)])
            g = (self.model.dists[-1].dnatsuff_from_dnatsuff(dnat, suff) - dnorm)
            self.wake_data["dlogp%d"%(nl_obs)] = g
            gs.insert(0, g.mean(0))
        else:
            gs.insert(0, np.zeros_like(self.model.dists[-1].ps))
        
        if self.layer_plastic[-2]:
            
            fun_name  = "x%d->x%d" % (nl_obs, self.nlayer-2)
            grad_name = "x%d->dlogp%d"%(self.nlayer-2, self.nlayer-2)
            
            g = self.approx_E(mean_name_higher, grad_name)
            self.wake_data["dlogp%d"%(self.nlayer-2)] = g
            gs.insert(0,g.mean(0))
        else:
            gs.insert(0,np.zeros_like(self.model.dists[-2].ps))
            
            
        for i in range(self.nlayer-3,-1,-1):
            mean_name = "mx%d_x%d" % (i,nl_obs)
            fun_name  = "mx%d_x%d->x%d" % (i+1,nl_obs,i)
            grad_name = "x%d->dlogp%d"%(i,i)

            self.wake_data[mean_name] = ddc.predict(self.wake_data, fun_name)
            
            if self.layer_plastic[i]:
                g = self.approx_E(mean_name, grad_name)
                self.wake_data["dlogp%d"%i] = g
                gs.insert(0,g.mean(0))
                
            else:
                gs.insert(0,np.zeros_like(self.model.dists[i].ps))

        gs = np.concatenate(gs)

        self.gradient_step(gs, it)

class LogpHM_Joint(DDCHM):
    g_type = "joint"
    
    def _merge_array(self, *args):
        
        self.sleep_data["".join(args)] = np.concatenate([self.sleep_data[k] for k in args],-1)
    
    def sleep(self, nvalid=0):
        reg = self.reg
        self.sleep_data = self.model.sample(self.nsleep)
        
        nl_obs = self.nlayer-1
        for i in range(self.nlayer-2,-1,-1):
            
            self._merge_array("x%d"%(i), "x%d"%(i+1))
            if i == self.nlayer-2:
                reg.train_weights(self.sleep_data, "x%d->x%dx%d" % (i+1,i,i+1), nvalid=nvalid)

            if i < self.nlayer-2:
                reg.train_weights(self.sleep_data, "x%dx%d->x%dx%d" % (i+1,i+2,i,i+1), nvalid=nvalid)
                reg.copy_weights("x%dx%d->x%dx%d" % (i+1,i+2,i,i+1), "mx%dx%d_x%d->x%dx%d" % (i+1,i+2,nl_obs,i,i+1) )
                
        reg.train_weights(self.sleep_data, "x0x1->x0", nvalid=nvalid)
        reg.copy_weights("x0x1->x0", "mx0x1_x%d->x0"%nl_obs)
                
        self.compute_approximate_gradients(nvalid=nvalid)
                
    def compute_approximate_gradients(self, nvalid=0):
        
        if self.layer_plastic[0]:
            self.sleep_data["dlogp0"] = self.model.dists[0].dlogp(self.sleep_data["x0"])
            self.reg.train_weights(self.sleep_data, "x0->dlogp0",  nvalid=nvalid)

        for i in range(1, self.model.depth):
            if not self.layer_plastic[i]:
                continue
            self.sleep_data["dlogp%d" % i] = self.model.dists[i].dlogp(self.sleep_data["x%d"%(i-1)], self.sleep_data["x%d"%(i)])
            self.reg.train_weights(self.sleep_data, "x%dx%d->dlogp%d"%(i-1,i,i),  nvalid=nvalid)
        
        self.reg.train_weights(self.sleep_data, "x%d->dlogp%d"%(self.nlayer-1, self.nlayer-1),  nvalid=nvalid)
        
    def wake(self, wake_data, it):

        reg = self.reg
        self.wake_data = wake_data.copy()

        gs = []
        
        nl_obs = (self.nlayer-1)
        mean_name = "mx%dx%d_x%d" % (nl_obs-1, nl_obs, nl_obs)
        fun_name  = "x%d->x%dx%d" % (nl_obs, nl_obs-1, nl_obs)
        self.wake_data[mean_name] = reg.predict(self.wake_data, fun_name)

        if self.layer_plastic[-1]:
            grad_name = "x%d->dlogp%d" % (nl_obs, nl_obs)
            g = reg.predict(self.wake_data, grad_name)
            #g = self.approx_E(mean_name, grad_name)
            self.wake_data["dlogp%d"%(nl_obs)] = g
            gs.insert(0, g.mean(0))
        else:
            gs.insert(0, np.zeros_like(self.model.dists[-1].ps))
        

        for i in range(nl_obs-1,0,-1):
            
            mean_name = "mx%dx%d_x%d" % (i-1,i,nl_obs)
            fun_name  = "mx%dx%d_x%d->x%dx%d" % (i,i+1,nl_obs,i-1,i)
            self.wake_data[mean_name] = reg.predict(self.wake_data, fun_name)
            if self.layer_plastic[i]:

                grad_name = "x%dx%d->dlogp%d"%(i-1,i,i)
                g = self.approx_E(mean_name, grad_name)
                self.wake_data["dlogp%d"%i] = g
                gs.insert(0,g.mean(0))

            else:
                gs.insert(0,np.zeros_like(self.model.dists[i].ps))
                
        if self.layer_plastic[0]:
            mean_name = "mx0_x%d" % (nl_obs)
            fun_name  = "mx0x1_x%d->x0" % (nl_obs)
            self.wake_data[mean_name] = reg.predict(self.wake_data, fun_name)
            g = self.approx_E(mean_name, "x0->dlogp0")
            self.wake_data["dlogp0"] = g
            gs.insert(0,g.mean(0))        
        else:
            gs.insert(0,np.zeros_like(self.model.dists[0].ps))

            
        gs = np.concatenate(gs)

        self.gradient_step(gs, it)

class ExpFamNormHM(DDCHM):
    
    g_type = "enorm"

    def compute_approximate_gradients(self, nvalid=0):

        if self.layer_plastic[0]:

            self.sleep_data["fsuff0"] = self.model.dists[0].suff(self.sleep_data["x0"])
            self.reg.train_weights(self.sleep_data, "x0->fsuff0",  nvalid=nvalid)

        for i in range(1, self.model.depth):

            if not self.layer_plastic[i]:
                continue

            n = self.nsleep

            nl_obs = self.nlayer-1

            self.sleep_data["dnorm%d" % i] = self.model.dists[i].dnorm(self.sleep_data["x%d"%(i-1)])
            self.reg.train_weights(self.sleep_data, "x%d->dnorm%d"%(i,i),  nvalid=nvalid)
            self.reg.copy_weights("x%d->dnorm%d"%(i,i), "mx%d_x%d->dnorm%d"%(i,nl_obs,i))

            if i == (self.model.depth-1):
                self.sleep_data["dnat%d" % i] = self.model.dists[i].dnat(self.sleep_data["x%d"%(i-1)])
                self.reg.train_weights(self.sleep_data, "x%d->dnat%d"%(i,i),  nvalid=nvalid)
            else:
                self.sleep_data["dnatsuff%d" % i] = self.model.dists[i].dnatsuff(self.sleep_data["x%d"%(i-1)], self.sleep_data["x%d"%(i)])
                self.reg.train_weights(self.sleep_data, "x%d->dnatsuff%d"%(i,i),  nvalid=nvalid)
                
    def wake(self, wake_data, it):
        
        ddc = self.reg
        self.wake_data = wake_data.copy()

        gs = []

        nl_obs = self.nlayer-1

        mean_name_higher= "mx%d_x%d"%(self.nlayer-2, nl_obs)
        fun_name = "x%d->x%d"%(nl_obs, self.nlayer-2)
        self.wake_data[mean_name_higher] = ddc.predict(self.wake_data, fun_name)

        if self.layer_plastic[-1]:            


            grad_name = "x%d->dnat%d"%(nl_obs, nl_obs)
            dnat      = ddc.predict(self.wake_data, grad_name)

            grad_name = "x%d->dnorm%d"%(nl_obs, nl_obs)
            dnorm     = ddc.predict(self.wake_data, grad_name)
            self.wake_data["dnorm%d"%(nl_obs)] = dnorm

            suff = self.model.dists[-1].suff(self.wake_data["x%d"%(nl_obs)])
            dnatsuff = self.model.dists[-1].dnatsuff_from_dnatsuff(dnat, suff)
            self.wake_data["dnatsuff%d"%(nl_obs)] = dnatsuff

            g = (dnatsuff - dnorm)
            self.wake_data["dlogp%d"%(nl_obs)] = g
            gs.insert(0, g.mean(0))
            

        else:
            gs.insert(0, np.zeros_like(self.model.dists[-1].ps))
        
        if self.nlayer > 1:
            
            for i in range(self.nlayer-2,0,-1):
            
                mean_name_lower = mean_name_higher
                mean_name_higher= "mx%d_x%d" % (i-1,nl_obs)
                fun_name  = "mx%d_x%d->x%d" % (i,nl_obs,i-1)
                self.wake_data[mean_name_higher] = ddc.predict(self.wake_data, fun_name)

                if self.layer_plastic[i]:

                    grad_name = "x%d->dnatsuff%d"%(i,i)
                    dnatsuff = self.approx_E(mean_name_lower, grad_name)
                    self.wake_data["dnatsuff%d"%i] = dnatsuff
                    #dnatsuff = self.model.dists[i].dnatsuff(self.wake_data["x%d"%(i-1)], self.wake_data["x%d"%(i)])

                    grad_name = "mx%d_x%d->dnorm%d"%(i,nl_obs,i)
                    dnorm     = ddc.predict(self.wake_data, grad_name)
                    self.wake_data["dnorm%d"%i] = dnorm
                    #dnorm    = self.model.dists[i].dnorm(self.wake_data["x%d"%(i-1)])

                    g = (dnatsuff - dnorm)
                    #g2 = (self.model.dists[i].dlogp(self.wake_data["x%d"%(i-1)], self.wake_data["x%d"%(i)])).mean(0)
                    #assert np.allclose(g, g2)
                    self.wake_data["dlogp%d"%i] = g
                    gs.insert(0,g.mean(0))

                else:
                    gs.insert(0,np.zeros_like(self.model.dists[i].ps))

            if self.layer_plastic[0]:

                grad_name = "x0->fsuff0"
                suff = self.approx_E(mean_name_higher, grad_name)
                #suff = self.model.dists[0].suff(self.wake_data["x0"])

                dnat = self.model.dists[0].dnat()
                dnatsuff = self.model.dists[0].dnatsuff_from_dnatsuff(dnat, suff)
                self.wake_data["dnatsuff0"] = dnatsuff

                dnorm = self.model.dists[0].dnorm(n=dnatsuff.shape[0])
                self.wake_data["dnorm0"] = dnorm
                g    = (dnatsuff - dnorm)
                #g    = self.model.dists[0].dlogp(self.wake_data["x0"]).mean(0)
                #assert np.allclose(g_exp,g)
                self.wake_data["dlogp0"] = g
                gs.insert(0, g.mean(0))

            else:
                gs.insert(0,np.zeros_like(self.model.dists[0].ps))

        gs = np.concatenate(gs)

        self.gradient_step(gs, it)

class ExpFamTensorHM(DDCHM):
    
    g_type = "eten"

    def compute_approximate_gradients(self, nvalid=0):

        if self.layer_plastic[0]:

            self.sleep_data["fsuff0"] = self.model.dists[0].suff(self.sleep_data["x0"])
            self.reg.train_weights(self.sleep_data, "x0->fsuff0",  nvalid=nvalid)

        for i in range(1, self.model.depth):

            if not self.layer_plastic[i]:
                continue
            n = self.nsleep
            self.sleep_data["dnorm%d" % i] = self.model.dists[i].dnorm(self.sleep_data["x%d"%(i-1)])
            self.reg.train_weights(self.sleep_data, "x%d->dnorm%d"%(i-1,i),  nvalid=nvalid)

            if i == (self.model.depth-1):

                data = self.sleep_data
                lam  = self.reg.lam

                z_feat = self.reg.transform_data(data, ["x%d"%(i-1)])["x%d"%(i-1)]
                x_suff = self.model.dists[-1].suff(data["x%d"%i])

                x = np.einsum("ij,ik->ijk",z_feat, x_suff).reshape(self.nsleep, -1)
                y = z_feat
                f = self.model.dists[-1].dlogp(data["x%d"%(i-1)], data["x%d"%(i)])
                Mx = np.eye(self.nsleep)-x.dot(np.linalg.solve(x.T.dot(x)+np.eye(x.shape[1])*lam, x.T))
                My = np.eye(self.nsleep)-y.dot(np.linalg.solve(y.T.dot(y)+np.eye(y.shape[1])*lam, y.T))
                A =  np.linalg.solve((x.T.dot(My).dot(x))+np.eye(x.shape[1])*lam, x.T.dot(My).dot(f))
                B = -np.linalg.solve((y.T.dot(Mx).dot(y))+np.eye(y.shape[1])*lam, y.T.dot(Mx).dot(f))

                self.reg.Ws["A"] = A
                self.reg.Ws["B"] = B
            
            else:
                self.sleep_data["dnatsuff%d" % i] = self.model.dists[i].dnatsuff(self.sleep_data["x%d"%(i-1)], self.sleep_data["x%d"%(i)])
                self.reg.train_weights(self.sleep_data, "x%d->dnatsuff%d"%(i,i),  nvalid=nvalid)
                
    def wake(self, wake_data, it):
        
        ddc = self.reg
        self.wake_data = wake_data.copy()

        gs = []

        nl_obs = self.nlayer-1

        mean_name_higher= "mx%d_x%d"%(self.nlayer-2, nl_obs)
        fun_name = "x%d->x%d"%(nl_obs, self.nlayer-2)
        self.wake_data[mean_name_higher] = ddc.predict(self.wake_data, fun_name)

        if self.layer_plastic[-1]:            

                    
            A = self.reg.Ws["A"]
            B = self.reg.Ws["B"]

            z_mean = self.wake_data["mx%d_x%d"%(nl_obs-1, nl_obs)]
            x_suff = self.model.dists[-1].suff(self.wake_data["x%d"%nl_obs])
            x = np.einsum("ij,ik->ijk",z_mean, x_suff).reshape(z_mean.shape[0], -1)
            y = z_mean
            
            dnatsuff = x.dot(A)
            dnorm    = y.dot(B)
            g = dnatsuff - dnorm

            self.wake_data["dnatsuff%d"%nl_obs] = dnatsuff
            self.wake_data["dnorm%d"%nl_obs] = dnorm
            self.wake_data["dlogp%d"%(nl_obs)] = g
            gs.insert(0, g.mean(0))
            

        else:
            gs.insert(0, np.zeros_like(self.model.dists[-1].ps))
        
        if self.nlayer > 1:
            
            for i in range(self.nlayer-2,0,-1):
            
                mean_name_lower = mean_name_higher
                mean_name_higher= "mx%d_x%d" % (i-1,nl_obs)
                fun_name  = "mx%d_x%d->x%d" % (i,nl_obs,i-1)
                self.wake_data[mean_name_higher] = ddc.predict(self.wake_data, fun_name)

                if self.layer_plastic[i]:

                    grad_name = "x%d->dnatsuff%d"%(i,i)
                    dnatsuff = self.approx_E(mean_name_lower, grad_name)
                    self.wake_data["dnatsuff%d"%i] = dnatsuff
                    #dnatsuff = self.model.dists[i].dnatsuff(self.wake_data["x%d"%(i-1)], self.wake_data["x%d"%(i)])

                    grad_name = "x%d->dnorm%d"%(i-1,i)
                    dnorm    = self.approx_E(mean_name_higher, grad_name)
                    self.wake_data["dnorm%d"%i] = dnorm
                    #dnorm    = self.model.dists[i].dnorm(self.wake_data["x%d"%(i-1)])

                    g = (dnatsuff - dnorm)
                    #g2 = (self.model.dists[i].dlogp(self.wake_data["x%d"%(i-1)], self.wake_data["x%d"%(i)])).mean(0)
                    #assert np.allclose(g, g2)
                    self.wake_data["dlogp%d"%i] = g
                    gs.insert(0,g.mean(0))

                else:
                    gs.insert(0,np.zeros_like(self.model.dists[i].ps))

            if self.layer_plastic[0]:

                grad_name = "x0->fsuff0"
                suff = self.approx_E(mean_name_higher, grad_name)
                #suff = self.model.dists[0].suff(self.wake_data["x0"])

                dnat = self.model.dists[0].dnat()
                dnatsuff = self.model.dists[0].dnatsuff_from_dnatsuff(dnat, suff)
                self.wake_data["dnatsuff0"] = dnatsuff

                dnorm = self.model.dists[0].dnorm(n=dnatsuff.shape[0])
                self.wake_data["dnorm0"] = dnorm
                g    = (dnatsuff - dnorm)
                #g    = self.model.dists[0].dlogp(self.wake_data["x0"]).mean(0)
                #assert np.allclose(g_exp,g)
                self.wake_data["dlogp0"] = g
                gs.insert(0, g.mean(0))

            else:
                gs.insert(0,np.zeros_like(self.model.dists[0].ps))

        gs = np.concatenate(gs)

        self.gradient_step(gs, it)


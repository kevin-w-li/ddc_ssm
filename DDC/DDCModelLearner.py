import autograd.numpy as np
from Features import *

class DDCModelLearner(object):

    def __init__(self, model=model, filt=filt, s_data=s_data, w_data=w_data):

        self.filt = filt
        self.model = model
        self.fs   = self.ddc.fs
        self.s_data  = s_data
        self.w_data  = w_data
        self.lr = lr
        self.filt.online_initialise(s_data, w_data)

    def sleep(self):
        
        self.s_data = self.model.step(self.s_data["zt"], 0)
        self.filt.online_train(self.s_data)

    def wake(self)

        self.filt.online_test(self.w_data)
        g = self.model.observation.dlogp(self.s_data["zt"][:,None,:], self.w_data["xt"].[None,:,:])
        f = self.fs["zt"]
        Eg = 0 

        nwake = self.w_data["xt"].shape[0]

        for wi, wd in enumerate(self.w_data["xt"]):
            f.learn_fun("obs_grad", self.s_data["zt"], wd)
            Eg += f.approx_E("obs_grad", self.filt.test_data["mz_x1:t"])
        model.observation.ps += self.lr * Eg / nwake


    def propagate(self, w_data):
        
        self.filt.online_propagate(self.s_data, w_data)
        self.w_data = w_data



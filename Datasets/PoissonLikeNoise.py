import numpy as np
from SSM import *
from scipy.stats import norm

class PoissonLikeObservation(Observation):
    
    def __init__(self, theta_n):

        self.theta_n = theta_n
        
    def conditional_param(self, y):
        raise NotImplementedError
        
    def sample_chain(self, y):
        mu = self.conditional_param(y,0)
        return np.random.randn(*mu.shape) * self.theta_n + mu
    
    def loglik(self, y, x):
        # for a batch of y in [batchsize, nsample, ndim]
        #                x in [batchszie, 1]
        
        mu = self.conditional_param(y)
        return norm.logpdf(mu[:,:,0], loc = x, scale=self.theta_n)

    def conditional_param(self, zt, t):
        raise NotImplementedError

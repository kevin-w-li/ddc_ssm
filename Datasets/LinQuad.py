import numpy as np
from SSM import *
from scipy.stats import norm

class LinQuadLatent(Observaiton):

    def __init__(self, theta_l, theta_q, theta_c, theta_w, theta_n):
        super(LinQuadLatent, self).__init__(1, theta_n)

        self.theta_l = theta_l
        self.theta_q = theta_q
        self.theta_c = theta_c
        self.theta_w = theta_w

    def conditional_param(self, zt, t):  
        return self.theta_l * zt + self.theta_q * zt/(1.0+zt**2) + self.theta_c * np.cos(self.theta_w * (t))
    


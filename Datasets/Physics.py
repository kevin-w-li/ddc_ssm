import numpy as np

class Spring(object):
    
    def __init__(self, pix, omega, latent_noise=0.0, init_std = 1.0, margin=3, pixel_noise=0.0):
        
        self.t = 0
        self.omega = omega
        self.latent_noise = latent_noise
        self.init_std = init_std
        self.pix = pix
        self.A = pix/2-margin
        self.phi = None
        self.pixel_noise = pixel_noise

    def sample(self, T, C):
        
        self.phi = np.random.randn(C) * self.init_std
        im = np.zeros((T, C, self.pix))
        self.t = 0
        z = self.A * np.sin(self.phi) + self.pix/2
        z = z.astype("int")
        im[0, range(C), z] = 1
        for t in range(1,T):
            self.step(im[t]) 
            self.t = t
        if self.pixel_noise != 0:
            im += np.random.randn(*im.shape) * self.pixel_noise
        return im

    def step(self, im, t=None):
        if t == None:
            self.t += 1
            t = self.t
        im *= 0
        C = im.shape[0] 
        if self.phi is None or len(self.phi) != C:
            self.phi = np.random.randn(C) * self.init_std
            self.t = 0
        z = self.A * np.sin(self.omega*t + self.phi + 
                                np.random.randn(C)*self.latent_noise ) + self.pix/2

        z = z.astype("int")
        im[range(C), z] = 1.0
        if self.pixel_noise != 0:
            im += np.random.randn(*im.shape) * self.pixel_noise
        return im

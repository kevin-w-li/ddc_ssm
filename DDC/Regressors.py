import torch, numpy as np 
from numpy import prod as prod_numpy
from settings import CTX
from collections import defaultdict
import warnings

prod = lambda x: prod_numpy(x).item()

class Regressor(object):
    
    def __init__(self, fs, lam=1e-5, tensor_op='*', lr = 1e-5, weight_decay = 0.0, max_M=np.inf, adaptive_lam=False):
        
        self.vars = list(fs.keys())
        self.fs = fs
        self.Ws = {}    
        self.Ns = {}    
        self.R2s = {}
        self.out_shapes={}
        self.lam_idx = {}
        self.max_M = max_M
        self.lam = torch.tensor([np.log(lam)]*20, requires_grad=adaptive_lam, **CTX)
        self.lam_count = 0
        self.tensor_op = tensor_op
        if len(self.params)!=0:
            self.opt = torch.optim.SGD(self.params, lr = lr, weight_decay = weight_decay, momentum=0.9)
        else:
            self.opt = None
        self.R2=defaultdict(list)
        if self.lam.requires_grad:
            self.opt_lam = torch.optim.SGD([self.lam], lr=lr)

    def gradient_step(self):

        s = 0
        for  k, v in self.R2s.items():
            self.R2[k].append(v.cpu().detach().numpy())
            s = s + v
        if len(self.params)!=0:
            self.opt.zero_grad()
        if self.lam.requires_grad:
            self.opt_lam.zero_grad()
        
        if self.opt is not None or self.lam.requires_grad:
            s.backward(retain_graph=True)
        if self.opt is not None:
            self.opt.step()
        if self.lam.requires_grad:
            self.opt_lam.step()
        self.constrain_params()
        self.R2s = {}

    @property
    def params(self):
        p = []
        for v in self.fs.values():
            if v.adaptive:
                p += v.params
        return p

    def constrain_params(self):
        for v in self.fs.values():
            v.constrain_params() 
        #self.lam.data = torch.clamp(self.lam, min=np.log(1e-3))
        return 
    
    @staticmethod
    def _parse_mapping(s):

        arrow = s.find(">")
        
        output = s[arrow+1:]
        output = output.split(",")
        append_ones = []
        
        if arrow > 0:
            inputs = s[:arrow-1]
            inputs = inputs.split(",")
            for i,inp in enumerate(inputs):
                if inp[-2:]=="+1":
                    inputs[i] = inp[:-2]
                    append_ones.append(True)
                else:
                    append_ones.append(False)

                    
        else:
            inputs = []

        return inputs, output, append_ones
        
    def _pivot(self, A, d):
        
        if self.tensor_op == "c":
            return A
        s = A.shape
        m = prod(s[:d])
        n = prod(s[d:])
        return A.reshape(m,n)

    @staticmethod
    def _flatten(A, d):
        
        s = A.shape
        m = s[:d]
        
        if len(m)==0:
            return A
        
        m = prod(m)
        n = s[d:]
        return A.reshape((m,)+n)
    
    def _unpivot(self, A, shape1, shape2=(-1,)):

        if self.tensor_op == "c":
            return A
        
        return A.reshape(shape1+shape2)
        

    def transform_data(self, data, keys=None, feats = None):

        if feats is None:
            feats = {}
        
        if keys is None:
            keys = self.fs.keys()
            
        for k in keys:
            
            # if maps to a mean,  the data is gradient (function) values, or a function value
            if k[0] in ["m", "d", "f"]:
            
                if k in data.keys():
                    # if m is already in data (precomputed) then put in feats
                    feats[k] = data[k]
                    continue
                
                if k[0] in ["m", "f"]:
                    d = data[k[1:]]
                else:
                    # if a function value, it has to be provided in data
                    assert k in data.keys(), k+" not in data"
            else:
                d = data[k]

            dshape = d.shape
            
            if k[0] == "m":
                d_flat = self._flatten(d,-2)
                f = self.fs[k[1:]].map_mean(d_flat)
                f = self._unpivot(f, dshape[:-2])
            elif k[0] == "f":
                f = d
            else:
                f = self.fs[k](d)
            
            feats[k] = f

        return feats
    
    
    def _form_input_tensor(self, feats, inputs, clip=False):

        f = torch.ones([1], **CTX)
        ni = len(inputs)
        op = self.tensor_op
        # assume data arranged in the same way, dim is last
        input_shape = feats[inputs[0]].shape[:-1]
        
        fshape = ()
        
        if clip:
            M  =  np.prod([feats[i].shape[-1] for i in inputs])
            if M > self.max_M and self.tensor_op=="*":
                M = int(self.max_M**(1.0/len(inputs)))
                for i in inputs:
                    feats[i] = feats[i][:,-M:]
        
        
        if op == "*":
            for ii, i in enumerate(inputs):
                
                fi = feats[i].reshape(-1, feats[i].shape[-1])
                fshape += (fi.shape[-1],)
                #f  = np.einsum('i...,ik->i...k', f, fi, optimize=False)
                f  = f[...,None] * fi.reshape([-1]+[1]*(ii)+[fi.shape[-1]])
            f = f.reshape(input_shape + fshape)
        elif op == "c":

            input_feats = [feats[i] for i in inputs]
            min_dim     = min([inp.shape[-1] for inp in input_feats])
            input_prod  = torch.prod(torch.stack([inp[...,:min_dim] for inp in input_feats]), 0)

            f = torch.cat(input_feats, -1)
            if len(inputs) > 1:
                f = torch.cat([f, input_prod], -1)

        return f
    
    def copy_weights(self, m1, m2):
        self.Ws[m2]=self.Ws[m1]
        self.out_shapes[m2]=self.out_shapes[m1]

    def learn_fun(self, fun, data, mapping, lam=1e-9):
        
        inputs, outputs, append_ones = self._parse_mapping(mapping)
        input_args = [data[i] for i in inputs]
        for i ,inp in enumerate(input_args):
            #if append_ones[i]:
            input_args[i] = torch.cat([inp, torch.ones(inp.shape[:-1]+(1,), **CTX)], -1)
        data[outputs[0]]   = fun(*input_args)
        
        self.train_weights(data, mapping, lam=lam)
        del data[outputs[0]]

    def prep_features(self, data, mapping, prep_output = True, transform = True):

        inputs, outputs, append_ones = self._parse_mapping(mapping)
        
        if prep_output:
            if transform:
                feats = self.transform_data(data, keys=inputs+outputs)
                for i, k in enumerate(inputs):
                    #if append_ones[i]:
                    f = feats[k]
                    feats[k] = torch.cat([f, torch.ones(f.shape[:-1]+(1,), **CTX)], -1)
                return feats, inputs, outputs
            else:
                return data, inputs, outputs
        else:   
            if transform:
                feats = self.transform_data(data, keys=inputs)
                for i, k in enumerate(inputs):
                    #if append_ones[i]:
                    f = feats[k]
                    feats[k] = torch.cat([f, torch.ones(f.shape[:-1]+(1,), **CTX)], -1)
                return feats, inputs, None
            else:
                return data, inputs, None
    
    def prep_train_test(self, data, mapping, nvalid, clip):
    
        feats, inputs, outputs = self.prep_features(data, mapping)

        f = self._form_input_tensor(feats, inputs, clip)
        t = self._form_input_tensor(feats, outputs)

        self.out_shapes[mapping] = t.shape[-len(outputs):]

        ni = len(inputs)
        no = len(outputs)
        
        f = self._pivot(f, d=-ni)
        t = self._pivot(t, d=-no)

        
        t = t.detach()
        
        if nvalid != 0:
            f_tr = f[:-nvalid].detach()
            t_tr = t[:-nvalid].detach()
            f_te = f[-nvalid:]
            t_te = t[-nvalid:]
            return f_tr, t_tr, f_te, t_te
        else:
            f_tr = f
            t_tr = t
            return f_tr, t_tr, None, None
        
    def train_weights(self, x, y, lam=None, nvalid=0):
        raise NotImplementedError

    def predict(self, x, mapping, lam=None):
        raise NotImplementedError

    def _get_lam(self, mapping, lam):
    
        if mapping not in self.lam_idx:
            self.lam_idx[mapping] = self.lam_count
            if lam is not None:
                if isinstance(lam, torch.Tensor):
                    lam = np.log(lam.cpu().numpy())
                    self.lam.data[self.lam_count] = torch.tensor(lam, **CTX)
                else:
                    self.lam.data[self.lam_count] = torch.tensor(np.log(lam), **CTX)
            lam = self.lam[self.lam_count]
            self.lam_count += 1
        else:
            lam = self.lam[self.lam_idx[mapping]]
        return lam    
        
        

class LinearRegressor(Regressor):

    r_type="lin"

    def train_weights_closed(self, *args, **kwargs):
        return self.train_weights(*args, **kwargs)
        
    def train_weights(self, data, mapping, nvalid=0, clip=False, lam=None):
        
        lam = self._get_lam(mapping, lam)     
        f_tr, t_tr, f_te, t_te = self.prep_train_test(data, mapping, nvalid, clip)
        
        n_tr = f_tr.shape[0]
        if n_tr < 1000:
            warnings.warn("using less than 1000 sleep samples", UserWarning)
            
        W = torch.gesv(f_tr.t().matmul(t_tr), 
                       f_tr.t().matmul(f_tr)  + torch.eye(f_tr.shape[1], **CTX) * torch.exp(lam))[0].detach()
        
        if nvalid>0:
            pred = torch.matmul(f_te, W)
            R2 = torch.mean( (pred - t_te)**2) / (t_te.var())
            if mapping in self.R2s:
                self.R2s[mapping] += R2
            else:   
                self.R2s[mapping] = R2
            #print("validation R2 of %10s: %7.5f" % (mapping, R2))
        
        self.Ws[mapping] = W

        return W
        
    def predict(self, data, mapping, inputs=None, detach=True, clip=False):
        
        if inputs is not None:
            new_mapping = inputs + "->" + mapping.split("->")[-1]
        else:
            new_mapping = mapping
        feats, inputs, _ = self.prep_features(data, new_mapping, prep_output=False)

        ni = len(inputs)

        f = self._form_input_tensor(feats, inputs, clip)

        f_flat = self._pivot(f, -ni)
        
        o_flat = f_flat.matmul(self.Ws[mapping])
        
        ''' 
        m, s = self.Ns[mapping]
        o_flat = o_flat*s + m
        ''' 

        o = self._unpivot(o_flat, f.shape[:-ni], self.out_shapes[mapping])
        
        if detach:
            return o.detach()
        else:
            return o
    
    def _kbr_string(self, mapping):
        
        var  = mapping.split(',')
        cond = var[0].split('->')
        mapping = cond[0]+'->'+var[-1]+','+var[-1]
        return mapping

    def train_kbr(self, data, mapping, lam=None):

        lam = self._get_lam(mapping, lam)     
        
        feats, inputs, outputs = self.prep_features(data, mapping)

        W = self.train_weights_closed(data, mapping, lam=torch.exp(lam))
        mapping = self._kbr_string(mapping)
        W = self.train_weights_closed(data, mapping, lam=torch.exp(lam))
        assert torch.all(torch.isfinite(W))
        

    def predict_kbr(self, data, mapping, inputs, lam=None):

        lam = self._get_lam(mapping, lam)     
        feats, inputs, _ = self.prep_features(data, inputs+"->", prep_output=False)
        
        cond  = feats[inputs[1]][...,None]
        
        Czy = self.predict(data, mapping, inputs=inputs[0])
        assert torch.all(torch.isfinite(Czy))

        mapping = self._kbr_string(mapping)

        Cyy = self.predict(data, mapping, inputs=inputs[0])
        assert torch.all(torch.isfinite(Cyy))
        Cyy_phi = torch.gesv( torch.matmul(Cyy, cond), 
                            torch.matmul(Cyy, Cyy) + torch.exp(lam) * torch.eye(Cyy.shape[-1], **CTX))[0]

        return torch.matmul(Czy, Cyy_phi)[...,0]

class LinearRegressorStep(Regressor):
    
    r_type="step"
    def __init__(self, fs, lr=1e-3, momentum=0.0, lam=1e-5, niter = 1, tensor_op="*"):
        
        super(LinearRegressorStep, self).__init__(fs, lam, tensor_op=tensor_op)
        self.lr = lr
        self.momentum=momentum
        self.deltas={}
        self.niter= niter
        
    def train_weights(self, data, mapping, lr=None, nvalid=0, tensor_op = "*", clip=False, lam=None):

        if lr is None: 
            lr = self.lr

        f_tr, t_tr, f_te, t_te = self.prep_train_test(data, mapping, nvalid, clip)
        
        if mapping not in self.Ws:
            delta  = (f_tr.t() @ ( - t_tr))/f_tr.shape[0]
            W = - lr * delta
            self.deltas[mapping] = torch.zeros_like(W)
            self.Ws[mapping] = W
        else:
            W = self.Ws[mapping]
            for i in range(self.niter):
                delta  = (f_tr.t() @ (f_tr @ W - t_tr))/f_tr.shape[0]
                norm = torch.norm(delta)
                if norm > 10:
                    delta /= norm/10

                self.deltas[mapping] *= self.momentum
                self.deltas[mapping] += (1-self.momentum) * delta
                self.Ws[mapping] -= self.deltas[mapping] * lr
        
        return W


    def train_weights_closed(self, data, mapping, lam=None, nvalid=0, tensor_op = "*", clip=False):

        lam = self._get_lam(mapping, lam)     
        f_tr, t_tr, f_te, t_te = self.prep_train_test(data, mapping, nvalid, clip)
        
            
        W = torch.gesv(f_tr.t().matmul(t_tr), 
                       f_tr.t().matmul(f_tr)  + torch.eye(f_tr.shape[1], **CTX) * torch.exp(lam))[0].detach()
        
        if nvalid>0:
            pred = torch.matmul(f_te, W)
            R2 = torch.mean( (pred - t_te)**2) / (t_te.var())
            if mapping in self.R2s:
                self.R2s[mapping] += R2
            else:   
                self.R2s[mapping] = R2
            #print("validation R2 of %10s: %7.5f" % (mapping, R2))
        
        self.Ws[mapping] = W

        return W

    def predict(self, data, mapping, inputs=None, detach=True, clip=False):
        
        if inputs is not None:
            new_mapping = inputs + "->" + mapping.split("->")[-1]
        else:
            new_mapping = mapping
        feats, inputs, _ = self.prep_features(data, new_mapping, prep_output=False)

        ni = len(inputs)

        f = self._form_input_tensor(feats, inputs, clip)

        f_flat = self._pivot(f, -ni)
        
        o_flat = f_flat.matmul(self.Ws[mapping])
        
        ''' 
        m, s = self.Ns[mapping]
        o_flat = o_flat*s + m
        ''' 

        o = self._unpivot(o_flat, f.shape[:-ni], self.out_shapes[mapping])
        
        if detach:
            return o.detach()
        else:
            return o

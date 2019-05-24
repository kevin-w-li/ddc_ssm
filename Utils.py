import numpy as np
import cvxpy as cp

def samples_to_hist(samples, weights, edges):
    
    # length, nchain and sample size
    T, C, N = weights.shape
    E = len(edges)-1
    D = samples.shape[-1]

    posts = np.zeros((T, C, )+(E,)*D)
    for ci in range(C):
        for t in range(T):
            if D == 1:
                posts[t,ci] = np.histogram(samples[t,ci,:,0],
                                              weights=weights[t,ci,:],bins=edges, 
                                              density=True)[0]
            elif D== 2:
                posts[t,ci] = np.histogram2d(samples[t,ci,:,0],samples[t,ci,:,1],
                              weights=weights[t,ci,:],bins=edges, 
                              density=True)[0]
    return posts

def decode(m, f, grid):

    ngrid = len(grid)
    f    = f(grid)

    x = cp.Variable(ngrid)
    objective = cp.Minimize(cp.sum_squares(f.T*x - m))
    constraints = [ x >=0, cp.sum(x) == 1]
    prob = cp.Problem(objective, constraints)

    # The optimal objective is returned by prob.solve().
    result = prob.solve()
    
    xv = x.value
    return xv

def r2(x,y,axis=None):
    assert x.shape == y.shape
    error = np.mean((x-y)**2, axis=axis)
    return 1-error / x.var(axis=axis)


def sort_dim_with_corr(x):
    c = np.corrcoef(x.reshape(-1, x.shape[-1]).T)
    D = x.shape[-1]
    corr_order = np.argsort((c.clip(0,np.inf).sum(-1)))[::-1]
    order = []
    max_i = corr_order[0]
    c[max_i,:] = -np.inf
    order.append(max_i)
    for i in range(D-1):
        max_i = np.argmax(c[:,max_i])
        order.append(max_i)
        c[max_i,:] = -np.inf
    return order

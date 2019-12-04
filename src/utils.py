import numpy as np
from numpy import *

def shuffle_graphadjs(graphadjs):
    graphadjsshuffled = []
    for adj in graphadjs:
        idx = np.arange(adj.shape[0])
        np.random.shuffle(idx)
        graphadjsshuffled.append(adj[idx][:,idx])
    return graphadjsshuffled

def get_incidence(a):
    """make incidence matrix from adjacency matrix"""
    ix = np.where(np.triu(a))
    edg = array(list(zip(*ix)))
    incid = np.zeros((len(a), len(edg)), dtype=np.float32)
    for _,[i,j] in enumerate(edg): 
        incid[i,_] = 1
        incid[j,_] = -1
    return incid

def incidence_shuffle(incid):
    neg = np.where(incid < 0)
    # preserve the rows, to preserve node degree, but shuffle columns, to shuffle links
    np.random.shuffle(neg[1])
    incid2 = 1.* (incid> 0 )
    incid2[neg] -= 1
    return incid2

def configuration_model_1(a, weighted = True):
    incid = get_incidence(a)
    incid2 = incidence_shuffle(incid)
    lap = incid2.dot(incid2.T)
    deg = lap.diagonal() 
    a2 = np.diag(deg)-lap
    if not weighted:
        a2 = (1*(a2>0))
    return a2

def trim_mat(a,b):
    """remove random links from 'a' until it has the same number of links as 'b'"""
    i = (a.sum(0)!= b.sum(0))
    ms = i[:,np.newaxis]*i[np.newaxis]
    masked_a = a*np.triu(ms)
    ix = np.where(masked_a)
    # num missing links
    d = int(a.sum()-b.sum())//2
    # randomly choose r indices to knockout
    r = np.argsort(np.random.rand(len(ix[0])))[:d]
    # make trimmed graph
    c = triu(a) + 0.
    for i in r:
        c[ix[0][i],ix[1][i]]=0
    c += c.T 
    return c
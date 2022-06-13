#!/usr/bin/env python
import json
import numpy as np
from scipy import stats
from scipy.linalg import expm
from scipy.spatial.distance import pdist, squareform
import os
import sys
from pathlib import Path
#Functions
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
def retrieve_shortest_path(s, t, hops, Pmat):
    path_length = hops[s, t]
    if path_length != 0:
        path = np.zeros((int(path_length + 1), 1), dtype='int')
        path[0] = s
        for ind in range(1, len(path)):
            s = Pmat[s, t]
            path[ind] = s
    else:
        path = []

    return path


def distance_wei_floyd(adjacency, transform=None):
    if transform is not None:
        if transform == 'log':
            if np.logical_or(adjacency > 1, adjacency < 0).any():
                raise ValueError("Connection strengths must be in the " +
                                 "interval [0,1) to use the transform " +
                                 "-log(w_ij).")
            SPL = -np.log(adjacency)
        elif transform == 'inv':
            SPL = 1. / adjacency
        else:
            raise ValueError("Unexpected transform type. Only 'log' and " +
                             "'inv' are accepted")
    else:
        SPL = adjacency.copy().astype('float')
        SPL[SPL == 0] = np.inf

    n = adjacency.shape[1]

    flag_find_paths = True
    hops = np.array(adjacency != 0).astype('float')
    Pmat = np.repeat(np.atleast_2d(np.arange(0, n)), n, 0)

    for k in range(n):
        i2k_k2j = np.repeat(SPL[:, [k]], n, 1) + np.repeat(SPL[[k], :], n, 0)

        if flag_find_paths:
            path = SPL > i2k_k2j
            i, j = np.where(path)
            hops[path] = hops[i, k] + hops[k, j]
            Pmat[path] = Pmat[i, k]

        SPL = np.min(np.stack([SPL, i2k_k2j], 2), 2)

    I = np.eye(n) > 0
    SPL[I] = 0

    if flag_find_paths:
        hops[I], Pmat[I] = 0, 0

    return SPL, hops, Pmat



def search_information(adjacency, transform=None, has_memory=False):
    N = len(adjacency)

    if np.allclose(adjacency, adjacency.T):
        flag_triu = True
    else:
        flag_triu = False

    T = np.linalg.solve(np.diag(np.sum(adjacency, axis=1)), adjacency)
    _, hops, Pmat = distance_wei_floyd(adjacency, transform)

    SI = np.zeros((N, N))
    SI[np.eye(N) > 0] = np.nan

    for i in range(N):
        for j in range(N):
            if (j > i and flag_triu) or (not flag_triu and i != j):
                path = retrieve_shortest_path(i, j, hops, Pmat)
                lp = len(path) - 1
                if flag_triu:
                    if np.any(path):
                        pr_step_ff = np.zeros(lp)
                        pr_step_bk = np.zeros(lp)
                        if has_memory:
                            pr_step_ff[0] = T[path[0], path[1]]
                            pr_step_bk[lp-1] = T[path[lp], path[lp-1]]
                            for z in range(1, lp):
                                pr_step_ff[z] = T[path[z], path[z+1]] / (1 - T[path[z-1], path[z]])
                                pr_step_bk[lp-z-1] = T[path[lp-z], path[lp-z-1]] / (1 - T[path[lp-z+1], path[lp-z]])
                        else:
                            for z in range(lp):
                                pr_step_ff[z] = T[path[z], path[z+1]]
                                pr_step_bk[z] = T[path[z+1], path[z]]

                        prob_sp_ff = np.prod(pr_step_ff)
                        prob_sp_bk = np.prod(pr_step_bk)
                        SI[i, j] = -np.log2(prob_sp_ff)
                        SI[j, i] = -np.log2(prob_sp_bk)
                else:
                    if np.any(path):
                        pr_step_ff = np.zeros(lp)
                        if has_memory:
                            pr_step_ff[0] = T[path[0], path[1]]
                            for z in range(1, lp):
                                pr_step_ff[z] = T[path[z], path[z+1]] / (1 - T[path[z-1], path[z]])
                        else:
                            for z in range(lp):
                                pr_step_ff[z] = T[path[z], path[z+1]]

                        prob_sp_ff = np.prod(pr_step_ff)
                        SI[i, j] = -np.log2(prob_sp_ff)
                    else:
                        SI[i, j] = np.inf

    return SI



def path_transitivity(W,transform=None):        
    n=len(W)
    m=np.zeros((n,n))
    T=np.zeros((n,n))
    
    for i in range(n-1):
        for j in range(i+1,n):
            x=0
            y=0
            z=0
            for k in range(n):
                if W[i,k]!=0 and W[j,k]!=0 and k!=i and k!=j:
                    x=x+W[i,k]+W[j,k]
                
                if k!=j:
                    y=y+W[i,k]
                if k!=i:
                    z=z+W[j,k]
            m[i,j]=x/(y+z)
    m=m+m.transpose()
    
    _,hops,Pmat = distance_wei_floyd(W,transform)
    
    #% --- path transitivity ---%%
    for i in range(n-1):
        for j in range(i+1,n):
            x=0
            path = retrieve_shortest_path(i,j,hops,Pmat)
            K=len(path)
            
            for t in range(K-1):
                for l in range(t+1,K):
                    x=x+m[path[t],path[l]]
            T[i,j]=2*x/(K*(K-1))
    T=T+T.transpose()
    
    return T


def communicability_wei(CIJ):
    N = np.size(CIJ,1)
    B = sum(CIJ.transpose())
    C = np.power(B, -0.5)
    D = np.diag(C)
    E = np.matmul(np.matmul(D,CIJ),D)
    F = expm(E)
    F[np.diag_indices_from(F)] = 0
    return F



def mean_first_passage_time(adjacency):
    P = np.linalg.solve(np.diag(np.sum(adjacency, axis=1)), adjacency)

    n = len(P)
    D, V = np.linalg.eig(P.T)

    aux = np.abs(D - 1)
    index = np.where(aux == aux.min())[0]

    if aux[index] > 10e-3:
        raise ValueError("Cannot find eigenvalue of 1. Minimum eigenvalue " +
                         "value is {0}. Tolerance was ".format(aux[index]+1) +
                         "set at 10e-3.")

    w = V[:, index].T
    w = w / np.sum(w)

    W = np.real(np.repeat(w, n, 0))
    I = np.eye(n)

    Z = np.linalg.inv(I - P + W)

    mfpt = (np.repeat(np.atleast_2d(np.diag(Z)), n, 0) - Z) / W

    return mfpt


def matching_ind_und(CIJ0):
    """
    M0 = MATCHING_IND_UND(CIJ) computes matching index for undirected
    graph specified by adjacency matrix CIJ. Matching index is a measure of
    similarity between two nodes' connectivity profiles (excluding their
    mutual connection, should it exist).
    Parameters
    ----------
    CIJ : NxN :obj:`numpy.ndarray`
        undirected adjacency matrix
    Returns
    -------
    M0 : NxN :obj:`numpy.ndarray`
        matching index matrix
    """
    K = np.sum(CIJ0, axis=0)
    n = len(CIJ0)
    R = (K != 0)
    N = np.sum(R)
    xR, = np.where(R == 0)
    CIJ = np.delete(np.delete(CIJ0, xR, axis=0), xR, axis=1)
    inv_eye = np.logical_not(np.eye(N))
    M = np.zeros((N, N))

    for i in range(N):
        c1 = CIJ[i, :]
        use = np.logical_or(c1, CIJ)
        use[:, i] = 0
        use *= inv_eye

        ncon1 = c1 * use
        ncon2 = CIJ * use
        ncon = np.sum(ncon1 + ncon2, axis=1)
        #print(ncon)

        M[:, i] = 2 * np.sum(np.logical_and(ncon1, ncon2), axis=1) / ncon

    M *= inv_eye
    M[np.isnan(M)] = 0
    M0 = np.zeros((n, n))
    yR, = np.where(R)
    M0[np.ix_(yR, yR)] = M
    return M0

def matching_ind(CIJ):
    """
    For any two nodes u and v, the matching index computes the amount of
    overlap in the connection patterns of u and v. Self-connections and
    u-v connections are ignored. The matching index is a symmetric
    quantity, similar to a correlation or a dot product.
    Parameters
    ----------
    CIJ : NxN :obj:`numpy.ndarray`
        adjacency matrix
    Returns
    -------
    Min : NxN :obj:`numpy.ndarray`
        matching index for incoming connections
    Mout : NxN :obj:`numpy.ndarray`
        matching index for outgoing connections
    Mall : NxN :obj:`numpy.ndarray`
        matching index for all connections
    Notes
    -----
    Does not use self- or cross connections for comparison.
    Does not use connections that are not present in BOTH u and v.
    All output matrices are calculated for upper triangular only.
    """
    n = len(CIJ)

    Min = np.zeros((n, n))
    Mout = np.zeros((n, n))
    Mall = np.zeros((n, n))

    # compare incoming connections
    for i in range(n - 1):
        for j in range(i + 1, n):
            c1i = CIJ[:, i]
            c2i = CIJ[:, j]
            usei = np.logical_or(c1i, c2i)
            usei[i] = 0
            usei[j] = 0
            nconi = np.sum(c1i[usei]) + np.sum(c2i[usei])
            if not nconi:
                Min[i, j] = 0
            else:
                Min[i, j] = 2 * \
                    np.sum(np.logical_and(c1i[usei], c2i[usei])) / nconi

            c1o = CIJ[i, :]
            c2o = CIJ[j, :]
            useo = np.logical_or(c1o, c2o)
            useo[i] = 0
            useo[j] = 0
            ncono = np.sum(c1o[useo]) + np.sum(c2o[useo])
            if not ncono:
                Mout[i, j] = 0
            else:
                Mout[i, j] = 2 * \
                    np.sum(np.logical_and(c1o[useo], c2o[useo])) / ncono

            c1a = np.ravel((c1i, c1o))
            c2a = np.ravel((c2i, c2o))
            usea = np.logical_or(c1a, c2a)
            usea[i] = 0
            usea[i + n] = 0
            usea[j] = 0
            usea[j + n] = 0
            ncona = np.sum(c1a[usea]) + np.sum(c2a[usea])
            if not ncona:
                Mall[i, j] = 0
            else:
                Mall[i, j] = 2 * \
                    np.sum(np.logical_and(c1a[usea], c2a[usea])) / ncona

    Min = Min + Min.T
    Mout = Mout + Mout.T
    Mall = Mall + Mall.T

    return Mall

def binarize(W, copy=True):
    if copy:
        W = W.copy()
    W[W != 0] = 1
    return W

def distance_bin(G):
    G = binarize(G, copy=True)
    D = np.eye(len(G))
    n = 1
    nPATH = G.copy()  # n path matrix
    L = (nPATH != 0)  # shortest n-path matrix

    while np.any(L):
        D += n * L
        n += 1
        nPATH = np.dot(nPATH, G)
        L = (nPATH != 0) * (D == 0)

    D[D == 0] = np.inf  # disconnected nodes are assigned d=inf
    np.fill_diagonal(D, 0)
    return D
    ############################


# def error(msg):
# 	global results
# 	results['errors'].append(msg) 
# 	#results['brainlife'].append({"type": "error", "msg": msg}) 
# 	print(msg)


# Choosing config file  ##change "config.json" to "config-sample.json" to test your code locally
configFilename = "config-sample.json"
argCount = len(sys.argv)
if(argCount > 1):
    configFilename = sys.argv[1]

# Defining paths
outputDirectory = "output/csv"

if(not os.path.exists(outputDirectory)):
    os.makedirs(outputDirectory)

# Reading config file
with open(configFilename, "r") as fd:
    config = json.load(fd)


indexFilename = config["index"]
labelFilename = config["label"]
CSVDirectory = config["csv"]

with open(indexFilename, "r") as fd:
    indexData = json.load(fd)

with open(labelFilename, "r") as fd:
    labelData = json.load(fd)
    labelDataHasHeader = False

for entry in indexData:
    entryFilename = entry["filename"]
    a = np.loadtxt(os.path.join(CSVDirectory, entryFilename),delimiter=",")


K = np.sum(a, axis=0)
R = (K != 0)
xR, = np.where(R == 0)
yR, =np.where(R != 0)
if config["clean data"]=="true":
    a = np.delete(np.delete(a, xR, axis=0), xR, axis=1)
# if len(xR)>0 and config["clean data"]=="false":
#     error("connectivity matrix (network) should be fully connected")



abin=a.copy()
abin[abin>0]=1
n = len(a)




gammavals=config['gammavals']




print("binary predictors...")
PLbin = distance_bin(abin)                      # path length
Gbin = expm(abin)                               # communicability
Cosbin = 1 - squareform(pdist(abin,'cosine'))   # cosine distance
mfptbin = mean_first_passage_time(abin) # mean first passage time
MIbin = matching_ind_und(abin)                  # matching index
SIbin = search_information(abin,'inv',False)    # search info
PTbin = path_transitivity(abin,'inv')           # path transitivity





print("weighted predictors...")
Gwei = communicability_wei(a)                  # communicabi
Coswei = 1 - squareform(pdist(a,'cosine'))     # cosine distance
mfptwei = mean_first_passage_time(a)  # mean first passage time
MIwei = matching_ind(a)                    # matching index                          
L = a**-gammavals                # convert weight to cost
PLwei = distance_wei_floyd(L)[0]               # path length
L[np.isinf(L) ]= 0
SIwei = search_information(L,transform=None,has_memory=False)      # search info
PTwei = path_transitivity(L,transform=None)             # path transitivity

print("Saving csv file (individual predictors)...")

np.savetxt('output/csv/PLbin.csv',PLbin,delimiter=',')  
np.savetxt('output/csv/PLwei.csv',PLwei,delimiter=',') 
np.savetxt('output/csv/Gwei.csv',Gwei,delimiter=',') 
np.savetxt('output/csv/Gbin.csv',Gbin,delimiter=',')
np.savetxt('output/csv/Coswei.csv',Coswei,delimiter=',') 
np.savetxt('output/csv/Cosbin.csv',Cosbin,delimiter=',')
np.savetxt('output/csv/SIbin.csv',SIbin,delimiter=',')
np.savetxt('output/csv/SIwei.csv',SIwei,delimiter=',')
np.savetxt('output/csv/PTbin.csv',PTbin,delimiter=',')
np.savetxt('output/csv/PTwei.csv',PTwei,delimiter=',')
np.savetxt('output/csv/MIwei.csv',MIwei,delimiter=',')
np.savetxt('output/csv/MIbin.csv',MIbin,delimiter=',')
np.savetxt('output/csv/mfptwei.csv',mfptwei,delimiter=',')
np.savetxt('output/csv/mfptbin.csv',mfptbin,delimiter=',')

label={"column"+str(i):"node"+str(j) for i,j in enumerate(yR)}
with open('output/label.json', 'w') as outfile:
    json.dump(label,outfile)


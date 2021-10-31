import json
import numpy as np
import pandas as pd
from scipy import stats
from scipy.linalg import expm
import os
import sys
from pathlib import Path



# Choosing config file
configFilename = "config-sample.json"
argCount = len(sys.argv)
if(argCount > 1):
    configFilename = sys.argv[1]

# Defining paths
outputDirectory = "output"

if(not os.path.exists(outputDirectory)):
    os.makedirs(outputDirectory)

# Reading config file
with open(configFilename, "r") as fd:
    config = json.load(fd)


data_file = str(config['conmat'])


print("Loading connectivity matrix...")

CIJ = pd.read_csv(data_file) #load data 

#weighted communicability
N = np.size(CIJ,1)
B = sum(CIJ.transpose())
C = np.power(B, -0.5)
D = np.diag(C)
E = np.matmul(np.matmul(D,CIJ),D)
F = expm(E)
F[np.diag_indices_from(F)] = 0

#MEan fisrt passage time
P = np.linalg.solve(np.diag(np.sum(CIJ, axis=1)), CIJ)

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

print("Saving csv file...")
np.savetxt('outputDirectory/wei_comm.csv',F,delimiter=',') 
np.savetxt('outputDirectory/mfpt.csv',mfpt,delimiter=',') 


# output type is conmat.
import os
import pickle
import time
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import csc_matrix, csr_matrix, dok_matrix
from scipy.optimize import curve_fit
from scipy.io import loadmat

from sgp import GGPgraphmcmc

mat = loadmat('../data/yeast/yeast.mat')
graph = mat['Problem'][0][0][2]

modelparam = dict()
mcmcparam = dict()

modelparam['alpha'] = (0, 0)
modelparam['sigma'] = (0, 0)
modelparam['tau'] = (0, 0)

mcmcparam['niter'] = 100
mcmcparam['nburn'] = 50
mcmcparam['thin'] = 1
mcmcparam['leapfrog.L'] = 5
mcmcparam['leapfrog.epsilon'] = 0.1
mcmcparam['leapfrog.nadapt'] = 25
mcmcparam['latent.MH_nb'] = 1
mcmcparam['hyper.MH_nb'] = 2
mcmcparam['hyper.rw_std'] = [0.02, 0.02]
mcmcparam['store_w'] = True

typegraph='undirected' # or simple

samples, stats = GGPgraphmcmc(graph, modelparam, mcmcparam, typegraph, verbose=True)

print(samples['sigma'])
from scipy.io import loadmat

from sgp import MixGGPgraphmcmc

mat = loadmat('../data/yeast/yeast.mat')
graph = mat['Problem'][0][0][2]

modelparam = dict()
mcmcparam = dict()

modelparam['alpha'] = (0, 0)
modelparam['sigma'] = (-0.1, 0.5)
modelparam['tau'] = (1, 1)
modelparam['n_mixture'] = 2
modelparam['estimate_alpha'] = True
modelparam['estimate_sigma'] = True
modelparam['estimate_tau'] = True
modelparam['alpha_a'] = 1.
modelparam['alpha_b'] = 1.
modelparam['tau_a'] = 1.
modelparam['tau_b'] = 1.
modelparam['dir_alpha'] = 10

mcmcparam['niter'] = 100
mcmcparam['nburn'] = 50
mcmcparam['thin'] = 1
mcmcparam['leapfrog.L'] = 5
mcmcparam['leapfrog.epsilon'] = 0.1
mcmcparam['leapfrog.nadapt'] = 25
mcmcparam['latent.MH_nb'] = 1
mcmcparam['hyper.MH_nb'] = 20
mcmcparam['hyper.rw_std'] = [0.02, 0.02]
mcmcparam['store_w'] = True
mcmcparam['j.niter'] = 1
mcmcparam['u.MH_nb'] = 1

typegraph = 'undirected'  # or simple

MixGGPgraphmcmc(graph, modelparam, mcmcparam, typegraph, verbose=True)

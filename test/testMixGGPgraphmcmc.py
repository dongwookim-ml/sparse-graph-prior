from scipy.io import loadmat

from sgp import MixGGPgraphmcmc
from sgp import GGPrnd, BSgraphrnd, GGPgraphrnd


# mat = loadmat('../data/yeast/yeast.mat')
# graph = mat['Problem'][0][0][2]

# generating a random graph
alpha = 1000
sigma = -1
tau = 3
graph, D, w, w_rem, alpha, sigma, tau = GGPgraphrnd(alpha, sigma, tau)

modelparam = dict()
mcmcparam = dict()

modelparam['alpha'] = (0, 0)
modelparam['sigma'] = (-0.1, 0.5)
modelparam['tau'] = (1, 1)
modelparam['n_mixture'] = 1
# modelparam['n_mixture'] = 2
modelparam['estimate_alpha'] = True
modelparam['estimate_sigma'] = True
modelparam['estimate_tau'] = True
modelparam['alpha_a'] = 1.
modelparam['alpha_b'] = 1.
modelparam['tau_a'] = 1.
modelparam['tau_b'] = 1.
modelparam['dir_alpha'] = 1

mcmcparam['niter'] = 500
mcmcparam['nburn'] = 250
mcmcparam['thin'] = 1
mcmcparam['latent.MH_nb'] = 1
mcmcparam['hyper.MH_nb'] = 10
mcmcparam['hyper.rw_std'] = [0.02, 0.02]
mcmcparam['store_w'] = True
mcmcparam['j.niter'] = 1
mcmcparam['u.MH_nb'] = 1

typegraph = 'undirected'  # or simple

MixGGPgraphmcmc(graph, modelparam, mcmcparam, typegraph, verbose=True)



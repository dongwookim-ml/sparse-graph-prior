"""
Simulating a sparse network and mixture network with various configuration
and save the output network via pickle.
"""
import itertools
import pickle
import os

import numpy as np

from sgp import GGPgraphrnd, GGPmixtureGraphrnd

n_samples = 10 # number of network generated from a given parameter set

alphas = [100]
sigmas = [0.5]
taus = [0.1]

d_alphas = [100]
d_sigmas = [-1]
d_taus = [0.1]

dest = '../result/random_network/mixture/'
if not os.path.exists(dest):
    os.makedirs(dest, exist_ok=True)

sdest = '../result/random_network/sparse/'
if not os.path.exists(sdest):
    os.makedirs(sdest, exist_ok=True)

for i in range(n_samples):
    for s_alpha, s_sigma, s_tau, d_alpha, d_sigma, d_tau in itertools.product(alphas, sigmas, taus,
                                                                              d_alphas, d_sigmas, d_taus):

        file_name = '%d_%.2f_%.2f_%.2f_%.2f_%.2f_%.2f.pkl' % (i, s_alpha, s_sigma, s_tau, d_alpha, d_sigma, d_tau)

        if not os.path.exists(os.path.join(dest, file_name)):
            try:
                sample = GGPmixtureGraphrnd(s_alpha=s_alpha, s_sigma=s_sigma, s_tau=s_tau,
                                            d_alpha=d_alpha, d_sigma=d_sigma,
                                            d_tau=d_tau)
                pickle.dump(sample, open(os.path.join(dest, file_name), 'wb'))
                print('Done', file_name, sample[0].shape[0])
                del sample
            except Exception:
                print('Fail', file_name)

    for s_alpha, s_sigma, s_tau in itertools.product(alphas, sigmas, taus):

        file_name = '%d_%.2f_%.2f_%.2f.pkl' % (i, s_alpha, s_sigma, s_tau)

        if not os.path.exists(os.path.join(sdest, file_name)):
            try:
                sample = GGPgraphrnd(s_alpha, s_sigma, s_tau)
                pickle.dump(sample, open(os.path.join(sdest, file_name), 'wb'))
                print('Done', file_name, sample[0].shape[0])
                del sample
            except Exception:
                print('Fail', file_name)

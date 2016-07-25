import time

import numpy as np
from numpy import log, exp
from scipy.sparse import triu, csr_matrix
from scipy.special import gammaln

from .NCRMmcmc import NGGPmcmc
from .GGPgraphmcmc import tpoissonrnd


def MixGGPgraphmcmc(G, modelparam, mcmcparam, typegraph, verbose=True):
    """
    Run MCMC for the GGP graph model


    Convert the same function used in BNPGraph matlab package by Francois Caron
    http://www.stats.ox.ac.uk/~caron/code/bnpgraph/index.html

    :param G:sparse logical adjacency matrix
    :param modelparam: dictionary of model parameters with the following fields:
        -  alpha: if scalar, the value of alpha. If vector of length
           2, parameters of the gamma prior over alpha
        -  sigma: if scalar, the value of sigma. If vector of length
           2, parameters of the gamma prior over (1-sigma)
        -  tau: if scalar, the value of tau. If vector of length
           2, parameters of the gamma prior over tau
    :param mcmcparam: dictionary of mcmc parameters with the following fields:
        - niter: number of MCMC iterations
        - nburn: number of burn-in iterations
        - thin: thinning of the MCMC output
        - leapfrog.L: number of leapfrog steps
        - leapfrog.epsilon: leapfrog stepsize
        - leapfrog.nadapt: number of iterations for adaptation (default:nburn/2)
        - latent.MH_nb: number of MH iterations for latent (if 0: Gibbs update)
        - hyper.MH_nb: number of MH iterations for hyperparameters
        - hyper.rw_std: standard deviation of the random walk
        - store_w: logical. If true, returns MCMC draws of w
    :param typegraph: type of graph ('undirected' or 'simple') simple graph does
        not contain any self-loop
    :param verbose: logical. If true (default), print information
    :return:
        - samples: dictionary with the MCMC samples for the variables
            - w
            - w_rem
            - alpha
            - logalpha
            - sigma
            - tau
        - stats: dictionary with summary stats about the MCMC algorithm
            - w_rate: acceptance rate of the HMC step at each iteration
            - hyper_rate: acceptance rate of the MH for the hyperparameters at
                each iteration
    """

    n_mixture = modelparam['n_mixture']

    if typegraph is 'simple':
        issimple = True
    else:
        issimple = False

    if modelparam['estimate_alpha']:
        alpha = 100. * np.random.random(size=n_mixture)
        if verbose:
            print('Random Init: alpha', alpha)
    else:
        alpha = modelparam['alpha']

    if modelparam['estimate_sigma']:
        sigma = 2. * np.random.random(size=n_mixture) - 1.
    else:
        sigma = modelparam['sigma']

    if modelparam['estimate_tau']:
        tau = 10. * np.random.random(size=n_mixture)
    else:
        tau = modelparam['tau']

    u = exp(np.random.normal(0, 1 / 4, size=n_mixture))

    K = G.shape[0]  # nodes

    pi = np.random.randint(0, n_mixture, size=K)

    if issimple:
        G2 = triu(G + G.T, k=1)
    else:
        G2 = triu(G + G.T, k=0)

    ind1, ind2 = G2.nonzero()

    n = np.random.randint(1, 5, size=len(ind1))
    count = csr_matrix((n, (ind1, ind2)), shape=(K, K), dtype=int)
    N = count.sum(0).T + count.sum(1)

    niter = mcmcparam['niter']
    nburn = mcmcparam['nburn']
    thin = mcmcparam['thin']

    J = np.zeros(K)
    J_rem = np.zeros(n_mixture)

    n_samples = int((niter - nburn) / thin)
    w_st = np.zeros((n_samples, K))
    w_rem_st = np.zeros((n_samples, n_mixture))
    alpha_st = np.zeros((n_samples, n_mixture))
    tau_st = np.zeros((n_samples, n_mixture))
    sigma_st = np.zeros((n_samples, n_mixture))

    rate = np.zeros(niter)
    rate2 = np.zeros(niter)

    tic = time.time()
    for iter in range(niter):
        if verbose:
            print('Iteration=%d' % iter, flush=True)
            print('\talpha =', alpha, flush=True)
            print('\tsigma =', sigma, flush=True)
            print('\ttau   =', tau, flush=True)

        # update node membership
        logdist = np.zeros(n_mixture)
        for m in range(n_mixture):
            logdist[m] = joint_logdist(N[pi == m], alpha[m], sigma[m], tau[m], u[m])

        for k in range(K):
            prev_m = pi[k]

            logdist[prev_m] += -log(u[prev_m]) - gammaln(N[k] - sigma[prev_m]) + gammaln(1 - sigma[prev_m])

            tmp = np.zeros(n_mixture)
            for m in range(n_mixture):
                tmp[m] = logdist[m] + log(u[m]) + gammaln(N[k] - sigma[m]) - gammaln(1 - sigma[m])

            tmp = log_normalise(tmp)
            pi[k] = np.random.multinomial(1, tmp).argmax()
            new_m = pi[k]

            logdist[new_m] += log(u[new_m]) + gammaln(N[k] - sigma[new_m]) - gammaln(1 - sigma[new_m])

        # update jump size
        for m in range(n_mixture):
            J[pi == m], J_rem[m], u[m] = NGGPmcmc(np.sum(N[pi == m]), N[pi == m], alpha[m], sigma[m], tau[m], u[m],
                                                  mcmcparam)
        logJ = log(J)

        # update hyperparam
        alpha, sigma, tau = update_hyper(N, pi, alpha, sigma, tau, u, n_mixture, modelparam, mcmcparam)

        # update latent count n
        lograte_poi = log(2.) + logJ[ind1] + logJ[ind2]
        lograte_poi[ind1 == ind2] = 2. * logJ[ind1[ind1 == ind2]]
        n = tpoissonrnd(lograte_poi)
        count = csr_matrix((n, (ind1, ind2)), (K, K))
        N = count.sum(0).T + count.sum(1)

        if iter == 10:
            toc = (time.time() - tic) * niter / 10.
            hours = np.floor(toc / 3600)
            minutes = (toc - hours * 3600.) / 60.
            print('-----------------------------------', flush=True)
            print('Start MCMC for GGP graphs', flush=True)
            print('Nb of nodes: %d - Nb of edges: %d' % (K, G2.sum()), flush=True)
            print('Number of iterations: %d' % niter, flush=True)
            print('Estimated computation time: %.0f hour(s) %.0f minute(s)' % (hours, minutes), flush=True)
            print('Estimated end of computation: ', time.strftime('%b %dth, %H:%M:%S', time.localtime(tic + toc)),
                  flush=True)
            print('-----------------------------------', flush=True)

        if iter > nburn and (iter - nburn) % thin == 0:
            ind = int((iter - nburn) / thin)
            if mcmcparam['store_w']:
                w_st[ind] = J
            w_rem_st[ind] = J_rem
            alpha_st[ind] = alpha
            sigma_st[ind] = sigma
            tau_st[ind] = tau


def update_hyper(N, pi, alpha, sigma, tau, u, n_mixture, modelparam, mcmcparam):
    pi_m = np.array([np.sum(pi == m) for m in range(n_mixture)])

    if modelparam['estimate_alpha']:
        alpha_a = modelparam['alpha_a']
        alpha_b = modelparam['alpha_b']
        for m in range(n_mixture):
            alpha[m] = np.random.gamma(alpha_a + pi_m[m], tau[m] / (alpha_b + ((u[m] + tau[m]) ** sigma[m] - tau[m] ** sigma[m])))

    if modelparam['estimate_sigma']:
        pass

    if modelparam['estimate_tau']:
        pass

    return alpha, sigma, tau


def log_normalise(log_prob):
    log_prob -= np.max(log_prob)
    return exp(log_prob)


def joint_logdist(pi, alpha, sigma, tau, u):
    abs_pi = len(pi)
    n = np.sum(pi)
    tmp = abs_pi * log(alpha) + (n - 1.) * log(u) - gammaln(n) - (n - sigma * abs_pi) * log(u + tau) \
          - (alpha / sigma) * ((u + tau) ** sigma - tau ** sigma)
    tmp += np.sum(gammaln(pi - sigma) - gammaln(1. - sigma))
    return tmp

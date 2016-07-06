import time

import numpy as np
from scipy.stats import poisson
from scipy.sparse import triu, csr_matrix
from scipy.special import gammaln
from numpy import log, exp

from .GGPutils import GGPpsi, GGPkappa, GGPsumrnd


def tpoissonrnd(rate):
    # sample from a zero-truncated poisson distribution (support: strictly positive integer)
    x = np.ones(rate.shape)
    ind = rate > 1e-5 # below this value, x=1 w. very high prob.
    rate_ind = rate[ind]
    x[ind] = poisson.ppf(exp(-rate_ind) + np.random.random(rate_ind.shape) * (1. - exp(-rate_ind)), rate_ind)
    return x


def grad_U(N, w, w_rem, sigma, tau):
    return -(N - sigma) + w * (tau + 2. * np.sum(w) + 2. * w_rem)


def update_n_Gibbs(logw, K, ind1, ind2):
    lograte_poi = log(2.) + logw[ind1] + logw[ind2]
    lograte_poi[ind1 == ind2] = 2. * logw[ind1[ind1 == ind2]]
    d = tpoissonrnd(exp(lograte_poi))
    count = csr_matrix((d, (ind1, ind2)), (K, K))
    N = np.sum(count, 0) + np.sum(count, 1)

    return N, d, count


def update_n_MH(logw, d, K, count, ind1, ind2, nbMH):
    for i in range(nbMH):
        lograte_poi = log(2.) + logw[ind1] + logw[ind2]
        lograte_poi[ind1 == ind2] = 2. * logw[ind1[ind1 == ind2]]
        ind = d == 1
        dprop = d
        dprop[ind] = 2
        if np.sum(ind == 0) > 0:
            dprop[ind == 0] += 2. * np.random.randint(0, 1, size=np.sum(ind == 0)) - 3

        logqprop = np.zeros(ind.shape)
        logqprop[ind == 0] = log(.5)

        indbis = dprop == 1
        logq = np.zeros(indbis.shape)
        if np.sum(indbis == 0) > 0:
            logq[indbis == 0] = log(.5)

        diff_d = (dprop - d)
        logaccept_d = diff_d * lograte_poi - gammaln(dprop + 1.) + gammaln(d + 1.) - logqprop + logq

        indaccept = log(np.random.random(logaccept_d.shape)) < logaccept_d
        d[indaccept] = dprop[indaccept]
        count += csr_matrix((diff_d[indaccept], (ind1[indaccept], ind2[indaccept])), (K, K))
        N = np.sum(count, 0) + np.sum(count, 1)

    return N, d, count


def update_w(w, logw, w_rem, N, L, epsilon, sigma, tau, issimple):
    sum_w = np.sum(w)
    sumall_w = sum_w + w_rem

    K = len(w)
    logwprop = logw
    p = np.random.normal(size=(K))
    grad1 = grad_U(N, w, w_rem, sigma, tau)
    pprop = p - epsilon * grad1 / 2.

    for lp in range(L):
        logwprop = logwprop + epsilon * pprop
        if lp != L - 1:
            pprop = pprop - epsilon * grad_U(N, exp(logwprop), w_rem, sigma, tau)

    wprop = exp(logwprop)
    pprop = pprop - epsilon / 2. * grad_U(N, wprop, w_rem, sigma, tau)

    sum_wprop = np.sum(wprop)
    sumall_wprop = sum_wprop + w_rem
    temp1 = -sumall_wprop ** 2. + sumall_w ** 2 + np.sum(
        (N - sigma - 1.) * (logwprop - logw) - tau * (sum_wprop - sum_w))

    logaccept = temp1 - .5 * np.sum(pprop ** 2. - p ** 2) - np.sum(logw) + np.sum(logwprop)
    if issimple:
        logaccept += np.sum(wprop ** 2.) - np.sum(w ** 2.)

    if np.isnan(logaccept):
        logaccept = -np.Inf

    if log(np.random.random()) < logaccept:
        w = wprop
        logw = logwprop

    rate = np.min(1, exp(logaccept))
    return w, logw, rate


def update_hyper(w, logw, w_rem, alpha, logalpha, sigma,
                 tau, nbMH, rw_std, estimate_alpha, estimate_sigma, estimate_tau,
                 hyper_alpha, hyper_sigma, hyper_tau, rw_alpha):
    K = len(w)
    for nn in range(nbMH):
        sum_w = np.sum(w)
        sumall_w = sum_w + w_rem

        # Sample (alpha, sigma, tau, w*) from the proposal distribution
        if estimate_sigma:
            sigmaprop = 1. - exp(log(1 - sigma) + rw_std[0] * np.random.normal())
        else:
            sigmaprop = sigma

        if estimate_tau:
            tauprop = exp(log(tau) + rw_std[1] * np.random.normal())
        else:
            tauprop = tau

        if sigmaprop > -1.:
            if estimate_alpha:
                if rw_alpha == 0:
                    alphaprop = np.random.gamma(K, 1. / (GGPpsi(2. * sum_w + 2. * w_rem, 1., sigmaprop, tauprop)))
                else:
                    alphaprop = alpha * exp(.02 * np.random.normal())
                logalphaprop = log(alphaprop)
            else:
                alphaprop = alpha
                logalphaprop = logalpha

            wprop_rem = GGPsumrnd(alphaprop, sigmaprop, tauprop + 2. * sum_w + 2. * w_rem)
        else:
            if estimate_alpha:
                if rw_alpha == 0:
                    alpha2prop = np.random.gamma(K,
                                                 1. / (GGPpsi((2. * sum_w + 2. * w_rem) / tauprop, 1., sigmaprop, 1.)))
                    logalphaprop = log(alpha2prop) - sigmaprop * log(tauprop)
                else:
                    logalphaprop = logalpha + .02 * np.random.normal()
                alphaprop = exp(logalphaprop)
                rate_K = exp(logalphaprop - log(-sigmaprop) + sigmaprop * log(tauprop + 2. * sum_w + 2. * w_rem))
                num_clust = np.random.poisson(rate_K)
                wprop_rem = np.random.gamma(-sigmaprop * num_clust, 1. / (tauprop + 2. * sum_w + 2. * w_rem))
            else:
                alphaprop = alpha
                logalphaprop = logalpha
                wprop_rem = GGPsumrnd(alphaprop, sigmaprop, tauprop + 2. * sum_w + 2. * w_rem)

        # compute the acceptance probability
        sum_wprop = np.sum(w)
        sumall_wprop = sum_wprop + wprop_rem

        temp1 = -sumall_wprop ** 2. + sumall_w ** 2. + (sigma - sigmaprop) * np.sum(logw) \
                - (tauprop - tau - 2. * wprop_rem + 2. * w_rem) * sum_w
        temp2 = K * (gammaln(1. - sigma) - gammaln(1. - sigmaprop))

        logaccept = temp1 + temp2
        if estimate_alpha:
            if rw_alpha == 0:
                logaccept += K * (
                    log(GGPpsi((2. * sum_wprop + 2. * wprop_rem) / tau, 1., sigma, 1.)) + sigma * log(tau) - log(
                        (GGPpsi((2. * sum_w + 2. * w_rem) / tauprop, 1., sigmaprop, 1.))) - sigmaprop * log(tauprop))
            else:
                logaccept -= exp(logalphaprop + sigmaprop * log(tauprop)) * GGPpsi(
                    (2. * sum_w + 2. * w_rem) / tauprop, 1., sigmaprop, 1.) + exp(logalpha + sigma * log(tau)) * GGPpsi(
                    (2. * sum_wprop + 2. * wprop_rem) / tau, 1., sigma, 1.)
                + K * (logalphaprop - logalpha)
            if hyper_alpha[0] > 0:
                logaccept += hyper_alpha[0] * (logalphaprop - logalpha)
            if hyper_alpha[1] > 0:
                logaccept -= hyper_alpha[1] * (alphaprop - alpha)

        logaccept -= GGPpsi(2. * sum_w + 2. * w_rem, alphaprop, sigmaprop, tauprop) + GGPpsi(
            2. * sum_wprop + 2. * wprop_rem, alpha, sigma, tau)

        if estimate_tau:
            logaccept += hyper_tau[0] * (log(tauprop) - log(tau)) - hyper_tau[1] * (tauprop - tau)
        if estimate_sigma:
            logaccept += hyper_sigma[0] * (log(1. - sigmaprop) - log(1. - sigma)) - hyper_sigma[1] * (
                1. - sigmaprop - 1. + sigma)

        if np.isnan(logaccept):
            raise Exception("Cannot compute acceptance probability")

        if log(np.random.random()) < logaccept:
            w_rem = wprop_rem
            alpha = alphaprop
            logalpha = logalphaprop
            sigma = sigmaprop
            tau = tauprop
    rate2 = min(1., exp(logaccept))

    return w_rem, alpha, logalpha, sigma, tau, rate2


def GGPgraphmcmc(G, modelparam, mcmcparam, typegraph, verbose=True):
    """
    Run MCMC for the GGP graph model


    Convert the same function used in BNPGraph matlab package by Francois Caron
    http://www.stats.ox.ac.uk/~caron/code/bnpgraph/index.html

    :param G:sparse logical adjacency matrix
    :param modelparam: structure of model parameters with the following fields:
        -  alpha: if scalar, the value of alpha. If vector of length
           2, parameters of the gamma prior over alpha
        -  sigma: if scalar, the value of sigma. If vector of length
           2, parameters of the gamma prior over (1-sigma)
        -  tau: if scalar, the value of tau. If vector of length
           2, parameters of the gamma prior over tau
    :param mcmcparam: structure of mcmc parameters with the following fields:
        - niter: number of MCMC iterations
        - nburn: number of burn-in iterations
        - thin: thinning of the MCMC output
        - leapfrog.L: number of leapfrog steps
        - leapfrog.epsilon: leapfrog stepsize
        - latent.MH_nb: number of MH iterations for latent (if 0: Gibbs update)
        - hyper.MH_nb: number of MH iterations for hyperparameters
        - hyper.rw_std: standard deviation of the random walk
        - store_w: logical. If true, returns MCMC draws of w
    :param typegraph: type of graph ('undirected' or 'simple')
    :param verbose: logical. If true (default), print informatio
    :return:
        - samples: structure with the MCMC samples for the variables
            - w
            - w_rem
            - alpha
            - logalpha
            - sigma
            - tau
        - stats: structure with summary stats about the MCMC algorithm
            - rate: acceptance rate of the HMC step at each iteration
            - rate2: acceptance rate of the MH for the hyperparameters at
                each iteration
    """

    if not G is G.T or not len(np.unique(G)) != 2:
        raise Exception("Adjacency matrix G must be a symmetric binary matrix")

    if typegraph is 'simple':
        issimple = True
    else:
        issimple = False

    if not modelparam.alpha is None:
        alpha = modelparam.alpha
        estimated_alpha = 0
    else:
        alpha = 100. * np.random.random()
        estimated_alpha = 1

    logalpha = log(alpha)

    if not modelparam.sigma is None:
        sigma = modelparam.sigma
        estimated_sigma = 0
    else:
        sigma = 2. * np.random.random() - 1.
        estimated_sigma = 1

    if not modelparam.tau is None:
        tau = modelparam.tau
        estimated_tau = 0
    else:
        sigma = 10. * np.random.random()
        estimated_tau = 1

    K = G.shape[0]  # nodes

    if issimple:
        G2 = triu(G + G.T, k=1)
    else:
        G2 = triu(G + G.T, k=0)

    ind1, ind2 = G2.nonzero()

    n = np.random.randint(0, 9, size=len(ind1))
    count = csr_matrix((n, (ind1, ind2)), (K, K), dtype=int)
    N = np.sum(count, 0) + np.sum(count, 1)
    w = np.random.gamma(1., 1., size=K)
    logw = log(w)
    w_rem = np.random.gamma(1., 1.)

    # parameters of the MCMC algorithm
    niter = mcmcparam.niter
    nburn = mcmcparam.nburn
    thin = mcmcparam.thin
    L = mcmcparam.leapfrog_L
    epsilon = mcmcparam.leapfrog_epsilon / K ** (1. / 4.)
    # Choice of update for the latent (Gibbs/MH)
    if mcmcparam.latent_MH_nb == 0:
        update_n = update_n_Gibbs
    else:
        update_n = update_n_MH

    # To store MCMC samples
    n_samples = (niter - nburn) / thin
    w_st = np.zeros((n_samples, K))
    w_rem_st = np.zeros(n_samples)
    alpha_st = np.zeros(n_samples)
    logalpha_st = np.zeros(n_samples)
    tau_st = np.zeros(n_samples)
    sigma_st = np.zeros(n_samples)

    rate = np.zeros(niter)
    rate2 = np.zeros(niter)

    tic = time.time()
    for i in range(niter):
        if verbose and i % 2000 == 0:
            print('i=%d\n' % i)
            print('alpha=%.2f\n' % alpha)
            print('sigma=%.3f\n' % sigma)
            print('tau=%.2f\n' % tau)

        # update w using Hamiltonian Monte Carlo
        w, logw, rate[i] = update_w(w, logw, w_rem, N, L, epsilon, sigma, tau, issimple)
        if i < mcmcparam.leapfrog_nadapt:
            epsilon = exp(log(epsilon) + .01 * (np.mean(rate[:i]) - 0.6))

        # update w_rem and hyperparameters using Metropolis-Hastings
        if i % 2 == 0:
            rw_alpha = True
        else:
            rw_alpha = False

        w_rem, alpha, logalpha, sigma, tau, rate2[i] = update_hyper(w, logw, w_rem, alpha, logalpha, sigma, tau,
                                                                    mcmcparam.hyper_MH_nb, mcmcparam.hyper_rw_std,
                                                                    estimated_alpha, estimated_sigma, estimated_tau,
                                                                    modelparam.alpha, modelparam.sigma, modelparam.tau,
                                                                    rw_alpha)

        N, n, count = update_n(logw, n, K, count, ind1, ind2)

        if np.isnan(alpha):
            raise Exception('alpha is not a number')

        # Print text at start
        if i == 10:
            toc = (time.time() - tic) * niter / 10.
            hours = np.floor(toc / 3600)
            minutes = (toc - hours * 3600.) / 60.
            print('-----------------------------------\n')
            print('Start MCMC for GGP graphs\n')
            print('Nb of nodes: %d - Nb of edges: %d\n' % (K, np.sum(G2)))
            print('Number of iterations: %d\n' % niter)
            print('Estimated computation time: %.0f hour(s) %.0f minute(s)\n' % (hours, minutes))
            # print('Estimated end of computation: %s \n'% datestr(now + time/3600/24))
            print('-----------------------------------\n')

        if i > nburn and (i - nburn) % thin == 0:
            ind = ((i - nburn) / thin)
            if mcmcparam.store_w:
                w_st[ind] = w
            w_rem_st[ind] = w_rem
            logalpha_st[ind] = logalpha
            alpha_st[ind] = alpha
            sigma_st[ind] = sigma
            tau_st[ind] = tau

    samples = [w_st, w_rem_st, alpha_st, logalpha_st, sigma_st, tau_st]
    stats = [rate, rate2]

    toc = time.time() - tic
    hours = np.floor(toc / 3600)
    minutes = (toc - hours * 3600.) / 60.
    print('-----------------------------------\n')
    print('End MCMC for GGP graphs\n')
    print('Computation time: %.0f hour(s) %.0f minute(s)\n' % (hours, minutes))
    # print('End of computation: %s \n', datestr(now))
    print('-----------------------------------\n')

    return samples, stats

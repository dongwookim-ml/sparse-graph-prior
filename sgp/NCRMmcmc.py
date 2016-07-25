import numpy as np
from numpy import log, exp
from scipy.stats import norm

from .GGPutils import GGPsumrnd, GGPkappa


def log_density_v(v, n, C, alpha, sigma, tau):
    return v * n - (n - alpha * C) * log(exp(v) + tau) - (alpha / sigma) * ((exp(v) + tau) ** sigma - tau ** sigma)


def sampling_u(u, n, C, alpha, sigma, tau, n_steps=1):
    """
    Metropolis Hasting for auxiliary variable u

    :param u: previous u
    :param n: number of observations
    :param C: number of clusters
    :param alpha: strictly positive scalar
    :param sigma: (-infty, 1)
    :param tau: positive scalar
    :return:
    """

    for i in range(n_steps):
        v = log(u)
        var = 1. / 4.
        std = np.sqrt(var)
        prop_v = np.random.normal(v, 1. / 4.)

        # compute acceptance probability
        log_rate = log_density_v(prop_v, n, C, alpha, sigma, tau) + norm.logpdf(v, prop_v, std) \
                   - log_density_v(v, n, C, alpha, sigma, tau) - norm.logpdf(prop_v, v, std)

        rate = min(1, np.exp(log_rate))

        if np.random.random() < rate:
            v = prop_v

    return exp(v), rate


def NGGPmcmc(n, pi, alpha, sigma, tau, u, MCMCparams):
    """
    Sampling posterior distribution of the underlying GGP given observations from NGGP

    :param n: number of observations
    :param pi: size of each cluster
    :param alpha: strictly positive scalar
    :param sigma: (-infty, 1)
    :param tau: positive scalar
    :param u: strictly positive scalar
    :param MCMCparams:
        - j.niter: number of MCMC iterations for j
        - u.MH_nb: number of MH iterations for auxiliary variable u
    :return:
        - J: jump size for each cluster
        - J_rem: remaining jump from GGP
        - u: auxiliary variable
    """

    C = pi.size
    J = np.zeros(C)

    for iter in MCMCparams['j.niter']:
        for i in range(C):
            u = sampling_u(u, n, C, alpha, sigma, tau, MCMCparams['u.MH_nb'])
            J[i] = np.random.gamma(pi[i] - sigma, u + tau)

        u = sampling_u(u, n, C, alpha, sigma, tau, MCMCparams['u.MH_nb'])
        J_rem = GGPsumrnd(alpha, sigma, u + tau)

    return J, J_rem, u

import numpy as np
from numpy import log, exp
from scipy.stats import norm, lognorm
from scipy.special import gammaln

from .GGPutils import GGPsumrnd


# when sigma < 0, this is not a valid log density of v
def log_density_v(v, n, abs_pi, alpha, sigma, tau):
    return v * n - (n - alpha * abs_pi) * log(exp(v) + tau) - (alpha / sigma) * ((exp(v) + tau) ** sigma - tau ** sigma)


def log_density_tau(logtau, alpha, sigma, u, n, abs_pi, tau_a, tau_b):
    return (tau_a - 1) * logtau - exp(logtau) * tau_b \
           - (alpha / sigma) * ((u + exp(logtau)) ** sigma - exp(sigma * logtau)) \
           - sigma * abs_pi * logtau - (n - sigma * abs_pi) * log(u + exp(logtau))


def log_density_sigma(alpha, sigma, tau, u, abs_pi, n, pi):
    return -(alpha / sigma) * ((u + tau) ** sigma - tau ** sigma) - sigma * abs_pi * log(tau) \
           - (n - sigma * abs_pi) * log(u + tau) - log(1 - sigma) + np.sum(gammaln(pi - sigma)) - abs_pi * gammaln(
        1 - sigma)


def update_hyper(n, pi, alpha, sigma, tau, u, modelparam, mcmcparam):
    abs_pi = pi.size

    if modelparam['estimate_alpha']:
        alpha_a = modelparam['alpha_a']
        alpha_b = modelparam['alpha_b']
        alpha = np.random.gamma(alpha_a + abs_pi,
                                1. / (alpha_b + ((u + tau) ** sigma
                                                 - tau ** sigma) / sigma))

    if modelparam['estimate_sigma']:
        # std = np.sqrt(1. / 4.)
        std = 0.1
        prop_sigma = 1 - np.random.lognormal(log(1 - sigma), std)

        log_rate = log_density_sigma(alpha, prop_sigma, tau, u, abs_pi, n, pi) \
                   + lognorm.logpdf(1 - sigma, std, scale=(1 - prop_sigma)) \
                   - log_density_sigma(alpha, sigma, tau, u, abs_pi, n, pi) \
                   - lognorm.logpdf(1 - prop_sigma, std, scale=(1 - sigma))

        if np.isnan(log_rate):
            log_rate = -np.Inf

        if np.isinf(prop_sigma):
            log_rate = -np.Inf

        rate = min(1, np.exp(log_rate))

        if np.random.random() < rate:
            sigma = prop_sigma

    if modelparam['estimate_tau']:
        tau_a = modelparam['tau_a']
        tau_b = modelparam['tau_b']
        std = np.sqrt(1. / 4.)
        logtau = log(tau)
        prop_logtau = np.random.normal(logtau, std)

        # compute acceptance probability
        log_rate = log_density_tau(prop_logtau, alpha, sigma, u, n, abs_pi, tau_a,
                                   tau_b) + norm.logpdf(logtau, prop_logtau, std) \
                   - log_density_tau(logtau, alpha, sigma, u, n, abs_pi, tau_a,
                                     tau_b) - norm.logpdf(prop_logtau, logtau, std)

        if np.isnan(log_rate):
            log_rate = -np.Inf

        if np.isinf(exp(prop_logtau)) or np.isnan(exp(prop_logtau)):
            log_rate = -np.Inf

        rate = min(1, np.exp(log_rate))

        if np.random.random() < rate:
            tau = exp(prop_logtau)

    return alpha, sigma, tau


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
        prop_v = np.random.normal(v, std)

        # compute acceptance probability
        log_rate = log_density_v(prop_v, n, C, alpha, sigma, tau) + norm.logpdf(v, prop_v, std) \
                   - log_density_v(v, n, C, alpha, sigma, tau) - norm.logpdf(prop_v, v, std)

        if np.isnan(log_rate):
            log_rate = -np.Inf

        # prevent following problems caused by infinity
        if np.isinf(exp(prop_v)):
            log_rate = -np.Inf

        rate = min(1, exp(log_rate))

        if np.random.random() < rate:
            v = prop_v
            u = exp(v)

    return u, rate


def NGGPmcmc(n, pi, alpha, sigma, tau, u, modelparam, mcmcparam, verbose=False, isinfinite=True):
    """
    Sampling posterior distribution of the underlying GGP given observations from NGGP

    :param n: number of observations
    :param pi: size of each cluster
    :param alpha: strictly positive scalar
    :param sigma: (-infty, 1)
    :param tau: positive scalar
    :param u: strictly positive scalar
    :param mcmcparam:
        - j.niter: number of MCMC iterations for j
        - u.MH_nb: number of MH iterations for auxiliary variable u
    :return:
        - J: jump size for each cluster
        - J_rem: remaining jump from GGP
        - u: auxiliary variable
    """

    C = pi.size
    J = np.zeros(C)

    if isinfinite:
        for i in range(mcmcparam['j.niter']):
            for m in range(C):
                u, rate = sampling_u(u, n, C, alpha, sigma, tau, mcmcparam['u.MH_nb'])
                scale = 1. / (u + tau)
                if scale < 1e-100:
                    J[m] = np.random.gamma(pi[m] - sigma, 1e-100)
                else:
                    J[m] = np.random.gamma(pi[m] - sigma, scale)

            u, rate = sampling_u(u, n, C, alpha, sigma, tau, mcmcparam['u.MH_nb'])

            J_rem = GGPsumrnd(alpha, sigma, u + tau)

            for j in range(mcmcparam['hyper.MH_nb']):
                alpha, sigma, tau = update_hyper(n, pi, alpha, sigma, tau, u, modelparam, mcmcparam)

            if verbose:
                print("%d: %.2f\t%.2f\t%.2f\t%.2f" % (i, alpha, sigma, tau, u))

    else:
        for i in range(mcmcparam['j.niter']):
            for m in range(C):
                scale = 1. / tau

    return J, J_rem, alpha, sigma, tau, u

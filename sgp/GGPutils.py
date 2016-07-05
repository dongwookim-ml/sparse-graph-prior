from scipy.special import gammaln
import numpy as np
from .etstablernd import etstablernd


def GGPsumrnd(alpha, sigma, tau):
    """
    GGPsumrnd samples from the distribution of the total mass of a GGP

    It generates a realisation of the random variable S with Laplace transform in t
    E[e^-(t*S)] = exp(-alpha/sigma * [(t+tau)^sigma - tau^sigma]

    Convert the same function used in BNPGraph matlab package by Francois Caron
    http://www.stats.ox.ac.uk/~caron/code/bnpgraph/index.html

    :param alpha: positive scalar
    :param sigma: real in (-Inf, 1)
    :param tau: positive scalar
    :return: positive scalar
    """

    if sigma < -10 ** -8:
        # Compound Poisson case
        K = np.random.poisson(-alpha / sigma / tau ** (-sigma))
        S = np.random.gamma(-sigma * K, 1 / tau)
    elif sigma < 10 ** -8:
        # Gamma process case
        # S is gamma distributed
        S = np.random.gamma(alpha, 1 / tau)
    elif sigma == 0.5 and tau == 0:
        # Inverse Gaussian process case
        # S is distributed from an inverse Gaussian distribution
        _lambda = 2*alpha**2
        mu = alpha/np.sqrt(tau)
        S = np.random.wald(mu, _lambda)
    else:
        # General case
        # S is distributed from an exponentially tilted stable distribution
        S = etstablernd(alpha/sigma, sigma, tau)

    return S


def GGPkappa(n, z, alpha, sigma, tau):
    """
    Compute nth moment of a tilted GGP

    kappa(n,z) = = int_0^infty w^n * exp(-zw) * rho(w)dw
              = alpha * (z+tau)^(n-sigma) * gamma(n-sigma)/gamma(1-sigma)
       where rho(w) = alpha/gamma(1-sigma) * w^(-1-sigma) * exp(-tau*w)
       is the Levy measure of a generalized gamma process

    Convert the same function used in BNPGraph matlab package by Francois Caron
    http://www.stats.ox.ac.uk/~caron/code/bnpgraph/index.html

    :param n: strictly positive integer
    :param z: positive scalar
    :param alpha: positive scalar
    :param sigma: real in (-Inf, 1)
    :param tau: positive scalar
    :return:
        kappa: kappa(n,z)
        log_kappa: log(kappa(n,z))
    """

    log_kappa = np.log(alpha) - (n-sigma) * np.log(z+ tau) + gammaln(n-sigma) - gammaln(1.-sigma)
    return np.exp(log_kappa), log_kappa


def GGPpsi(t, alpha, sigma, tau):
    """
    Compute the laplace exponent of a GGP
    The Laplace exponent of a GGP evaluated at t is
    psi(t) = -log ( E[exp(-t * sum_i w_i)] )
           = alpha/sigma * ( (t+tau)^sigma - tau^sigma))
        where
        (w_i)_{i=1,2,..} are the points of a Poisson process on R_+ of mean measure
        rho(dw) = alpha/Gamma(1-sigma) * w^{-1-sigma} * exp(-tau*w)dw

    Convert the same function used in BNPGraph matlab package by Francois Caron
    http://www.stats.ox.ac.uk/~caron/code/bnpgraph/index.html

    :param t: vector of positive scalars of length n
    :param alpha: positive scalar
    :param sigma: real in (-inf, 1)
    :param tau: positive scalar
    :return:
        lp: Laplace exponent evaluated at the values t
    """

    if sigma == 0:
        lp = alpha * np.log(1.+t/tau)
    else:
        lp = alpha/sigma * ((t+tau)**sigma - tau ** sigma)
    return lp
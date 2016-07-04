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



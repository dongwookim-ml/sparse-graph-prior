import numpy as np
from scipy.special import gamma, gammaln


def W(t, x, alpha, sigma, tau):
    if tau > 0:
        logout = np.log(alpha) + np.log(1. - np.exp(-tau * (x - t))) + (-1 - sigma) * np.log(t) + (-t * tau) - np.log(
            tau) - gammaln(1. - sigma)
    else:
        logout = np.log(alpha) - gammaln(1. - sigma) - np.log(sigma) + np.log(t ** (-sigma) - x ** (-sigma))
    return np.exp(logout)


def inv_W(t, x, alpha, sigma, tau):
    if tau > 0:
        out = t - 1. / tau * np.log(1. - gamma(1. - sigma) * x * tau / (alpha * t ** (-1. - sigma) * np.exp(-t * tau)))
    else:
        logout = -1. / sigma * np.log(t ** (-sigma) - sigma * gamma(1. - sigma) / alpha * x)
        out = np.exp(logout)
    return out


def GGPrnd(alpha, sigma, tau, T=0):
    """
    GGPrnd samples points of a generalised gamma process

    Samples the points of the GGP with Levy measure
      alpha/Gamma(1-sigma) * w^(-1-sigma) * exp(-tau*w)

    For sigma>=0, it samples points above the threshold T>0 using the adaptive
    thinning strategy described in Favaro and Teh (2013).

    Convert the same function used in BNPGraph matlab package by Francois Caron
    http://www.stats.ox.ac.uk/~caron/code/bnpgraph/index.html

    Reference:
    S. Favaro and Y.W. Teh. MCMC for normalized random measure mixture
        models. Statistical Science, vol.28(3), pp.335-359, 2013.
    :param alpha: positive scalar
    :param sigma: real in (-Inf, 1)
    :param tau: positive scalar
    :param T: truncation threshold; positive scalar
    :return:
        N: weights from the GGP
        T: threshold
    """

    # finite activity GGP, don't need to truncate
    if sigma < 0:
        rate = np.exp(np.log(alpha) - np.log(-sigma) + sigma + np.log(tau))
        K = np.random.poisson(rate)
        N = np.random.gamma(-sigma, scale=1. / tau, size=K)
        T = 0
        return N, T

    # infinite activity GGP
    if T == 0:
        # set the threshold automatically
        # Number of jumps of order alpha/sigma/Gamma(1-sigma) * T^{-sigma} for sigma > 0
        # and alpha*log(T) for sigma = 0
        if sigma > .1:
            Njumps = 20000
            T = np.exp(1. / sigma * (np.log(alpha) - np.log(sigma) - gammaln(1. - sigma) - np.log(Njumps)))
        else:
            T = 1e-10
            if sigma > 0:
                Njumps = np.floor(alpha / sigma / gamma(1. - sigma) * T ** (-sigma))
            else:
                Njumps = np.floor(-alpha * np.log(T))
    else:
        if T <= 0:
            raise ValueError("Threshold T must be strictly positive")
        if sigma > 1e-3:
            Njumps = np.floor(alpha / sigma / gamma(1. - sigma) * T ** (-sigma))
        else:
            Njumps = np.floor(-alpha * np.log(T))
        if Njumps > 1e7:
            raise Warning("Expected number of jumps = %d" % Njumps)

    # Adaptive thinning strategy
    N = np.zeros(int(np.ceil(Njumps + 3 * np.sqrt(Njumps))))
    k = 0
    t = T
    count = 0
    while True:
        e = -np.log(np.random.random())  # Sample exponential random variable of unit rate

        if e > W(t, np.inf, alpha, sigma, tau):
            N = N[0:k]
            return N, T
        else:
            t_new = inv_W(t, e, alpha, sigma, tau)

        if tau == 0 or np.log(np.random.random()) < (-1. - sigma) * np.log(t_new / t):
            # if tau>0, adaptive thinning - otherwise accept always
            if k > len(N):
                N = np.append(N, np.zeros(Njumps))
            N[k] = t_new
            k += 1

        t = t_new
        count += 1
        if count > 10e8:
            # If too many computation, we lower the threshold T and rerun
            T /= 10.
            N, T = GGPrnd(alpha, sigma, tau, T)

    return N, T

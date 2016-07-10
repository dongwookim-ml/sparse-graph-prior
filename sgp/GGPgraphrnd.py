import collections
import numpy as np
from scipy.sparse import csc_matrix
from .GGPrnd import GGPrnd


def GGPgraphrnd(alpha, sigma, tau, T=0):
    """
    Generate (sample) a random graph.

    :param alpha: positive scalar
    :param sigma: real in (-inf, 1)
    :param tau: positive scalar
    :param T: truncation threshold; positive scalar
    :return:
        G: undirected graph
        D: directed multigraph used to generate G
        w: sociability param of each nodes
        w_rem: sum of sociability of unactivated nodes
        alpha: parameter used to generate graph
        sigma: parameter
        tau: parameter
    """

    if isinstance(alpha, collections.Iterable):
        hyper_alpha = alpha
        alpha = np.random.gamma(hyper_alpha[0], 1./hyper_alpha[1])
    if isinstance(sigma, collections.Iterable):
        hyper_sigma = sigma
        sigma = 1. - np.random.gamma(hyper_sigma[0], 1./hyper_sigma[1])
    if isinstance(tau, collections.Iterable):
        hyper_tau = tau
        tau = np.random.gamma(hyper_tau[0], 1./hyper_tau[1])

    w, T = GGPrnd(alpha, sigma, tau, T)

    if len(w) == 0:
        raise Exception("GGP has no atom %.2f %.2f %.2f" % (alpha, sigma, tau))

    cumsum_w = np.cumsum(w)
    w_star = cumsum_w[-1]
    d_star = np.random.poisson(w_star ** 2)
    if d_star == 0:
        raise Exception("No edge in graph")
    tmp = w_star * np.random.random(size=(d_star, 2))
    idx = np.digitize(tmp, cumsum_w)
    active_nodes_idx, inv_idx = np.unique(idx.flatten(), return_inverse=True)
    w_rem = np.sum(w) - np.sum(w[active_nodes_idx])
    w = w[active_nodes_idx]
    new_idx = inv_idx.reshape(idx.shape).T

    g_size = len(active_nodes_idx)
    D = csc_matrix((np.ones(len(new_idx[0])), (new_idx[0], new_idx[1])), shape=(g_size, g_size))  # directed multigraph
    G = D + D.T  # undirected multigraph
    nnz = G.nonzero()
    G = csc_matrix((np.ones(len(nnz[0])), (nnz)), shape=(g_size, g_size)) # undirected simple graph

    return G, D, w, w_rem, alpha, sigma, tau

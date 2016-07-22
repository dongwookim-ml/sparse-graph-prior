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
        alpha = np.random.gamma(hyper_alpha[0], 1. / hyper_alpha[1])
    if isinstance(sigma, collections.Iterable):
        hyper_sigma = sigma
        sigma = 1. - np.random.gamma(hyper_sigma[0], 1. / hyper_sigma[1])
    if isinstance(tau, collections.Iterable):
        hyper_tau = tau
        tau = np.random.gamma(hyper_tau[0], 1. / hyper_tau[1])

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
    G = csc_matrix((np.ones(len(nnz[0])), (nnz)), shape=(g_size, g_size))  # undirected simple graph

    return G, D, w, w_rem, alpha, sigma, tau


def GGPmixtureGraphrnd(s_alpha=1., s_sigma=0.5, s_tau=1., d_alpha=1., d_sigma=-1., d_tau=1., T=0):
    if isinstance(s_alpha, collections.Iterable):
        hyper_alpha = s_alpha
        s_alpha = np.random.gamma(hyper_alpha[0], 1. / hyper_alpha[1])
    if isinstance(s_sigma, collections.Iterable):
        hyper_sigma = s_sigma
        s_sigma = 1. - np.random.gamma(hyper_sigma[0], 1. / hyper_sigma[1])
    if isinstance(s_tau, collections.Iterable):
        hyper_tau = s_tau
        s_tau = np.random.gamma(hyper_tau[0], 1. / hyper_tau[1])

    if isinstance(d_alpha, collections.Iterable):
        hyper_alpha = d_alpha
        d_alpha = np.random.gamma(hyper_alpha[0], 1. / hyper_alpha[1])
    if isinstance(d_sigma, collections.Iterable):
        hyper_sigma = d_sigma
        d_sigma = 1. - np.random.gamma(hyper_sigma[0], 1. / hyper_sigma[1])
    if isinstance(d_tau, collections.Iterable):
        hyper_tau = d_tau
        d_tau = np.random.gamma(hyper_tau[0], 1. / hyper_tau[1])

    s_w, s_T = GGPrnd(s_alpha, s_sigma, s_tau, T)
    d_w, d_T = GGPrnd(d_alpha, d_sigma, d_tau, T)

    if len(s_w) == 0 or len(d_w) == 0:
        raise Exception("GGP has no atom")

    cumsum_s_w = np.cumsum(s_w)
    cumsum_d_w = np.cumsum(d_w)
    s_w_star = cumsum_s_w[-1]
    d_w_star = cumsum_d_w[-1]
    d_star = np.random.poisson((s_w_star + d_w_star) ** 2)
    if d_star == 0:
        raise Exception("No edge in graph")

    w = np.concatenate((s_w, d_w))
    cumsum_w = np.cumsum(w)

    tmp = (s_w_star + d_w_star) * np.random.random(size=(d_star, 2))
    idx = np.digitize(tmp, cumsum_w)
    active_nodes_idx, inv_idx = np.unique(idx.flatten(), return_inverse=True)  # active_nodes_idx[inv_idx] = idx.flatten
    w_rem = np.sum(w) - np.sum(w[active_nodes_idx])
    w = w[active_nodes_idx]
    new_idx = inv_idx.reshape(idx.shape).T

    sparse_nodes = [i for i, node in enumerate(active_nodes_idx) if node < len(s_w)]
    dense_nodes = [i for i, node in enumerate(active_nodes_idx) if node >= len(s_w)]

    g_size = len(active_nodes_idx)
    D = csc_matrix((np.ones(len(new_idx[0])), (new_idx[0], new_idx[1])), shape=(g_size, g_size))  # directed multigraph
    G = D + D.T  # undirected multigraph
    nnz = G.nonzero()
    G = csc_matrix((np.ones(len(nnz[0])), (nnz)), shape=(g_size, g_size))  # undirected simple graph
    params = (s_alpha, s_sigma, s_tau, d_alpha, d_sigma, d_tau)

    return G, D, w, w_rem, params, sparse_nodes, dense_nodes

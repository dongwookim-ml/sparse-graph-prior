import itertools
import collections
import numpy as np
from scipy.sparse import csc_matrix
from .GGPrnd import GGPrnd
from .GGPgraphrnd import GGPgraphrnd

def BSgraphrnd(alpha, sigma, tau, K, eta, beta_0, T=0):
    """
    generate a block-structred sparse graph

    reference: Herlau, T., Schmidt, M. N., & Mørup, M. (2015). Completely random measures for modelling block-structured
               networks, (1), 1–15. Retrieved from http://arxiv.org/abs/1507.02925
    :param alpha: positive scalar
    :param sigma: real in (-inf, 1)
    :param tau: positive scalar
    :param K: integer, number of blocks
    :param eta: K x K matrix or 1x2 positive scalars
    :param beta_0: positive scalar
    :param T: truncation threshold; positive scalar
    :return:
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

    K = int(K)
    if K == 1:
        return GGPgraphrnd(alpha, sigma, tau, T)

    if len(eta) == 2:
        hyper_eta = eta
        eta = np.random.gamma(hyper_eta[0], 1. / hyper_eta[1], size=(K, K))

    w, T = GGPrnd(alpha, sigma, tau, T)
    u = np.random.random(size=w.shape)
    beta = np.random.dirichlet([beta_0 / K] * K)

    if len(w) == 0:
        raise Exception("GGP has no atom %.2f %.2f %.2f" % (alpha, sigma, tau))

    cumsum_w = np.cumsum(w)
    w_star = cumsum_w[-1]
    cumsum_beta = np.cumsum(beta)
    group = np.digitize(u, cumsum_beta)

    row_idx = list()
    col_idx = list()
    active_nodes_idx = np.ndarray(shape=0, dtype=int)

    w_stars = np.zeros(K)
    cumsum_groups = np.zeros([K, len(w)])

    for k in range(K):
        w_tmp = np.zeros_like(w)
        w_tmp[group == k] = w[group == k]
        cumsum_groups[k] = np.cumsum(w_tmp)
        w_stars[k] = cumsum_groups[k, -1]

    for k1, k2 in itertools.product(range(K), repeat=2):
        d_star = np.random.poisson(eta[k1, k2] * w_stars[k1] * w_stars[k2])

        tmp = w_stars[k1] * np.random.random(size=d_star)
        idx1 = np.digitize(tmp, cumsum_groups[k1])
        tmp = w_stars[k2] * np.random.random(size=d_star)
        idx2 = np.digitize(tmp, cumsum_groups[k2])

        row_idx = np.append(row_idx, idx1)
        col_idx = np.append(col_idx, idx2)
        active_nodes_idx = np.union1d(active_nodes_idx, idx1)
        active_nodes_idx = np.union1d(active_nodes_idx, idx2)

    if len(row_idx) == 0:
        raise Exception("No edge in graph")

    w_rem = w_star - np.sum(w[active_nodes_idx])
    w = w[active_nodes_idx]
    group = group[active_nodes_idx]
    _, new_idx = np.unique(np.append(row_idx, col_idx), return_inverse=True)
    new_idx = np.reshape(new_idx, (len(row_idx), 2)).T

    g_size = len(active_nodes_idx)
    D = csc_matrix((np.ones(len(row_idx)), (new_idx[0], new_idx[1])), shape=(g_size, g_size))  # directed multigraph

    return D, w, w_rem, alpha, sigma, tau, eta, group

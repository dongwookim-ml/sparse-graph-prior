import collections
import numpy as np
from scipy.sparse import csc_matrix

from .GGPrnd import GGPrnd


def KGgraphrnd(alpha, beta, sigma_alpha, sigma_beta, tau_alpha, tau_beta, T=0):
    """
    Generating a random knowledge graph from two completely random measures

    :param alpha: positive scalar
    :param sigma_alpha: real in (-inf, 1)
    :param tau_alpha: positive scalar
    :param beta: positive scalar
    :param sigma_beta: real in (-inf, 1)
    :param tau_beta: positive scalar
    :param T: truncation threshold; positive scalar
    :return:
        G: a list of undirected graph as a sparse matrix
        w_alpha: sociability param of each entity
        w_alpha_rem: sum of sociability of unactivated entities
        alpha: parameter used to generate graph
        sigma_alpha: parameter
        tau_alpha: parameter
        w_beta: sociability param of each relation
        w_beta_rem: sum of sociability of unactivated relations
        beta: parameter used to generate graph
        sigma_beta: parameter
        tau_beta: parameter
    """

    if isinstance(alpha, collections.Iterable):
        hyper_alpha = alpha
        alpha = np.random.gamma(hyper_alpha[0], 1. / hyper_alpha[1])
    if isinstance(sigma_alpha, collections.Iterable):
        hyper_sigma = sigma_alpha
        sigma_alpha = 1. - np.random.gamma(hyper_sigma[0], 1. / hyper_sigma[1])
    if isinstance(tau_alpha, collections.Iterable):
        hyper_tau = tau_alpha
        tau_alpha = np.random.gamma(hyper_tau[0], 1. / hyper_tau[1])

    if isinstance(beta, collections.Iterable):
        hyper_beta = beta
        beta = np.random.gamma(hyper_beta[0], 1. / hyper_beta[1])
    if isinstance(sigma_beta, collections.Iterable):
        hyper_sigma = sigma_beta
        sigma_beta = 1. - np.random.gamma(hyper_sigma[0], 1. / hyper_sigma[1])
    if isinstance(tau_beta, collections.Iterable):
        hyper_tau = tau_beta
        tau_beta = np.random.gamma(hyper_tau[0], 1. / hyper_tau[1])

    w_alpha, T = GGPrnd(alpha, sigma_alpha, tau_alpha, T)
    w_beta, T = GGPrnd(beta, sigma_beta, tau_beta, T)

    if len(w_alpha) == 0:
        raise Exception("GGP has no entity %.2f %.2f %.2f" % (alpha, sigma_alpha, tau_alpha))
    if len(w_beta) == 0:
        raise Exception("GGP has no relation %.2f %.2f %.2f" % (beta, sigma_beta, tau_beta))

    cumsum_w_alpha = np.cumsum(w_alpha)
    cumsum_w_beta = np.cumsum(w_beta)
    w_alpha_star = cumsum_w_alpha[-1]
    w_beta_star = cumsum_w_beta[-1]
    d_star = np.random.poisson(w_alpha_star ** 2 * w_beta_star)
    if d_star == 0:
        raise Exception("No edge in graph")

    tmp = w_beta_star * np.random.random(size=(d_star))
    relation_idx = np.digitize(tmp, cumsum_w_beta)
    active_relation_idx, inv_relation_idx = np.unique(relation_idx, return_inverse=True)
    w_beta_rem = np.sum(w_beta) - np.sum(w_beta[active_relation_idx])
    w_beta = w_beta[active_relation_idx]
    new_relation_idx = inv_relation_idx.reshape(relation_idx.shape).T

    tmp = w_alpha_star * np.random.random(size=(d_star, 2))
    entity_idx = np.digitize(tmp, cumsum_w_alpha)
    active_entity_idx, inv_entity_idx = np.unique(entity_idx.flatten(), return_inverse=True)
    w_alpha_rem = np.sum(w_alpha) - np.sum(w_alpha[active_entity_idx])
    w_alpha = w_alpha[active_entity_idx]
    new_entity_idx = inv_entity_idx.reshape(entity_idx.shape).T

    G = list()
    g_size = len(active_entity_idx)
    for ri in range(len(active_relation_idx)):
        relation_entities = inv_relation_idx == ri

        D = csc_matrix((np.ones(np.sum(relation_entities)),
                        (new_entity_idx[0][relation_entities], new_entity_idx[1][relation_entities])),
                       shape=(g_size, g_size))  # directed multigraph

        nz = D.nonzero()
        _G = csc_matrix((np.ones(len(nz[0])), (nz)), shape=(g_size, g_size)) #directed simple graph
        G.append(_G)

    return G, w_alpha, w_alpha_rem, alpha, sigma_alpha, tau_alpha, w_beta, w_beta_rem, beta, sigma_beta, tau_beta

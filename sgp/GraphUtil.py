import numpy as np
from collections import defaultdict
from scipy.sparse import csr_matrix, csc_matrix, triu


def compute_growth_rate(G, n_repeat=10):
    """
    Compute the growth rate of graph G

    :param G: sparse matrix (csc_matrix or csr_matrix)
    :param n_repeat: int
    :return:
    """
    n_n = G.shape[0]
    nnz = G.nonzero()

    n_link = defaultdict(list)

    for si in range(n_repeat):
        rnd_nodes = np.arange(n_n, dtype=int)
        np.random.shuffle(rnd_nodes)
        node_dic = {i: n for i, n in enumerate(rnd_nodes)}

        row_idx = list(map(lambda x: node_dic[x], nnz[0]))
        col_idx = list(map(lambda x: node_dic[x], nnz[1]))

        rnd_row = csr_matrix((G.data, (row_idx, col_idx)), shape=G.shape)
        rnd_col = csc_matrix((G.data, (row_idx, col_idx)), shape=G.shape)

        n_link[0].append(0)

        for i in range(1, n_n):
            # counting triples by expanding tensor
            cnt = 0
            cnt += rnd_row.getrow(i)[:, :i].nnz
            cnt += rnd_col.getcol(i)[:i - 1, :].nnz
            n_link[i].append(cnt + n_link[i - 1][-1])

    return np.array([np.mean(n_link[x]) for x in range(n_n)])


def degree_distribution(G):
    d = defaultdict(int)
    # degree = triu(G).sum(0)

    degree = G.sum(0) + G.sum(1)
    degree /= 2

    max_d = degree.max()
    for _d in degree.tolist()[0]:
        d[int(_d)] += 1

    return d, [d[i] for i in range(int(max_d))]


def degree_one_nodes(G):
    return np.sum(G.sum(0) / 2 == 1)

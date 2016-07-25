from sgp import GGPgraphrnd
from sgp import BSgraphrnd
from sgp import GGPmixtureGraphrnd
from sgp.GraphUtil import compute_growth_rate, degree_distribution, degree_one_nodes

# G, D, w, w_rem, alpha, sigma, tau = GGPgraphrnd(5, 0.5, 1)

#G, D, w, w_rem, alpha, sigma, tau = GGPgraphrnd((1,1), (1,1), (1,1))

# print(G.toarray())

# K = 4
# alpha = 20.
# tau = 1.
# sigma = 0.5
# D, w, w_rem, alpha, sigma, tau, eta, group, i_count = BSgraphrnd(alpha, sigma, tau, K, (1.,1.), float(K))
#
# print(D.shape)
# print(D.toarray())
# print(group)

s_alpha = 20
s_tau = 1
s_sigma = 0.5

d_alpha = 20
d_tau = 1
d_sigma = -1.

G, D, w, w_rem, params, sparse_nodes, dense_nodes = GGPmixtureGraphrnd(s_alpha, s_sigma, s_tau, d_alpha, d_sigma, d_tau)

print(G.toarray())
print(G.shape)
print('# dense nodes', len(dense_nodes), dense_nodes)
print('# sparse nodes', len(sparse_nodes))


compute_growth_rate(G, n_repeat=1)
d, d_list = degree_distribution(G)
one = degree_one_nodes(G)
print(one)

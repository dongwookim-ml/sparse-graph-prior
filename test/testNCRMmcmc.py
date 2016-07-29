import numpy as np

from sgp import GGPrnd, NGGPmcmc
from sgp.NCRMmcmc import log_density_v
from scipy.stats import entropy

modelparam = dict()
# modelparam['estimate_alpha'] = False
# modelparam['estimate_sigma'] = False
# modelparam['estimate_tau'] = False
modelparam['estimate_alpha'] = True
modelparam['estimate_sigma'] = True
modelparam['estimate_tau'] = True
modelparam['alpha_a'] = 1.
modelparam['alpha_b'] = 100.
modelparam['tau_a'] = 1.
modelparam['tau_b'] = 100.

mcmcparam = dict()
mcmcparam['j.niter'] = 1
mcmcparam['u.MH_nb'] = 1
mcmcparam['hyper.MH_nb'] = 1

# generating a random CRM from GGP with given parameters using adaptive thinning
alpha = 300
sigma = 0.5
tau = 1
w, T = GGPrnd(alpha, sigma, tau)

# normalised random measure
mu = w/np.sum(w)

# random samples drawn from NRM
D = np.random.multinomial(np.sum(w), mu)

_w = w[D.nonzero()]
_w_rem = np.sum(w) - np.sum(_w)

mu = np.zeros(_w.size + 1)
mu[:_w.size] = _w
mu[-1] = _w_rem
mu /= np.sum(mu)

print('# atoms', _w.size, _w.sum())
print('Remaining', _w_rem)

# posterior sampling of NRM using auxiliary MCMC algorithm
pi = D[D.nonzero()]
n = np.sum(pi)
alpha = 10000. * np.random.random()
sigma = 1 - np.random.lognormal(1, 1)
tau = 10000. * np.random.random()

sigma = 0.9
v = np.linspace(-500, 500, 1000)
y = log_density_v(v, n, pi.size, alpha, sigma, tau)
u = np.exp(v[y.argmax()])

print(alpha, sigma, tau)
print('init u', u)

for i in range(1000):
    J, J_rem, alpha, sigma, tau, u = NGGPmcmc(n, pi, alpha, sigma, tau, u, modelparam, mcmcparam, verbose=True)

    mu_bar = np.zeros(J.size + 1)
    mu_bar[:J.size] = J
    mu_bar[-1] = J_rem
    mu_bar = mu_bar / np.sum(mu_bar)

    print(i)
    print('entropy between mu / mu_bar', entropy(mu, mu_bar))


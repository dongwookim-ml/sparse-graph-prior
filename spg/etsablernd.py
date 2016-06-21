import numpy as np

"""
Random samples from exponentially tilted stable distribution.

Convert the same function used in BNPGraph matlab package by Francois Caron
http://www.stats.ox.ac.uk/~caron/code/bnpgraph/index.html
Original reference from (Hofert, 2011).

samples = etstablernd(V0, alpha, tau, n) returns a n*1 vector of numbers
distributed fron an exponentially tilted stable distribution with Laplace
transform (in z)
          exp(-V0 * ((z + tau)^alpha - tau^alpha))

References:
- Luc Devroye. Random variate generation for exponentially and polynomially
tilted stable distributions. ACM Transactions on Modeling and Computer
Simulation, vol. 19(4), 2009.
- Marius Hofert. Sampling exponentially tilted stable distributions.
ACM Transactions on Modeling and Computer Simulation, vol. 22(1), 2011.
"""


def gen_U(w1, w2, w3, gamma):
    V = np.random.random()
    W_p = np.random.random()
    if gamma >= 1:
        if V < w1 / (w1 + w2):
            U = np.abs(np.random.standard_normal()) / np.sqrt(gamma)
        else:
            U = np.pi * (1. - W_p ** 2.)
    else:
        if V < w3 / (w3 + w2):
            U = np.pi * W_p
        else:
            U = np.pi * (1. - W_p ** 2)
    return U


def sinc(x):
    return np.sin(x) / x


def ratio_B(x, sigma):
    return sinc(x) / (sinc(sigma * x)) ** sigma / (sinc((1. - sigma) * x)) ** (1 - sigma)


def zolotarev(u, sigma):
    return ((np.sin(sigma * u)) ** sigma * (np.sin((1. - sigma) * u)) ** (1.0 - sigma) / np.sin(u)) ** (
        1. / (1. - sigma))


def etstablernd(V0, alpha, tau, n=1):
    """
    Samples from the exponentially tilted stable distribution
    :param V0:  positive scalar
    :param alpha: real in (0,1)
    :param tau: positive scalar
    :param n: integer
    :return: samples - vector of length n
    """
    if alpha <= 0 or alpha >= 1:
        raise Exception("alpha must be in (0,1)")
    if tau < 0:
        raise Exception("tau must be >= 0 ")
    if V0 <= 0:
        raise Exception("V0 must be > 0")

    lambda_alpha = tau ** alpha * V0

    # Now we sample from an exponentially tilted distribution of parameters
    # sigma, lambda, as in (Devroye, 2009)
    gamma = lambda_alpha * alpha * (1. - alpha)

    s_gamma = np.sqrt(gamma)
    s_pi = np.sqrt(np.pi)

    c1 = np.sqrt(np.pi / 2.)
    c2 = 2. + c1
    c3 = s_gamma * c2

    # xi = 1. / np.pi * ((2. + np.sqrt(np.pi / 2.)) * np.sqrt(2. * gamma) + 1.)  # Correction in Hofert
    # psi = 1. / np.pi * np.exp(-gamma * np.pi ** 2. / 8.) * (2. + np.sqrt(np.pi / 2.)) * np.sqrt(gamma * np.pi)

    xi = (1. + np.sqrt(2.) * c3) / np.pi
    psi = c3 * np.exp(-gamma * np.pi ** 2 / 8) / s_pi

    # w1 = xi * np.sqrt(np.pi / 2. / gamma)
    w1 = c1 * xi / s_gamma
    w2 = 2. * psi * s_pi
    w3 = xi * np.pi
    b = (1. - alpha) / alpha

    samples = np.zeros(n)
    for i in range(n):
        while True:
            while True:
                U = gen_U(w1, w2, w3, gamma)
                zeta = np.sqrt(ratio_B(U, alpha))
                W = np.random.random()
                z = 1. / (1. - (1. + alpha * zeta / s_gamma) ** (-1. / alpha))

                # rho = np.pi * np.exp(-lambda_alpha * (1. - zeta ** (-2.))) * (xi * np.exp(-gamma * U ** 2. / 2.)) * (
                #     U >= 0) * (gamma >= 1) + psi / np.sqrt(np.pi - U) * (U > 0) * (U < np.pi) + xi * (U >= 0) * (
                #         U <= np.pi) * (gamma < 1.) / ((1. + c1) * s_gamma / zeta + z)

                rho = np.pi * np.exp(-lambda_alpha * (1. - zeta ** (-2.))) / ((1. + c1) * s_gamma / zeta + z)

                if np.isnan(rho) or np.isinf(rho):
                    pass

                d = 0
                if U >= 0 and gamma >= 1:
                    d = xi * np.exp(-gamma * U ** 2. / 2.)
                if 0 < U < np.pi:
                    d += psi / (np.sqrt(np.pi - U))
                if 0 <= U <= np.pi and gamma < 1.:
                    d += xi

                rho *= d

                if U < np.pi and W * rho <= 1:
                    break

            a = zolotarev(U, alpha)
            m = (b / a) ** alpha * lambda_alpha
            delta = np.sqrt(m * alpha / a)
            a1 = delta * np.sqrt(np.pi / 2.)
            a2 = a1 + delta
            a3 = z / a
            s = a1 + delta + a3
            V_p = np.random.random()
            N_p = np.random.standard_normal()
            # E_p = -np.log(np.random.random())
            E_p = np.random.exponential()

            if V_p < a1 / s:
                X = m - delta * np.abs(N_p)
            elif V_p < a2 / s:
                X = delta * np.random.random() + m
            else:
                X = m + delta + a3 * E_p

            E = -np.log(np.random.random())

            # cond = (a * (X - m) + np.exp(1. / alpha * np.log(lambda_alpha) - b * np.log(m)) * (
            #     (m / X) ** b - 1.) - N_p ** 2. / 2. * (X < m) - E_p * (X > m + delta))

            cond = a * (X - m) + np.exp((1. / alpha) * np.log(lambda_alpha) - b * np.log(m)) * ((m / X) ** b - 1.)
            if np.isnan(cond) or np.isinf(cond):
                pass

            if X < m:
                cond -= N_p ** 2. / 2.
            elif X > m + delta:
                cond -= E_p

            if X >= 0 and cond <= E:
                break
        samples[i] = np.exp(1. / alpha * np.log(V0) - b * np.log(X))

    return samples


if __name__ == '__main__':
    samples = etstablernd(100, 0.9, 1, 10000)
    print(np.mean(samples), np.std(samples))

from sgp import GGPrnd
import numpy as np


alpha = 20.
tau = 1.
sigma = 0.5

for i in range(100000):
    w, T = GGPrnd(alpha, sigma, tau)
    if i%100 == 0:
        print('pass', i)

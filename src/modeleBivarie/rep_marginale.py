import numpy as np

from statsmodels.distributions.empirical_distribution import ECDF

def F_evt(losses, u, xi, beta, N, Nu):
    losses = np.asarray(losses)
    ecdf = ECDF(losses)
    U = np.zeros_like(losses, dtype=float)

    mask_normal = losses <= u
    U[mask_normal] = ecdf(losses[mask_normal])

    # Zone extrême (GPD)
    mask_extreme = ~mask_normal
    x = losses[mask_extreme]
    if len(x) > 0:
        y = (x - u) / beta
        # Cas limite xi ~ 0
        if abs(xi) < 1e-6:
            U[mask_extreme] = 1 - (Nu / N) * np.exp(-y)

        else:
            U[mask_extreme] = 1 - (Nu / N) * (1 + xi * y) ** (-1 / xi)

    # Sécurité numérique
    eps = 1e-10
    U = np.clip(U, eps, 1 - eps)

    return U
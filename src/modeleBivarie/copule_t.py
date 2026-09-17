# import numpy as np
# from scipy.stats import t
# from scipy.special import gamma

from scipy.optimize import minimize

from scipy.stats import multivariate_t, t
import numpy as np
from scipy.stats import kendalltau

def estimate_rho_kendall(U):
    """
    Estimation robuste de rho via Kendall tau
    """
    U = np.asarray(U)
    u1 = U[:, 0]
    u2 = U[:, 1]

    tau, _ = kendalltau(u1, u2)

    rho = np.sin(np.pi * tau / 2)

    return rho


def log_likelihood_t_copula(nu, U, rho):
    """
    Log-vraisemblance profile pour nu (rho fixé)
    """
    T, d = U.shape

    Sigma = np.array([[1, rho],
                      [rho, 1]])

    if nu <= 2:
        return -1e18

    X = t.ppf(U, df=nu)

    loglik = 0.0

    for i in range(T):

        x = X[i]

        # densité multivariée t
        joint = multivariate_t.logpdf(
            x,
            loc=np.zeros(d),
            shape=Sigma,
            df=nu
        )

        # marges indépendantes t
        marg = np.sum(t.logpdf(x, df=nu))

        loglik += joint - marg
    return loglik


def estimate_nu(U, rho):
    """
    estimation stable de nu
    """

    def obj(nu):
        return -log_likelihood_t_copula(nu[0], U, rho)

    res = minimize(
        obj,
        x0=np.array([6.0]),
        bounds=[(2.1, 50)]
    )

    return res.x[0]

# def student_t_multivariate_pdf(x, Sigma, nu):
#     """
#     x : (d,) vecteur
#     Sigma : (d,d) matrice de corrélation
#     nu : degrés de liberté
#     """
#     d = len(x)
#     inv_Sigma = np.linalg.inv(Sigma)
#     det_Sigma = np.linalg.det(Sigma)

#     Q = x @ inv_Sigma @ x   # @ est l'opérateur produit matriciel

#     coef = gamma((nu + d) / 2) / (
#         gamma(nu / 2) * (nu * np.pi) ** (d / 2) * np.sqrt(det_Sigma)
#     )

#     return coef * (1 + Q / nu) ** (-(nu + d) / 2)

# # Densité marginale t de Student univariée
# def student_t_pdf(x, nu):
#     return (
#         gamma((nu + 1) / 2)
#         / (np.sqrt(nu * np.pi) * gamma(nu / 2))
#         * (1 + x**2 / nu) ** (-(nu + 1) / 2)
#     )


# # log-vraisemblance de la copule t
# def log_likelihood_t_copula(params, U):
#     """
#     U : (T, d) pseudo-observations
#     params : [rho_12, rho_13, ..., nu]
#     """
#     T, d = U.shape # nombre d'observations et de dimensions
#     nu = params[-1]

#     if nu <= 2:
#         return np.inf

#     # Construire Sigma
#     Sigma = np.eye(d) # matrice identité dxd
#     idx = 0
#     for i in range(d):
#         for j in range(i+1, d):
#             Sigma[i, j] = Sigma[j, i] = params[idx]
#             idx += 1

#     # Vérification positive définie
#     if np.min(np.linalg.eigvals(Sigma)) <= 0:
#         return np.inf

#     # transformation copule
#     X = t.ppf(U, nu)

#     # sécurité numérique
#     eps = 1e-10
#     X = np.clip(X, -10, 10)

#     # Calcul de la log-vraisemblance
#     inv_Sigma = np.linalg.inv(Sigma)
#     det_Sigma = np.linalg.det(Sigma)
#     log_lik = 0.0
#     for k in range(T):
#         x = X[k]

#         # quadratic form
#         Q = x @ inv_Sigma @ x

#         log_joint = (
#             gamma((nu + d) / 2).log()
#             - gamma(nu / 2).log()
#             - (d / 2) * np.log(nu * np.pi)
#             - 0.5 * np.log(det_Sigma)
#             - ((nu + d) / 2) * np.log(1 + Q / nu)
#         )

#         log_marginals = np.sum(
#             np.log(student_t_pdf(x, nu))
#         )

#         log_lik += log_joint - log_marginals

#     return -log_lik


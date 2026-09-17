import numpy as np
from itertools import combinations
from scipy.stats import kendalltau, multivariate_t, t
from scipy.optimize import minimize

EPS = 1e-10

# NB: Les méthodes dont le nom commence par "_" sont des méthodes internes 
# (non destinées à être utilisées directement par l'utilisateur) : Équivalent de "private" en Java ou C++.


class StudentTCopula:
    """
    Copule t-Student multivariée pour k actifs (k ≥ 2).

    Généralise copule_t.py (bivarié) à la dimension k :
        - Estimation de la matrice de corrélation \Sigma via les tau de Kendall
          (une paire (i,j) à la fois : plus robuste que le MLE joint)
        - Estimation des degrés de liberté \nu par maximum de vraisemblance profil

    Paramètres ajustés :
        rho  : np.ndarray (kxk) — matrice de corrélation
        nu   : float           — degrés de liberté

    Simulation :
        U suit t_k(0, \Sigma, \nu)  puis  U_i = F_\nu(Z_i)  (transformation PIT marginale)
    """

    def __init__(self):
        self.rho = None   # matrice de corrélation \Sigma (kxk)
        self.nu  = None   # degrés de liberté \nu
        self.k   = None   # dimension (nombre d'actifs)

    # ------------------------------------------------------------------ #
    #  Estimation de \Sigma via les tau de Kendall (paire par paire)     #
    # ------------------------------------------------------------------ #

    def _estimate_rho_kendall(self, U: np.ndarray) -> np.ndarray:
        """
        Construit la matrice de corrélation \Sigma à partir des tau de Kendall.

        La relation entre rho et tau est : rho = sin(pi * tau/2)
        (valable pour la copule t et la copule gaussienne).
        """
        k = U.shape[1]
        Sigma = np.eye(k)
        for i, j in combinations(range(k), 2):
            tau, _ = kendalltau(U[:, i], U[:, j])
            rho_ij = np.sin(np.pi * tau / 2.0)
            Sigma[i, j] = rho_ij
            Sigma[j, i] = rho_ij

        # S'assurer que \Sigma est définie positive
        eigvals = np.linalg.eigvalsh(Sigma)
        if np.any(eigvals <= 0):
            Sigma = _nearest_positive_definite(Sigma)

        return Sigma

    # ------------------------------------------------------------------ #
    #  Log-vraisemblance de la copule t ( \nu fixé, \Sigma fixé)         #
    # ------------------------------------------------------------------ #

    def _log_likelihood(self, nu: float, U: np.ndarray, Sigma: np.ndarray) -> float:
        """
        Log-vraisemblance profil de la copule t pour des degrés de liberté \nu.

        log c(u_1,...,u_k) = log f_t(F^{-1}_\nu(u_1),...,F^{-1}_\nu(u_k) ; \Sigma, \nu)
                           - \sum_i log f_{t,\nu}(F^{-1}_\nu(u_i))

        où f_t est la densité t multivariée et f_{t,\nu} la densité t univariée.
        """
        T, d = U.shape
        if nu <= 2:
            return -1e18

        # Transformation inverse : Z_i = t^{-1}_nu(U_i)
        X = t.ppf(np.clip(U, EPS, 1 - EPS), df=nu)

        loglik = 0.0
        for i in range(T):
            x = X[i]
            # Densité t multivariée conjointe
            joint = multivariate_t.logpdf(x, loc=np.zeros(d), shape=Sigma, df=nu)
            # Somme des densités t marginales indépendantes (pour chaque actifs séparément)
            marg  = np.sum(t.logpdf(x, df=nu))
            loglik += joint - marg

        return loglik

    # ------------------------------------------------------------------ #
    #  Ajustement                                                        #
    # ------------------------------------------------------------------ #

    def fit(self, U: np.ndarray) -> "StudentTCopula":
        """
        Ajuste la copule t-Student aux pseudo-observations U (T x k).

        Étape 1 : \Sigma via tau de Kendall
        Étape 2 : nu via maximisation de la vraisemblance profil (numérique)

        Parameters
        ----------
        U : np.ndarray (T x k)
            Pseudo-observations uniformes issues du PIT (\in (0,1))
        """
        U = np.asarray(U)
        self.k = U.shape[1]

        # Étape 1 : estimation de \Sigma
        self.rho = self._estimate_rho_kendall(U)

        # Étape 2 : estimation de nu
        def obj(nu_arr):
            return -self._log_likelihood(nu_arr[0], U, self.rho)

        res = minimize(obj, x0=np.array([6.0]), bounds=[(2.1, 50.0)],
                       method="L-BFGS-B")
        self.nu = float(res.x[0])
        return self

    # ------------------------------------------------------------------ #
    #  Simulation                                                        #
    # ------------------------------------------------------------------ #

    def simulate(self, n_sim: int) -> np.ndarray:
        """
        Simule n_sim vecteurs de la copule t-Student.

        Algorithme :
            1. Z suit t_k(0, Sigma, nu)           (vecteur t multivarié)
            2. U_i = F_nu(Z_i)             (CDF t marginale -> uniforme sur (0,1))

        Returns
        -------
        np.ndarray (n_sim x k) : pseudo-observations uniformes
        """
        Z = multivariate_t.rvs(
            loc=np.zeros(self.k),
            shape=self.rho,
            df=self.nu,
            size=n_sim
        )
        Z = np.atleast_2d(Z)   # Sécurité si k=1 ou n_sim=1
        U = t.cdf(Z, df=self.nu)
        return np.clip(U, EPS, 1 - EPS)

    # ------------------------------------------------------------------ #
    #  Résumé                                                            #
    # ------------------------------------------------------------------ #

    def summary(self) -> dict:
        return {"type": "Student-t", "rho": self.rho, "nu": self.nu, "k": self.k}


# ====================================================================== #
#  Utilitaire : plus proche matrice définie positive (algorithme de Higham)
# ====================================================================== #

def _nearest_positive_definite(A: np.ndarray) -> np.ndarray:
    """
    Retourne la plus proche matrice symétrique définie positive de A.
    Utilisée pour corriger les matrices de corrélation estimées qui ne sont
    pas numériquement définies positives (à cause des tau de Kendall).
    """
    B = (A + A.T) / 2
    _, s, Vt = np.linalg.svd(B)
    H  = Vt.T @ np.diag(s) @ Vt
    A2 = (B + H) / 2
    A3 = (A2 + A2.T) / 2

    if _is_positive_definite(A3):
        return A3

    spacing = np.spacing(np.linalg.norm(A))
    k = 1
    while not _is_positive_definite(A3):
        mineig = np.min(np.real(np.linalg.eigvals(A3)))
        A3 += np.eye(A.shape[0]) * (-mineig * k**2 + spacing)
        k  += 1

    return A3


def _is_positive_definite(B: np.ndarray) -> bool:
    try:
        np.linalg.cholesky(B)
        return True
    except np.linalg.LinAlgError:
        return False

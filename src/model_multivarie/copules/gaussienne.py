import numpy as np
import pandas as pd
from copulas.multivariate import GaussianMultivariate


class GaussianCopula:
    """
    Copule gaussienne multivariée pour k actifs.

    La copule gaussienne ne capture pas la dépendance en queue (tail dependence = 0),
    ce qui la rend moins adaptée aux risques extrêmes. Elle est incluse ici
    à titre de comparaison avec les copules t-Student et Gumbel.

    Utilise la librairie copulas (GaussianMultivariate).
    """

    def __init__(self):
        self._copula     = None   # objet GaussianMultivariate interne
        self.correlation = None   # matrice de corrélation estimée
        self._col_names  = None   # noms de colonnes utilisés lors du fit

    def fit(self, U: np.ndarray, col_names: list = None) -> "GaussianCopula":
        """
        Ajuste la copule gaussienne aux pseudo-observations U (T x k).

        Parameters
        ----------
        U : np.ndarray (T x k)
            Pseudo-observations uniformes issues du PIT.
        col_names : list of str, optional
            Noms des colonnes (pour la cohérence avec la simulation).
            Si None, utilise des noms génériques "u0", "u1", ...
        """
        U = np.asarray(U)
        k = U.shape[1]

        # GaussianMultivariate attend un DataFrame
        self._col_names = col_names if col_names else [f"u{i}" for i in range(k)]
        df = pd.DataFrame(U, columns=self._col_names)

        self._copula     = GaussianMultivariate()
        self._copula.fit(df)
        self.correlation = self._copula.correlation
        return self

    def simulate(self, n_sim: int) -> np.ndarray:
        """
        Simule n_sim vecteurs de la copule gaussienne.

        Returns
        -------
        np.ndarray (n_sim x k) : pseudo-observations uniformes
        """
        if self._copula is None:
            raise RuntimeError("La copule doit d'abord être ajustée via fit().")
        samples = self._copula.sample(n_sim)
        # sample() retourne un DataFrame -> convertir en numpy
        return samples[self._col_names].values

    def summary(self) -> dict:
        return {"type": "Gaussienne", "correlation": self.correlation}

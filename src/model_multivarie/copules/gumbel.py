import numpy as np
from copulas.bivariate import Gumbel as BivarGumbel


class GumbelCopula:
    """
    Copule de Gumbel bivariée (k = 2 actifs uniquement).

    La copule de Gumbel est une copule archimédienne dont la structure de
    dépendance est asymétrique : elle capture la dépendance en queue supérieure
    (co-krachs simultanés) mieux que la copule de Clayton.

    Paramètre : theta >= 1
        - theta = 1 : indépendance totale
        - theta -> infinity : dépendance totale

    IMPORTANT : Pour k > 2 actifs, utiliser StudentTCopula qui se généralise
    naturellement à la dimension k. La copule de Gumbel bivariée de la librairie
    copulas ne supporte que d = 2.

    Généralise simulation_gumbel.py.
    """

    def __init__(self):
        self.theta  = None   # paramètre de dépendance theta
        self.loglik = None   # log-vraisemblance sur les données d'entraînement
        self._gumbel = None  # objet interne copulas.bivariate.Gumbel

    def fit(self, U: np.ndarray) -> "GumbelCopula":
        """
        Ajuste la copule de Gumbel aux pseudo-observations U (T x 2).

        Parameters
        ----------
        U : np.ndarray (T x 2)
            Pseudo-observations uniformes issues du PIT.

        Raises
        ------
        ValueError si U n'a pas exactement 2 colonnes.
        """
        U = np.asarray(U)
        if U.shape[1] != 2:
            raise ValueError(
                f"La copule de Gumbel ne supporte que k=2 actifs, "
                f"mais k={U.shape[1]} a été fourni. "
                f"Pour k>2, utilisez copula_type='student'."
            )

        self._gumbel = BivarGumbel()
        self._gumbel.fit(U)
        self._gumbel.tau = 1 - 1 / self._gumbel.theta

        self.theta  = self._gumbel.theta
        self.loglik = float(np.sum(self._gumbel.log_probability_density(U)))
        return self

    def simulate(self, n_sim: int) -> np.ndarray:
        """
        Simule n_sim paires de la copule de Gumbel.

        Returns
        -------
        np.ndarray (n_sim x 2) : pseudo-observations uniformes
        """
        if self._gumbel is None:
            raise RuntimeError("La copule doit d'abord être ajustée via fit().")
        return self._gumbel.sample(n_sim)

    def summary(self) -> dict:
        return {"type": "Gumbel", "theta": self.theta, "loglik": self.loglik}

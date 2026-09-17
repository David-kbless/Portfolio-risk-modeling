import numpy as np
import pandas as pd
from scipy.stats import genpareto
from statsmodels.distributions.empirical_distribution import ECDF


class MarginalEVT:
    """
    Modélise la distribution marginale d'un actif via l'EVT (Peaks-Over-Threshold + GPD).

    Distribution hybride :
        - Sous le seuil u  : distribution empirique (ECDF)
        - Au-dessus de u   : queue GPD (Generalized Pareto Distribution)

    Généralise rep_marginale.py (bivarié) à un actif quelconque.

    Exemple :
        evt = MarginalEVT(losses_spy, "SPY")
        evt.select_threshold(percentile=95.0)  # ou evt.set_threshold(0.025)
        U = evt.F_evt()  # pseudo-observations uniformes
    """

    def __init__(self, losses: np.ndarray, ticker: str):
        self.losses = np.asarray(losses, dtype=float)
        self.ticker = ticker
        self.N = len(self.losses)

        # Paramètres GPD (remplis après select_threshold ou set_threshold)
        self.u = None      # seuil
        self.xi = None     # paramètre de forme (shape)
        self.beta = None   # paramètre d'échelle (scale)
        self.Nu = None     # nombre d'excès au-dessus de u
        self.p_u = None    # fraction des observations au-dessus de u (Nu / N)

    # ------------------------------------------------------------------ #
    #  Sélection et ajustement du seuil                                  #
    # ------------------------------------------------------------------ #

    def select_threshold(self, percentile: float = 95.0) -> float:
        """
        Sélectionne automatiquement le seuil u au percentile donné,
        puis ajuste la GPD.

        Parameters
        ----------
        percentile : float
            Percentile pour le seuil (ex: 95.0 -> 95e percentile des pertes)

        Returns
        -------
        float : valeur du seuil u sélectionné
        """
        self.u = np.percentile(self.losses, percentile)
        self._fit_gpd()
        return self.u

    def set_threshold(self, u: float):
        """
        Définit manuellement le seuil u puis ajuste la GPD.

        Parameters
        ----------
        u : float
            Valeur du seuil (ex: 0.025 pour 2.5% de perte journalière)
        """
        self.u = u
        self._fit_gpd()

    def _fit_gpd(self):
        """
        Ajuste la loi GPD aux excès X - u pour les observations X > u,
        via maximum de vraisemblance (floc=0 car les excès commencent à 0).
        """
        excesses = self.losses[self.losses > self.u] - self.u
        self.Nu = len(excesses)
        self.p_u = self.Nu / self.N

        if self.Nu < 5:
            raise ValueError(
                f"[{self.ticker}] Seuil u={self.u:.6f} trop élevé : "
                f"seulement {self.Nu} excès (minimum requis : 5). "
                f"Abaissez le percentile."
            )

        # La loi GPD est ajustée via le maximum de vraisemblance
        # genpareto.fit retourne (xi, loc, beta) avec floc=0 (excès centrés en 0)
        self.xi, _, self.beta = genpareto.fit(excesses, floc=0)

    # ------------------------------------------------------------------ #
    #  Excès moyen (pour le graphique de sélection du seuil)             #
    # ------------------------------------------------------------------ #

    def mean_excess(self, seuils: np.ndarray) -> np.ndarray:
        """
        Calcule l'excès moyen E[L - u | L > u] pour chaque seuil u.

        Si la courbe est approximativement linéaire croissante,
        cela confirme que les excès suivent une GPD avec xi > 0 (queue lourde).
        """
        me = []
        for u in seuils:
            excesses = self.losses[self.losses > u] - u
            me.append(excesses.mean() if len(excesses) > 0 else np.nan)
        return np.array(me)

    # ------------------------------------------------------------------ #
    #  Transformation PIT (Probability Integral Transform)               #
    # ------------------------------------------------------------------ #

    def F_evt(self, x: np.ndarray = None) -> np.ndarray:
        """
        Calcule F(x) = P(L <= x) via la distribution hybride ECDF + GPD.

        Si x est None, transforme la série de pertes interne self.losses.

        - Zone normale (x <= u)  : F(x) = ECDF(x) (fonction de répartition empirique)
        - Zone extrême (x > u)  : F(x) = 1 - (Nu/N) * (1 + xi*(x-u)/beta)^(-1/xi)

        Returns
        -------
        np.ndarray : pseudo-observations U \in (0, 1)
        """
        if x is None:
            x = self.losses
        x = np.asarray(x, dtype=float)

        ecdf = ECDF(self.losses)
        U = np.zeros_like(x, dtype=float)

        # Zone normale : distribution empirique
        mask_normal = x <= self.u
        U[mask_normal] = ecdf(x[mask_normal])

        # Zone extrême : formule GPD semi-paramétrique
        mask_extreme = ~mask_normal # x > u
        if np.any(mask_extreme):
            y = (x[mask_extreme] - self.u) / self.beta
            if abs(self.xi) < 1e-6:
                # Cas limite xi -> 0 : GPD dégénère en exponentielle
                U[mask_extreme] = 1 - self.p_u * np.exp(-y)
            else:
                U[mask_extreme] = 1 - self.p_u * (1 + self.xi * y) ** (-1.0 / self.xi)

        # Protection numérique : U doit être strictement dans (0, 1)
        eps = 1e-10
        return np.clip(U, eps, 1 - eps)

    # ------------------------------------------------------------------ #
    #  Sérialisation                                                     #
    # ------------------------------------------------------------------ #

    def to_dict(self) -> dict:
        """Retourne les paramètres sous forme de dictionnaire (pour DataFrame)."""
        return {
            "ticker": self.ticker,
            "u":      self.u,
            "xi":     self.xi,
            "beta":   self.beta,
            "N":      self.N,
            "Nu":     self.Nu,
            "Pu":     self.p_u
        }

    @classmethod
    def from_params(cls, ticker: str, losses: np.ndarray,
                    u: float, xi: float, beta: float, N: int, Nu: int) -> "MarginalEVT":
        """
        Reconstruit un MarginalEVT à partir de paramètres déjà estimés.
        Utile pour charger des paramètres depuis un CSV sauvegardé.
        """
        obj = cls(losses, ticker)
        obj.u    = u
        obj.xi   = xi
        obj.beta = beta
        obj.N    = N
        obj.Nu   = Nu
        obj.p_u  = Nu / N
        return obj

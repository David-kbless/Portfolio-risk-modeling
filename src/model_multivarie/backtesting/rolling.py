import numpy as np
import pandas as pd
from scipy.stats import norm


class RollingBacktest:
    """
    Backtesting glissant de la VaR du portefeuille multivarié.

    Stratégie retenue :
        - La VaR "multivariée statique" est calculée une seule fois sur
          l'échantillon complet (par la simulation Monte-Carlo du modèle)
          puis comparée à toutes les pertes historiques -> test de Kupiec.

        - Pour la visualisation dynamique (rolling), deux méthodes rapides
          sont calculées sur une fenêtre glissante à partir des pertes
          historiques du portefeuille (série univariée) :
              * VaR historique      : quantile empirique de la fenêtre
              * VaR EVT univariée   : GPD ajustée sur la fenêtre courante

    Paramètres de la fenêtre adaptative :
        window_base : fenêtre de référence (en jours ouvrés, ex: 250 = 1 an)
        window_min  : fenêtre minimale (ex: 150)
        window_max  : fenêtre maximale (ex: 350)
        La fenêtre s'élargit quand la volatilité est faible (marché calme)
        et se rétrécit quand la volatilité augmente (marché agité).
    """

    def __init__(
        self,
        portfolio_losses: np.ndarray,
        dates: pd.DatetimeIndex,
        alpha: float = 0.99,
        window_base: int = 250,
        window_min:  int = 150,
        window_max:  int = 350,
        threshold_percentile: float = 95.0,
    ):
        """
        Parameters
        ----------
        portfolio_losses     : np.ndarray — série de pertes du portefeuille (pondérées)
        dates                : pd.DatetimeIndex — dates correspondantes
        alpha                : float — niveau de confiance pour la VaR
        window_base / min / max : int — paramètres de la fenêtre adaptative
        threshold_percentile : float — percentile pour le seuil EVT en rolling
        """
        self.L      = np.asarray(portfolio_losses)
        self.dates  = dates
        self.alpha  = alpha
        self.window_base = window_base
        self.window_min  = window_min
        self.window_max  = window_max
        self.threshold_percentile = threshold_percentile

    # ------------------------------------------------------------------ #
    #  Fenêtre adaptative selon la volatilité                            #
    # ------------------------------------------------------------------ #

    def _adaptive_windows(self) -> list:
        """
        Calcule la fenêtre glissante adaptative pour chaque jour t.

        Si sigma_t > sigma_base (période de crise) -> fenêtre réduite (plus réactif).
        Si sigma_t < sigma_base (période calme)    -> fenêtre élargie (plus de données).

        window_t = window_base x (sigma_base / sigma_t)  clampé dans [window_min, window_max]
        """
        sigma_base = np.std(self.L[:self.window_base])
        windows = []
        for t in range(self.window_base, len(self.L)):
            sigma_t = np.std(self.L[t - 20:t])   # volatilité locale sur 20 jours
            window_t = int(self.window_base * (sigma_base / sigma_t)) if sigma_t > 0 else self.window_base
            window_t = max(self.window_min, min(self.window_max, window_t))
            windows.append(window_t)
        return windows

    # ------------------------------------------------------------------ #
    #  VaR historique et EVT univariée sur la fenêtre courante           #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _var_historique(losses: np.ndarray, alpha: float) -> float:
        """VaR historique = quantile empirique de la fenêtre courante."""
        return float(np.quantile(losses, alpha))

    @staticmethod
    def _var_normale(losses: np.ndarray, alpha: float) -> float:
        """VaR sous hypothèse de normalité (mu + sigma * z_alpha)."""
        mu    = np.mean(losses)
        sigma = np.std(losses)
        return float(mu + sigma * norm.ppf(alpha))

    @staticmethod
    def _var_evt_univarie(losses: np.ndarray, alpha: float, percentile: float) -> float:
        """
        VaR EVT univariée via GPD ajustée sur la fenêtre courante.

        Formule analytique (POT) :
            VaR_alpha = u + (scale/shape) * [((1-alpha)/p_u)^{-shape} - 1]
        """
        from scipy.stats import genpareto
        u    = np.percentile(losses, percentile)
        exc  = losses[losses > u] - u
        Nu   = len(exc)
        N    = len(losses)
        if Nu < 5:
            return float(np.quantile(losses, alpha))   # repli sur historique

        xi, _, beta = genpareto.fit(exc, floc=0)
        p_u = Nu / N

        if abs(xi) < 1e-6:
            return float(u - beta * np.log((1 - alpha) / p_u))
        return float(u + (beta / xi) * (((1 - alpha) / p_u) ** (-xi) - 1))

    # ------------------------------------------------------------------ #
    #  Lancement du backtesting glissant                                 #
    # ------------------------------------------------------------------ #

    def run(self) -> pd.DataFrame:
        """
        Lance le backtesting sur l'ensemble de la période disponible.

        Pour chaque jour t (à partir de window_max), calcule :
            - VaR_historique  : quantile empirique de la fenêtre courante
            - VaR_normale     : VaR gaussienne de la fenêtre courante
            - VaR_EVT_1D      : VaR EVT univariée de la fenêtre courante

        Returns
        -------
        pd.DataFrame avec index=dates, colonnes=[portfolio_loss, VaR_historique,
                                                  VaR_normale, VaR_EVT_1D]
        """
        adaptive_wins = self._adaptive_windows()
        t_start = self.window_max

        var_hist_series = []
        var_norm_series = []
        var_evt_series  = []

        for t in range(t_start, len(self.L)):
            w_t      = adaptive_wins[t - self.window_base]
            past     = self.L[t - w_t:t]

            var_hist_series.append(self._var_historique(past, self.alpha))
            var_norm_series.append(self._var_normale(past, self.alpha))
            var_evt_series.append(self._var_evt_univarie(past, self.alpha, self.threshold_percentile))

        results = pd.DataFrame(index=self.dates[t_start:])
        results["portfolio_loss"] = self.L[t_start:]
        results["VaR_historique"] = var_hist_series
        results["VaR_normale"]    = var_norm_series
        results["VaR_EVT_1D"]     = var_evt_series

        return results.dropna()

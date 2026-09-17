import numpy as np
from arch import arch_model
from scipy.stats import genpareto
from statsmodels.distributions.empirical_distribution import ECDF


class GarchEVT:
    """
    Modélise la marginale d'un actif via GARCH(1,1) + EVT-GPD sur les résidus.

    Problème résolu :
        Le modèle EVT standard (MarginalEVT) suppose une VaR constante dans le temps.
        Pendant les crises, la volatilité explose -> les violations de VaR se regroupent
        -> le test de Christoffersen est rejeté.

    Solution (McNeil & Frey, 2000) :
        1. GARCH(1,1) filtre la volatilité conditionnelle sigma_t
        2. Résidus standardisés : z_t = L_t / sigma_t
        3. EVT-GPD ajustée sur z_t au lieu des pertes brutes
        4. VaR conditionnelle : VaR_t = sigma_t * VaR_GPD(z)
           -> elle monte quand le marché s'agite, descend en période calme

    Interface compatible avec MarginalEVT :
        Mêmes méthodes publiques (select_threshold, F_evt, to_dict, mean_excess)
        + méthodes spécifiques GARCH (var_conditionnel, forecast_sigma)

    Exemple :
        garch = GarchEVT(losses_spy, "SPY")
        garch.fit_garch()
        garch.select_threshold(percentile=95.0)
        U = garch.F_evt()                    # PIT sur résidus z_t -> pour la copule
        var_t = garch.var_conditionnel()     # série VaR_t = sigma_t * VaR_z
        sigma_T1 = garch.forecast_sigma()    # sigma_{T+1} pour la simulation
    """

    def __init__(self, losses: np.ndarray, ticker: str):
        self.losses = np.asarray(losses, dtype=float)
        self.ticker = ticker
        self.N = len(self.losses)

        # ---- Résultats GARCH ----
        self.garch_result = None   # objet résultat arch
        self.sigma_t      = None   # volatilités conditionnelles sigma_t (même longueur que losses)
        self.residuals    = None   # résidus standardisés z_t = L_t / sigma_t

        # ---- Paramètres EVT sur résidus z_t ----
        self.u    = None   # seuil sur z_t
        self.xi   = None   # paramètre de forme GPD
        self.beta = None   # paramètre d'échelle GPD
        self.Nu   = None   # nombre d'excès z_t > u
        self.p_u  = None   # Nu / N

    # ================================================================== #
    #  Étape 1 : GARCH(1,1)                                              #
    # ================================================================== #

    def fit_garch(self) -> "GarchEVT":
        """
        Ajuste un GARCH(1,1) sur les pertes et extrait sigma_t et z_t.

        Les pertes sont multipliées par 100 (en %) pour la stabilité numérique
        du GARCH, puis les volatilités sont ramenées aux unités originales.

        Modèle :
            L_t   = sigma_t * z_t     z_t suit i.i.d. N(0,1) (filtré)
            sigma²_t  = omega + alpha*epsilon²_{t-1} + beta*sigma²_{t-1}

        Après filtrage, les z_t sont approximativement i.i.d. mais encore
        à queue lourde -> l'EVT s'applique sur z_t.
        """
        # On travaille en % pour la stabilité numérique du GARCH
        model = arch_model(
            self.losses * 100,
            vol='Garch',
            p=1, q=1,
            dist='normal',
            rescale=False
        )
        self.garch_result = model.fit(disp='off')

        # sigma_t en unités originales (pas en %)
        cv = self.garch_result.conditional_volatility
        self.sigma_t = (cv.values if hasattr(cv, "values") else np.asarray(cv)) / 100.0

        # Résidus standardisés : z_t = L_t / sigma_t
        self.residuals = self.losses / self.sigma_t

        return self

    # ================================================================== #
    #  Étape 2 : EVT-GPD sur les résidus z_t                             #
    # ================================================================== #

    def select_threshold(self, percentile: float = 95.0) -> float:
        """
        Sélectionne le seuil u sur les résidus z_t (pas sur les pertes brutes)
        et ajuste la GPD.

        Parameters
        ----------
        percentile : float
            Percentile de z_t pour le seuil. Ex: 95.0 -> 95e percentile des résidus.
        """
        if self.residuals is None:
            raise RuntimeError("Appeler fit_garch() avant select_threshold().")
        self.u = np.percentile(self.residuals, percentile)
        self._fit_gpd()
        return self.u

    def set_threshold(self, u: float):
        """Définit manuellement le seuil sur z_t et ajuste la GPD."""
        if self.residuals is None:
            raise RuntimeError("Appeler fit_garch() avant set_threshold().")
        self.u = u
        self._fit_gpd()

    def _fit_gpd(self):
        """Ajuste la GPD aux excès z_t > u via maximum de vraisemblance."""
        excesses = self.residuals[self.residuals > self.u] - self.u
        self.Nu  = len(excesses)
        self.p_u = self.Nu / self.N

        if self.Nu < 5:
            raise ValueError(
                f"[{self.ticker}] Seuil u={self.u:.4f} trop élevé sur résidus : "
                f"seulement {self.Nu} excès. Abaissez le percentile."
            )
        # La loi GPD est ajustée aux excès via le maximum de vraisemblance
        # floc=0 : excès déjà centrés en 0 (on a soustrait u)
        self.xi, _, self.beta = genpareto.fit(excesses, floc=0)

    # ================================================================== #
    #  Excès moyen (sur résidus, pour le graphique de sélection du seuil)#
    # ================================================================== #

    def mean_excess(self, seuils: np.ndarray) -> np.ndarray:
        """
        Calcule l'excès moyen E[z - u | z > u] pour chaque seuil u.
        Opère sur les résidus standardisés z_t (pas sur les pertes brutes).
        """
        if self.residuals is None:
            raise RuntimeError("Appeler fit_garch() d'abord.")
        me = []
        for u in seuils:
            exc = self.residuals[self.residuals > u] - u
            me.append(exc.mean() if len(exc) > 0 else np.nan)
        return np.array(me)

    # ================================================================== #
    #  Transformation PIT sur les résidus                                #
    # ================================================================== #

    def F_evt(self, x: np.ndarray = None) -> np.ndarray:
        """
        Applique le PIT sur les résidus standardisés z_t.

        Si x est None, transforme self.residuals.
        Les pseudo-observations U = F(z) sont utilisées pour ajuster la copule.

        Distribution hybride :
            - z <= u : ECDF empirique des résidus
            - z > u  : formule GPD semi-paramétrique
        """
        if x is None:
            x = self.residuals
        x = np.asarray(x, dtype=float)

        ecdf = ECDF(self.residuals)
        U = np.zeros_like(x, dtype=float)

        # Zone normale : distribution empirique des résidus
        mask_normal  = x <= self.u
        U[mask_normal] = ecdf(x[mask_normal])

        # Zone extrême : GPD semi-paramétrique sur les résidus
        mask_extreme = ~mask_normal
        if np.any(mask_extreme):
            y = (x[mask_extreme] - self.u) / self.beta
            if abs(self.xi) < 1e-6:
                # Cas limite xi -> 0 : GPD -> exponentielle
                U[mask_extreme] = 1 - self.p_u * np.exp(-y)
            else:
                U[mask_extreme] = 1 - self.p_u * (1 + self.xi * y) ** (-1.0 / self.xi)

        eps = 1e-10
        return np.clip(U, eps, 1 - eps)

    # ================================================================== #
    #  VaR et ES conditionnelles                                         #
    # ================================================================== #

    def var_residuel(self, alpha: float = 0.99) -> float:
        """
        VaR analytique du résidu standardisé z suit GPD.

        Formule inverse GPD :
            VaR_z = u + (scale/shape) * [((1-alpha)/p_u)^{-shape} - 1]

        C'est un scalaire, indépendant du temps.
        """
        if abs(self.xi) < 1e-6:
            return float(self.u - self.beta * np.log((1 - alpha) / self.p_u))
        return float(self.u + (self.beta / self.xi) * (((1 - alpha) / self.p_u) ** (-self.xi) - 1))

    def var_conditionnel(self, alpha: float = 0.99) -> np.ndarray:
        """
        Série de VaR conditionnelles à chaque date t.

        VaR_t = sigma_t * VaR_z(alpha)

        La VaR est dynamique :
            - sigma_t élevé (crise)  -> VaR_t grande
            - sigma_t faible (calme) -> VaR_t petite
        Returns
        -------
        np.ndarray (N,) : VaR conditionnelle à chaque date
        """
        if self.sigma_t is None:
            raise RuntimeError("Appeler fit_garch() d'abord.")
        return self.sigma_t * self.var_residuel(alpha)

    def es_conditionnel(self, alpha: float = 0.99) -> np.ndarray:
        """
        ES conditionnel analytique à chaque date t.

        ES_z  = (VaR_z + scale - shape*u) / (1 - shape)
        ES_t  = sigma_t * ES_z
        """
        var_z = self.var_residuel(alpha)
        es_z  = float((var_z + self.beta - self.xi * self.u) / (1 - self.xi))
        return self.sigma_t * es_z

    def forecast_sigma(self, horizon: int = 1) -> float:
        """
        Prévision de sigma_{T+horizon} par le GARCH.

        Utilisé pour la simulation Monte-Carlo conditionnelle :
            L_i_simulée = sigma_{i,T+1} * GPD_inverse(U_i, z_i)

        Returns
        -------
        float : volatilité prévue à T+horizon (dans les unités originales des pertes)
        """
        if self.garch_result is None:
            raise RuntimeError("Appeler fit_garch() d'abord.")
        forecast = self.garch_result.forecast(horizon=horizon, reindex=False)
        # conditional_volatility du forecast est en % -> diviser par 100
        return float(np.sqrt(forecast.variance.values[-1, 0]) / 100.0)

    # ================================================================== #
    #  Sérialisation (compatible avec MarginalEVT)                       #
    # ================================================================== #

    def to_dict(self) -> dict:
        """Retourne les paramètres EVT (sur résidus) pour le DataFrame résumé."""
        return {
            "ticker": self.ticker,
            "u":      self.u,
            "xi":     self.xi,
            "beta":   self.beta,
            "N":      self.N,
            "Nu":     self.Nu,
            "Pu":     self.p_u
        }

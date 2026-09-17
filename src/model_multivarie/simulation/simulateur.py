import numpy as np


class Simulateur:
    """
    Simule des pertes conjointes pour k actifs via Monte-Carlo.

    Pipeline de simulation :
        1. La copule génère n_sim vecteurs de pseudo-observations U \in (0,1)^k
        2. Pour chaque actif i, on inverse la distribution marginale hybride :
               - U_i <= 1 - p_u : quantile empirique historique (zone normale)
               - U_i > 1 - p_u  : formule inverse GPD analytique (zone extrême)
        3. On calcule la perte du portefeuille : L = \sum w_i * L_i

    Généralise simulation_gumbel.py et simulation_student.py à k actifs.
    """

    # ------------------------------------------------------------------ #
    #  Inversion de la marginale hybride (ECDF + GPD)                    #
    # ------------------------------------------------------------------ #

    @staticmethod
    def inverse_marginale(
        U_asset: np.ndarray,
        u: float,
        xi: float,
        beta: float,
        p_u: float,
        losses_hist: np.ndarray,
        sigma_forecast: float = 1.0
    ) -> np.ndarray:
        """
        Inverse la distribution marginale hybride F_EVT pour un actif.

        Deux zones :
            - Zone normale (U <= 1 - p_u) :
                On rescale U sur [0, 1] et on prend le quantile empirique
                des pertes historiques sous le seuil u.
            - Zone extrême (U > 1 - p_u) :
                Formule analytique inverse de la GPD :
                    L = u + (scale/shape) * [((1-U)/p_u)^{-shape} - 1]

        Mode GARCH-EVT (sigma_forecast != 1.0) :
            losses_hist contient les résidus standardisés z_t (pas les pertes brutes).
            La sortie est multipliée par sigma_forecast :
                L_simulée = sigma_{T+1} * GPD_inverse(U, z)

        Parameters
        ----------
        U_asset        : np.ndarray (n_sim,) - pseudo-observations de l'actif
        u              : float - seuil EVT (sur z_t en mode GARCH, sur L_t sinon)
        xi             : float - paramètre de forme GPD
        beta           : float - paramètre d'échelle GPD
        p_u            : float - P(L > u) = Nu/N
        losses_hist    : np.ndarray - pertes ou résidus historiques
        sigma_forecast : float - sigma_{T+1} prédit par GARCH (1.0 = mode standard sans GARCH)

        Returns
        -------
        np.ndarray (n_sim,) : pertes simulées pour l'actif
        """
        simulated = np.zeros_like(U_asset, dtype=float)
        seuil_prob = 1.0 - p_u   # probabilité de dépasser u

        # Zone normale : quantile empirique avec rescaling
        mask_normal = U_asset <= seuil_prob
        if np.any(mask_normal):
            # Rescaling : on ramène U \in [0, seuil_prob] vers [0, 1]
            U_rescaled = U_asset[mask_normal] / seuil_prob
            # On isole uniquement les observations sous le seuil u (pertes ou résidus)
            sous_seuil = losses_hist[losses_hist <= u]
            simulated[mask_normal] = np.quantile(sous_seuil, U_rescaled)

        # Zone extrême : inverse analytique GPD
        mask_extreme = U_asset > seuil_prob
        if np.any(mask_extreme):
            U_ext = U_asset[mask_extreme]
            if abs(xi) < 1e-6:
                # Cas limite xi -> 0 : GPD -> exponentielle
                simulated[mask_extreme] = u - beta * np.log((1 - U_ext) / p_u)
            else:
                simulated[mask_extreme] = u + (beta / xi) * (((1 - U_ext) / p_u) ** (-xi) - 1)

        # Mode GARCH-EVT : scaling par la volatilité conditionnelle prévue σ_{T+1}
        # L_simulée = σ_{T+1} × z_simulé  (sigma_forecast = 1.0 en mode standard)
        return simulated * sigma_forecast

    # ------------------------------------------------------------------ #
    #  Simulation des pertes conjointes                                  #
    # ------------------------------------------------------------------ #

    @staticmethod
    def simulate_joint_losses(
        copula,
        evts: list,
        losses_dict: dict,
        tickers: list,
        n_sim: int,
        sigma_forecasts: list = None
    ) -> np.ndarray:
        """
        Simule les pertes conjointes pour k actifs.

        Mode standard (sigma_forecasts=None) :
            losses_dict contient les pertes brutes L_t.
            Sortie : pertes simulées en unités originales.

        Mode GARCH-EVT (sigma_forecasts fourni) :
            losses_dict contient les résidus standardisés z_t.
            sigma_forecasts[i] = sigma_{i,T+1} prédit par le GARCH de l'actif i.
            Sortie : L_simulée = sigma_{i,T+1} * z_simulé  (pertes conditionnelles)

        Parameters
        ----------
        copula          : objet copule ajusté
        evts            : list[MarginalEVT ou GarchEVT] - un EVT par actif
        losses_dict     : dict ticker -> np.ndarray - pertes brutes ou résidus z_t
        tickers         : list[str]
        n_sim           : int - nombre de scénarios
        sigma_forecasts : list[float], optional - sigma_{i,T+1} par actif (mode GARCH-EVT)

        Returns
        -------
        np.ndarray (n_sim x k) : pertes simulées par actif
        """
        # Étape 1 : simulation de la copule -> pseudo-observations U (n_sim x k)
        U = copula.simulate(n_sim)

        k = len(tickers)
        joint_losses = np.zeros((n_sim, k))

        # Étape 2 : inversion de la marginale hybride pour chaque actif
        for i, (ticker, evt) in enumerate(zip(tickers, evts)):
            losses_hist = losses_dict[ticker]
            # sigma_forecast = 1.0 en mode standard (pas de scaling)
            sigma = sigma_forecasts[i] if sigma_forecasts is not None else 1.0
            joint_losses[:, i] = Simulateur.inverse_marginale(
                U[:, i],
                evt.u,
                evt.xi,
                evt.beta,
                evt.p_u,
                losses_hist,
                sigma_forecast=sigma
            )

        return joint_losses

    # ------------------------------------------------------------------ #
    #  Perte du portefeuille et mesures de risque                        #
    # ------------------------------------------------------------------ #

    @staticmethod
    def portfolio_losses(joint_losses: np.ndarray, weights: np.ndarray) -> np.ndarray:
        """
        Calcule la perte agrégée du portefeuille.

        L_portfolio = \sum_i w_i * L_i  (produit matriciel : joint_losses @ weights)
        """
        return joint_losses @ weights

    @staticmethod
    def var_es(losses: np.ndarray, alpha: float = 0.99) -> tuple:
        """
        Calcule la VaR et l'ES empiriques à partir des pertes simulées.

        VaR_alpha = quantile alpha de la distribution des pertes
        ES_alpha  = espérance des pertes qui dépassent VaR_alpha

        Parameters
        ----------
        losses : np.ndarray - vecteur de pertes (univarié)
        alpha  : float - niveau de confiance (0.99 = 99%)

        Returns
        -------
        (VaR, ES) : tuple de floats
        """
        var = np.quantile(losses, alpha)
        es  = losses[losses >= var].mean()
        return float(var), float(es)

    @staticmethod
    def joint_extreme_prob(joint_losses: np.ndarray, alpha: float = 0.99) -> float:
        """
        Calcule la probabilité empirique de co-krach.

        Définition : P(L_1 > VaR_alpha(L_1) ET L_2 > VaR_alpha(L_2) ET ... ET L_k > VaR_alpha(L_k))

        Une probabilité élevée indique une forte dépendance en queue supérieure
        (tous les actifs chutent en même temps lors d'une crise).
        """
        # Quantile individuel par actif au niveau alpha
        quantiles = np.quantile(joint_losses, alpha, axis=0)   # (k,)
        # Indicateur : tous les actifs dépassent simultanément leur VaR individuelle
        above_all = np.all(joint_losses > quantiles, axis=1)
        return float(np.mean(above_all))

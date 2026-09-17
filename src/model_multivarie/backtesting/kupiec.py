import numpy as np
from scipy.stats import chi2


class KupiecTest:
    """
    Tests statistiques de validité des modèles de VaR.

    Deux tests complémentaires :
        1. Test de Kupiec (POF - Proportion of Failures) :
           Vérifie que le taux de violation empirique est conforme au niveau alpha.
           H0 : pi_hat = 1 - alpha  (taux théorique de violations)

        2. Test de Christoffersen (indépendance des violations) :
           Vérifie que les violations ne se regroupent pas dans le temps.
           H0 : les violations sont indépendantes (pas de clustering)

    Les deux tests s'appliquent à n'importe quelle série (perte univariée,
    perte de portefeuille, perte par actif).

    Référence :
        Kupiec, P. (1995). "Techniques for Verifying the Accuracy of Risk Measurement Models."
        Christoffersen, P. (1998). "Evaluating Interval Forecasts."
    """

    @staticmethod
    def kupiec(losses: np.ndarray, var_series, alpha: float = 0.99) -> dict:
        """
        Test de Kupiec (POF).

        Parameters
        ----------
        losses     : np.ndarray — pertes réalisées (historiques)
        var_series : float ou np.ndarray
            VaR à tester. Si float : VaR statique (même valeur pour toute la période).
            Si ndarray : VaR dynamique (une valeur par jour).
        alpha      : float — niveau de confiance de la VaR (ex: 0.99)

        Returns
        -------
        dict avec les clés :
            N         : nombre de violations observées
            T         : longueur de la série
            pi_hat    : taux de violation empirique
            LR        : statistique de test (loi khi²(1) sous H0)
            p_value   : p-valeur du test
            rejete_H0 : True si on rejette H0 au seuil 5%
            interpretation : texte explicatif
        """
        p = 1.0 - alpha   # taux théorique de violation

        losses     = np.asarray(losses)
        var_series = np.full_like(losses, var_series) if np.isscalar(var_series) else np.asarray(var_series)

        violations = losses > var_series
        T   = int(len(losses))
        N   = int(np.sum(violations))
        pi_hat = N / T

        # Cas dégénérés (pas de violation ou 100%)
        if pi_hat == 0 or pi_hat == 1:
            return {
                "N": N, "T": T, "pi_hat": pi_hat,
                "LR": np.nan, "p_value": np.nan, "rejete_H0": None,
                "interpretation": "Test impossible (pi_hat = 0 ou 1)"
            }

        # Statistique de rapport de vraisemblance
        # LR suit khi²(1) sous H0
        LR = -2.0 * np.log(
            ((1 - p) ** (T - N) * p ** N) /
            ((1 - pi_hat) ** (T - N) * pi_hat ** N)
        )
        p_value = 1.0 - chi2.cdf(LR, df=1)
        rejete  = bool(p_value < 0.05)

        if rejete:
            if pi_hat > p:
                interp = f"Modèle sous-estime le risque (trop peu de couverture : {pi_hat:.2%} > {p:.2%} théorique)"
            else:
                interp = f"Modèle sur-estime le risque (trop conservateur : {pi_hat:.2%} < {p:.2%} théorique)"
        else:
            interp = f"Modèle valide : taux de violation {pi_hat:.2%} conforme à {p:.2%} théorique"

        return {
            "N": N, "T": T, "pi_hat": pi_hat,
            "LR": float(LR), "p_value": float(p_value),
            "rejete_H0": rejete,
            "interpretation": interp
        }

    @staticmethod
    def christoffersen(losses: np.ndarray, var_series, alpha: float = 0.99) -> dict:
        """
        Test de Christoffersen (indépendance conditionnelle des violations).

        Vérifie que les violations ne se regroupent pas dans le temps
        (ce qui indiquerait que le modèle réagit trop lentement aux crises).

        H0 : pi_{01} = pi_{11} (les violations sont indépendantes d'un jour à l'autre)

        Parameters
        ----------
        losses     : np.ndarray — pertes réalisées
        var_series : float ou np.ndarray — VaR à tester
        alpha      : float — niveau de confiance

        Returns
        -------
        dict avec LR_ind, p_value, rejete_H0, interpretation
        """
        losses     = np.asarray(losses)
        var_series = np.full_like(losses, var_series) if np.isscalar(var_series) else np.asarray(var_series)

        violations = (losses > var_series).astype(int)
        T = len(violations)

        # Comptage des transitions entre états (0 = pas de violation, 1 = violation)
        n_00 = int(np.sum((violations[:-1] == 0) & (violations[1:] == 0)))
        n_01 = int(np.sum((violations[:-1] == 0) & (violations[1:] == 1)))
        n_10 = int(np.sum((violations[:-1] == 1) & (violations[1:] == 0)))
        n_11 = int(np.sum((violations[:-1] == 1) & (violations[1:] == 1)))

        # Probabilités conditionnelles de violation
        pi_01 = n_01 / (n_00 + n_01) if (n_00 + n_01) > 0 else 0.0
        pi_11 = n_11 / (n_10 + n_11) if (n_10 + n_11) > 0 else 0.0
        pi    = (n_01 + n_11) / T if T > 0 else 0.0

        if pi_01 == 0 or pi_11 == 0 or pi == 0 or pi == 1:
            return {
                "LR_ind": np.nan, "p_value": np.nan, "rejete_H0": None,
                "pi_01": pi_01, "pi_11": pi_11,
                "interpretation": "Test impossible (transitions insuffisantes)"
            }

        LR_ind = -2.0 * np.log(
            ((1 - pi) ** (n_00 + n_10) * pi ** (n_01 + n_11)) /
            ((1 - pi_01) ** n_00 * pi_01 ** n_01 * (1 - pi_11) ** n_10 * pi_11 ** n_11)
        )
        p_value = 1.0 - chi2.cdf(LR_ind, df=1)
        rejete  = bool(p_value < 0.05)

        if rejete:
            interp = (
                f"Les violations se regroupent dans le temps (clustering). "
                f"P(violation | violation hier) = {pi_11:.2%} vs "
                f"P(violation | pas de violation hier) = {pi_01:.2%}"
            )
        else:
            interp = "Les violations sont indépendantes : pas de clustering détecté"

        return {
            "LR_ind":    float(LR_ind),
            "p_value":   float(p_value),
            "rejete_H0": rejete,
            "pi_01":     pi_01,
            "pi_11":     pi_11,
            "interpretation": interp
        }

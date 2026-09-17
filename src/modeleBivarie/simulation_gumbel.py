import numpy as np
import pandas as pd
from scipy.stats import gumbel_r
from scipy.stats import levy_stable
from copulas.bivariate import Gumbel

import numpy as np

# Dans notre cas on a 2 actifs donc d=2
d = 2

def simulate_gumbel_copula(theta, n):
    """
    Simulation d'une copule de Gumbel en dimension d
    
    theta : paramètre de dépendance
    n     : nombre de scénarios
    d     : nombre d'actifs
    """

    copula = Gumbel()
    copula.theta = theta
    copula.tau = 1 - 1/theta

    U = copula.sample(n)

    return U

    # if theta < 1:
    #     raise ValueError("Pour une copule de Gumbel il faut theta >= 1")

    # alpha = 1.0 / theta

    # # Variable latente commune (choc systémique global)
    # W = levy_stable.rvs(
    #     alpha=alpha,
    #     beta=1,
    #     size=n
    # )

    # # Protection numérique
    # W = np.maximum(W, 1e-10)

    # # Exponentielles indépendantes ( E est un choc idiosyncratique)
    # E = np.random.exponential(scale=1.0, size=(n, d))

    # # Construction des U
    # # On divise E par W car tous les actifs doivent partager le même choc systémique
    # U = np.exp(-(E / W[:, None]) ** alpha)  # shape (n, d)

    # return U  # shape (n, d)

def evt_quantile(u, xi, beta, p, p_u):
    """
    Quantile EVT (POT + GPD)
    """
    return u + (beta / xi) * (( (1 - p) / p_u ) ** (-xi) - 1)


def simulate_joint_losses(
    params_evt,
    n,
    theta,
    data_historique
):
    """
    Simule des pertes conjointes complètes via Gumbel (Zone Normale + Zone GPD)
    """
    U = simulate_gumbel_copula(theta, n)
    assets = list(params_evt.index)
    d = len(assets)
    losses = np.zeros((n, d))

    for i, asset in enumerate(assets):
        p_params = params_evt.loc[asset]
        
        u = p_params["u"]
        xi = p_params["xi"]
        beta = p_params["beta"]
        p_u = p_params["Pu"]
        seuil_prob = 1 - p_u
        
        col_name = "loss_sp500" if asset == "sp500" else "loss_eurostoxx"
        losses_hist_asset = data_historique[col_name].values
        
        U_asset = U[:, i]
        
        # 1. Zone normale (U <= seuil_prob)
        mask_normal = U_asset <= seuil_prob
        if np.any(mask_normal):
            # ON RESCALE de [0, seuil_prob] vers [0, 1]
            U_rescaled = U_asset[mask_normal] / seuil_prob
            
            # On isole UNIQUEMENT les données historiques sous le seuil u
            losses_normales_hist = losses_hist_asset[losses_hist_asset <= u]
            
            # On passe U_rescaled pour mapper correctement
            losses[mask_normal, i] = np.quantile(losses_normales_hist, U_rescaled)
            
        # 2. Zone extrême (U > seuil_prob)
        mask_extreme = U_asset > seuil_prob
        if np.any(mask_extreme):
            U_ext = U_asset[mask_extreme]
            losses[mask_extreme, i] = u + (beta / xi) * (((1 - U_ext) / p_u) ** (-xi) - 1)

    return losses

# Construction de la perte du portefeuille
def portfolio_losses(joint_losses, weights):
    """
    Calcule la perte du portefeuille
    """
    return joint_losses @ weights

# Calcul de VaR et ES empiriques
def var_es(losses, alpha=0.99):
    """
    Calcule VaR et ES empiriques
    """
    var = np.quantile(losses, alpha)
    es = losses[losses >= var].mean()
    return var, es

import numpy as np
from scipy.stats import multivariate_t, t
import pandas as pd

# Le nombre d'actifs d
d=2
# Protection numérique
EPS = 1e-10


def simulate_t_copula(n_sim, rho, nu):
    """
    Simulation d'une copule t multivariée
    """
    Sigma = np.full((d, d), rho) # On remplit une matrice de taille dxd avec la valeur de rho
    np.fill_diagonal(Sigma, 1.0) # On remplace toutes les valeurs de la diagonale par 1

    # Vérification rapide de validité
    eigvals = np.linalg.eigvals(Sigma)
    if np.any(eigvals <= 0):
        raise ValueError("La matrice de corrélation n'est pas définie positive")

    # multivariate_t.rvs(...) génère des points aléatoires qui suivent la distribution de student multivariée
    Z = multivariate_t.rvs(
        loc=np.zeros(d),
        shape=Sigma,
        df=nu,
        size=n_sim
    )
    # Sécurité dimensionnelle
    Z = np.atleast_2d(Z)

    # Fonction de répartition qui transforme les valeurs de Z en U comprises entre 0 et 1
    U = t.cdf(Z, df=nu)
    # Protection numérique EVT
    U = np.clip(U, EPS, 1 - EPS)

    return U

def gpd_inverse(U, u, xi, beta, p_u, losses_historiques):
    """
    Inversion de la distribution marginale complète (Hybride) :
    - Si U <= 1 - p_u : Rendement normal -> Quantile empirique des données historiques
    - Si U > 1 - p_u  : Rendement extrême -> Formule mathématique GPD inverse
    """
    # On prépare un tableau de zéros de la même taille que U
    simulated_losses = np.zeros_like(U)
    
    # Seuil de probabilité à partir duquel on entre dans la GPD
    seuil_prob = 1 - p_u
    
    # 1. Zone normale (U <= seuil_prob)
    mask_normal = U <= seuil_prob
    if np.any(mask_normal):
        # ON RESCALE : On ramène U sur l'échelle [0, 1]
        U_rescaled = U[mask_normal] / seuil_prob
        # On extrait uniquement la partie sous le seuil u de l'historique
        losses_normales_hist = losses_historiques[losses_historiques <= u]
        simulated_losses[mask_normal] = np.quantile(losses_normales_hist, U_rescaled)
        
    # 2. Zone extrême / GPD (U > seuil_prob)
    mask_extreme = U > seuil_prob
    if np.any(mask_extreme):
        U_ext = U[mask_extreme]
        if abs(xi) < 1e-6:

            simulated_losses[mask_extreme] = (
                u
                - beta * np.log((1 - U_ext) / p_u)
            )
        else:
            # La formule correcte de l'inverse de la GPD conditionnelle :
            simulated_losses[mask_extreme] = u + (beta / xi) * ( ( (1 - U_ext) / p_u ) ** (-xi) - 1 )
            
    return simulated_losses

def simulate_losses_evt_t(
    n_sim,
    params_evt,   # dict par actif
    rho,
    nu,
    weights,
    data_historique # Ajout de la base de données historique (.csv ou DataFrame)
):
    assets = list(params_evt.index)
    d = len(assets)

    U = simulate_t_copula(n_sim, rho, nu)
    losses = np.zeros((n_sim, d))

    for i, asset in enumerate(assets):
        p = params_evt.loc[asset]
        
        # On récupère les pertes historiques spécifiques à cet actif pour la zone normale
        col_name = "loss_sp500" if asset == "sp500" else "loss_eurostoxx"
        losses_hist_asset = data_historique[col_name].values
        
        losses[:, i] = gpd_inverse(
            U[:, i],
            p["u"],
            p["xi"],
            p["beta"],
            p["Pu"],
            losses_hist_asset
        )

    portfolio_losses = losses @ weights
    return portfolio_losses

def var_es(losses, alpha=0.99):
    VaR = np.quantile(losses, alpha)
    ES = losses[losses >= VaR].mean()
    return VaR, ES

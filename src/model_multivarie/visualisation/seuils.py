import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from scipy.stats import genpareto


def plot_mean_excess(tickers: list, evts: list, losses_dict: dict):
    """
    Trace les Mean Excess Plots pour chaque actif du portefeuille.

    Le Mean Excess Plot (MEP) aide à sélectionner le seuil u :
        - Au-delà du seuil optimal, la courbe doit être approximativement
          linéaire croissante -> confirme une queue lourde GPD avec shape > 0.
        - Une droite de tendance est tracée sur la zone 95e-98e percentile.
        - Le seuil retenu u est affiché en vert pointillé.

    """
    k = len(tickers)
    fig, axes = plt.subplots(1, k, figsize=(6 * k, 4))
    if k == 1:
        axes = [axes]

    for ax, ticker, evt in zip(axes, tickers, evts):
        losses = losses_dict[ticker]
        thresholds = np.linspace(np.percentile(losses, 90), np.percentile(losses, 99), 50)
        me = evt.mean_excess(thresholds)

        # Droite de tendance sur la zone 95e – 98e percentile
        zone_mask = (
            (thresholds >= np.percentile(losses, 95)) &
            (thresholds <= np.percentile(losses, 98))
        )
        zone_t  = thresholds[zone_mask].reshape(-1, 1)
        zone_me = me[zone_mask]

        ax.plot(thresholds, me, marker='o', markersize=3, label=f'Mean Excess {ticker}')

        if len(zone_t) > 1:
            reg  = LinearRegression().fit(zone_t, zone_me)
            line = reg.predict(zone_t)
            ax.plot(zone_t, line, color='red', linewidth=2, label='Droite de tendance')

        if evt.u is not None:
            ax.axvline(evt.u, color='green', linestyle='--', linewidth=1.5,
                       label=f'Seuil u = {evt.u:.4f}')

        ax.set_xlabel("Seuil u")
        ax.set_ylabel("Excès moyen E[L-u | L>u]")
        ax.set_title(f"Mean Excess Plot – {ticker}")
        ax.legend(fontsize=8)
        ax.grid(True)

    plt.tight_layout()
    plt.show()


def plot_gpd_parameters(tickers: list, losses_dict: dict, percentile_range=(90, 99), n_points=20):
    """
    Trace la stabilité des paramètres GPD (shape et scale) en fonction du seuil u.

    Un bon seuil est caractérisé par des valeurs stables de shape et scale dans
    un intervalle de seuils. On cherche la zone de "plateau".

    Pour chaque actif : deux sous-graphes (shape en haut, scale en bas).

    Parameters
    ----------
    percentile_range : tuple (low, high) — plage de percentiles à explorer
    n_points : int — nombre de seuils à tester dans la plage
    """
    k = len(tickers)
    fig, axes = plt.subplots(2, k, figsize=(6 * k, 8))
    if k == 1:
        axes = axes.reshape(2, 1)

    for j, ticker in enumerate(tickers):
        losses   = losses_dict[ticker]
        u_values = np.linspace(
            np.percentile(losses, percentile_range[0]),
            np.percentile(losses, percentile_range[1]),
            n_points
        )
        xi_list, beta_list = [], []

        for u in u_values:
            excesses = losses[losses > u] - u
            if len(excesses) < 5:
                xi_list.append(np.nan)
                beta_list.append(np.nan)
                continue
            # La loi GPD est ajustée aux excès via le maximum de vraisemblance
            # floc=0 car les excès sont déjà centrés en 0
            xi, _, beta = genpareto.fit(excesses, floc=0)
            xi_list.append(xi)
            beta_list.append(beta)

        axes[0, j].plot(u_values, xi_list, marker='o', markersize=4)
        axes[0, j].set_xlabel("Seuil u")
        axes[0, j].set_ylabel("ξ (shape)")
        axes[0, j].set_title(f"Stabilité ξ(u) – {ticker}")
        axes[0, j].grid(True)

        axes[1, j].plot(u_values, beta_list, marker='o', markersize=4, color='orange')
        axes[1, j].set_xlabel("Seuil u")
        axes[1, j].set_ylabel("β (scale)")
        axes[1, j].set_title(f"Stabilité β(u) – {ticker}")
        axes[1, j].grid(True)

    plt.tight_layout()
    plt.show()

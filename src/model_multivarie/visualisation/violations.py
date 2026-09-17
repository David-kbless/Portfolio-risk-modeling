import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# Couleurs et labels par méthode (cohérence visuelle)
_PALETTE = {
    "VaR_multivarie": ("red",       "VaR Multivariée EVT+Copule"),
    "VaR_historique": ("blue",      "VaR Historique"),
    "VaR_normale":    ("green",     "VaR Normale"),
    "VaR_EVT_1D":     ("purple",    "VaR EVT Univariée"),
    "VaR_GARCH_EVT":  ("darkorange","VaR GARCH-EVT"),
}


def plot_var_vs_losses(
    results: pd.DataFrame,
    alpha: float = 0.99,
    var_statique: float = None,
    label_statique: str = "VaR Modèle (statique)"
):
    """
    Trace les pertes historiques du portefeuille face aux séries de VaR.

    Affiche :
        - Pertes journalières du portefeuille (gris)
        - Séries de VaR dynamiques (rolling window) depuis results
        - VaR statique du modèle multivarié (tiret horizontal) si fournie
        - Violations marquées par des croix rouges

    Parameters
    ----------
    results       : DataFrame - sortie de RollingBacktest.run()
    alpha         : float - niveau de confiance (pour le titre)
    var_statique  : float, optional - VaR statique du modèle Monte-Carlo complet
    label_statique: str - label pour la VaR statique
    """
    fig, ax = plt.subplots(figsize=(14, 5))

    # Pertes historiques
    ax.plot(results.index, results["portfolio_loss"],
            color="gray", alpha=0.6, linewidth=0.8, label="Perte portefeuille")

    # Séries de VaR dynamiques
    for col, (color, label) in _PALETTE.items():
        if col in results.columns:
            ax.plot(results.index, results[col],
                    color=color, linewidth=1.5, label=f"{label} {alpha*100:.0f}%")

    # VaR statique du modèle (ligne horizontale)
    if var_statique is not None:
        ax.axhline(var_statique, color="red", linestyle="--", linewidth=2,
                   label=f"{label_statique} {alpha*100:.0f}%")
        violations_statique = results["portfolio_loss"][results["portfolio_loss"] > var_statique]
        ax.scatter(violations_statique.index, violations_statique.values,
                   color="red", marker="x", s=50, zorder=5, label="Violations (modèle statique)")

    # Violations pour chaque méthode dynamique
    for col, (color, label) in _PALETTE.items():
        if col in results.columns:
            viol = results[results["portfolio_loss"] > results[col]]
            if not viol.empty:
                ax.scatter(viol.index, viol["portfolio_loss"],
                           color=color, marker="x", s=30, zorder=5, alpha=0.7)

    ax.set_xlabel("Date")
    ax.set_ylabel("Perte")
    ax.set_title(f"Backtesting : Pertes du portefeuille vs VaR {alpha*100:.0f}%")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(True, alpha=0.4)
    plt.tight_layout()
    plt.show()


def plot_cumul_violations(
    results: pd.DataFrame,
    alpha: float = 0.99,
    var_statique: float = None,
    label_statique: str = "Modèle statique"
):
    """
    Trace le cumul des violations dans le temps pour chaque méthode.

    La ligne noire théorique représente (1 - alpha) x nombre de jours écoulés.
    Un modèle bien calibré doit rester proche de cette ligne.

    Si la courbe est au-dessus : le modèle sous-estime le risque.
    Si la courbe est en-dessous : le modèle sur-estime le risque.

    """
    fig, ax = plt.subplots(figsize=(12, 4))
    n = len(results)

    # Ligne théorique de référence : (1-alpha) * n_jours
    theoretical = np.arange(1, n + 1) * (1 - alpha)
    ax.plot(results.index, theoretical, "--", color="black",
            linewidth=1.5, label=f"Taux théorique {(1-alpha)*100:.1f}%")

    # Méthodes dynamiques
    for col, (color, label) in _PALETTE.items():
        if col in results.columns:
            viol_cum = (results["portfolio_loss"] > results[col]).cumsum()
            ax.plot(results.index, viol_cum,
                    color=color, linewidth=1.5,
                    label=f"{label} ({int(viol_cum.iloc[-1])} violations)")

    # Méthode statique
    if var_statique is not None:
        viol_static = (results["portfolio_loss"] > var_statique).cumsum()
        ax.plot(results.index, viol_static, "--", color="red", linewidth=2,
                label=f"{label_statique} ({int(viol_static.iloc[-1])} violations)")

    ax.set_xlabel("Date")
    ax.set_ylabel("Violations cumulées")
    ax.set_title(f"Cumul des violations VaR {alpha*100:.0f}%")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(True, alpha=0.4)
    plt.tight_layout()
    plt.show()


def plot_portfolio_losses_distribution(
    portfolio_losses_sim: np.ndarray,
    VaR: float,
    ES: float,
    alpha: float = 0.99
):
    """
    Trace la distribution des pertes simulées du portefeuille avec VaR et ES.

    Utile pour visualiser la queue de distribution et la position de la VaR.
    """
    fig, ax = plt.subplots(figsize=(10, 5))

    ax.hist(portfolio_losses_sim, bins=200, density=True,
            color="steelblue", alpha=0.7, label="Pertes simulées")

    ax.axvline(VaR, color="red", linewidth=2,
               label=f"VaR {alpha*100:.0f}% = {VaR:.4f}")
    ax.axvline(ES, color="orange", linewidth=2, linestyle="--",
               label=f"ES {alpha*100:.0f}% = {ES:.4f}")

    ax.set_xlabel("Perte du portefeuille")
    ax.set_ylabel("Densité")
    ax.set_title(f"Distribution des pertes simulées – Portefeuille")
    ax.legend()
    ax.grid(True, alpha=0.4)
    plt.tight_layout()
    plt.show()

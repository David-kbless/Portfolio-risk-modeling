import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations


def plot_pit_histograms(tickers: list, U_dict: dict):
    """
    Trace les histogrammes des pseudo-observations uniformes U (PIT) par actif.

    Si la transformation PIT (Probability Integral Transform) est correcte,
    les U doivent suivre une loi uniforme sur [0, 1].
    Un histogramme plat confirme que la marginale EVT est bien spécifiée.

    """
    k = len(tickers)
    fig, axes = plt.subplots(1, k, figsize=(6 * k, 4))
    if k == 1:
        axes = [axes]

    for ax, ticker in zip(axes, tickers):
        ax.hist(U_dict[ticker], bins=30, edgecolor='black', color='steelblue', alpha=0.8)
        # Ligne de référence : densité uniforme théorique
        T = len(U_dict[ticker])
        ax.axhline(T / 30, color='red', linestyle='--', linewidth=1.5, label='Uniforme théorique')
        ax.set_title(f'Histogramme PIT – {ticker}')
        ax.set_xlabel('U')
        ax.set_ylabel('Fréquence')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.4)

    plt.tight_layout()
    plt.show()


def plot_copula_scatter(tickers: list, U_dict: dict):
    """
    Trace les nuages de points des pseudo-observations pour chaque paire d'actifs.

    Visualise la structure de dépendance capturée par la copule.
    Pour k actifs, trace k(k-1)/2 paires.

    """
    pairs = list(combinations(tickers, 2))
    n = len(pairs)
    if n == 0:
        return

    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5))
    if n == 1:
        axes = [axes]

    for ax, (t1, t2) in zip(axes, pairs):
        ax.scatter(U_dict[t1], U_dict[t2], alpha=0.3, s=3, color='steelblue')
        ax.set_xlabel(f'U – {t1}')
        ax.set_ylabel(f'U – {t2}')
        ax.set_title(f'Pseudo-observations copule\n{t1} vs {t2}')
        ax.grid(True, alpha=0.4)

    plt.tight_layout()
    plt.show()

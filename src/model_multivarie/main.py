"""
PortfolioRiskModel — Modèle de risque de portefeuille multivarié

Pipeline complet :
    1. Collecte et traitement des données de marché (k actifs)
    2. Sélection des seuils EVT par actif (Mean Excess Plot + stabilité GPD)
    3. Ajustement des marginales EVT (GPD) par actif
    4. Transformation PIT -> pseudo-observations uniformes
    5. Ajustement de la copule choisie (Student-t / Gumbel / Gaussienne)
    6. Simulation Monte-Carlo des pertes conjointes
    7. Calcul VaR et ES du portefeuille
    8. Backtesting (Kupiec, Christoffersen) + visualisations

Exécution :
    cd risk_extreme_project
    source venv/bin/activate
    python -m implementations.model_multivarie.main
    # ou directement :
    python implementations/model_multivarie/main.py
"""

import sys
import os
import re
import pickle
import warnings

import numpy as np
import pandas as pd

# Assurer que la racine du projet est dans sys.path pour les imports absolus
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from implementations.model_multivarie.data.loader import DataLoader, sanitize
from implementations.model_multivarie.marginales.evt import MarginalEVT
from implementations.model_multivarie.marginales.garch_evt import GarchEVT
from implementations.model_multivarie.copules.student_t import StudentTCopula
from implementations.model_multivarie.copules.gumbel import GumbelCopula
from implementations.model_multivarie.copules.gaussienne import GaussianCopula
from implementations.model_multivarie.simulation.simulateur import Simulateur
from implementations.model_multivarie.backtesting.kupiec import KupiecTest
from implementations.model_multivarie.backtesting.rolling import RollingBacktest
from implementations.model_multivarie.visualisation import seuils as viz_seuils
from implementations.model_multivarie.visualisation import pit as viz_pit
from implementations.model_multivarie.visualisation import violations as viz_violations


class PortfolioRiskModel:
    """
    Modèle de risque de portefeuille multivarié basé sur l'EVT et les copules.

    Généralise le modèle bivarié (SPY/FEZ de datas.py) à k actifs quelconques.

    Parameters
    ----------
    tickers : list of str
        Tickers Yahoo Finance des k actifs.
        Ex : ["SPY", "FEZ"]  (bivarié )
             ["SPY", "FEZ", "QQQ", "GLD"]  (k=4)
    start_date : str
        Date de début (format "YYYY-MM-DD"). Ex : "2005-01-01"
    weights_file : str, optional
        Chemin vers un fichier .pkl ou .csv contenant les poids du portefeuille.
        Si None -> poids égaux (1/k).
    copula_type : str
        Type de copule : "student" | "gumbel" (k=2 uniquement) | "gaussian"
        - "student"  : t-Student multivariée, recommandée pour k > 2
        - "gumbel"   : Gumbel bivariée, capture la dépendance en queue supérieure
        - "gaussian" : Gaussienne (queue légère, référence)
    n_sim : int
        Nombre de simulations Monte-Carlo (défaut : 200 000)
    alpha : float
        Niveau de confiance pour VaR et ES (défaut : 0.99 = 99%)
    threshold_percentile : float
        Percentile pour la sélection automatique du seuil EVT (défaut : 95.0)
    output_dir : str
        Répertoire de sortie pour les CSV intermédiaires (défaut : ".")
    use_garch : bool
        Si True, utilise GARCH(1,1) + EVT sur résidus (GarchEVT) au lieu de l'EVT
        standard (MarginalEVT). Corrige le clustering des violations (test de
        Christoffersen) en rendant la VaR conditionnelle et dynamique.
        - VaR_t = sigma_t x VaR_GPD(z)  (dépend de la volatilité courante)
        - La simulation utilise sigma_{T+1} prévu par le GARCH
        Défaut : False (comportement original)
    """

    def __init__(
        self,
        tickers: list,
        start_date: str = "2005-01-01",
        weights_file: str = None,
        copula_type: str = "student",
        n_sim: int = 200_000,
        alpha: float = 0.99,
        threshold_percentile: float = 95.0,
        output_dir: str = ".",
        use_garch: bool = False,
    ):
        self.tickers              = tickers
        self.k                    = len(tickers)
        self.start_date           = start_date
        self.copula_type          = copula_type
        self.n_sim                = n_sim
        self.alpha                = alpha
        self.threshold_percentile = threshold_percentile
        self.use_garch            = use_garch
        self.output_dir           = output_dir

        # Chargement des poids du portefeuille
        self.weights = self._load_weights(weights_file)

        # Composant de chargement des données
        self._loader = DataLoader(tickers, start_date)

        # ---- État interne (rempli par les méthodes du pipeline) ----
        self.data                 = None   # DataFrame (returns + losses)
        self.losses_dict          = None   # dict ticker -> np.ndarray des pertes
        self.evts                 = None   # list[MarginalEVT], un par actif
        self.U_dict               = None   # dict ticker -> np.ndarray des pseudo-obs
        self.U_matrix             = None   # np.ndarray (T * k) des pseudo-obs
        self.copula               = None   # objet copule ajusté
        self.param_gpd            = None   # DataFrame résumé des paramètres GPD
        self.joint_losses_sim     = None   # np.ndarray (n_sim * k)
        self.portfolio_losses_sim = None   # np.ndarray (n_sim,)
        self.VaR                  = None   # float
        self.ES                   = None   # float
        self.backtesting_results  = None   # DataFrame

    # ================================================================== #
    #  Chargement des poids                                              #
    # ================================================================== #

    def _load_weights(self, weights_file: str) -> np.ndarray:
        """
        Charge les poids depuis un fichier .pkl ou .csv.

        Si weights_file est None -> poids égaux (1/k).
        Le fichier .pkl doit contenir un np.ndarray ou une liste de k floats.
        Le fichier .csv doit avoir une seule colonne avec k valeurs.

        Les poids sont automatiquement normalisés si leur somme != 1.
        """
        if weights_file is None:
            w = np.ones(self.k) / self.k
            print(f"[Poids] Aucun fichier fourni -> poids égaux : {np.round(w, 4)}")
            return w

        if not os.path.exists(weights_file):
            raise FileNotFoundError(f"Fichier de poids introuvable : {weights_file}")

        if weights_file.endswith(".pkl"):
            with open(weights_file, "rb") as f:
                w = pickle.load(f)
        elif weights_file.endswith(".csv"):
            df = pd.read_csv(weights_file, index_col=0)
            w  = df.values.flatten()
        else:
            raise ValueError(f"Format non supporté : '{weights_file}'. Utiliser .pkl ou .csv.")

        w = np.asarray(w, dtype=float)

        if len(w) != self.k:
            raise ValueError(
                f"Le fichier de poids contient {len(w)} valeurs "
                f"mais {self.k} actifs sont définis."
            )

        if not np.isclose(w.sum(), 1.0):
            warnings.warn(f"Poids normalisés (somme = {w.sum():.4f} différent de 1.0).")
            w = w / w.sum()

        print(f"[Poids] Chargés depuis '{weights_file}' : {dict(zip(self.tickers, np.round(w, 4)))}")
        return w

    # ================================================================== #
    #  Étape 1 : Collecte et traitement des données                      #
    # ================================================================== #

    def load_data(self, from_csv: str = None) -> "PortfolioRiskModel":
        """
        Télécharge les données de marché ou les charge depuis un CSV existant.

        Parameters
        ----------
        from_csv : str, optional
            Chemin vers un CSV généré lors d'une exécution précédente.
            Si None -> téléchargement via yfinance.

        Notes
        -----
        Le CSV généré contient les colonnes :
            return_{ticker}  et  loss_{ticker}  pour chaque actif.
        """
        csv_path = os.path.join(self.output_dir, "donnees_multivarie.csv")

        if from_csv:
            print(f"[Données] Chargement depuis {from_csv}...")
            self.data = self._loader.load(from_csv)
        else:
            print(f"[Données] Téléchargement de {self.tickers} depuis {self.start_date}...")
            self.data = self._loader.download(output_csv=csv_path)

        self.losses_dict = self._loader.get_losses_dict(self.data)
        print(f"[Données] {len(self.data)} observations chargées pour {self.k} actifs.")
        return self

    # ================================================================== #
    #  Étape 2 : Sélection des seuils EVT                                #
    # ================================================================== #

    def select_thresholds(
        self,
        manual_thresholds: dict = None,
        show_plots: bool = True
    ) -> "PortfolioRiskModel":
        """
        Sélectionne le seuil EVT u par actif, puis affiche les graphiques
        d'aide au choix (Mean Excess Plot et stabilité des paramètres GPD).

        Parameters
        ----------
        manual_thresholds : dict, optional
            Dictionnaire ticker -> float pour fixer manuellement certains seuils.
            Ex : {"SPY": 0.025, "FEZ": 0.022}
            Pour un actif absent du dict (ou valeur None) -> seuil auto au percentile.
        show_plots : bool
            Si True, affiche Mean Excess Plot et stabilité shape(u) / scale(u).
        """
        self.evts = []

        for ticker in self.tickers:
            losses = self.losses_dict[ticker]

            if self.use_garch:
                # Mode GARCH-EVT : filtrage de la volatilité avant l'EVT
                evt = GarchEVT(losses, ticker)
                evt.fit_garch()
                print(f"[GARCH] {ticker} -> GARCH(1,1) ajusté (sigma_T+1 = {evt.forecast_sigma():.6f})")
            else:
                # Mode standard : EVT directement sur les pertes brutes
                evt = MarginalEVT(losses, ticker)

            # Seuil : manuel s'il est fourni et non-None, sinon percentile automatique
            if manual_thresholds and manual_thresholds.get(ticker) is not None:
                u = manual_thresholds[ticker]
                evt.set_threshold(u)
                src = "manuel"
            else:
                u = evt.select_threshold(self.threshold_percentile)
                src = f"percentile {self.threshold_percentile}%"

            label = "résidus z_t" if self.use_garch else "pertes brutes"
            print(f"[Seuil] {ticker} -> u = {u:.6f}  ({src}, sur {label})")

            self.evts.append(evt)

        if show_plots:
            # En mode GARCH les graphiques sont tracés sur les résidus z_t
            effective_dict = (
                {t: e.residuals for t, e in zip(self.tickers, self.evts)}
                if self.use_garch else self.losses_dict
            )
            viz_seuils.plot_mean_excess(self.tickers, self.evts, effective_dict)
            viz_seuils.plot_gpd_parameters(self.tickers, effective_dict)

        return self

    # ================================================================== #
    #  Étape 3 : Ajustement des marginales EVT                           #
    # ================================================================== #

    def fit_evt_marginals(self) -> "PortfolioRiskModel":
        """
        Sauvegarde les paramètres GPD de chaque actif dans un DataFrame
        et exporte le résumé dans evt_gpd_parameters.csv.

        Les paramètres sont déjà calculés par select_thresholds().
        Cette étape les consolide et les affiche.
        """
        records = [evt.to_dict() for evt in self.evts]
        self.param_gpd = pd.DataFrame(records).set_index("ticker")

        csv_path = os.path.join(self.output_dir, "evt_gpd_parameters.csv")
        self.param_gpd.to_csv(csv_path)

        print("\n[EVT] Paramètres GPD par actif :")
        print(self.param_gpd.to_string())
        return self

    # ================================================================== #
    #  Étape 4 : Transformation PIT                                      #
    # ================================================================== #

    def fit_pit(self, show_plots: bool = True) -> "PortfolioRiskModel":
        """
        Applique la transformation PIT (Probability Integral Transform) :
            U_i = F_EVT(L_i)   \in (0, 1)   pour chaque actif i

        Les U_i doivent suivre des lois uniformes si la marginale EVT est
        bien spécifiée (vérifiable par les histogrammes).

        Sauvegarde les pseudo-observations dans marginales_evt_pit.csv.
        Affiche les histogrammes et le nuage de points de la copule.
        """
        # Calcul des pseudo-observations uniformes via F_evt
        self.U_dict   = {ticker: evt.F_evt() for ticker, evt in zip(self.tickers, self.evts)}
        self.U_matrix = np.column_stack([self.U_dict[t] for t in self.tickers])

        U_df = pd.DataFrame(
            self.U_matrix,
            index=self.data.index,
            columns=[f"U_{sanitize(t)}" for t in self.tickers]
        )
        U_df.to_csv(os.path.join(self.output_dir, "marginales_evt_pit.csv"))

        if show_plots:
            viz_pit.plot_pit_histograms(self.tickers, self.U_dict)
            viz_pit.plot_copula_scatter(self.tickers, self.U_dict)

        return self

    # ================================================================== #
    #  Étape 5 : Ajustement de la copule                                 #
    # ================================================================== #

    def fit_copula(self) -> "PortfolioRiskModel":
        """
        Ajuste la copule choisie aux pseudo-observations U (T * k).

        Copules disponibles :
            "student"  : t-Student multivariée (k quelconque)
                         Recommandée : capture la dépendance en queue (tail dependence)
            "gumbel"   : Gumbel bivariée (k = 2 uniquement)
                         Dépendance plus forte en queue supérieure (co-krachs)
            "gaussian" : Gaussienne (k quelconque, tail dependence = 0)
                         Utile comme référence mais sous-estime les risques extrêmes

        Raises
        ------
        ValueError si copula_type = "gumbel" et k > 2
        """
        print(f"\n[Copule] Ajustement de la copule '{self.copula_type}' sur {self.k} actifs...")

        if self.copula_type == "student":
            self.copula = StudentTCopula()
            self.copula.fit(self.U_matrix)
            print(f"  rho (Kendall -> Pearson) :\n{np.round(self.copula.rho, 4)}")
            print(f"  nu (degrés de liberté) : {self.copula.nu:.2f}")

        elif self.copula_type == "gumbel":
            if self.k > 2:
                raise ValueError(
                    f"Copule de Gumbel : k=2 requis, mais k={self.k}. "
                    f"Utiliser copula_type='student' pour k > 2."
                )
            self.copula = GumbelCopula()
            self.copula.fit(self.U_matrix)
            print(f"  Theta (dépendance) : {self.copula.theta:.4f}")
            print(f"  Log-vraisemblance : {self.copula.loglik:.2f}")

        elif self.copula_type == "gaussian":
            col_names = [f"U_{sanitize(t)}" for t in self.tickers]
            self.copula = GaussianCopula()
            self.copula.fit(self.U_matrix, col_names=col_names)
            print(f"  Corrélation estimée :\n{np.round(self.copula.correlation, 4)}")

        else:
            raise ValueError(
                f"copula_type='{self.copula_type}' invalide. "
                f"Choisir parmi : 'student', 'gumbel', 'gaussian'."
            )

        return self

    # ================================================================== #
    #  Étape 6 : Simulation et calcul de VaR / ES                        #
    # ================================================================== #

    def simulate(self) -> "PortfolioRiskModel":
        """
        Simule n_sim scénarios de pertes conjointes via la copule et les
        marginales EVT, puis calcule VaR et ES du portefeuille.

        La distribution des pertes simulées peut être visualisée avec :
            viz_violations.plot_portfolio_losses_distribution(...)
        """
        mode = "GARCH-EVT" if self.use_garch else "EVT standard"
        print(f"\n[Simulation] {self.n_sim:,} scénarios – copule '{self.copula_type}' – mode {mode}...")

        if self.use_garch:
            # Mode GARCH-EVT :
            #   - losses_dict de simulation = résidus standardisés z_t
            #   - sigma_forecasts = sigma_{i,T+1} prévu par le GARCH de chaque actif
            #   - Pertes simulées : L_i = sigma_{i,T+1} x GPD_inverse(U_i, z_i)
            sim_losses_dict  = {t: e.residuals for t, e in zip(self.tickers, self.evts)}
            sigma_forecasts  = [e.forecast_sigma() for e in self.evts]
            print(f"  sigma_{{T+1}} par actif : { {t: f'{s:.6f}' for t, s in zip(self.tickers, sigma_forecasts)} }")
            self.joint_losses_sim = Simulateur.simulate_joint_losses(
                self.copula, self.evts, sim_losses_dict, self.tickers, self.n_sim,
                sigma_forecasts=sigma_forecasts
            )
        else:
            # Mode standard : pertes brutes L_t directement
            self.joint_losses_sim = Simulateur.simulate_joint_losses(
                self.copula, self.evts, self.losses_dict, self.tickers, self.n_sim
            )

        # Perte du portefeuille : L = w^T L_i (produit scalaire pondéré)
        self.portfolio_losses_sim = Simulateur.portfolio_losses(
            self.joint_losses_sim, self.weights
        )

        # VaR et ES empiriques
        self.VaR, self.ES = Simulateur.var_es(self.portfolio_losses_sim, self.alpha)

        # Probabilité de co-krach (toutes les pertes dépassent leur VaR individuelle)
        prob_cokrach = Simulateur.joint_extreme_prob(self.joint_losses_sim, self.alpha)

        print(f"\n[Risque] VaR {self.alpha*100:.0f}%  : {self.VaR:.6f}")
        print(f"[Risque] ES  {self.alpha*100:.0f}%  : {self.ES:.6f}")
        print(f"[Risque] Probabilité de co-krach : {prob_cokrach:.4%}")

        # Visualisation de la distribution des pertes simulées
        viz_violations.plot_portfolio_losses_distribution(
            self.portfolio_losses_sim, self.VaR, self.ES, self.alpha
        )
        return self

    # ================================================================== #
    #  Étape 7 : Backtesting                                             #
    # ================================================================== #

    def backtest(
        self,
        window_base: int = 250,
        window_min:  int = 150,
        window_max:  int = 350,
        show_plots:  bool = True
    ) -> "PortfolioRiskModel":
        """
        Backtesting de la VaR du portefeuille.

        Deux niveaux :

        A) Backtesting statique :
           La VaR du modèle complet (calculée sur tout l'échantillon) est
           comparée à toutes les pertes historiques pondérées.
           -> Test de Kupiec + Christoffersen sur cette VaR fixe.

        B) Backtesting dynamique (rolling window) :
           Fenêtre glissante adaptative sur les pertes historiques du portefeuille.
           Calcul de trois VaR de référence :
               - VaR historique (quantile empirique)
               - VaR normale (hypothèse gaussienne)
               - VaR EVT univariée (GPD sur les pertes du portefeuille)
           -> Tests de Kupiec + Christoffersen pour chaque méthode.

        Parameters
        ----------
        window_base / window_min / window_max : int
            Paramètres de la fenêtre glissante adaptative (en jours ouvrés).
        show_plots : bool
            Affiche les graphiques de violations et le cumul des violations.
        """
        print("\n[Backtesting] Démarrage...")

        # Reconstruction de la série de pertes historiques du portefeuille
        loss_cols = [self._loader.get_loss_col(t) for t in self.tickers]
        L_matrix  = np.column_stack([self.data[c].values for c in loss_cols])
        L_hist    = L_matrix @ self.weights   # pertes pondérées historiques

        # ----------------------------------------------------------------- #
        #  GARCH-EVT portefeuille (si use_garch=True)                       #
        #  On ajuste un GARCH(1,1) sur L_hist et on calcule la VaR_t        #
        #  conditionnelle pour chaque date historique.                      #
        #  VaR_t = sigma_t x VaR_GPD(z) : monte en crise, baisse en période #
        #  calme -> corrige le clustering des violations (Christoffersen).  #
        # ----------------------------------------------------------------- #
        garch_pf      = None
        var_garch_all = None
        if self.use_garch:
            print("[GARCH-EVT Portefeuille] Ajustement GARCH(1,1) sur les pertes du portefeuille...")
            garch_pf = GarchEVT(L_hist, "Portfolio")
            garch_pf.fit_garch()
            garch_pf.select_threshold(self.threshold_percentile)
            # VaR conditionnelle à chaque date : alpha_t * VaR_GPD(z, alpha)
            var_garch_all = garch_pf.var_conditionnel(self.alpha)
            print(f"  Seuil sur résidus : u = {garch_pf.u:.4f}  "
                  f"(shape = {garch_pf.xi:.4f}, scale = {garch_pf.beta:.4f})")

        # --------------------------------------------------------------- #
        #  A) Backtesting statique : VaR du modèle vs pertes historiques  #
        # --------------------------------------------------------------- #
        print("\n--- A) Backtesting statique (VaR modèle complet) ---")

        # Test de Kupiec sur la VaR scalaire du modèle Monte-Carlo
        res_k = KupiecTest.kupiec(L_hist, self.VaR, self.alpha)
        res_c = KupiecTest.christoffersen(L_hist, self.VaR, self.alpha)

        print(f"  Kupiec   -> N={res_k['N']}/{res_k['T']}, "
              f"LR={res_k['LR']:.3f}, p={res_k['p_value']:.4f}  "
              f"({'REJETÉ' if res_k['rejete_H0'] else 'ACCEPTÉ'})")
        print(f"  {res_k['interpretation']}")
        print(f"  Christoffersen -> LR={res_c['LR_ind']:.3f}, p={res_c['p_value']:.4f}  "
              f"({'REJETÉ' if res_c['rejete_H0'] else 'ACCEPTÉ'})")
        print(f"  {res_c['interpretation']}")

        if self.use_garch and var_garch_all is not None:
            # Test statique GARCH-EVT : série VaR_t vs toute la série L_hist
            res_gk = KupiecTest.kupiec(L_hist, var_garch_all, self.alpha)
            res_gc = KupiecTest.christoffersen(L_hist, var_garch_all, self.alpha)
            print(f"\n  [GARCH-EVT statique] Kupiec -> N={res_gk['N']}/{res_gk['T']}, "
                  f"LR={res_gk['LR']:.3f}, p={res_gk['p_value']:.4f}  "
                  f"({'REJETÉ' if res_gk['rejete_H0'] else 'ACCEPTÉ'})")
            print(f"  {res_gk['interpretation']}")
            print(f"  Christoffersen -> LR={res_gc['LR_ind']:.3f}, p={res_gc['p_value']:.4f}  "
                  f"({'REJETÉ' if res_gc['rejete_H0'] else 'ACCEPTÉ'})")
            print(f"  {res_gc['interpretation']}")

        # ------------------------------------------------------------- #
        #  B) Backtesting dynamique (rolling)                           #
        # ------------------------------------------------------------- #
        print("\n--- B) Backtesting dynamique (rolling window) ---")

        roller = RollingBacktest(
            portfolio_losses=L_hist,
            dates=self.data.index,
            alpha=self.alpha,
            window_base=window_base,
            window_min=window_min,
            window_max=window_max,
            threshold_percentile=self.threshold_percentile,
        )
        self.backtesting_results = roller.run()

        if self.use_garch and var_garch_all is not None:
            # Alignement de la série VaR_t GARCH-EVT sur les dates du rolling backtest
            # (le rolling commence à window_max, donc on sélectionne les dates correspondantes)
            var_garch_series = pd.Series(var_garch_all, index=self.data.index)
            self.backtesting_results["VaR_GARCH_EVT"] = (
                var_garch_series.loc[self.backtesting_results.index].values
            )

        csv_path = os.path.join(self.output_dir, "backtesting_results.csv")
        self.backtesting_results.to_csv(csv_path)
        print(f"  Résultats sauvegardés dans : {csv_path}")

        # Tests de Kupiec et Christoffersen pour chaque méthode rolling
        rolling_methods = [
            ("VaR_historique", "Historique"),
            ("VaR_normale",    "Normale"),
            ("VaR_EVT_1D",     "EVT Univariée"),
        ]
        if self.use_garch:
            rolling_methods.append(("VaR_GARCH_EVT", "GARCH-EVT Portefeuille"))

        for col, label in rolling_methods:
            if col not in self.backtesting_results.columns:
                continue
            L_sub    = self.backtesting_results["portfolio_loss"].values
            var_sub  = self.backtesting_results[col].values
            rk = KupiecTest.kupiec(L_sub, var_sub, self.alpha)
            rc = KupiecTest.christoffersen(L_sub, var_sub, self.alpha)
            print(f"\n  [{label}]  N={rk['N']}/{rk['T']}  "
                  f"LR={rk['LR']:.3f}  p={rk['p_value']:.4f}  "
                  f"({'REJETÉ' if rk['rejete_H0'] else 'ACCEPTÉ'})")
            print(f"    {rk['interpretation']}")
            print(f"    Christoffersen : p={rc['p_value']:.4f}  "
                  f"({'REJETÉ' if rc['rejete_H0'] else 'ACCEPTÉ'})")

        # ------------------------------------------------------------- #
        #  Graphiques                                                   #
        # ------------------------------------------------------------- #
        if show_plots:
            viz_violations.plot_var_vs_losses(
                self.backtesting_results,
                alpha=self.alpha,
                var_statique=self.VaR,
                label_statique=f"VaR Modèle ({self.copula_type})"
            )
            viz_violations.plot_cumul_violations(
                self.backtesting_results,
                alpha=self.alpha,
                var_statique=self.VaR,
                label_statique=f"Modèle ({self.copula_type})"
            )

        return self

    # ================================================================== #
    #  Pipeline complet                                                  #
    # ================================================================== #

    def run(
        self,
        from_csv: str = None,
        manual_thresholds: dict = None,
        show_plots: bool = True,
        run_backtest: bool = True,
    ) -> "PortfolioRiskModel":
        """
        Lance le pipeline complet :
            load_data -> select_thresholds -> fit_evt_marginals ->
            fit_pit -> fit_copula -> simulate -> backtest

        Parameters
        ----------
        from_csv : str, optional
            Charge les données depuis ce CSV au lieu de télécharger.
        manual_thresholds : dict, optional
            Seuils manuels par ticker. Ex : {"SPY": 0.025, "FEZ": None}
            None = utiliser le percentile automatique.
        show_plots : bool
            Active/désactive tous les graphiques.
        run_backtest : bool
            Si False, saute l'étape de backtesting (plus rapide).
        """
        self.load_data(from_csv=from_csv)
        self.select_thresholds(manual_thresholds=manual_thresholds, show_plots=show_plots)
        self.fit_evt_marginals()
        self.fit_pit(show_plots=show_plots)
        self.fit_copula()
        self.simulate()
        if run_backtest:
            self.backtest(show_plots=show_plots)
        return self

    # ================================================================== #
    #  Résumé des résultats                                              #
    # ================================================================== #

    def summary(self) -> pd.DataFrame:
        """
        Retourne un DataFrame résumant les résultats principaux du modèle.
        """
        return pd.DataFrame({
            "Actifs":     [", ".join(self.tickers)],
            "Poids":      [", ".join(f"{w:.2%}" for w in self.weights)],
            "Copule":     [self.copula_type],
            "Mode EVT":   ["GARCH-EVT" if self.use_garch else "Standard"],
            f"VaR {self.alpha*100:.0f}%": [f"{self.VaR:.6f}" if self.VaR is not None else "N/A"],
            f"ES {self.alpha*100:.0f}%":  [f"{self.ES:.6f}"  if self.ES  is not None else "N/A"],
        })


# ====================================================================== #
#  Point d'entrée                                                        #
# ====================================================================== #

if __name__ == "__main__":

    # ------------------------------------------------------------------ #
    #  Exemple 1 : Reproduction du cas bivarié original (SPY + FEZ)      #
    # ------------------------------------------------------------------ #
    model = PortfolioRiskModel(
        tickers=["SPY", "FEZ"],
        start_date="2005-01-01",
        weights_file=None,           # poids égaux 0.5 / 0.5
        copula_type="student",       # ou "gumbel" (k=2) ou "gaussian"
        n_sim=200_000,
        alpha=0.99,
        threshold_percentile=95.0,   # percentile pour la sélection auto du seuil
        output_dir=".",              # les CSV sont sauvegardés ici
        use_garch=True,              # True : GARCH-EVT (corrige clustering violations)
                                     # False : EVT standard
    )

    model.run(
        from_csv=None,               # None = télécharger via yfinance
                                     # Passer le chemin CSV pour éviter de retélécharger :
                                     # from_csv="donnees_multivarie.csv"
        manual_thresholds={          # Seuils manuels (optionnel)
            "SPY": None,             # None = percentile automatique
            "FEZ": None,
        },
        show_plots=True,
        run_backtest=True,
    )

    print("\n" + "="*60)
    print("RÉSUMÉ DES RÉSULTATS")
    print("="*60)
    print(model.summary().to_string(index=False))

    # ------------------------------------------------------------------ #
    #  Exemple 2 : Portefeuille de 3 actifs (décommenter pour tester)    #
    # ------------------------------------------------------------------ #
    # model3 = PortfolioRiskModel(
    #     tickers=["SPY", "FEZ", "QQQ"],
    #     start_date="2005-01-01",
    #     copula_type="student",   # Gumbel ne supporte pas k > 2
    #     n_sim=200_000,
    #     alpha=0.99,
    #     use_garch=True,
    # )
    # model3.run(show_plots=True, run_backtest=True)
    # print("\n" + "="*60)
    # print("RÉSUMÉ DES RÉSULTATS")
    # print("="*60)
    # print(model3.summary().to_string(index=False))

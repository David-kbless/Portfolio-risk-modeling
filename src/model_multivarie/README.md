# Modèle de Risque de Portefeuille Multivarié — EVT + Copules

Modélisation du risque extrême d'un portefeuille de **k actifs** via la théorie des valeurs extrêmes (EVT) et les copules. Calcul de la VaR et de l'ES avec backtesting (tests de Kupiec et Christoffersen).

---

## Structure du projet

```
model_multivarie/
│
├── main.py                        ← Point d'entrée : classe PortfolioRiskModel
│
├── data/
│   └── loader.py                  ← DataLoader : téléchargement et traitement (k actifs)
│
├── marginales/
│   └── evt.py                     ← MarginalEVT : seuil + GPD + PIT par actif
│
├── copules/
│   ├── student_t.py               ← StudentTCopula : t-Student (k quelconque)
│   ├── gumbel.py                  ← GumbelCopula : Gumbel bivariée (k=2 uniquement)
│   └── gaussienne.py              ← GaussianCopula : Gaussienne (référence)
│
├── simulation/
│   └── simulateur.py              ← Simulateur : Monte-Carlo + VaR/ES
│
├── backtesting/
│   ├── kupiec.py                  ← KupiecTest + Christoffersen
│   └── rolling.py                 ← RollingBacktest : fenêtre glissante adaptative
│
├── visualisation/
│   ├── seuils.py                  ← Mean Excess Plot, stabilité GPD
│   ├── pit.py                     ← Histogrammes PIT, nuage de points copule
│   └── violations.py              ← Pertes vs VaR, cumul violations, distribution
│
│   (fichiers originaux conservés)
├── datas.py                       ← Code original bivarié (SPY/FEZ)
├── rep_marginale.py               ← F_evt original (bivarié)
├── copule_t.py                    ← Copule t originale (bivarié)
├── simulation_gumbel.py           ← Simulation Gumbel originale (bivarié)
└── simulation_student.py          ← Simulation Student originale (bivarié)
```

---

## Prérequis — Activation de l'environnement

**Toujours exécuter depuis la racine du projet :**

```bash
cd /home/kossiy/risk_extreme_project
source venv/bin/activate
```

Librairies requises (déjà installées dans le venv) :
- `numpy`, `pandas`, `scipy`, `matplotlib`
- `yfinance` — téléchargement des données
- `scikit-learn` — régression linéaire (Mean Excess Plot)
- `statsmodels` — ECDF
- `copulas` — copules Gumbel et Gaussienne

---

## Compilation et exécution

### Lancement rapide (cas bivarié SPY + FEZ, reproduit datas.py)

```bash
cd /home/kossiy/risk_extreme_project
source venv/bin/activate
python -m implementations.model_multivarie.main
```

Ou directement :

```bash
python implementations/model_multivarie/main.py
```

---

## Utilisation dans du code Python

### Cas 1 : Portefeuille de 2 actifs (bivarié — cas original)

```python
from implementations.model_multivarie.main import PortfolioRiskModel

model = PortfolioRiskModel(
    tickers=["SPY", "FEZ"],
    start_date="2005-01-01",
    weights_file=None,        # poids égaux : 0.5 / 0.5
    copula_type="student",    # ou "gumbel" ou "gaussian"
    n_sim=200_000,
    alpha=0.99,
    threshold_percentile=95.0,
    output_dir=".",
)

model.run(
    from_csv=None,            # None = télécharger ; sinon : "donnees_multivarie.csv"
    manual_thresholds={       # seuils manuels (optionnel)
        "SPY": 0.0247,        # None = auto (percentile)
        "FEZ": None,
    },
    show_plots=True,
    run_backtest=True,
)

print(model.summary())
```

### Cas 2 : Portefeuille de 3 actifs (généralisation à k=3)

```python
model3 = PortfolioRiskModel(
    tickers=["SPY", "FEZ", "QQQ"],
    start_date="2005-01-01",
    copula_type="student",    # Gumbel ne supporte pas k > 2
    n_sim=200_000,
    alpha=0.99,
)
model3.run(show_plots=True, run_backtest=True)
print(model3.summary())
```

### Cas 3 : Poids depuis un fichier

```python
# Fichier CSV avec une seule colonne de k valeurs
# poids.csv :
#   SPY, 0.4
#   FEZ, 0.35
#   QQQ, 0.25

model = PortfolioRiskModel(
    tickers=["SPY", "FEZ", "QQQ"],
    weights_file="poids.csv",   # ou un fichier .pkl contenant np.array([0.4, 0.35, 0.25])
    copula_type="student",
)
model.run()
```

### Étapes individuelles (pipeline manuel)

```python
model = PortfolioRiskModel(tickers=["SPY", "FEZ"], start_date="2005-01-01")

model.load_data()                    # Téléchargement
model.select_thresholds()            # Seuils EVT + graphiques
model.fit_evt_marginals()            # Paramètres GPD → CSV
model.fit_pit()                      # PIT → marginales_evt_pit.csv
model.fit_copula()                   # Ajustement copule
model.simulate()                     # Monte-Carlo + VaR/ES
model.backtest()                     # Kupiec + Christoffersen + graphiques
```

---

## Pipeline détaillé

| Étape | Méthode | Description |
|-------|---------|-------------|
| 1 | `load_data()` | Télécharge les prix, calcule rendements log et pertes |
| 2 | `select_thresholds()` | Sélectionne le seuil u par actif (Mean Excess Plot) |
| 3 | `fit_evt_marginals()` | Ajuste la GPD sur les excès, sauvegarde les paramètres |
| 4 | `fit_pit()` | Transforme les pertes en U ∈ (0,1) via F_EVT |
| 5 | `fit_copula()` | Ajuste la copule choisie aux pseudo-observations |
| 6 | `simulate()` | Simule n_sim scénarios, calcule VaR et ES |
| 7 | `backtest()` | Tests de Kupiec et Christoffersen, graphiques |

---

## Choix de la copule

| Copule | `copula_type` | Dimension | Tail dependence | Usage recommandé |
|--------|--------------|-----------|-----------------|-----------------|
| t-Student | `"student"` | k ≥ 2 | Oui | **Défaut** — risques extrêmes |
| Gumbel | `"gumbel"` | k = 2 | Oui (queue sup.) | Co-krachs 2 actifs |
| Gaussienne | `"gaussian"` | k ≥ 2 | Non | Référence uniquement |

**Règle** : pour k > 2 actifs, seule `"student"` ou `"gaussian"` sont disponibles.

---

## Backtesting

Le backtesting se déroule en deux parties :

### A — Backtesting statique
La VaR du modèle complet (estimée sur tout l'échantillon) est comparée à l'ensemble des pertes historiques du portefeuille.
- **Test de Kupiec** : le taux de violation empirique est-il conforme à 1 - α ?
- **Test de Christoffersen** : les violations sont-elles indépendantes (pas de clustering) ?

### B — Backtesting dynamique (rolling window adaptative)
Fenêtre glissante dont la taille s'adapte à la volatilité courante :

```
window_t = window_base × (σ_base / σ_t)
       clampé dans [window_min, window_max]
```

Trois méthodes de référence sont calculées par rolling :
- **VaR Historique** : quantile empirique de la fenêtre
- **VaR Normale** : VaR gaussienne (µ + σ · z_α)
- **VaR EVT Univariée** : GPD ajustée sur les pertes du portefeuille (1D)

Les tests de Kupiec et Christoffersen sont appliqués aux trois méthodes.

---

## Sorties générées

| Fichier | Description |
|---------|-------------|
| `donnees_multivarie.csv` | Rendements et pertes journalières par actif |
| `evt_gpd_parameters.csv` | Paramètres GPD par actif (u, ξ, β, N, Nu, Pu) |
| `marginales_evt_pit.csv` | Pseudo-observations uniformes U ∈ (0,1) |
| `backtesting_results.csv` | Pertes et séries de VaR rolling |

---

## Graphiques produits

| Graphique | Module | Description |
|-----------|--------|-------------|
| Mean Excess Plot | `visualisation/seuils.py` | Aide au choix du seuil u |
| Stabilité ξ(u) / β(u) | `visualisation/seuils.py` | Robustesse des paramètres GPD |
| Histogrammes PIT | `visualisation/pit.py` | Validation des marginales |
| Nuage de points copule | `visualisation/pit.py` | Structure de dépendance |
| Distribution pertes simulées | `visualisation/violations.py` | VaR et ES sur la queue |
| Pertes vs VaR rolling | `visualisation/violations.py` | Violations dans le temps |
| Cumul des violations | `visualisation/violations.py` | Calibration du modèle |

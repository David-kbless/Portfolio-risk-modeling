# import sys
# import os

# # 1. On calcule le chemin vers la racine 'risk_extreme_project'
# # On remonte de deux niveaux depuis 'implementations/model_multivarie/'
# current_dir = os.path.dirname(os.path.abspath(__file__))
# project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))

# # 2. On ajoute ce chemin au système pour que Python "voit" tout le projet
# if project_root not in sys.path:
#     sys.path.insert(0, project_root)

# from implementations.model_univarie import exces_moyen as em
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from scipy.stats import genpareto
from copulas.multivariate import GaussianMultivariate
from copulas.bivariate import Clayton
from copulas.bivariate import Gumbel
import implementations.modeleBivarie.copule_t as tm
from scipy.optimize import minimize


import simulation_gumbel as sg
import rep_marginale as rm
import simulation_student as st

#==================================================================================
# Traitement des données multivariées
#==================================================================================
def traitement_donnees_multivarie(tickers, start_date):

    data_prices = yf.download(
        tickers,
        start=start_date,
        auto_adjust=True 
    )["Close"]

    data_prices.dropna(inplace=True)
    # print(data_prices.head())
    # print(data_prices.tail())

    # Calcul des rendements logarithmiques
    data_returns = np.log(data_prices / data_prices.shift(1))
    data_returns.dropna(inplace=True)

    # definir les pertes comme l'opposé des rendements
    losses = - data_returns
    losses.describe()
    losses.corr()

    data = pd.DataFrame(
        index=data_returns.index,
        data={
            "returns_sp500": data_returns["SPY"],
            "returns_eurostoxx": data_returns["FEZ"],
            "loss_sp500": losses["SPY"],
            "loss_eurostoxx": losses["FEZ"]
        }
    )

    data.to_csv("donnees_multivarie.csv")
#--------------------------------------------------------------------------------
# tickers = ["SPY", "FEZ"]
# traitement_donnees_multivarie(tickers, "2005-01-01")
#--------------------------------------------------------------------------------

data = pd.read_csv("donnees_multivarie.csv", index_col=0, parse_dates=True)
losses_sp = data["loss_sp500"].values
losses_eu = data["loss_eurostoxx"].values

#==================================================================================
# Definition des fonctions de visualisation
#==================================================================================


def show_mean_excess_plot(thresholds_sp, me_sp500, thresholds_eu, me_eurostoxx):
    # Zones mask pour les droites de tendance
    zone_mask_sp = (thresholds_sp >= np.percentile(losses_sp, 95)) & \
                   (thresholds_sp <= np.percentile(losses_sp, 98))
    zone_mask_eu = (thresholds_eu >= 0.018) & \
                   (thresholds_eu <= np.percentile(losses_eu, 94.5))
    
    zone_thresholds_sp = thresholds_sp[zone_mask_sp].reshape(-1, 1)
    zone_thresholds_eu = thresholds_eu[zone_mask_eu].reshape(-1, 1)

    zone_me_sp = me_sp500[zone_mask_sp]
    zone_me_eu = me_eurostoxx[zone_mask_eu]

    # Régressions linéaires
    reg_sp = LinearRegression().fit(zone_thresholds_sp, zone_me_sp)
    reg_eu = LinearRegression().fit(zone_thresholds_eu, zone_me_eu)

    line_sp = reg_sp.predict(zone_thresholds_sp)
    line_eu = reg_eu.predict(zone_thresholds_eu)

    # Tracé des deux mean excess plots côte à côte
    plt.figure(figsize=(10,4))

    plt.subplot(1,2,1)
    plt.plot(thresholds_sp, me_sp500, marker='o', label='Mean Excess S&P 500')
    plt.plot(zone_thresholds_sp, line_sp, color='red', label='Trend Line S&P 500')
    plt.xlabel("Seuil u")
    plt.ylabel("Mean Excess")
    plt.title("Mean Excess Plot - S&P 500")
    plt.grid(True)

    plt.subplot(1,2,2)
    plt.plot(thresholds_eu, me_eurostoxx, marker='s', label='Mean Excess Euro Stoxx 50')
    plt.plot(zone_thresholds_eu, line_eu, color='red', label='Trend Line Euro Stoxx 50')
    plt.xlabel("Seuil u")
    plt.ylabel("Mean Excess")
    plt.title("Mean Excess Plot - Euro Stoxx 50")
    plt.grid(True)

    plt.tight_layout() # Ajuste les sous-graphes pour éviter le chevauchement
    plt.show()

def show_gpd_parameters(u_values_sp, xi_list_sp, beta_list_sp, u_values_eu, xi_list_eu, beta_list_eu):
    # Tracé des paramètres
    plt.figure(figsize=(10, 10))

    plt.subplot(2,2,1)
    plt.plot(u_values_sp, xi_list_sp, marker='o', label="S&P 500")
    plt.legend()
    plt.xlabel('Seuil u')
    plt.ylabel('ξ (shape)')
    plt.title('Parameter Stability Plot : ξ(u) - S&P 500')
    plt.grid(True)

    plt.subplot(2,2,2)
    plt.plot(u_values_eu, xi_list_eu, marker='s', label="Euro Stoxx 50")
    plt.xlabel('Seuil u')
    plt.ylabel('ξ (shape)')
    plt.title('Parameter Stability Plot : ξ(u) - Euro Stoxx 50')
    plt.grid(True)

    plt.subplot(2,2,3)
    plt.plot(u_values_sp, beta_list_sp, marker='o', color='orange', label="S&P 500")
    plt.xlabel('Seuil u')
    plt.ylabel('β (scale)')
    plt.title('Parameter Stability Plot : β(u) - S&P 500')
    plt.grid(True)

    plt.subplot(2,2,4)
    plt.plot(u_values_eu, beta_list_eu, marker='s', color='orange', label="Euro Stoxx 50")
    plt.xlabel('Seuil u')
    plt.ylabel('β (scale)')
    plt.title('Parameter Stability Plot : β(u) - Euro Stoxx 50')
    plt.grid(True)

    plt.tight_layout() # Ajuste les sous-graphes pour éviter le chevauchement
    plt.show()

def show_PIT_histograms(U_sp500, U_euro):
    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.hist(U_sp500, bins=30, edgecolor='black')
    plt.title('Histogramme des U - S&P 500')
    plt.xlabel('U values')
    plt.ylabel('Frequency')
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.hist(U_euro, bins=30, edgecolor='black')
    plt.title('Histogramme des U - Euro Stoxx 50')
    plt.xlabel('U values')
    plt.ylabel('Frequency')
    plt.grid(True)

    plt.tight_layout()
    plt.show()

def show_copula_scatter():
    plt.figure(figsize=(6,6))
    plt.scatter(U_data["U_sp500"], U_data["U_eurostoxx"], alpha=0.5)
    plt.title('Pseudo-observations (EVT marginals)')
    plt.xlabel('U - S&P 500')
    plt.ylabel('U - Euro Stoxx 50')
    plt.grid(True)
    plt.show()



#==================================================================================
# Calcul des mean excess et visualisation
#==================================================================================
def mean_excess(pertes, seuils):
    mean_excess_values = []
    
    for u in seuils:
        # Calcul de l'excès moyen des pertes au-dessus du seuil u
        excesses = pertes[pertes > u] - u
        mean_excess_values.append(excesses.mean())
        
    return np.array(mean_excess_values)

# Les seuils
thresholds_sp = np.linspace(
        np.percentile(losses_sp, 90),
        np.percentile(losses_sp, 99),
        50
    )

thresholds_eu = np.linspace(
        np.percentile(losses_eu, 90),
        np.percentile(losses_eu, 99),
        50
    )

# mean excess pour le S&P 500
me_sp500 = mean_excess(losses_sp, thresholds_sp)

# mean excess pour le eurostoxx
me_eurostoxx = mean_excess(losses_eu, thresholds_eu)

# Affichage des résultats
#----------------------------------------------------------------------------------
#show_mean_excess_plot(thresholds_sp, me_sp500, thresholds_eu, me_eurostoxx)
#----------------------------------------------------------------------------------

#==================================================================================
# Visualisation des paramètres GPD
#==================================================================================
u_values_sp = np.linspace(
        np.percentile(losses_sp, 90),
        np.percentile(losses_sp, 99),
        20
    )
u_values_eu = np.linspace(
        np.percentile(losses_sp, 90),
        np.percentile(losses_eu, 99),
        20
    )

def calculate_gpd_parameters(u_values_sp, u_values_eu):
    # les seuils à tester d'après le tracé de l'excès moyen
    xi_list_sp = []
    beta_list_sp = []
    xi_list_eu = []
    beta_list_eu = []

    # Maintenant on ne va travailler que sur les pertes au dessus de chaque seuil u
    for u in u_values_sp:
        excesses = losses_sp[losses_sp > u] - u
        # La loi GPD est ajustée aux excès via le maximum de vraisemblance et renvoie \xi, \beta et loc qui est le décalage (ici 0 puisque on a déjà soustrait u)
        xi, loc, beta = genpareto.fit(excesses, floc=0)
        xi_list_sp.append(xi)
        beta_list_sp.append(beta)

    for u in u_values_eu:
        excesses = losses_eu[losses_eu > u] - u
        # La loi GPD est ajustée aux excès via le maximum de vraisemblance et renvoie \xi, \beta et loc qui est le décalage (ici 0 puisque on a déjà soustrait u)
        xi, loc, beta = genpareto.fit(excesses, floc=0)
        xi_list_eu.append(xi)
        beta_list_eu.append(beta)
    
    return xi_list_sp, beta_list_sp, xi_list_eu, beta_list_eu

xi_list_sp, beta_list_sp, xi_list_eu, beta_list_eu = calculate_gpd_parameters(u_values_sp, u_values_eu)

#--------------------------------------------------------------------------------
#show_gpd_parameters(u_values_sp, xi_list_sp, beta_list_sp, u_values_eu, xi_list_eu, beta_list_eu)
#--------------------------------------------------------------------------------

# Finalement on garde seuil_sp au percentile 95 pour le S&P 500 et seuil_eu au percentile 96.8 pour l'Euro Stoxx 50
u_sp = np.percentile(losses_sp, 97.1)
u_eu = np.percentile(losses_eu, 93.0)
#print(f"sp : {u_sp} \n eu : {u_eu}")

def final_gpd_parameters(u_sp, u_eu):
    excesses_sp = losses_sp[losses_sp > u_sp] - u_sp
    xi_sp, loc_sp, beta_sp = genpareto.fit(excesses_sp, floc=0)

    excesses_eu = losses_eu[losses_eu > u_eu] - u_eu
    xi_eu, loc_eu, beta_eu = genpareto.fit(excesses_eu, floc=0)

    evt_params = pd.DataFrame(
        index=["sp500", "Euro"],
        data={
            "u": [u_sp, u_eu],
            "xi": [xi_sp, xi_eu],
            "beta": [beta_sp, beta_eu],
            "N": [len(losses_sp), len(losses_eu)],
            "Nu": [len(excesses_sp), len(excesses_eu)],
            "Pu" : [len(excesses_sp)/len(losses_sp), len(excesses_eu)/len(losses_eu)]
        }
    )
    evt_params.to_csv("evt_gpd_parameters.csv")
#----------------------------------------------------------------------------------
# final_gpd_parameters(u_sp, u_eu)
#----------------------------------------------------------------------------------

#==================================================================================
# Construction des marginales EVT et transformation PIT
#==================================================================================
param_gpd = pd.read_csv("evt_gpd_parameters.csv", index_col=0)

def construct_marginales_evt(param_gpd):
    U_sp500 = rm.F_evt(losses_sp,
                    u_sp,
                    param_gpd.loc["sp500", "xi"],
                    param_gpd.loc["sp500", "beta"],
                    param_gpd.loc["sp500", "N"],
                    param_gpd.loc["sp500", "Nu"]
    )

    U_euro = rm.F_evt(losses_eu,
                    u_eu,
                    param_gpd.loc["Euro", "xi"],
                    param_gpd.loc["Euro", "beta"],
                    param_gpd.loc["Euro", "N"],
                    param_gpd.loc["Euro", "Nu"]
    )

    U_data = pd.DataFrame(
        index=data.index,
        data={
            "U_sp500": U_sp500,
            "U_eurostoxx": U_euro
        }
    )

    U_data.to_csv("marginales_evt_pit.csv")

#----------------------------------------------------------------------------------
# construct_marginales_evt(param_gpd)
#----------------------------------------------------------------------------------

U_data = pd.read_csv("marginales_evt_pit.csv", index_col=0, parse_dates=True)
#print(U_data.describe())

#----------------------------------------------------------------------------------
# show_PIT_histograms(U_data["U_sp500"].values, U_data["U_eurostoxx"].values)
#----------------------------------------------------------------------------------

#==================================================================================
# Les copules
#==================================================================================
# copule gaussienne
def fit_copula_gaussian(U_data):
    cop_gauss = GaussianMultivariate()
    cop_gauss.fit(U_data)
    print(f"Corrélation estimée : {cop_gauss.correlation}")

#----------------------------------------------------------------------------------
# fit_copula_gaussian(U_data)
#----------------------------------------------------------------------------------

# copule t-Student
def fit_copula_student(U_data):
    # 1. rho via Kendall
    rho = tm.estimate_rho_kendall(U_data)

    # 2. nu via likelihood profil
    nu = tm.estimate_nu(U_data, rho)

    print("=== t-copula fit ===")
    print("rho :", rho)
    print("nu  :", nu)

    return rho, nu

#----------------------------------------------------------------------------------
# fit_copula_student(U_data)
# show_copula_scatter()
#----------------------------------------------------------------------------------

# Copules Archimédiennes (Clayton et Gumbel)

def clayton_copula_fit(U):
    clayton = Clayton()
    clayton.fit(U)

    theta_clayton = clayton.theta
    loglik_clayton = np.sum(clayton.log_probability_density(U))

    print("\nCopule Clayton")
    print("Theta estimé :", theta_clayton)
    return loglik_clayton, theta_clayton

def gumbel_copula_fit(U):
    gumbel = Gumbel()
    gumbel.fit(U)

    theta_gumbel = gumbel.theta
    loglik_gumbel = np.sum(gumbel.log_probability_density(U))

    print("\nCopule Gumbel")
    print("Theta estimé :", theta_gumbel)
    return loglik_gumbel, theta_gumbel

theta = 0
def fit_copules_archimediennes(U_data):
    global theta
    U = U_data[["U_sp500", "U_eurostoxx"]].values
    # Copule Clayton
    loglik_clayton, theta_clayton = clayton_copula_fit(U)
    print("--------------------------")
    # Copule Gumbel
    loglik_gumbel, theta_gumbel = gumbel_copula_fit(U)
    print("--------------------------")
    print("\nComparaison des copules")
    print(f"Log-vraisemblance Clayton : {loglik_clayton:.2f}")
    print(f"Log-vraisemblance Gumbel  : {loglik_gumbel:.2f}")

    if loglik_clayton > loglik_gumbel:
        theta = theta_clayton
        # Clayton = Queue gauche des "pertes" -> Dépendance quand les pertes sont faibles (marché normal ou haussier)
        print("Il y a une dépendance plus forte lors des gains ou des pertes normales (queue gauche des pertes)")
    elif loglik_clayton < loglik_gumbel:
        theta = theta_gumbel
        # Gumbel = Queue droite des "pertes" -> Dépendance quand les pertes sont très élevées (krach simultané)
        print("Il y a une dépendance plus forte lors des pertes extrêmes / Krachs (queue droite des pertes)")
    else:
        theta = theta_gumbel
        print("Il y a une dépendance symétrique")
#----------------------------------------------------------------------------------
# fit_copules_archimediennes(U_data)
#----------------------------------------------------------------------------------

#==================================================================================
# Simulations
#==================================================================================
results = pd.DataFrame({
    "Copule": ["Gumbel", "Student t"],
    "VaR 99%": [0.0, 0.0],
    "ES 99%":  [0.0,  0.0]
})

n_sim = 200_000

def simule_gumbel(param_gpd, n_sim, results, data):
    weights = np.array([0.5, 0.5])
    # Simulation
    joint_losses = sg.simulate_joint_losses(param_gpd, n_sim, theta, data)

    print("Perte Min Gumbel :", joint_losses.min())
    print("Perte Max Gumbel :", joint_losses.max())

    q = 0.99

    u1 = joint_losses[:,0]
    u2 = joint_losses[:,1]

    thr1 = np.quantile(u1, q)
    thr2 = np.quantile(u2, q)

    joint_extreme = np.mean((u1 > thr1) & (u2 > thr2))

    print("Probabilité empirique de co-krach :", joint_extreme)
    
    # Portefeuille
    L = sg.portfolio_losses(joint_losses, weights)

    # Risque
    VaR_99, ES_99 = sg.var_es(L, alpha=0.99)

    idx = results[results["Copule"] == "Gumbel"].index[0]
    results.at[idx, "VaR 99%"] = VaR_99
    results.at[idx, "ES 99%"] = ES_99

    print("Gumbel")
    print("VaR 99% :", VaR_99)
    print("ES 99%  :", ES_99)

#----------------------------------------------------------------------------------
# fit_copules_archimediennes()
# simule_gumbel(param_gpd, n_sim, results)
#----------------------------------------------------------------------------------

def simule_student(param_gpd, rho, nu, n_sim, results, data_losses):
    weights = np.array([0.5, 0.5])

    losses_t = st.simulate_losses_evt_t(
        n_sim,
        param_gpd,
        rho,
        nu,
        weights=weights,
        data_historique=data_losses
    )

    VaR_t, ES_t = st.var_es(losses_t)

    idx = results[results["Copule"] == "Student t"].index[0]
    results.at[idx, "VaR 99%"] = VaR_t
    results.at[idx, "ES 99%"] = ES_t

    print("Copule t")
    print("VaR 99% :", VaR_t)
    print("ES  99% :", ES_t)

#----------------------------------------------------------------------------------
# rho, nu = fit_copula_student()
# simule_student(param_gpd, rho, nu, n_sim, results)
# results.to_csv("resultats_simulation.csv")
#----------------------------------------------------------------------------------

def obtenir_VaR_ES_finaux():
    tickers = ["SPY", "FEZ"]
    start_date = "2005-01-01"
    # "2005-01-01" : la date de début à considérer pour les données
    traitement_donnees_multivarie(tickers, start_date)
    data_losses = pd.read_csv("donnees_multivarie.csv", index_col=0, parse_dates=True)

    # Les seuils
    thresholds_sp = np.linspace(np.percentile(losses_sp, 90), np.percentile(losses_sp, 99), 50)
    thresholds_eu = np.linspace(np.percentile(losses_eu, 90), np.percentile(losses_eu, 99), 50)
    #mean excess pour le S&P 500
    me_sp500 = mean_excess(losses_sp, thresholds_sp)
    #mean excess pour le eurostoxx
    me_eurostoxx = mean_excess(losses_eu, thresholds_eu)
    #show_mean_excess_plot(thresholds_sp, me_sp500, thresholds_eu, me_eurostoxx)

    xi_list_sp, beta_list_sp, xi_list_eu, beta_list_eu = calculate_gpd_parameters(u_values_sp, u_values_eu)
    #show_gpd_parameters(u_values_sp, xi_list_sp, beta_list_sp, u_values_eu, xi_list_eu, beta_list_eu)

    # Les seuils ci-dessus sont en fonction de l'observation de la fonction show_gpd_parameters(...)
    u_sp = np.percentile(losses_sp, 97.1)
    u_eu = np.percentile(losses_eu, 93.0)
    #print(f"sp : {u_sp} \n eu : {u_eu}")
    final_gpd_parameters(u_sp, u_eu)
    param_gpd = pd.read_csv("evt_gpd_parameters.csv", index_col=0)
    construct_marginales_evt(param_gpd)
    U_data = pd.read_csv("marginales_evt_pit.csv", index_col=0, parse_dates=True)
    #show_PIT_histograms(U_data["U_sp500"].values, U_data["U_eurostoxx"].values)

    fit_copula_gaussian(U_data)
    rho, nu = fit_copula_student(U_data)
    #show_copula_scatter()
    fit_copules_archimediennes(U_data)

    # Simulations
    n_sim = 200_000
    results = pd.DataFrame({
        "Copule": ["Gumbel", "Student t"],
        "VaR 99%": [0.0, 0.0],
        "ES 99%":  [0.0,  0.0]
    })

    simule_gumbel(param_gpd, n_sim, results, data_losses)
    simule_student(param_gpd, rho, nu, n_sim, results, data_losses)

    results.to_csv("resultats_simulation.csv")

#----------------------------------------------------------------------------------
obtenir_VaR_ES_finaux()
#----------------------------------------------------------------------------------








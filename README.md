# Portfolio-risk-modeling
Estimation du risque extrême d'un portefeuille

# Création d'un environnement virtuel pour le projet de risque extrême

- Installer les outils nécessaires (une seule fois)
pip install yfinance numpy pandas matplotlib scipy

- Créer un environnement virtuel
mkdir risk_extreme_project
cd risk_extreme_project

python3 -m venv venv

- Activer l’environnement
source venv/bin/activate

- Installer les librairies DANS le venv
pip install yfinance numpy pandas matplotlib scipy

# installer scikit-learn (pour la régression linéaire)
pip install scikit-learn

pip install statsmodels

pip install copulas

POUR SUPPRIMER L'ENV
deactivate 
rm -rf .venv

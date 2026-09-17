import re
import numpy as np
import pandas as pd
import yfinance as yf


def sanitize(ticker: str) -> str:
    """Convertit un ticker Yahoo Finance en nom de colonne Python valide (minuscules, sans caractères spéciaux)."""
    return re.sub(r'[^a-zA-Z0-9]', '_', ticker).strip('_').lower()


class DataLoader:
    """
    Télécharge et prépare les données de marché pour k actifs.

    Pour chaque ticker, génère deux colonnes dans le DataFrame :
        - return_{col}  : rendement logarithmique journalier
        - loss_{col}    : perte journalière (= -rendement)

    où col = sanitize(ticker).

    Exemple :
        loader = DataLoader(["SPY", "FEZ"], "2005-01-01")
        data = loader.download(output_csv="donnees_multivarie.csv")
        losses = loader.get_losses_dict(data)  # {"SPY": np.array([...]), "FEZ": np.array([...])}
    """

    def __init__(self, tickers: list, start_date: str):
        self.tickers = tickers
        self.start_date = start_date
        # Mapping ticker -> nom de colonne sécurisé
        self.col_names = {t: sanitize(t) for t in tickers}

    def download(self, output_csv: str = None) -> pd.DataFrame:
        """
        Télécharge les prix via yfinance, calcule rendements log et pertes.

        Parameters
        ----------
        output_csv : str, optional
            Si fourni, sauvegarde le DataFrame dans ce fichier CSV.

        Returns
        -------
        pd.DataFrame
            DataFrame indexé par date avec colonnes return_{col} et loss_{col}.
        """
        prices = yf.download(
            self.tickers,
            start=self.start_date,
            auto_adjust=True,
            progress=False
        )["Close"]

        # Cas d'un seul actif : yfinance retourne une Series
        if isinstance(prices, pd.Series):
            prices = prices.to_frame(name=self.tickers[0])

        prices.dropna(inplace=True)

        # Rendements logarithmiques : r_t = log(P_t / P_{t-1})
        returns = np.log(prices / prices.shift(1)).dropna()

        # Pertes = opposé des rendements
        losses = -returns

        data = pd.DataFrame(index=returns.index)
        for t in self.tickers:
            col = self.col_names[t]
            data[f"return_{col}"] = returns[t]
            data[f"loss_{col}"] = losses[t]

        if output_csv:
            data.to_csv(output_csv)
            print(f"[DataLoader] Données sauvegardées dans : {output_csv}")

        return data

    def load(self, csv_path: str) -> pd.DataFrame:
        """Charge un CSV généré par download()."""
        return pd.read_csv(csv_path, index_col=0, parse_dates=True)

    def get_losses_dict(self, data: pd.DataFrame) -> dict:
        """
        Extrait les séries de pertes du DataFrame.

        Returns
        -------
        dict : ticker -> np.ndarray des pertes journalières
        """
        return {t: data[f"loss_{self.col_names[t]}"].values for t in self.tickers}

    def get_loss_col(self, ticker: str) -> str:
        """Retourne le nom de la colonne de perte pour un ticker donné."""
        return f"loss_{self.col_names[ticker]}"

"""
Teste la robustesse de la stratégie sur des régimes de marché spécifiques
et historiquement difficiles pour les stratégies factorielles :

- 2008 : crise financière globale (stress cross-asset généralisé)
- 2013 : taper tantrum (choc de duration sur les taux)
- 2020 : crash COVID (vol extrême, corrélations qui explosent vers 1)
- 2022 : hausse de taux agressive (momentum et value en tension)

Une stratégie qui a un bon Sharpe global mais qui s'effondre sur 2-3 de ces
régimes n'est pas robuste -- ce module rend ça visible explicitement plutôt
que caché dans une moyenne long terme.
"""

import numpy as np
import pandas as pd

from backtest.metrics import summary_table

REGIMES = {
    "gfc_2008": ("2008-01-01", "2009-06-30"),
    "taper_tantrum_2013": ("2013-05-01", "2013-09-30"),
    "covid_crash_2020": ("2020-02-15", "2020-04-30"),
    "rate_hike_2022": ("2022-01-01", "2022-12-31"),
}

# BacktestEngine produit des rendements journaliers (voir backtest/engine.py).
_PERIODS_PER_YEAR = 252


def performance_by_regime(returns: pd.Series,
                           regimes: dict = REGIMES) -> pd.DataFrame:
    """
    Calcule les métriques de performance (backtest/metrics.py) sur chaque
    fenêtre de régime définie ci-dessus.
    """
    if returns.empty:
        raise ValueError("returns ne peut pas être vide.")
    if not regimes:
        raise ValueError("regimes ne peut pas être vide.")

    metric_names = summary_table(returns, _PERIODS_PER_YEAR).index

    rows = {}
    n_observations = {}
    for name, (start, end) in regimes.items():
        window = returns.loc[pd.Timestamp(start):pd.Timestamp(end)].dropna()
        n_observations[name] = len(window)
        if window.empty:
            # Régime hors de la période couverte par `returns` (ex: gfc_2008
            # alors que config.yaml démarre le backtest en 2010 par défaut) :
            # NaN explicite plutôt qu'une erreur qui empêcherait d'analyser
            # les autres régimes effectivement couverts.
            rows[name] = pd.Series(np.nan, index=metric_names)
        else:
            rows[name] = summary_table(window, _PERIODS_PER_YEAR)["value"]

    table = pd.DataFrame(rows).T
    table["n_observations"] = pd.Series(n_observations)
    return table

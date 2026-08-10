"""
Décompose la performance totale du portefeuille par contribution de chaque
facteur -- essentiel pour comprendre SI la stratégie fonctionne, mais surtout
POURQUOI, et si sa performance dépend excessivement d'un seul facteur
(fragilité) ou est bien diversifiée entre facteurs (robustesse).

C'est typiquement ce qu'une équipe quant institutionnelle présente en review :
pas juste "le Sharpe est de 0.8" mais "70% de la performance vient du momentum,
concentré sur 2022, ce qui nous inquiète sur la robustesse".
"""

import numpy as np
import pandas as pd

from backtest.engine import BacktestEngine
from backtest.metrics import summary_table

# BacktestEngine produit des rendements journaliers (voir backtest/engine.py) ;
# ce module hérite de cette hypothèse pour annualiser les métriques.
_PERIODS_PER_YEAR = 252


def _returns_to_prices(returns: pd.DataFrame) -> pd.DataFrame:
    """
    BacktestEngine.run() prend des prix, pas des rendements. On reconstruit
    une série de prix synthétique (base 1.0) dont le pct_change() redonne
    exactement `returns`, y compris sur la toute première date (grâce à une
    ligne fictive antérieure à base 1.0).
    """
    prior_date = returns.index[0] - pd.Timedelta(days=1)
    initial_row = pd.DataFrame(1.0, index=[prior_date], columns=returns.columns)
    prices = pd.concat([initial_row, (1.0 + returns).cumprod()])
    return prices.sort_index()


def single_factor_backtest(factor_name: str, factor_scores: pd.DataFrame,
                            returns: pd.DataFrame, config: dict) -> pd.Series:
    """
    Lance un backtest avec un seul facteur actif (les autres neutralisés),
    pour isoler sa contribution marginale à la performance du portefeuille
    multi-facteurs complet.
    """
    if factor_scores.empty:
        raise ValueError("factor_scores ne peut pas être vide.")
    if returns.empty:
        raise ValueError("returns ne peut pas être vide.")

    prices = _returns_to_prices(returns)
    # Un seul facteur dans le dict passé au moteur = les autres sont
    # "neutralisés" de fait (combine_factors n'a rien d'autre à combiner).
    results = BacktestEngine(config).run(prices, {factor_name: factor_scores})
    return results["portfolio_return"]


def performance_attribution(multi_factor_returns: pd.Series,
                             single_factor_returns: dict[str, pd.Series]) -> pd.DataFrame:
    """
    Compare la performance multi-facteurs à la somme des performances
    mono-facteur, pour quantifier l'effet de diversification/combinaison.
    """
    if multi_factor_returns.empty:
        raise ValueError("multi_factor_returns ne peut pas être vide.")
    if not single_factor_returns:
        raise ValueError("single_factor_returns ne peut pas être vide.")

    rows = {name: summary_table(series, _PERIODS_PER_YEAR)["value"]
            for name, series in single_factor_returns.items()}
    rows["multi_factor"] = summary_table(multi_factor_returns, _PERIODS_PER_YEAR)["value"]

    # Moyenne (pas somme brute) des rendements mono-facteur : chaque backtest
    # mono-facteur est déjà un portefeuille ~100% investi via risk parity : les
    # additionner littéralement gonflerait artificiellement l'exposition et
    # rendrait la comparaison de vol/Sharpe avec multi_factor_returns non
    # significative. La moyenne équipondérée isole l'effet de diversification
    # (corrélations) du simple effet d'échelle.
    common_index = multi_factor_returns.index
    for series in single_factor_returns.values():
        common_index = common_index.intersection(series.index)
    aligned = pd.DataFrame({name: series.loc[common_index] for name, series in single_factor_returns.items()})
    naive_average_returns = aligned.mean(axis=1)
    rows["naive_average_of_factors"] = summary_table(naive_average_returns, _PERIODS_PER_YEAR)["value"]

    table = pd.DataFrame(rows).T

    # Contribution de chaque facteur au rendement annualisé total (ex: "70%
    # de la performance vient du momentum"), non définie pour les lignes
    # composites (multi_factor, naive_average_of_factors).
    factor_returns = table.loc[list(single_factor_returns), "annualized_return"]
    total = factor_returns.sum()
    table["contribution_pct"] = np.nan
    if total != 0:
        table.loc[factor_returns.index, "contribution_pct"] = factor_returns / total

    return table

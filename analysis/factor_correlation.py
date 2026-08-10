"""
Étudie la stabilité de la corrélation entre facteurs dans le temps.
Point clé du factor investing : les facteurs supposés décorrélés (ex: value
et momentum) peuvent devenir fortement corrélés en période de stress --
la diversification "sur papier" disparaît exactement quand on en a le plus besoin.

Ce module produit les diagnostics pour détecter ce phénomène plutôt que de
supposer une corrélation stable calculée sur tout l'historique.
"""

import numpy as np
import pandas as pd

from portfolio.scoring import zscore_cross_sectional


def _factor_pairs(names: list[str]) -> list[tuple[str, str]]:
    return [(names[i], names[j]) for i in range(len(names)) for j in range(i + 1, len(names))]


def factor_mimicking_returns(factor_scores: pd.DataFrame, returns: pd.DataFrame) -> pd.Series:
    """
    Rendement d'un portefeuille "factor-mimicking" : à chaque date, expose
    chaque actif proportionnellement à son z-score cross-sectionnel du jour
    précédent (conviction brute, sans risk parity ni tilt). Sert à isoler le
    signal propre d'un facteur -- backtest/attribution.single_factor_backtest
    fait transiter le signal par la base risk parity commune à tous les
    facteurs, ce qui gonfle artificiellement la corrélation mesurée entre
    facteurs (leurs rendements partagent alors la même base). Ici, seule la
    conviction du facteur pilote l'exposition.
    """
    if factor_scores.empty:
        raise ValueError("factor_scores ne peut pas être vide.")
    if returns.empty:
        raise ValueError("returns ne peut pas être vide.")

    common_columns = factor_scores.columns.intersection(returns.columns)
    if common_columns.empty:
        raise ValueError("Aucun ticker commun entre factor_scores et returns.")

    z = zscore_cross_sectional(factor_scores[common_columns])
    gross = z.abs().sum(axis=1)
    weights = z.div(gross.replace(0.0, np.nan), axis=0)

    # Conviction connue à la clôture de t, appliquée au rendement réalisé de
    # t+1 (jamais celui de t lui-même) : pas de look-ahead.
    weights_lagged = weights.shift(1)
    return (weights_lagged * returns[common_columns]).sum(axis=1, min_count=1)


def rolling_factor_correlation(factor_returns: dict[str, pd.Series],
                                window_days: int = 126) -> pd.DataFrame:
    """
    Corrélation glissante entre chaque paire de facteurs, pour visualiser
    son évolution dans le temps plutôt qu'un chiffre statique.
    """
    if len(factor_returns) < 2:
        raise ValueError("Il faut au moins 2 facteurs pour calculer une corrélation.")
    if window_days < 2:
        raise ValueError(f"window_days doit être >= 2 (obtenu {window_days}).")

    # DataFrame(dict de Series) aligne automatiquement sur l'union des dates
    # (NaN là où une série n'a pas de valeur ce jour-là).
    aligned = pd.DataFrame(factor_returns)
    pairs = _factor_pairs(list(aligned.columns))

    # rolling().corr(other) : fenêtre [t-window+1, t] uniquement -> pas de
    # look-ahead.
    return pd.DataFrame({
        f"{a}__{b}": aligned[a].rolling(window_days, min_periods=window_days).corr(aligned[b])
        for a, b in pairs
    })


def correlation_regime_breakdown(factor_returns: dict[str, pd.Series],
                                  regimes: dict) -> pd.DataFrame:
    """
    Corrélation entre facteurs spécifiquement pendant les régimes de stress
    définis dans regime_analysis.py, comparée à la corrélation moyenne.
    """
    if len(factor_returns) < 2:
        raise ValueError("Il faut au moins 2 facteurs pour calculer une corrélation.")
    if not regimes:
        raise ValueError("regimes ne peut pas être vide.")

    aligned = pd.DataFrame(factor_returns)
    pairs = _factor_pairs(list(aligned.columns))

    def _pairwise_corr(window: pd.DataFrame) -> dict:
        corr_matrix = window.corr()
        return {f"{a}__{b}": corr_matrix.loc[a, b] for a, b in pairs}

    full_period = aligned.dropna()
    rows = {"full_period": _pairwise_corr(full_period)}
    n_observations = {"full_period": len(full_period)}

    for name, (start, end) in regimes.items():
        window = aligned.loc[pd.Timestamp(start):pd.Timestamp(end)].dropna()
        n_observations[name] = len(window)
        if len(window) < 2:
            # Régime hors de la période couverte, ou trop court pour une
            # corrélation significative : NaN explicite plutôt qu'une erreur
            # qui empêcherait d'analyser les autres régimes couverts (même
            # logique que regime_analysis.performance_by_regime).
            rows[name] = {f"{a}__{b}": np.nan for a, b in pairs}
        else:
            rows[name] = _pairwise_corr(window)

    table = pd.DataFrame(rows).T
    table["n_observations"] = pd.Series(n_observations)
    return table

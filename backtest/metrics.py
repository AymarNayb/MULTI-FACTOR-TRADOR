"""
Métriques standard d'évaluation d'une stratégie systématique.
Chaque fonction prend une série de rendements et retourne un scalaire,
pour faciliter les tests unitaires et la réutilisation dans attribution.py.
"""

import pandas as pd
import numpy as np


def _clean(returns: pd.Series) -> pd.Series:
    """Retire les NaN (ex: warmup avant le premier rebalancement) avant tout calcul."""
    cleaned = returns.dropna()
    if cleaned.empty:
        raise ValueError("returns ne contient aucune valeur exploitable (vide ou 100% NaN).")
    return cleaned


def _check_periods_per_year(periods_per_year: int) -> None:
    if periods_per_year <= 0:
        raise ValueError(f"periods_per_year doit être positif (obtenu {periods_per_year}).")


def annualized_return(returns: pd.Series, periods_per_year: int = 12) -> float:
    _check_periods_per_year(periods_per_year)
    returns = _clean(returns)
    compounded_growth = (1.0 + returns).prod()
    n_periods = len(returns)
    return compounded_growth ** (periods_per_year / n_periods) - 1.0


def annualized_volatility(returns: pd.Series, periods_per_year: int = 12) -> float:
    _check_periods_per_year(periods_per_year)
    returns = _clean(returns)
    return returns.std() * np.sqrt(periods_per_year)


def sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.0,
                  periods_per_year: int = 12) -> float:
    _check_periods_per_year(periods_per_year)
    returns = _clean(returns)
    # risk_free_rate est un taux ANNUEL ; conversion géométrique vers le taux
    # par période, cohérente avec la composition géométrique de annualized_return.
    rf_per_period = (1.0 + risk_free_rate) ** (1.0 / periods_per_year) - 1.0
    excess_returns = returns - rf_per_period
    vol = excess_returns.std()
    if vol == 0:
        return np.nan
    return (excess_returns.mean() / vol) * np.sqrt(periods_per_year)


def max_drawdown(returns: pd.Series) -> float:
    returns = _clean(returns)
    cumulative = (1.0 + returns).cumprod()
    running_max = cumulative.cummax()
    drawdown = cumulative / running_max - 1.0
    return drawdown.min()


def sortino_ratio(returns: pd.Series, periods_per_year: int = 12) -> float:
    """Comme Sharpe mais ne pénalise que la volatilité à la baisse."""
    _check_periods_per_year(periods_per_year)
    returns = _clean(returns)
    downside = returns.clip(upper=0.0)
    downside_deviation = np.sqrt((downside ** 2).mean()) * np.sqrt(periods_per_year)
    if downside_deviation == 0:
        return np.nan
    return annualized_return(returns, periods_per_year) / downside_deviation


def calmar_ratio(returns: pd.Series, periods_per_year: int = 12) -> float:
    """Rendement annualisé / max drawdown -- pertinent pour juger la robustesse."""
    _check_periods_per_year(periods_per_year)
    returns = _clean(returns)
    mdd = max_drawdown(returns)
    if mdd == 0:
        return np.nan
    return annualized_return(returns, periods_per_year) / abs(mdd)


def summary_table(returns: pd.Series, periods_per_year: int = 12) -> pd.DataFrame:
    """Regroupe toutes les métriques ci-dessus dans un tableau unique."""
    _check_periods_per_year(periods_per_year)
    returns = _clean(returns)
    metrics = {
        "annualized_return": annualized_return(returns, periods_per_year),
        "annualized_volatility": annualized_volatility(returns, periods_per_year),
        "sharpe_ratio": sharpe_ratio(returns, periods_per_year=periods_per_year),
        "max_drawdown": max_drawdown(returns),
        "sortino_ratio": sortino_ratio(returns, periods_per_year),
        "calmar_ratio": calmar_ratio(returns, periods_per_year),
    }
    return pd.DataFrame.from_dict(metrics, orient="index", columns=["value"])

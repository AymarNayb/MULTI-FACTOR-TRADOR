"""
Modélise les coûts réels d'implémentation, cruciaux en multi-actif à cause
du levier -- un backtest sans coûts de financement surestime systématiquement
la performance d'une stratégie avec du levier ou du short.

Deux composantes :
- Coûts de transaction : liés au turnover (achat/vente) à chaque rebalancement
- Coûts de financement : liés au levier net du portefeuille (emprunter pour
  être >100% investi, ou financer une position short), approximés par le
  taux sans risque + spread (voir FRED dans data/loaders/fred_loader.py)
"""

import pandas as pd


def transaction_costs(weights_before: pd.Series, weights_after: pd.Series,
                       cost_bps: float) -> float:
    """
    Coût = turnover (somme des variations absolues de poids) * cost_bps.
    """
    if cost_bps < 0:
        raise ValueError(f"cost_bps doit être positif ou nul (obtenu {cost_bps}).")

    common_index = weights_before.index.union(weights_after.index)
    before = weights_before.reindex(common_index).fillna(0.0)
    after = weights_after.reindex(common_index).fillna(0.0)

    turnover = (after - before).abs().sum()
    return turnover * (cost_bps / 10_000.0)


def financing_costs(net_leverage: float, risk_free_rate: float,
                     spread_bps: float = 20) -> float:
    """
    Coût annualisé du financement du levier net du portefeuille,
    à convertir en coût périodique selon la fréquence de rebalancement.
    """
    raise NotImplementedError

"""
Low volatility / defensive : favorise les actifs à faible volatilité relative.
Anomalie empirique bien documentée (les actifs à faible beta/vol ont un ratio
rendement/risque historiquement meilleur que prédit par le CAPM classique).

Sert aussi de facteur "stabilisateur" dans la combinaison multi-facteurs :
tend à être décorrélé voire anti-corrélé au momentum en période de stress.
"""

import pandas as pd
from factors.base import Factor


class LowVolFactor(Factor):
    name = "low_vol"

    def __init__(self, lookback_days: int = 63):
        self.lookback_days = lookback_days

    def compute(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        Score = volatilité réalisée inversée (signe négatif de la vol),
        cross-sectionnelle par classe d'actif à chaque date.

        Returns:
            DataFrame de scores de low_vol bruts par actif
        """
        if data.empty:
            raise ValueError("data ne peut pas être vide.")

        returns = data.pct_change()
        # rolling() ne regarde que [t-window+1, t] : pas de look-ahead.
        realized_vol = returns.rolling(window=self.lookback_days, min_periods=self.lookback_days).std()
        return -realized_vol

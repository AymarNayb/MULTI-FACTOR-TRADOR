"""
Value : cherté relative d'un actif par rapport à son propre historique ou à
un fondamental. La définition change selon la classe d'actif — c'est le
facteur le plus hétérogène du pipeline :

- Actions : P/E, P/B relatifs (nécessite données fondamentales, pas juste prix)
- Taux : real yield actuel vs moyenne historique (proxy de cherté obligataire)
- FX : écart au fair value PPP, ou carry-adjusted valuation
- Commodities : spot price vs coût de production estimé, ou vs moyenne long terme

Pour la V1 du projet (données ETF uniquement, pas de fondamentaux), on utilise
un proxy simplifié : écart du prix actuel à sa moyenne mobile long terme,
normalisé par la volatilité (mean-reversion statistique comme proxy de value).
Documenter clairement cette approximation dans le rapport final.
"""

import pandas as pd
from factors.base import Factor

TRADING_DAYS_PER_YEAR = 252


class ValueFactor(Factor):
    name = "value"

    def __init__(self, lookback_years: int = 5):
        self.lookback_years = lookback_years

    def compute(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        Proxy V1 : (prix actuel - moyenne mobile long terme) / vol réalisée,
        signe inversé (un actif "cher" par rapport à son historique = value
        score négatif).

        Returns:
            DataFrame de scores de value bruts par actif
        """
        if data.empty:
            raise ValueError("data ne peut pas être vide.")

        window = self.lookback_years * TRADING_DAYS_PER_YEAR

        # rolling() ne regarde que [t-window+1, t] : pas de look-ahead.
        # Écart-type du PRIX (pas des rendements) pour rester sur la même
        # échelle que le numérateur (prix - moyenne mobile) : un z-score
        # cohérent, pas un mélange dollars / pourcentages.
        rolling_mean = data.rolling(window=window, min_periods=window).mean()
        rolling_std = data.rolling(window=window, min_periods=window).std()

        z_richness = (data - rolling_mean) / rolling_std
        return -z_richness

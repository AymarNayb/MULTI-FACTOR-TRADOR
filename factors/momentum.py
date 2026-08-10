"""
Momentum time-series : rendement passé sur une fenêtre (typiquement 12 mois),
en excluant le mois le plus récent (12-1) pour éviter l'effet de reversal
court terme documenté empiriquement (Jegadeesh & Titman).

Applicable à toutes les classes d'actifs de la même façon (contrairement à
value ou carry qui nécessitent une définition différente par classe d'actif).
"""

import pandas as pd
from factors.base import Factor

# Approximation standard : ~21 jours ouvrés par mois. data provient de
# data_aligner.py (calendrier business-day déjà régulier), donc un décalage
# en nombre de lignes approxime un décalage calendaire en mois.
TRADING_DAYS_PER_MONTH = 21


class MomentumFactor(Factor):
    name = "momentum"

    def __init__(self, lookback_months: int = 12, skip_months: int = 1):
        self.lookback_months = lookback_months
        self.skip_months = skip_months

    def compute(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        Args:
            data: DataFrame de prix (index=date, colonnes=tickers)

        Returns:
            DataFrame de rendement cumulé sur la fenêtre [t-lookback, t-skip]
        """
        if data.empty:
            raise ValueError("data ne peut pas être vide.")

        skip_days = self.skip_months * TRADING_DAYS_PER_MONTH
        lookback_days = self.lookback_months * TRADING_DAYS_PER_MONTH

        # shift() ne regarde que le passé : le score à la date t ne peut donc
        # jamais dépendre d'une donnée postérieure à t (pas de look-ahead).
        end_price = data.shift(skip_days)
        start_price = data.shift(skip_days + lookback_days)
        return end_price / start_price - 1

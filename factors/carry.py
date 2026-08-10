"""
Carry : rendement qu'on gagne à détenir un actif si rien ne bouge.
Comme value, la définition varie par classe d'actif :

- Actions : dividend yield
- Taux : pente de courbe (roll-down), ou yield du future vs taux court
- FX : différentiel de taux d'intérêt entre les deux devises (interest rate parity)
- Commodities : structure de la courbe des futures (contango = carry négatif,
  backwardation = carry positif)

Nécessite des données complémentaires aux prix seuls (yields FRED pour taux/FX,
dividend yield pour actions). Voir data/loaders/fred_loader.py.
"""

import pandas as pd
from factors.base import Factor


class CarryFactor(Factor):
    name = "carry"

    def __init__(self, method: str = "yield_differential"):
        self.method = method

    def compute(self, data: pd.DataFrame, yields: pd.DataFrame | None = None,
                **kwargs) -> pd.DataFrame:
        """
        Args:
            data: prix des actifs
            yields: séries de taux/yields nécessaires selon la classe d'actif

        Returns:
            DataFrame de scores de carry bruts par actif
        """
        if self.method != "yield_differential":
            raise NotImplementedError(f"Méthode de carry '{self.method}' non implémentée.")

        if yields is None or yields.empty:
            raise ValueError(
                "Le facteur carry (méthode 'yield_differential') nécessite un "
                "DataFrame 'yields' (voir data/loaders/fred_loader.py) : "
                "le carry ne peut pas être déduit des prix seuls."
            )

        common_cols = data.columns.intersection(yields.columns)
        if common_cols.empty:
            raise ValueError(
                "Aucun ticker commun entre 'data' et 'yields' : impossible de "
                "calculer le carry. Vérifier que 'yields' est déjà mappé sur "
                "les tickers de 'data' (un différentiel par actif, pas par série FRED brute)."
            )

        # reindex + ffill uniquement : jamais de valeur empruntée au futur.
        aligned_yields = yields.reindex(data.index).ffill()
        return aligned_yields[common_cols].reindex(columns=data.columns)

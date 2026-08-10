"""
Interface commune à tous les facteurs. Chaque facteur (momentum, value, carry,
low_vol) hérite de cette classe et implémente compute().

Ça permet à portfolio/scoring.py de traiter tous les facteurs de façon
uniforme, sans connaître leur implémentation interne — on peut ajouter un
nouveau facteur sans toucher au reste du pipeline (Open/Closed principle).
"""

from abc import ABC, abstractmethod
import pandas as pd


class Factor(ABC):
    """Classe abstraite pour tout facteur du pipeline."""

    name: str

    @abstractmethod
    def compute(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        Calcule le score brut du facteur pour chaque actif à chaque date.

        Args:
            data: prix ou données nécessaires (dépend du facteur)

        Returns:
            DataFrame (index=date, colonnes=tickers) de scores bruts,
            AVANT standardisation (le z-score se fait dans scoring.py,
            de façon uniforme pour tous les facteurs)
        """
        raise NotImplementedError

    def __repr__(self) -> str:
        return f"<Factor: {self.name}>"

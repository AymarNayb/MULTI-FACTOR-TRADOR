"""
Tests unitaires des modules factors/. Priorité de test :
1. Absence de look-ahead bias (le score à la date t ne doit dépendre que de
   données <= t)
2. Comportement correct sur des cas limites (données manquantes, série trop
   courte pour la fenêtre de lookback)
3. Cohérence du signe (ex: un momentum positif doit correspondre à un
   rendement passé positif, trivial mais à vérifier explicitement)
"""

import pandas as pd
import pytest


def test_momentum_no_lookahead():
    """Le score momentum à la date t ne doit pas changer si on ajoute des
    données futures après t."""
    raise NotImplementedError


def test_momentum_sign_consistency():
    """Une série strictement croissante doit produire un score momentum positif."""
    raise NotImplementedError


def test_low_vol_handles_missing_data():
    """Le calcul ne doit pas planter silencieusement sur des NaN ponctuels."""
    raise NotImplementedError

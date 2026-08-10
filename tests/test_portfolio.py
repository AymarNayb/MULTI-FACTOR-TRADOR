"""
Tests unitaires des modules portfolio/. Priorité de test :
1. Les poids risk parity somment bien à l'exposition brute attendue
2. Les contraintes (leverage, position caps) sont bien respectées après
   application, pas juste "en théorie"
3. La redistribution après plafonnement ne casse pas la contrainte de levier
"""

import pandas as pd
import pytest


def test_risk_parity_weights_sum_to_target_leverage():
    raise NotImplementedError


def test_position_caps_respected_after_redistribution():
    raise NotImplementedError


def test_equal_risk_contribution_property():
    """Vérifie que chaque actif contribue effectivement de façon égale à la
    variance totale du portefeuille (propriété définissante de l'ERC)."""
    raise NotImplementedError

"""
Tests unitaires du moteur de backtest -- le module le plus critique à tester
rigoureusement, car un bug ici invalide silencieusement tous les résultats.

Priorité de test :
1. Look-ahead bias : les poids générés à la date t ne doivent utiliser
   aucune donnée postérieure à t (test à faire avec des données synthétiques
   où on connaît la réponse attendue)
2. Les coûts de transaction sont bien déduits de la performance
3. Le turnover est calculé correctement
4. Une stratégie "buy and hold" simple (poids constants) doit produire un
   résultat vérifiable à la main, pour valider le moteur indépendamment
   de la logique factorielle
"""

import pandas as pd
import pytest


def test_no_lookahead_in_backtest_loop():
    raise NotImplementedError


def test_transaction_costs_reduce_returns():
    raise NotImplementedError


def test_buy_and_hold_matches_manual_calculation():
    """Cas de référence simple pour valider le moteur indépendamment des facteurs."""
    raise NotImplementedError

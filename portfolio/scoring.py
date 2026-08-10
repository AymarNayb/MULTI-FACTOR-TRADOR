"""
Combine les scores bruts de plusieurs facteurs en un signal composite unique
par actif. C'est ici que se joue une bonne partie de la valeur ajoutée du
factor investing multi-facteurs vs mono-facteur.

Étapes :
1. Standardisation (z-score) de chaque facteur, cross-sectionnellement,
   à chaque date -> rend les facteurs comparables entre eux
2. Vérification de la corrélation entre facteurs standardisés (éviter de
   sur-pondérer implicitement deux facteurs redondants, ex: value et carry
   sont parfois très corrélés en FX)
3. Combinaison pondérée (poids égaux par défaut, ou optimisés/orthogonalisés
   en version avancée) -> signal composite

Ne fait PAS la construction de portefeuille elle-même (poids en $ ou en risque),
c'est le rôle de risk_parity.py -- ce module produit des CONVICTIONS, pas des POIDS.
"""

import numpy as np
import pandas as pd


def zscore_cross_sectional(factor_scores: pd.DataFrame) -> pd.DataFrame:
    """
    Standardise un DataFrame de scores bruts en z-score, actif par actif,
    à chaque date (cross-sectionnel : moyenne/std calculées sur la coupe
    transversale des actifs à un instant t, pas dans le temps).
    """
    if factor_scores.empty:
        raise ValueError("factor_scores ne peut pas être vide.")

    row_mean = factor_scores.mean(axis=1)
    row_std = factor_scores.std(axis=1)
    return factor_scores.sub(row_mean, axis=0).div(row_std, axis=0)


def combine_factors(factor_scores: dict[str, pd.DataFrame],
                     weights: dict[str, float] | None = None) -> pd.DataFrame:
    """
    Combine plusieurs facteurs standardisés en un signal composite.

    Args:
        factor_scores: dict {nom_facteur: DataFrame de z-scores}
        weights: dict {nom_facteur: poids}, None = poids égaux

    Returns:
        DataFrame du signal composite (conviction) par actif et par date
    """
    if not factor_scores:
        raise ValueError("factor_scores ne peut pas être vide.")

    factor_names = list(factor_scores.keys())
    if weights is None:
        weights = {name: 1.0 / len(factor_names) for name in factor_names}
    elif set(weights) != set(factor_names):
        raise ValueError(
            "weights doit couvrir exactement les facteurs de factor_scores "
            f"(attendu {sorted(factor_names)}, reçu {sorted(weights)})."
        )
    elif not np.isclose(sum(weights.values()), 1.0):
        raise ValueError(f"La somme des poids doit valoir 1.0 (obtenu {sum(weights.values())}).")

    first = factor_scores[factor_names[0]]
    index = first.index
    columns = first.columns
    for df in factor_scores.values():
        index = index.union(df.index)
        columns = columns.union(df.columns)

    weighted_sum = pd.DataFrame(0.0, index=index, columns=columns)
    weight_sum = pd.DataFrame(0.0, index=index, columns=columns)

    for name, df in factor_scores.items():
        aligned = df.reindex(index=index, columns=columns)
        available = aligned.notna()
        # Facteur manquant (ex: warmup d'une fenêtre longue) = exclu du calcul
        # à cette cellule, pas traité comme une conviction neutre à 0.
        weighted_sum += aligned.fillna(0.0) * weights[name]
        weight_sum += available * weights[name]

    # NaN uniquement là où AUCUN facteur n'est disponible.
    return weighted_sum / weight_sum.replace(0.0, np.nan)


def check_factor_correlation(factor_scores: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Diagnostic : matrice de corrélation entre facteurs standardisés, utile
    pour justifier (ou remettre en question) le choix des poids de combinaison.
    """
    if not factor_scores:
        raise ValueError("factor_scores ne peut pas être vide.")

    # stack() aplatit (date, ticker) -> une seule série par facteur ; concat
    # aligne ces séries sur leur MultiIndex commun, puis corr() calcule les
    # corrélations pairwise en ignorant les (date, ticker) manquants.
    flattened = {name: df.stack() for name, df in factor_scores.items()}
    combined = pd.concat(flattened, axis=1)
    return combined.corr()

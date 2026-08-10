"""
Applique les contraintes de gestion des risques définies dans config.yaml
aux poids bruts issus de risk_parity.py, avant exécution.

C'est une couche de sécurité indépendante du calcul de poids -- même si le
modèle de risk parity a un bug ou un cas limite, ces contraintes empêchent
des positions absurdes (levier excessif, concentration extrême).
"""

import numpy as np
import pandas as pd


def apply_leverage_cap(weights: pd.DataFrame, max_leverage: float) -> pd.DataFrame:
    """Renormalise les poids si la somme des expositions brutes dépasse max_leverage."""
    if weights.empty:
        raise ValueError("weights ne peut pas être vide.")
    if max_leverage <= 0:
        raise ValueError(f"max_leverage doit être positif (obtenu {max_leverage}).")

    gross_exposure = weights.abs().sum(axis=1)
    # clip(upper=1.0) : on ne redimensionne QUE si le levier est dépassé,
    # jamais pour "compléter" un portefeuille sous-exposé.
    scale = (max_leverage / gross_exposure).clip(upper=1.0)
    return weights.mul(scale, axis=0)


def _cap_with_redistribution(values: pd.Series, cap: float) -> pd.Series:
    """
    Plafonne |values| à `cap`. L'excédent des positions plafonnées est
    redistribué UNE SEULE FOIS, proportionnellement aux positions non
    plafonnées -- pas de cascade itérative jusqu'à saturation complète.
    Une cascade complète peut effacer toute différenciation entre positions
    (ex: 4 actifs, cap * 4 == exposition totale -> tout finit plafonné au
    même niveau, quelle que soit la conviction d'origine). Toute position
    que cette unique redistribution repousse elle-même au-dessus du cap est
    re-plafonnée sans nouvelle redistribution : l'exposition totale peut
    alors être légèrement inférieure à l'originale, ce qui est préférable à
    une perte totale d'information sur les positions relatives.
    """
    result = values.astype(float).copy()
    over = result.abs() > cap + 1e-9
    if not over.any():
        return result

    excess = (result[over].abs() - cap).sum()
    result[over] = np.sign(result[over]) * cap

    free = ~over
    free_abs_sum = result[free].abs().sum()
    if free.any() and free_abs_sum > 0:
        result[free] = result[free] * (1.0 + excess / free_abs_sum)

    still_over = result.abs() > cap + 1e-9
    result[still_over] = np.sign(result[still_over]) * cap

    return result


def _cap_row(row: pd.Series, asset_cap: float, class_cap: float,
             classes: pd.Series) -> pd.Series:
    valid = row.dropna()
    if valid.empty:
        return row

    valid_classes = classes.loc[valid.index]
    current = _cap_with_redistribution(valid.astype(float), asset_cap)

    class_totals = current.abs().groupby(valid_classes).sum()
    capped_class_totals = _cap_with_redistribution(class_totals, class_cap)
    # Le plafonnement par classe rescale chaque actif de la classe
    # proportionnellement, sans changer les poids relatifs à l'intérieur
    # de la classe elle-même.
    class_scale = (capped_class_totals / class_totals.replace(0.0, np.nan)).fillna(1.0)
    current = current * valid_classes.map(class_scale)

    # Ce rescale par classe peut repousser un actif au-dessus du cap
    # individuel : plafonnement final, sans nouvelle redistribution (pour ne
    # pas relancer un cycle classe <-> actif qui reproduirait la cascade).
    # Un clip ne peut que réduire une valeur, donc le cap de classe déjà
    # respecté ne peut pas être re-violé par cette étape.
    over_again = current.abs() > asset_cap + 1e-9
    current[over_again] = np.sign(current[over_again]) * asset_cap

    result = row.copy()
    result.loc[valid.index] = current
    return result


def apply_position_caps(weights: pd.DataFrame, max_weight_per_asset: float,
                         max_weight_per_asset_class: float,
                         asset_class_map: dict[str, str]) -> pd.DataFrame:
    """
    Plafonne le poids d'un actif individuel et le poids agrégé par classe
    d'actif, puis redistribue l'excédent proportionnellement aux autres
    positions.
    """
    if weights.empty:
        raise ValueError("weights ne peut pas être vide.")
    if max_weight_per_asset <= 0 or max_weight_per_asset_class <= 0:
        raise ValueError("max_weight_per_asset et max_weight_per_asset_class doivent être positifs.")
    if max_weight_per_asset > max_weight_per_asset_class:
        raise ValueError(
            "max_weight_per_asset ne peut pas dépasser max_weight_per_asset_class "
            "(sinon la contrainte individuelle serait inatteignable dans certaines classes)."
        )
    missing = set(weights.columns) - set(asset_class_map)
    if missing:
        raise ValueError(f"asset_class_map ne couvre pas tous les tickers de weights : {missing}")

    classes = pd.Series({ticker: asset_class_map[ticker] for ticker in weights.columns})

    return weights.apply(
        lambda row: _cap_row(row, max_weight_per_asset, max_weight_per_asset_class, classes),
        axis=1,
    )

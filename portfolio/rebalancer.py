"""
Orchestre le cycle complet de rebalancement à une fréquence donnée
(mensuel par défaut) : scoring -> risk parity -> contraintes -> poids finaux.

Point d'entrée unique utilisé par backtest/engine.py à chaque date de
rebalancement -- garde toute la logique métier hors du moteur de backtest,
qui ne doit gérer que la simulation temporelle et les coûts.
"""

import pandas as pd

from portfolio.scoring import zscore_cross_sectional, combine_factors
from portfolio.risk_parity import (
    equal_risk_contribution_weights,
    inverse_volatility_weights,
    apply_conviction_tilt,
    scale_to_target_volatility,
)
from portfolio.constraints import apply_leverage_cap, apply_position_caps

_WEIGHTING_METHODS = {
    "risk_parity": equal_risk_contribution_weights,  # ERC, voir config.yaml
    "inverse_volatility": inverse_volatility_weights,
}


def _asset_class_map(universe: dict) -> dict[str, str]:
    return {
        entry["ticker"]: asset_class
        for asset_class, entries in universe.items()
        for entry in entries
    }


def generate_target_weights(date: pd.Timestamp, factor_scores: dict,
                             returns_history: pd.DataFrame,
                             config: dict) -> pd.Series:
    """
    Calcule les poids cibles du portefeuille à une date de rebalancement donnée.

    Args:
        date: date de rebalancement
        factor_scores: scores des facteurs disponibles jusqu'à cette date
                        (aucune donnée future, critique pour éviter le look-ahead)
        returns_history: historique de rendements jusqu'à cette date
        config: configuration du projet (config.yaml chargé)

    Returns:
        pd.Série des poids cibles par actif
    """
    if not factor_scores:
        raise ValueError("factor_scores ne peut pas être vide.")
    if returns_history.empty:
        raise ValueError("returns_history ne peut pas être vide.")
    if date not in returns_history.index:
        raise ValueError(f"date ({date}) absente de returns_history.index.")
    if returns_history.index.max() > date:
        raise ValueError(
            "returns_history contient des dates postérieures à 'date' : "
            "risque de look-ahead bias (ce module ne doit voir que le passé)."
        )

    portfolio_cfg = config.get("portfolio", {})

    # 1. Scoring. Le z-score cross-sectionnel est déjà "local" à une date
    # (moyenne/std sur les actifs à t) : on ne standardise que la ligne
    # `date`, pas tout l'historique, puisque generate_target_weights est
    # appelé à répétition (une fois par date de rebalancement).
    standardized = {}
    for name, scores in factor_scores.items():
        if date not in scores.index:
            raise ValueError(f"Le facteur '{name}' n'a pas de score à la date {date}.")
        standardized[name] = zscore_cross_sectional(scores.loc[[date]])
    conviction = combine_factors(standardized)

    # 2. Risk parity : poids "neutres en risque" sur la fenêtre qui précède
    # immédiatement la date de rebalancement (pas tout l'historique : inutile
    # et coûteux, en particulier pour l'ERC, de ré-optimiser des dates qu'on
    # n'utilisera pas).
    method_name = portfolio_cfg.get("weighting_method", "risk_parity")
    if method_name not in _WEIGHTING_METHODS:
        raise ValueError(
            f"weighting_method '{method_name}' inconnu. "
            f"Méthodes disponibles : {sorted(_WEIGHTING_METHODS)}."
        )
    lookback_days = portfolio_cfg.get("risk_parity_lookback_days", 63)

    window = returns_history.loc[:date].tail(lookback_days)
    if len(window) < lookback_days:
        raise ValueError(
            f"returns_history ne contient que {len(window)} observations avant {date}, "
            f"besoin de {lookback_days} (portfolio.risk_parity_lookback_days)."
        )
    risk_parity_weights = _WEIGHTING_METHODS[method_name](window, lookback_days=lookback_days)
    base_weights = risk_parity_weights.loc[[date]]
    if base_weights.iloc[0].isna().all():
        raise ValueError(f"Impossible de calculer les poids risk parity à la date {date}.")

    # 3. Tilt de conviction (conviction manquante pour un actif = tilt
    # neutre, géré par apply_conviction_tilt lui-même).
    tilt_strength = portfolio_cfg.get("tilt_strength", 0.5)
    conviction_aligned = conviction.reindex(index=base_weights.index, columns=base_weights.columns)
    tilted_weights = apply_conviction_tilt(base_weights, conviction_aligned, tilt_strength=tilt_strength)

    # 4. Contraintes de risque (position caps puis levier, en dernier
    # rempart quel que soit ce qui précède).
    asset_class_map = _asset_class_map(config["universe"])
    missing = set(tilted_weights.columns) - set(asset_class_map)
    if missing:
        raise ValueError(
            "Tickers absents de config['universe'], impossible de déterminer "
            f"leur classe d'actif pour les contraintes : {missing}"
        )

    capped = apply_position_caps(
        tilted_weights,
        portfolio_cfg["max_weight_per_asset"],
        portfolio_cfg["max_weight_per_asset_class"],
        asset_class_map,
    )

    # 5. Scaling vers la volatilité cible : risk parity/ERC ne donne que des
    # poids relatifs (somme <= 1), sans lien garanti avec max_leverage. Le cap
    # de levier reste appliqué APRÈS, en dernier garde-fou si ce scaling
    # pousse au-delà de max_leverage.
    scaled_weights = scale_to_target_volatility(
        capped, window, portfolio_cfg["target_volatility"], lookback_days=lookback_days
    )
    final_weights = apply_leverage_cap(scaled_weights, portfolio_cfg["max_leverage"])

    return final_weights.loc[date]

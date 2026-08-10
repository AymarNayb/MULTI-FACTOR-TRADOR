"""
Point critique du pipeline : aligne toutes les séries (actions, taux, FX,
commodities, crédit) sur un calendrier commun avant tout calcul de facteur.

Pourquoi c'est important :
- Les marchés FX tradent 24h/5j, les actions suivent le calendrier boursier local,
  les données FRED sont parfois publiées avec un lag -> désalignement naïf =
  risque de look-ahead bias ou de faux signaux
- Un simple .join() sans réflexion sur le lag introduit des biais silencieux

Responsabilités :
- Réindexer toutes les séries sur un calendrier de référence (ex: jours ouvrés US)
- Forward-fill contrôlé (jamais de fill vers le futur)
- Décaler les séries macro (FRED) si publiées avec un lag pour éviter le look-ahead
"""

import logging

import pandas as pd

logger = logging.getLogger(__name__)

# Calendriers de référence supportés. "NYSE" est approximé par les jours ouvrés
# (Lun-Ven) car pandas_market_calendars n'est pas une dépendance du projet
# (voir requirements.txt) : les jours fériés US ne sont donc pas exclus.
_SUPPORTED_CALENDARS = {"NYSE"}


def align_to_common_calendar(price_dfs: dict[str, pd.DataFrame],
                              reference_calendar: str = "NYSE") -> pd.DataFrame:
    """
    Fusionne et aligne plusieurs DataFrames de prix sur un calendrier commun.

    Args:
        price_dfs: dict {asset_class: DataFrame} issus des loaders
        reference_calendar: calendrier de référence pour les jours ouvrés

    Returns:
        DataFrame unique aligné, colonnes multi-index (asset_class, ticker)
    """
    if not price_dfs:
        raise ValueError("price_dfs ne peut pas être vide.")
    if reference_calendar not in _SUPPORTED_CALENDARS:
        raise NotImplementedError(
            f"Calendrier de référence '{reference_calendar}' non supporté. "
            f"Calendriers disponibles : {sorted(_SUPPORTED_CALENDARS)}."
        )

    empty = [asset_class for asset_class, df in price_dfs.items() if df.empty]
    if empty:
        raise ValueError(f"DataFrame(s) vide(s) pour : {empty}")

    all_dates = pd.DatetimeIndex(sorted({d for df in price_dfs.values() for d in df.index}))
    reference_index = pd.bdate_range(all_dates.min(), all_dates.max(), name="date")
    logger.info(
        "Alignement de %d classes d'actifs sur %d jours ouvrés (%s -> %s)",
        len(price_dfs), len(reference_index), reference_index.min().date(), reference_index.max().date(),
    )

    aligned_blocks = []
    for asset_class, df in price_dfs.items():
        # ffill uniquement : une valeur ne peut se propager que vers le futur,
        # jamais être devinée à partir d'une observation postérieure (bfill interdit)
        block = df.reindex(reference_index).ffill()
        block.columns = pd.MultiIndex.from_product([[asset_class], block.columns])
        aligned_blocks.append(block)

    return pd.concat(aligned_blocks, axis=1)


def apply_publication_lag(macro_df: pd.DataFrame, lag_days: int = 1) -> pd.DataFrame:
    """
    Décale les séries macro publiées avec délai pour éviter le look-ahead bias
    (une donnée publiée le jour J n'est utilisable qu'à partir de J+lag).
    """
    if lag_days < 0:
        raise ValueError("lag_days doit être positif ou nul.")
    if macro_df.empty:
        raise ValueError("macro_df ne peut pas être vide.")

    # Décalage par position de ligne : suppose un index déjà régulier (ex: sortie
    # de align_to_common_calendar). Sur un calendrier irrégulier, un décalage par
    # ligne n'équivaut pas exactement à un décalage de `lag_days` jours calendaires.
    logger.info("Application d'un lag de publication de %d jour(s)", lag_days)
    return macro_df.shift(lag_days)

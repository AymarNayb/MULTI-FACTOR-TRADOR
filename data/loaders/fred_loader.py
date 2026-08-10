"""
Récupère les séries macro/taux depuis FRED (Federal Reserve Economic Data).
Utilisé pour : yields souverains, spreads de crédit, taux sans risque
(nécessaire pour le coût de financement du levier en backtest).

Nécessite une clé API FRED gratuite (https://fred.stlouisfed.org/docs/api/api_key.html)
stockée en variable d'environnement FRED_API_KEY, jamais en dur dans le code.
"""

import logging
import os
from pathlib import Path

import pandas as pd
from fredapi import Fred

logger = logging.getLogger(__name__)

RAW_DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "raw" / "fred"

# Tolérance calendaire : start_date/end_date peuvent tomber un jour non publié
# (weekend, jour férié, décalage de publication FRED). Sans cette marge, le
# cache serait jugé incomplet à chaque run. Même logique que yfinance_loader.
_CACHE_TOLERANCE = pd.Timedelta(days=5)

_fred_client: Fred | None = None


def _get_client() -> Fred:
    """Construit (une seule fois) le client fredapi à partir de FRED_API_KEY."""
    global _fred_client
    if _fred_client is not None:
        return _fred_client

    api_key = os.environ.get("FRED_API_KEY")
    if not api_key:
        raise RuntimeError(
            "La variable d'environnement FRED_API_KEY n'est pas définie. "
            "Obtenir une clé gratuite sur "
            "https://fred.stlouisfed.org/docs/api/api_key.html puis l'exporter "
            "(ex: export FRED_API_KEY=... ou $env:FRED_API_KEY='...')."
        )

    _fred_client = Fred(api_key=api_key)
    return _fred_client


def _cache_path(series_id: str) -> Path:
    return RAW_DATA_DIR / f"{series_id}.csv"


def _read_cache(series_id: str) -> pd.Series | None:
    """Lit le cache d'une série si présent et lisible, sinon None."""
    path = _cache_path(series_id)
    if not path.exists():
        return None
    try:
        cached = pd.read_csv(path, index_col=0, parse_dates=True)["value"]
    except (KeyError, pd.errors.EmptyDataError, ValueError) as exc:
        logger.warning("Cache illisible pour %s (%s), re-téléchargement", series_id, exc)
        return None
    return cached if not cached.empty else None


def _cache_covers_range(cached: pd.Series, start: pd.Timestamp, end: pd.Timestamp) -> bool:
    return (cached.index.min() <= start + _CACHE_TOLERANCE
            and cached.index.max() >= end - _CACHE_TOLERANCE)


def _write_cache(series_id: str, series: pd.Series) -> None:
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    series.rename("value").to_csv(_cache_path(series_id))


def _download_series(series_id: str, start_date: str, end_date: str | None) -> pd.Series:
    """Télécharge une série FRED brute via fredapi."""
    logger.info("Téléchargement FRED : %s (%s -> %s)", series_id, start_date, end_date or "aujourd'hui")
    client = _get_client()
    try:
        raw = client.get_series(series_id, observation_start=start_date, observation_end=end_date)
    except ValueError as exc:
        # fredapi lève déjà ValueError pour un series_id invalide ou une clé API rejetée
        raise ValueError(f"Échec du téléchargement FRED pour la série '{series_id}': {exc}") from exc
    except Exception as exc:
        raise RuntimeError(f"Échec du téléchargement FRED pour la série '{series_id}': {exc}") from exc

    if raw is None or raw.empty:
        end_label = end_date or "aujourd'hui"
        raise ValueError(
            f"Aucune donnée retournée par FRED pour la série '{series_id}' "
            f"sur la période {start_date} -> {end_label}. "
            "Vérifier que le code de série est valide."
        )

    series = raw.dropna()
    series.index.name = "date"
    series.name = series_id
    return series


def load_fred_series(series_id: str, start_date: str,
                      end_date: str | None = None, use_cache: bool = True) -> pd.Series:
    """
    Télécharge une série FRED unique (ex: "DGS10" pour le 10Y US).

    Returns:
        pd.Series indexée par date
    """
    if not series_id:
        raise ValueError("series_id ne peut pas être vide.")

    start_ts = pd.Timestamp(start_date)
    end_ts = pd.Timestamp(end_date) if end_date else pd.Timestamp.today().normalize()
    if end_ts < start_ts:
        raise ValueError(f"end_date ({end_date}) est antérieure à start_date ({start_date}).")

    cached = _read_cache(series_id) if use_cache else None
    if cached is not None and _cache_covers_range(cached, start_ts, end_ts):
        logger.info("Cache valide pour %s, téléchargement évité", series_id)
        series = cached
    else:
        series = _download_series(series_id, start_date, end_date)
        if use_cache:
            _write_cache(series_id, series)

    return series.loc[(series.index >= start_ts) & (series.index <= end_ts)]


def load_all_fred_series(series_map: dict[str, str], start_date: str) -> pd.DataFrame:
    """
    Télécharge toutes les séries FRED définies dans config["data"]["fred_series"].

    Args:
        series_map: dict {nom_lisible: code_fred}, ex: {"us_10y_yield": "DGS10"}

    Returns:
        DataFrame (index=date, colonnes=noms lisibles)
    """
    if not series_map:
        raise ValueError("series_map ne peut pas être vide.")

    columns_by_label = {}
    for label, series_id in series_map.items():
        logger.info("Chargement de la série FRED '%s' (%s)", label, series_id)
        columns_by_label[label] = load_fred_series(series_id, start_date)

    return pd.DataFrame(columns_by_label).sort_index()

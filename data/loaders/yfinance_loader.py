"""
Récupère les prix historiques pour tous les tickers définis dans config.yaml
via yfinance, et les sauvegarde en cache local sous data/raw/.

Responsabilités :
- Un seul point d'entrée pour toute donnée de prix ETF/FX depuis Yahoo Finance
- Gestion du cache pour éviter de re-télécharger à chaque run
- Retourne des DataFrames au format standard : index=date, colonnes=tickers

Ne fait PAS :
- L'alignement entre classes d'actifs (-> data_aligner.py)
- Le calcul de rendements ou de facteurs (-> factors/)
"""

import logging
from pathlib import Path

import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)

RAW_DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "raw" / "yfinance"


def _cache_path(ticker: str) -> Path:
    safe_ticker = ticker.replace("=", "_").replace("^", "_").replace("/", "_")
    return RAW_DATA_DIR / f"{safe_ticker}.csv"


def _read_cache(ticker: str) -> pd.Series | None:
    """Lit le cache d'un ticker s'il existe et est lisible, sinon None."""
    path = _cache_path(ticker)
    if not path.exists():
        return None
    try:
        cached = pd.read_csv(path, index_col=0, parse_dates=True)["close"]
    except (KeyError, pd.errors.EmptyDataError, ValueError) as exc:
        logger.warning("Cache illisible pour %s (%s), re-téléchargement", ticker, exc)
        return None
    return cached if not cached.empty else None


# Tolérance calendaire : start_date/end_date peuvent tomber un jour non ouvré
# (weekend, jour férié), auquel cas aucune donnée ne peut exister exactement
# à cette date. Sans cette marge, le cache serait jugé incomplet à chaque run.
_CACHE_TOLERANCE = pd.Timedelta(days=5)


def _cache_covers_range(cached: pd.Series, start: pd.Timestamp, end: pd.Timestamp) -> bool:
    return (cached.index.min() <= start + _CACHE_TOLERANCE
            and cached.index.max() >= end - _CACHE_TOLERANCE)


def _write_cache(ticker: str, series: pd.Series) -> None:
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    series.rename("close").to_csv(_cache_path(ticker))


def _download_ticker(ticker: str, start_date: str, end_date: str | None) -> pd.Series:
    """Télécharge la série de clôtures ajustées d'un ticker depuis yfinance."""
    logger.info("Téléchargement yfinance : %s (%s -> %s)", ticker, start_date, end_date or "aujourd'hui")
    try:
        raw = yf.download(
            ticker,
            start=start_date,
            end=end_date,
            auto_adjust=True,
            progress=False,
            threads=False,
        )
    except Exception as exc:
        raise RuntimeError(f"Échec du téléchargement yfinance pour '{ticker}': {exc}") from exc

    if raw is None or raw.empty:
        end_label = end_date or "aujourd'hui"
        raise ValueError(
            f"Aucune donnée retournée par yfinance pour le ticker '{ticker}' "
            f"sur la période {start_date} -> {end_label}. "
            "Vérifier que le symbole est valide."
        )

    close = raw["Close"]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    close = close.dropna()
    close.index.name = "date"
    close.name = ticker
    return close


def load_prices(tickers: list[str], start_date: str, end_date: str | None = None,
                 use_cache: bool = True) -> pd.DataFrame:
    """
    Télécharge les prix de clôture ajustés pour une liste de tickers.

    Args:
        tickers: liste de symboles yfinance (ex: ["SPY", "EWC"])
        start_date: format "YYYY-MM-DD"
        end_date: format "YYYY-MM-DD", None = aujourd'hui
        use_cache: si True, lit/écrit data/raw/ pour éviter les re-téléchargements

    Returns:
        DataFrame (index=date, colonnes=tickers) des prix de clôture ajustés
    """
    if not tickers:
        raise ValueError("La liste de tickers ne peut pas être vide.")

    start_ts = pd.Timestamp(start_date)
    end_ts = pd.Timestamp(end_date) if end_date else pd.Timestamp.today().normalize()
    if end_ts < start_ts:
        raise ValueError(f"end_date ({end_date}) est antérieure à start_date ({start_date}).")

    series_by_ticker: dict[str, pd.Series] = {}
    for ticker in tickers:
        cached = _read_cache(ticker) if use_cache else None
        if cached is not None and _cache_covers_range(cached, start_ts, end_ts):
            logger.info("Cache valide pour %s, téléchargement évité", ticker)
            series_by_ticker[ticker] = cached
            continue

        series = _download_ticker(ticker, start_date, end_date)
        if use_cache:
            _write_cache(ticker, series)
        series_by_ticker[ticker] = series

    prices = pd.DataFrame(series_by_ticker).sort_index()
    prices = prices.loc[(prices.index >= start_ts) & (prices.index <= end_ts)]
    return prices


def load_universe_from_config(config: dict) -> dict[str, pd.DataFrame]:
    """
    Charge les prix pour toutes les classes d'actifs définies dans config.yaml.

    Returns:
        dict {asset_class: DataFrame de prix}, ex: {"equities": df, "fx": df, ...}
    """
    universe = config.get("universe")
    if not universe:
        raise ValueError("config['universe'] est absent ou vide.")

    backtest_cfg = config.get("backtest", {})
    start_date = backtest_cfg.get("start_date")
    if not start_date:
        raise ValueError("config['backtest']['start_date'] est requis pour charger l'univers.")
    end_date = backtest_cfg.get("end_date")

    prices_by_asset_class = {}
    for asset_class, entries in universe.items():
        tickers = [entry["ticker"] for entry in entries]
        logger.info("Chargement de la classe d'actifs '%s' : %s", asset_class, tickers)
        prices_by_asset_class[asset_class] = load_prices(tickers, start_date, end_date)

    return prices_by_asset_class

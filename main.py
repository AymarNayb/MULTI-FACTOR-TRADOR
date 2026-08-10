"""
Point d'entrée : orchestre l'ensemble du pipeline de bout en bout.
Chaque étape appelle un module dédié -- ce fichier ne contient AUCUNE
logique métier, uniquement de l'orchestration. Toute la substance vit
dans les modules correspondants.
"""

from pathlib import Path

import yaml

from data.loaders.yfinance_loader import load_universe_from_config
from data.loaders.data_aligner import align_to_common_calendar

from factors.momentum import MomentumFactor
from factors.value import ValueFactor
from factors.low_vol import LowVolFactor

from backtest.engine import BacktestEngine
from backtest.metrics import summary_table

from analysis.regime_analysis import performance_by_regime
from analysis.plots import plot_cumulative_returns, plot_drawdown

# BacktestEngine produit des rendements journaliers (voir backtest/engine.py).
_PERIODS_PER_YEAR = 252

OUTPUT_DIR = Path(__file__).resolve().parent / "output"


def load_config(path: str = "config/config.yaml") -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    config = load_config()

    # 1. Données
    prices_by_class = load_universe_from_config(config)
    aligned_prices = align_to_common_calendar(prices_by_class)
    # Les facteurs et le moteur de backtest travaillent sur des colonnes
    # plates (ticker), pas sur le MultiIndex (asset_class, ticker) produit
    # par data_aligner.py.
    aligned_prices.columns = aligned_prices.columns.get_level_values(1)

    # 2. Facteurs. carry.py n'est pas branché ici : il nécessite un mapping
    # ticker -> série FRED (yields) par classe d'actif qui n'existe pas
    # encore dans config.yaml (voir factors/carry.py). Utilisable isolément
    # (voir tests), pas encore dans l'orchestration globale.
    factors = {
        "momentum": MomentumFactor(**config["factors"]["momentum"]),
        "value": ValueFactor(**config["factors"]["value"]),
        "low_vol": LowVolFactor(**config["factors"]["low_vol"]),
    }
    factor_scores = {name: f.compute(aligned_prices) for name, f in factors.items()}

    # 3. Backtest -- portfolio/rebalancer.py standardise et combine déjà les
    # facteurs en interne à chaque date de rebalancement : on lui passe les
    # scores bruts, pas des z-scores pré-calculés.
    engine = BacktestEngine(config)
    results = engine.run(aligned_prices, factor_scores)

    # 4. Évaluation
    print(summary_table(results["portfolio_return"], periods_per_year=_PERIODS_PER_YEAR))
    print(performance_by_regime(results["portfolio_return"]))

    OUTPUT_DIR.mkdir(exist_ok=True)
    plot_cumulative_returns(results["portfolio_return"]).savefig(OUTPUT_DIR / "cumulative_returns.png")
    plot_drawdown(results["portfolio_return"]).savefig(OUTPUT_DIR / "drawdown.png")


if __name__ == "__main__":
    main()

"""
Moteur de backtest walk-forward. C'est le module le plus sensible du projet :
toute erreur ici (look-ahead bias, mauvaise gestion des coûts) invalide
l'ensemble des résultats -- à tester en priorité (voir tests/test_backtest.py).

Principe walk-forward (par opposition à un simple split train/test) :
- Fenêtre d'entraînement glissante (ex: 5 ans) pour calibrer si besoin
- Fenêtre de test qui avance dans le temps (ex: 12 mois), jamais de retour
  en arrière
- Réentraînement/recalibrage périodique, simulant les conditions réelles
  où le modèle n'a jamais accès aux données futures
"""

import pandas as pd

from backtest.costs import transaction_costs
from portfolio.rebalancer import generate_target_weights

_FREQUENCY_ALIASES = {"monthly": "ME", "weekly": "W", "daily": "D"}


class BacktestEngine:
    def __init__(self, config: dict):
        self.config = config
        self.results: pd.DataFrame | None = None
        self._trading_calendar: pd.DatetimeIndex | None = None

    def run(self, prices: pd.DataFrame, factor_scores: dict) -> pd.DataFrame:
        """
        Exécute la simulation complète :
        1. Pour chaque date de rebalancement (walk-forward, jamais de fuite future)
        2. Génère les poids cibles (portfolio/rebalancer.py)
        3. Calcule les rendements réalisés entre deux rebalancements
        4. Applique les coûts de transaction (backtest/costs.py)
        5. Accumule la valeur du portefeuille dans le temps

        Returns:
            DataFrame avec au minimum : date, portfolio_return, portfolio_value,
            turnover, poids par actif
        """
        if prices.empty:
            raise ValueError("prices ne peut pas être vide.")
        if not factor_scores:
            raise ValueError("factor_scores ne peut pas être vide.")

        backtest_cfg = self.config.get("backtest", {})
        portfolio_cfg = self.config.get("portfolio", {})

        train_window_years = backtest_cfg.get("train_window_years", 5)
        cost_bps = backtest_cfg.get("transaction_cost_bps", 0.0)
        frequency = portfolio_cfg.get("rebalance_frequency", "monthly")

        sim_start = pd.Timestamp(backtest_cfg["start_date"]) if backtest_cfg.get("start_date") else prices.index.min()
        sim_end = pd.Timestamp(backtest_cfg["end_date"]) if backtest_cfg.get("end_date") else prices.index.max()

        # train_window_years = période de warmup avant le premier rebalancement,
        # pas un "entraînement" au sens ML (les facteurs sont des formules
        # déterministes, aucun paramètre à ajuster). Ça laisse le temps aux
        # facteurs à fenêtre longue (ex: value, 5 ans par défaut) de sortir de
        # leur période de warmup avant la toute première décision.
        first_possible_date = prices.index.min() + pd.DateOffset(years=train_window_years)
        effective_start = max(sim_start, first_possible_date)

        self._trading_calendar = prices.index
        rebalance_dates = self._rebalance_dates(effective_start, sim_end, frequency)
        if not rebalance_dates:
            raise ValueError(
                f"Aucune date de rebalancement entre {effective_start.date()} et {sim_end.date()} "
                f"(après {train_window_years} an(s) de warmup) : historique de prix insuffisant."
            )

        returns = prices.pct_change()
        rebalance_set = set(rebalance_dates)
        trading_days = prices.index[(prices.index >= rebalance_dates[0]) & (prices.index <= sim_end)]

        current_weights = pd.Series(0.0, index=prices.columns)
        portfolio_value = 1.0
        records = []

        for day in trading_days:
            # Le rendement du jour utilise TOUJOURS les poids décidés AVANT ce
            # jour, jamais ceux recalculés avec la clôture du jour même -- sinon
            # on utiliserait la donnée du jour pour "prédire" son propre
            # rendement (look-ahead bias).
            day_return = float((current_weights * returns.loc[day].reindex(prices.columns).fillna(0.0)).sum())
            portfolio_value *= (1.0 + day_return)

            turnover = 0.0
            if day in rebalance_set:
                returns_history = returns.loc[:day]
                target_weights = generate_target_weights(day, factor_scores, returns_history, self.config)
                target_weights = target_weights.reindex(prices.columns).fillna(0.0)

                turnover = float((target_weights - current_weights).abs().sum())
                cost = transaction_costs(current_weights, target_weights, cost_bps)
                portfolio_value *= (1.0 - cost)
                current_weights = target_weights

            records.append({
                "date": day,
                "portfolio_return": day_return,
                "portfolio_value": portfolio_value,
                "turnover": turnover,
                **{f"weight_{ticker}": w for ticker, w in current_weights.items()},
            })

        self.results = pd.DataFrame(records).set_index("date")
        return self.results

    def _rebalance_dates(self, start: str, end: str, frequency: str) -> list:
        """Génère les dates de rebalancement selon la fréquence configurée."""
        if frequency not in _FREQUENCY_ALIASES:
            raise ValueError(
                f"rebalance_frequency '{frequency}' non supportée. "
                f"Options : {sorted(_FREQUENCY_ALIASES)}."
            )
        if self._trading_calendar is None:
            raise RuntimeError(
                "_rebalance_dates() nécessite un calendrier de trading : appeler run() d'abord."
            )

        calendar = self._trading_calendar
        start_ts, end_ts = pd.Timestamp(start), pd.Timestamp(end)
        trading_days = calendar[(calendar >= start_ts) & (calendar <= end_ts)]
        if trading_days.empty:
            return []

        # Dernier jour de bourse RÉELLEMENT disponible de chaque période : on
        # ne génère jamais une date qui n'existe pas dans le calendrier (donc
        # pas de fill vers une date future qui n'a pas encore de prix).
        series = pd.Series(trading_days, index=trading_days)
        return series.resample(_FREQUENCY_ALIASES[frequency]).last().dropna().tolist()

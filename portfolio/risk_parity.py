"""
Transforme les convictions (signal composite de scoring.py) en poids de
portefeuille réels, en équilibrant la contribution au risque de chaque actif
plutôt que de pondérer naïvement par la force du signal.

Pourquoi : un signal fort sur un actif très volatil (ex: commodities) ne doit
pas dominer le portefeuille juste parce que son score est élevé -- sinon le
portefeuille est en réalité piloté par la volatilité, pas par la conviction.

Implémente deux approches :
- Inverse-volatility (simple, rapide) : poids proportionnel à 1/vol
- ERC (Equal Risk Contribution, plus rigoureux) : résout un problème
  d'optimisation pour que chaque actif contribue également à la variance
  totale du portefeuille, en tenant compte des corrélations
"""

import pandas as pd
import numpy as np
from scipy.optimize import minimize


def inverse_volatility_weights(returns: pd.DataFrame,
                                lookback_days: int = 63) -> pd.DataFrame:
    """
    Poids simples proportionnels à l'inverse de la volatilité réalisée.
    Ne tient pas compte des corrélations entre actifs (approximation rapide).
    """
    if returns.empty:
        raise ValueError("returns ne peut pas être vide.")
    if lookback_days < 2:
        raise ValueError("lookback_days doit être >= 2 pour estimer une volatilité.")

    # rolling() ne regarde que [t-lookback+1, t] : pas de look-ahead.
    realized_vol = returns.rolling(window=lookback_days, min_periods=lookback_days).std()
    inv_vol = 1.0 / realized_vol
    return inv_vol.div(inv_vol.sum(axis=1), axis=0)


def _solve_erc(cov: np.ndarray) -> np.ndarray:
    """Résout le problème ERC pour une matrice de covariance donnée (Maillard et al.)."""
    n = cov.shape[0]
    initial_guess = np.full(n, 1.0 / n)

    # L'ERC est invariant à un facteur d'échelle sur la covariance (les poids
    # optimaux ne changent pas), mais SLSQP a besoin d'un objectif bien
    # conditionné : sur des variances de rendements journaliers (~1e-4),
    # l'objectif brut (~1e-12) est sous le bruit numérique et l'optimiseur
    # déclare une convergence immédiate sans bouger de initial_guess.
    scale = 1.0 / np.mean(np.diag(cov))
    cov_scaled = cov * scale

    def objective(w):
        portfolio_var = w @ cov_scaled @ w
        marginal_contrib = cov_scaled @ w
        risk_contrib = w * marginal_contrib
        target = portfolio_var / n
        return np.sum((risk_contrib - target) ** 2)

    result = minimize(
        objective, initial_guess, method="SLSQP",
        bounds=[(0.0, 1.0)] * n,
        constraints=[{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}],
        options={"ftol": 1e-12, "maxiter": 1000},
    )
    if not result.success:
        raise RuntimeError(f"Optimisation ERC non convergée : {result.message}")
    return result.x


def equal_risk_contribution_weights(returns: pd.DataFrame,
                                     lookback_days: int = 63) -> pd.DataFrame:
    """
    Résout le problème d'optimisation ERC : trouve les poids tels que chaque
    actif contribue également à la variance totale du portefeuille.
    Utilise la matrice de covariance complète (pas juste les vols individuelles).
    """
    if returns.empty:
        raise ValueError("returns ne peut pas être vide.")
    if lookback_days < 2:
        raise ValueError("lookback_days doit être >= 2 pour estimer une matrice de covariance.")

    weights = pd.DataFrame(index=returns.index, columns=returns.columns, dtype=float)

    for t in range(lookback_days - 1, len(returns)):
        # Fenêtre [t-lookback+1, t] uniquement : pas de look-ahead.
        window = returns.iloc[t - lookback_days + 1: t + 1]
        if window.isna().any().any():
            continue  # fenêtre incomplète (warmup / données manquantes) -> pas de poids ce jour-là
        cov = window.cov().to_numpy()
        weights.iloc[t] = _solve_erc(cov)

    return weights


def scale_to_target_volatility(weights: pd.DataFrame, returns: pd.DataFrame,
                                target_volatility: float,
                                lookback_days: int = 63,
                                periods_per_year: int = 252) -> pd.DataFrame:
    """
    Scale les poids vers une volatilité de portefeuille annualisée cible :
    poids_finaux = poids * (vol_cible / vol_réalisée_du_portefeuille).

    La vol réalisée est estimée en appliquant les poids de chaque date (supposés
    constants sur la fenêtre) aux rendements de la fenêtre [date-lookback+1, date]
    qui précède immédiatement cette date -- même fenêtre que
    equal_risk_contribution_weights, donc pas de look-ahead. Sans cette étape,
    risk parity/ERC ne donne que des poids relatifs (somme <= 1) sans lien avec
    une exposition au risque absolue.
    """
    if weights.empty:
        raise ValueError("weights ne peut pas être vide.")
    if returns.empty:
        raise ValueError("returns ne peut pas être vide.")
    if target_volatility <= 0:
        raise ValueError(f"target_volatility doit être positif (obtenu {target_volatility}).")
    if lookback_days < 2:
        raise ValueError("lookback_days doit être >= 2 pour estimer une volatilité de portefeuille.")

    scaled = weights.copy()
    for date in weights.index:
        row = weights.loc[date]
        if row.isna().all():
            continue  # pas de poids à cette date (warmup) -> rien à scaler

        window = returns.loc[:date, row.index].tail(lookback_days)
        if len(window) < lookback_days or window.isna().any().any():
            continue  # fenêtre incomplète -> pas de scaling fiable ce jour-là

        portfolio_returns = window.to_numpy() @ row.fillna(0.0).to_numpy()
        realized_vol = portfolio_returns.std() * np.sqrt(periods_per_year)
        if realized_vol == 0.0:
            continue  # vol nulle (cas dégénéré) -> pas de scaling, poids inchangés

        scaled.loc[date] = row * (target_volatility / realized_vol)

    return scaled


def apply_conviction_tilt(risk_parity_weights: pd.DataFrame,
                           conviction_scores: pd.DataFrame,
                           tilt_strength: float = 0.5) -> pd.DataFrame:
    """
    Combine les poids risk parity (base neutre en risque) avec le signal
    composite de conviction (scoring.py) pour tilter le portefeuille vers
    les actifs à conviction plus forte, sans pour autant ignorer le risque.

    Args:
        tilt_strength: 0 = risk parity pur, 1 = conviction pure (non recommandé)
    """
    if risk_parity_weights.empty:
        raise ValueError("risk_parity_weights ne peut pas être vide.")
    if not 0.0 <= tilt_strength <= 1.0:
        raise ValueError(f"tilt_strength doit être dans [0, 1] (obtenu {tilt_strength}).")

    # Conviction manquante (actif/date absent de conviction_scores, ou warmup
    # d'un facteur) = tilt neutre (multiplicateur 1.0), pas de perte du poids
    # risk parity de base.
    aligned_conviction = conviction_scores.reindex(
        index=risk_parity_weights.index, columns=risk_parity_weights.columns
    ).fillna(0.0)

    tilt_multiplier = 1.0 + tilt_strength * aligned_conviction
    tilted = risk_parity_weights * tilt_multiplier

    # Renormalisation pour préserver l'exposition brute de la base risk parity
    # (le tilt redistribue le risque, il ne doit pas changer le levier global).
    gross_exposure = risk_parity_weights.abs().sum(axis=1)
    tilted_gross = tilted.abs().sum(axis=1)
    scale = gross_exposure.div(tilted_gross.replace(0.0, np.nan), axis=0)
    return tilted.mul(scale, axis=0)

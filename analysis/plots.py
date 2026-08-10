"""
Fonctions de visualisation standard pour l'évaluation de la stratégie.
Chaque fonction retourne une figure matplotlib, réutilisable en notebook
ou pour générer un rapport automatisé.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_cumulative_returns(returns: pd.Series, benchmark: pd.Series | None = None):
    """Courbe de performance cumulée, avec comparaison optionnelle à un benchmark."""
    if returns.empty:
        raise ValueError("returns ne peut pas être vide.")

    fig, ax = plt.subplots(figsize=(10, 5))
    cumulative = (1.0 + returns.dropna()).cumprod()
    ax.plot(cumulative.index, cumulative.values, label="Stratégie", color="tab:blue")

    if benchmark is not None:
        if benchmark.empty:
            raise ValueError("benchmark ne peut pas être vide s'il est fourni.")
        cumulative_bench = (1.0 + benchmark.dropna()).cumprod()
        ax.plot(cumulative_bench.index, cumulative_bench.values, label="Benchmark",
                color="tab:gray", linestyle="--")

    ax.set_title("Performance cumulée")
    ax.set_xlabel("Date")
    ax.set_ylabel("Valeur cumulée (base 1.0)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def plot_drawdown(returns: pd.Series):
    """Courbe de drawdown dans le temps (underwater plot)."""
    if returns.empty:
        raise ValueError("returns ne peut pas être vide.")

    cumulative = (1.0 + returns.dropna()).cumprod()
    running_max = cumulative.cummax()
    drawdown = cumulative / running_max - 1.0

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.fill_between(drawdown.index, drawdown.values, 0.0, color="tab:red", alpha=0.4)
    ax.plot(drawdown.index, drawdown.values, color="tab:red", linewidth=0.8)

    ax.set_title("Drawdown (underwater plot)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Drawdown")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def plot_factor_contribution(attribution_df: pd.DataFrame):
    """Barres empilées de contribution de chaque facteur à la performance totale."""
    if attribution_df.empty:
        raise ValueError("attribution_df ne peut pas être vide.")
    if "contribution_pct" not in attribution_df.columns:
        raise ValueError("attribution_df doit contenir une colonne 'contribution_pct' "
                          "(voir backtest.attribution.performance_attribution).")

    # Les lignes composites (multi_factor, naive_average_of_factors) ont un
    # contribution_pct=NaN par construction (performance_attribution) : elles
    # sont donc naturellement exclues ici, seuls les facteurs individuels restent.
    contributions = attribution_df["contribution_pct"].dropna()
    if contributions.empty:
        raise ValueError("Aucune contribution exploitable (contribution_pct entièrement NaN).")

    fig, ax = plt.subplots(figsize=(8, 2.5))
    left = 0.0
    for name, value in contributions.items():
        ax.barh(0, value, left=left, label=name)
        left += value

    ax.set_xlim(0, max(left, 1.0))
    ax.set_yticks([])
    ax.set_xlabel("Part de la performance totale")
    ax.set_title("Contribution de chaque facteur à la performance totale")
    ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5))
    fig.tight_layout()
    return fig


def plot_rolling_correlation_heatmap(correlation_df: pd.DataFrame):
    """Heatmap de la corrélation glissante entre facteurs."""
    if correlation_df.empty:
        raise ValueError("correlation_df ne peut pas être vide.")

    data = correlation_df.dropna(how="all").T  # paires en lignes, dates en colonnes
    if data.empty:
        raise ValueError("correlation_df ne contient aucune valeur exploitable (entièrement NaN).")

    fig, ax = plt.subplots(figsize=(12, max(2.0, 0.6 * len(data))))
    im = ax.imshow(data.to_numpy(dtype=float), aspect="auto", cmap="RdBu_r", vmin=-1.0, vmax=1.0)

    ax.set_yticks(range(len(data.index)))
    ax.set_yticklabels(data.index)

    n_dates = data.shape[1]
    n_ticks = min(10, n_dates)
    tick_positions = np.linspace(0, n_dates - 1, n_ticks).astype(int)
    ax.set_xticks(tick_positions)
    ax.set_xticklabels([pd.Timestamp(data.columns[i]).strftime("%Y-%m") for i in tick_positions],
                        rotation=45, ha="right")

    ax.set_title("Corrélation glissante entre facteurs")
    fig.colorbar(im, ax=ax, label="Corrélation")
    fig.tight_layout()
    return fig

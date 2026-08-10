# Factor Investing Multi-Actif

Stratégie systématique de factor investing appliquée à plusieurs classes d'actifs
(actions, taux, FX, commodities, crédit), construite avec pondération risk parity
et backtest walk-forward.

Trois facteurs actifs — **momentum**, **value** (proxy statistique), **low volatility**
— combinés en un signal composite, converti en poids de portefeuille par risk parity
(ERC), scalé vers une volatilité cible, contraint (levier, concentration), puis simulé
jour par jour sur un historique réel via yfinance/FRED.

## État du projet

Le pipeline complet (`data/` → `factors/` → `portfolio/` → `backtest/` → `analysis/`
→ `main.py`) est **implémenté et testé avec des données réelles** de bout en bout.
Dernier run complet (2015-01-30 → 2026-07-28, 16 tickers, rebalancement mensuel),
avec `target_volatility: 0.10` et `max_leverage: 2.75` :

| Métrique | Valeur |
|---|---|
| Rendement annualisé | 6.81% |
| Volatilité annualisée | 9.94% |
| Sharpe | 0.71 |
| Sortino | 0.95 |
| Calmar | 0.31 |
| Max drawdown | -22.09% |

Voir **Calibration du risk budget** ci-dessous pour l'historique des choix de
`target_volatility` / `max_leverage`, et **Limites connues** avant de tirer des
conclusions de ces chiffres — notamment l'absence de coût de financement du levier
(d'autant plus significative maintenant que le levier brut moyen tourne autour de
2.6-2.75) et le facteur carry non branché.

## Démarrage rapide

```bash
pip install -r requirements.txt
export FRED_API_KEY="votre_clé"   # gratuite : https://fred.stlouisfed.org/docs/api/api_key.html
python main.py
```

`main.py` télécharge (ou lit depuis le cache `data/raw/`) les prix de l'univers défini
dans `config/config.yaml`, calcule les facteurs, lance le backtest, affiche le tableau
de métriques + la performance par régime, et sauvegarde deux graphiques dans `output/`.

## Architecture

```
TRADOR/
├── config/
│   └── config.yaml           # Toute constante métier — jamais en dur dans le code
├── data/
│   ├── raw/                  # Cache local des téléchargements (non versionné)
│   ├── processed/            # Réservé aux données nettoyées (non utilisé actuellement)
│   └── loaders/
│       ├── yfinance_loader.py    # Prix ETF/FX, cache CSV par ticker
│       ├── fred_loader.py        # Séries macro FRED, clé via FRED_API_KEY
│       └── data_aligner.py       # Calendrier commun, lag de publication macro
├── factors/                  # Interface commune (factors/base.py) + 4 facteurs
│   ├── base.py
│   ├── momentum.py            # 12-1 mois, shift() pur (pas de look-ahead)
│   ├── value.py                # Proxy V1 : écart à la moyenne mobile 5 ans / vol
│   ├── carry.py                 # Implémenté mais PAS branché dans main.py (voir Limites)
│   └── low_vol.py
├── portfolio/
│   ├── scoring.py              # Z-score cross-sectionnel + combinaison pondérée
│   ├── risk_parity.py           # Inverse-vol, ERC (SLSQP), tilt de conviction,
│   │                             # scaling vers une volatilité cible
│   ├── constraints.py            # Levier max, caps individuel/classe (water-filling)
│   └── rebalancer.py              # Orchestration scoring → risk parity → scaling
│                                    # vol cible → contraintes
├── backtest/
│   ├── engine.py                # Simulation walk-forward jour par jour
│   ├── costs.py                  # Coûts de transaction (financing_costs non implémenté)
│   ├── metrics.py                 # Sharpe, Sortino, Calmar, max drawdown, etc.
│   └── attribution.py              # Backtest mono-facteur, contribution par facteur
├── analysis/
│   ├── regime_analysis.py          # Performance sur 4 régimes de stress historiques
│   ├── factor_correlation.py        # Corrélation glissante + par régime entre facteurs
│   └── plots.py                      # Courbes matplotlib réutilisables
├── notebooks/                 # Vide pour l'instant
├── tests/                      # Squelettes non implémentés (voir Limites)
├── output/                      # Graphiques générés par main.py
└── main.py                       # Point d'entrée, orchestration uniquement
```

## Flux de données (pipeline)

1. `data/loaders/` télécharge et met en cache les prix bruts (yfinance) et les
   séries macro (FRED)
2. `data/loaders/data_aligner.py` aligne les séries sur un calendrier commun
   (jours ouvrés) avec forward-fill contrôlé — jamais de valeur empruntée au futur
3. `factors/` calcule un score brut par facteur et par actif, uniquement à partir
   de données passées (vérifié explicitement par tests de non-lookahead)
4. `portfolio/scoring.py` standardise (z-score cross-sectionnel) et combine les
   facteurs en un signal composite (conviction) par actif
5. `portfolio/risk_parity.py` transforme la conviction en poids de portefeuille
   équilibrés en risque (ERC par défaut), applique un tilt de conviction, puis
   scale l'ensemble vers une volatilité de portefeuille cible
   (`scale_to_target_volatility`) — sans cette étape, l'ERC ne produit que des
   poids *relatifs* (somme ≈ 1), sans lien garanti avec le budget de risque
   défini par `max_leverage`
6. `portfolio/constraints.py` plafonne les positions individuelles et par classe
   d'actif, puis le levier brut — dernier rempart avant exécution, y compris si
   le scaling vers la vol cible pousse au-delà de `max_leverage`
7. `backtest/engine.py` simule la stratégie jour par jour, rebalance à la
   fréquence configurée, déduit les coûts de transaction
8. `backtest/metrics.py` et `attribution.py` évaluent la performance et la
   décomposent par facteur
9. `analysis/` produit les diagnostics (robustesse par régime, stabilité de la
   corrélation inter-facteurs dans le temps) et les visualisations

## Configuration

Tous les paramètres métier vivent dans `config/config.yaml` : univers, lookbacks
des facteurs, méthode de pondération, fenêtre de risk parity (`risk_parity_lookback_days`),
force du tilt de conviction (`tilt_strength`), **volatilité de portefeuille cible
(`target_volatility`)**, contraintes de risque (`max_leverage`, caps de position),
dates de backtest, coûts de transaction, mapping des séries FRED.

## Calibration du risk budget

`equal_risk_contribution_weights` (ERC) ne produit par construction que des poids
*relatifs* entre actifs (somme ≈ 1) — rien ne garantit que le portefeuille résultant
utilise le budget de risque autorisé par `max_leverage`. C'était le cas dans la
version initiale du pipeline : le portefeuille tournait à ~4% de vol annualisée quel
que soit `max_leverage`, faute d'étape de scaling vers une exposition absolue.

`scale_to_target_volatility` (`portfolio/risk_parity.py`) comble ce trou : les poids
finaux = poids ERC+tilt × (`target_volatility` / vol réalisée du portefeuille), estimée
sur la même fenêtre glissante que le reste du pipeline (`risk_parity_lookback_days`).
`apply_leverage_cap` reste appliqué *après*, comme garde-fou si ce scaling dépasse
`max_leverage`.

Deux paramètres ont été testés empiriquement pour caler ce nouveau mécanisme
(univers et période inchangés, voir tableau ci-dessus pour le run de référence) :

**`max_leverage`** — avec `target_volatility: 0.10` fixé, le cap de levier reste le
facteur limitant tant qu'il n'est pas assez haut pour laisser le scaling atteindre
sa cible :

| `max_leverage` | Rendement | Vol | Sharpe | Max DD | Cap actif (% jours) |
|---|---|---|---|---|---|
| 1.5 | 3.89% | 6.02% | 0.66 | -16.8% | 95% |
| 2.0 | 5.07% | 7.68% | 0.68 | -19.4% | 89% |
| 2.5 | 6.27% | 9.21% | 0.71 | -20.3% | 80% |
| **2.75 (retenu)** | **6.81%** | **9.94%** | **0.71** | **-22.1%** | 78% |
| 3.0 | 7.19% | 10.59% | 0.71 | -23.9% | 67% |

Le Sharpe et le Calmar plafonnent dès 2.5 (~0.71 / ~0.31) : au-delà, le levier
supplémentaire achète surtout du rendement brut et du drawdown, pas de meilleure
performance risque-ajustée. **2.75 est retenu comme le niveau minimal qui amène la
vol réalisée (9.94%) au niveau de la cible (10%)**, sans pousser plus loin que
nécessaire sur le plateau de Sharpe.

**`tilt_strength`** — hypothèse testée (et invalidée) : que `tilt_strength: 0.5`
diluerait un signal factoriel plus rentable en conviction pure. Résultat à 1.0
(pipeline sans scaling vol cible, pour comparaison directe avec la référence
2.66%/4%/0.67 d'avant l'ajout du scaling) : 2.58% / 3.99% / 0.66 / -11.8% — légèrement
*pire* que 0.5 sur toutes les métriques. La composante risk parity neutre (corrélations
via l'ERC) apporte donc de la valeur risque-ajustée ; `tilt_strength` reste à **0.5**.

## Limites connues

À lire avant d'interpréter les résultats ou d'étendre le projet.

- **Coût de financement du levier non modélisé.** `backtest/costs.py::financing_costs()`
  n'est pas implémenté (par choix, pour limiter le scope). Seuls les coûts de
  transaction (turnover × `transaction_cost_bps`) sont déduits dans `engine.py` —
  le coût d'emprunt pour maintenir une exposition >100% est absent. C'est
  particulièrement significatif maintenant : avec `max_leverage: 2.75`, le levier
  brut moyen réalisé tourne autour de 2.6-2.75 la majorité du temps (voir
  **Calibration du risk budget**), donc le rendement affiché surestime probablement
  la performance nette d'un coût de financement réel.
- **`max_leverage: 2.75` plafonne encore activement le scaling ~78% des jours.**
  La vol réalisée (9.94%) est proche mais légèrement en dessous de la cible (10%) :
  le régime "piloté uniquement par `target_volatility`" n'est pas encore pleinement
  atteint sur tout l'historique.
- **`factors/carry.py` n'est pas branché dans `main.py`.** Le facteur fonctionne et
  est testé isolément (nécessite un DataFrame `yields` déjà mappé par ticker), mais
  il n'existe pas encore de mapping ticker → série FRED par classe d'actif dans
  `config.yaml` pour l'alimenter automatiquement. `main.py` tourne avec momentum,
  value et low_vol uniquement.
- **Calendrier NYSE approximé par les jours ouvrés (Lun–Ven).** `data_aligner.py`
  n'exclut pas les jours fériés US réels — `pandas_market_calendars` n'est pas une
  dépendance du projet.
- **`value.py` est un proxy statistique, pas un facteur fondamental.** Écart à la
  moyenne mobile 5 ans normalisé par l'écart-type du prix (mean-reversion), faute
  de données P/E ou P/B (univers ETF uniquement en V1).
- **Les régimes GFC 2008 et taper tantrum 2013 ne sont pas couverts par défaut.**
  `config.yaml` démarre le backtest en 2010 avec 5 ans de warmup (`train_window_years`),
  donc la simulation ne commence qu'en 2015 — ces deux régimes retournent des
  métriques `NaN` (`n_observations=0`) tant que l'historique n'est pas étendu.
- **`apply_position_caps` redistribue en un seul passage, pas de façon itérative
  jusqu'à saturation complète.** Choix délibéré : un water-filling itératif complet
  peut faire converger tous les poids vers une distribution uniforme quand
  `cap × nb_actifs` s'approche de l'exposition totale, effaçant la conviction. La
  contrepartie : l'exposition brute finale peut être légèrement inférieure à la
  cible quand les contraintes sont très serrées.
- **`BacktestEngine._rebalance_dates` ne supporte que `monthly`, `weekly`, `daily`.**
  Pas de `quarterly` ni de fréquences personnalisées.
- **`test_window_months` (config.yaml) n'est pas utilisé.** Le walk-forward n'a pas
  de phase de ré-entraînement à proprement parler : les facteurs sont des formules
  déterministes, pas des modèles ajustés. `train_window_years` sert uniquement de
  période de warmup avant le premier rebalancement.
- **`tests/*.py` sont encore des squelettes (`raise NotImplementedError`).** Tout le
  pipeline a été validé manuellement avec des données réelles au fil du développement
  (voir l'historique de la session), mais aucun test automatisé n'est encore commité —
  pas de CI possible en l'état.
- **`notebooks/` est vide.**

## Principes de conception

- Chaque facteur hérite de `factors/base.py` (interface commune) pour permettre
  d'ajouter/retirer des facteurs sans toucher au reste du pipeline
- Séparation stricte signal (facteurs) / construction de portefeuille (risk parity)
  / évaluation (backtest) — aucune fuite de logique entre les couches
- Toute donnée de prix passe par `data_aligner.py` avant utilisation, pour éviter
  les biais de look-ahead liés à des calendriers désalignés entre classes d'actifs
- Erreurs explicites plutôt que silencieuses : données manquantes, contraintes
  inatteignables ou clé API absente lèvent une exception claire plutôt que de
  produire un résultat incorrect sans avertissement
- Toute constante métier vit dans `config.yaml`, jamais en dur dans le code

## Prochaines étapes

- Implémenter `financing_costs()` et l'intégrer à `backtest/engine.py` — priorité
  haute maintenant que le levier moyen tourne autour de 2.6-2.75
- Brancher `carry.py` dans `main.py` (mapping ticker → série FRED par classe d'actif)
- Écrire les tests dans `tests/*.py` (actuellement des squelettes) et mettre en place une CI
- Évaluer `pandas_market_calendars` pour un calendrier NYSE réel

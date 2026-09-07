"""Calibra K e vantagem de mando (HFA) do Elo contra as temporadas
historicas, em vez de deixa-los chutados. Roda uma vez (ou sempre que
quiser reconferir); o resultado alimenta as constantes em config.py que
src/elo.py usa para o Monte Carlo.

Metodologia (acordada em 2026-09-07):
- Rating inicial fixo em 1500, escala 400. So K e HFA sao otimizados
  por grid search: so a DIFERENCA entre ratings afeta a previsao, entao
  calibrar o valor inicial nao teria efeito.
- Avaliacao cronologica ("walk-forward"): os jogos de cada split sao
  processados em ordem de data, e o log-loss de cada jogo usa o rating
  ANTERIOR a ele. Nunca se usa o rating final da temporada pra prever
  um jogo do passado.
- 3 splits, um por temporada fechada (2023, 2024, 2025): a temporada de
  validacao entra por ultimo, com as outras duas processadas antes como
  aquecimento do rating (rating comeca do zero, 1500, em cada split).
  2026 fica de fora do grid search e so entra depois, uma unica vez,
  como conferencia final.
- Elo puro so da P(vitoria). A conversao pra 3 resultados usa o modelo
  de Davidson (1970), extensao classica do Bradley-Terry pra empates,
  com 1 parametro extra "nu" (forca do empate). nu e estimado por
  maxima verossimilhanca UMA VEZ, sobre a trajetoria baseline K=20/
  HFA=65, e mantido fixo durante o grid search de K/HFA -- nao vira uma
  terceira dimensao do grid. O update do rating em si NAO usa Davidson:
  continua o Elo padrao (S=1/0.5/0 pelo resultado real).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import PROCESSED, RAW  # noqa: E402

SCALE = 400.0
INITIAL_RATING = 1500.0
BASELINE_K = 20.0
BASELINE_HFA = 65.0
HISTORICAL_SEASONS = (2023, 2024, 2025)
HOLDOUT_SEASON = 2026
EPS = 1e-10

K_GRID = list(range(4, 61, 4))
HFA_GRID = list(range(0, 161, 10))


# --------------------------------------------------------------- dados

def load_played_matches() -> dict[int, list[tuple[int, int, int, int]]]:
    """Uma lista de (home_id, away_id, home_goals, away_goals) por temporada,
    ja ordenada por data. So partidas com placar preenchido (ver ingest.py
    sobre por que nao se usa o campo status pra decidir isso)."""
    df = pd.read_parquet(RAW / "matches.parquet")
    df = df[df["home_goals"].notna() & df["away_goals"].notna()].copy()
    df["date"] = pd.to_datetime(df["utc_date"])
    df = df.sort_values("date")

    by_season: dict[int, list[tuple[int, int, int, int]]] = {}
    for season, g in df.groupby("season"):
        by_season[int(season)] = list(
            zip(
                g["home_team_id"].astype(int),
                g["away_team_id"].astype(int),
                g["home_goals"].astype(int),
                g["away_goals"].astype(int),
            )
        )
    return by_season


# ----------------------------------------------------------------- elo

def expected_home(r_home: float, r_away: float, hfa: float) -> float:
    return 1.0 / (1.0 + 10 ** (-((r_home + hfa - r_away) / SCALE)))


def update_ratings(
    ratings: dict[int, float], home: int, away: int, hg: int, ag: int, k: float, hfa: float
) -> None:
    r_home = ratings.get(home, INITIAL_RATING)
    r_away = ratings.get(away, INITIAL_RATING)
    e_home = expected_home(r_home, r_away, hfa)
    s_home = 1.0 if hg > ag else (0.5 if hg == ag else 0.0)
    ratings[home] = r_home + k * (s_home - e_home)
    ratings[away] = r_away + k * ((1 - s_home) - (1 - e_home))


def davidson_probs(r_home: float, r_away: float, hfa: float, nu: float) -> tuple[float, float, float]:
    theta_h = 10 ** ((r_home + hfa) / SCALE)
    theta_a = 10 ** (r_away / SCALE)
    tie = nu * np.sqrt(theta_h * theta_a)
    denom = theta_h + theta_a + tie
    return theta_h / denom, tie / denom, theta_a / denom


def walk_forward(
    by_season: dict[int, list[tuple[int, int, int, int]]],
    warmup: list[int],
    eval_season: int,
    k: float,
    hfa: float,
) -> pd.DataFrame:
    """Processa aquecimento + temporada de avaliacao em sequencia, uma
    unica trajetoria de rating. Devolve so as linhas da temporada de
    avaliacao, com o rating PRE-jogo de cada uma."""
    ratings: dict[int, float] = {}
    for s in sorted(warmup):
        for home, away, hg, ag in by_season[s]:
            update_ratings(ratings, home, away, hg, ag, k, hfa)

    rows = []
    for home, away, hg, ag in by_season[eval_season]:
        r_home_pre = ratings.get(home, INITIAL_RATING)
        r_away_pre = ratings.get(away, INITIAL_RATING)
        outcome = "home" if hg > ag else ("draw" if hg == ag else "away")
        rows.append((r_home_pre, r_away_pre, outcome))
        update_ratings(ratings, home, away, hg, ag, k, hfa)

    return pd.DataFrame(rows, columns=["r_home_pre", "r_away_pre", "outcome"])


def three_splits(
    by_season: dict[int, list[tuple[int, int, int, int]]], k: float, hfa: float,
    seasons: tuple[int, ...] = HISTORICAL_SEASONS,
) -> dict[int, pd.DataFrame]:
    out = {}
    for val_season in seasons:
        warmup = [s for s in seasons if s != val_season]
        out[val_season] = walk_forward(by_season, warmup, val_season, k, hfa)
    return out


# ------------------------------------------------------- davidson (nu)

def _log_loss_from_probs(df: pd.DataFrame, p_home: np.ndarray, p_draw: np.ndarray, p_away: np.ndarray) -> float:
    p_actual = np.select(
        [df["outcome"] == "home", df["outcome"] == "draw", df["outcome"] == "away"],
        [p_home, p_draw, p_away],
    )
    p_actual = np.clip(p_actual, EPS, 1.0)
    return float(-np.log(p_actual).mean())


def fit_nu(rows: pd.DataFrame, hfa: float) -> float:
    """Maxima verossimilhanca de nu (forca do empate) por busca em grade
    fina -- sem depender de scipy, so pra evitar mais uma dependencia por
    um ajuste de 1 parametro escalar."""
    r_home = rows["r_home_pre"].to_numpy()
    r_away = rows["r_away_pre"].to_numpy()
    theta_h = 10 ** ((r_home + hfa) / SCALE)
    theta_a = 10 ** (r_away / SCALE)

    def nll(nu: float) -> float:
        tie = nu * np.sqrt(theta_h * theta_a)
        denom = theta_h + theta_a + tie
        p_home, p_draw, p_away = theta_h / denom, tie / denom, theta_a / denom
        return _log_loss_from_probs(rows, p_home, p_draw, p_away) * len(rows)

    coarse = np.linspace(0.01, 5.0, 200)
    losses = [nll(nu) for nu in coarse]
    best = coarse[int(np.argmin(losses))]

    fine = np.linspace(max(0.01, best - 0.05), best + 0.05, 200)
    losses_fine = [nll(nu) for nu in fine]
    return float(fine[int(np.argmin(losses_fine))])


def score_with_nu(rows: pd.DataFrame, hfa: float, nu: float) -> float:
    r_home = rows["r_home_pre"].to_numpy()
    r_away = rows["r_away_pre"].to_numpy()
    theta_h = 10 ** ((r_home + hfa) / SCALE)
    theta_a = 10 ** (r_away / SCALE)
    tie = nu * np.sqrt(theta_h * theta_a)
    denom = theta_h + theta_a + tie
    p_home, p_draw, p_away = theta_h / denom, tie / denom, theta_a / denom
    return _log_loss_from_probs(rows, p_home, p_draw, p_away)


# --------------------------------------------------------------- main

def main() -> None:
    by_season = load_played_matches()
    for s in (*HISTORICAL_SEASONS, HOLDOUT_SEASON):
        print(f"temporada {s}: {len(by_season.get(s, []))} partidas com placar")
    print()

    # 1) nu baseline, fixado a partir de K=20/HFA=65 (o baseline pedido).
    baseline_rows = three_splits(by_season, BASELINE_K, BASELINE_HFA)
    baseline_pooled = pd.concat(baseline_rows.values(), ignore_index=True)
    nu_baseline = fit_nu(baseline_pooled, BASELINE_HFA)
    print(f"nu baseline (K={BASELINE_K:.0f}, HFA={BASELINE_HFA:.0f}): {nu_baseline:.4f}")

    baseline_losses = [score_with_nu(baseline_rows[s], BASELINE_HFA, nu_baseline) for s in HISTORICAL_SEASONS]
    print(f"log-loss Elo baseline (sem calibrar) por split: {[round(x, 4) for x in baseline_losses]}")
    print(f"  media {np.mean(baseline_losses):.4f} +/- {np.std(baseline_losses):.4f}")
    print()

    # 2) grid search de K e HFA, nu fixo em nu_baseline.
    print(f"grid search: {len(K_GRID)} x {len(HFA_GRID)} = {len(K_GRID) * len(HFA_GRID)} combinacoes")
    best = None
    for k in K_GRID:
        for hfa in HFA_GRID:
            splits = three_splits(by_season, k, hfa)
            losses = [score_with_nu(splits[s], hfa, nu_baseline) for s in HISTORICAL_SEASONS]
            mean_loss = float(np.mean(losses))
            if best is None or mean_loss < best[0]:
                best = (mean_loss, k, hfa, losses)

    best_loss, best_k, best_hfa, best_losses = best
    print(f"melhor combinacao: K={best_k}, HFA={best_hfa}")
    print(f"log-loss por split: {[round(x, 4) for x in best_losses]}")
    print(f"  media {np.mean(best_losses):.4f} +/- {np.std(best_losses):.4f}")
    print()

    # 2b) grid completo, pra mapa de calor e checagem de plato.
    grid = np.zeros((len(K_GRID), len(HFA_GRID)))
    for i, k in enumerate(K_GRID):
        for j, hfa in enumerate(HFA_GRID):
            splits = three_splits(by_season, k, hfa)
            losses = [score_with_nu(splits[s], hfa, nu_baseline) for s in HISTORICAL_SEASONS]
            grid[i, j] = np.mean(losses)
    plot_heatmap(grid)

    print("=== K=20/HFA=65 vs K={}/HFA={}, split a split ===".format(best_k, best_hfa))
    for s, l_base, l_best in zip(HISTORICAL_SEASONS, baseline_losses, best_losses):
        diff = l_base - l_best
        print(f"  {s}: baseline {l_base:.4f}  |  calibrado {l_best:.4f}  |  ganho {diff:+.4f}")
    venceu_nos_tres = all(b > c for b, c in zip(baseline_losses, best_losses))
    print(f"  calibrado ganhou nos 3 splits: {venceu_nos_tres}")
    print()

    # 2c) HFA estimado direto dos dados (sem Elo): pontos em casa vs fora.
    print("=== HFA estimado direto dos dados (2023-2025) ===")
    all_games = [g for s in HISTORICAL_SEASONS for g in by_season[s]]
    n = len(all_games)
    home_wins = sum(1 for _, _, hg, ag in all_games if hg > ag)
    draws = sum(1 for _, _, hg, ag in all_games if hg == ag)
    away_wins = n - home_wins - draws
    home_pts = 3 * home_wins + draws
    away_pts = 3 * away_wins + draws
    print(f"  pontos (regra 3-1-0): casa {home_pts} ({100 * home_pts / (home_pts + away_pts):.1f}%) "
          f"vs fora {away_pts} ({100 * away_pts / (home_pts + away_pts):.1f}%)")
    s_home_avg = (home_wins + 0.5 * draws) / n
    hfa_implied = 400 * np.log10(s_home_avg / (1 - s_home_avg))
    print(f"  score esperado medio em casa (convencao Elo, empate=0.5): {s_home_avg:.4f}")
    print(f"  HFA implicado (400*log10(S/(1-S))): {hfa_implied:.1f}")
    print(f"  comparar com HFA do grid search: {best_hfa}")
    print()

    # 3) reestima nu na trajetoria final, checa se mudou muito.
    final_rows = three_splits(by_season, best_k, best_hfa)
    final_pooled = pd.concat(final_rows.values(), ignore_index=True)
    nu_final = fit_nu(final_pooled, best_hfa)
    rel_change = abs(nu_final - nu_baseline) / nu_baseline
    print(f"nu reestimado em K={best_k}/HFA={best_hfa}: {nu_final:.4f} "
          f"(baseline {nu_baseline:.4f}, mudanca relativa {100 * rel_change:.1f}%)")
    if rel_change > 0.25:
        print(
            "AVISO: nu mudou mais de 25% entre o baseline e a trajetoria final. "
            "A independencia entre nu e (K, HFA) que assumimos no grid search "
            "pode nao valer aqui -- confira antes de seguir para o Monte Carlo."
        )
    else:
        print("mudanca pequena: tratar nu como parametro de nuisance foi valido.")
    print()

    # 4) baselines: frequencia historica vs Elo fixo vs Elo calibrado.
    freq_losses = []
    for val_season in HISTORICAL_SEASONS:
        warmup_seasons = [s for s in HISTORICAL_SEASONS if s != val_season]
        warmup_outcomes = []
        for s in warmup_seasons:
            for _, _, hg, ag in by_season[s]:
                warmup_outcomes.append("home" if hg > ag else ("draw" if hg == ag else "away"))
        warmup_outcomes = pd.Series(warmup_outcomes)
        rates = warmup_outcomes.value_counts(normalize=True)
        p_home = rates.get("home", 0.0)
        p_draw = rates.get("draw", 0.0)
        p_away = rates.get("away", 0.0)

        val_outcomes = pd.Series(
            ["home" if hg > ag else ("draw" if hg == ag else "away") for _, _, hg, ag in by_season[val_season]]
        )
        p_actual = np.select(
            [val_outcomes == "home", val_outcomes == "draw", val_outcomes == "away"],
            [p_home, p_draw, p_away],
        )
        p_actual = np.clip(p_actual, EPS, 1.0)
        freq_losses.append(float(-np.log(p_actual).mean()))

    print("=== comparacao de log-loss (media +/- desvio entre os 3 splits) ===")
    print(f"  frequencia historica (casa/empate/fora fixos): {np.mean(freq_losses):.4f} +/- {np.std(freq_losses):.4f}")
    print(f"  Elo fixo K={BASELINE_K:.0f}/HFA={BASELINE_HFA:.0f} (sem calibrar): {np.mean(baseline_losses):.4f} +/- {np.std(baseline_losses):.4f}")
    print(f"  Elo calibrado K={best_k}/HFA={best_hfa}: {np.mean(best_losses):.4f} +/- {np.std(best_losses):.4f}")
    if np.mean(best_losses) < min(np.mean(freq_losses), np.mean(baseline_losses)):
        print("  -> calibracao venceu os dois baselines.")
    else:
        print("  -> AVISO: calibracao NAO venceu os dois baselines. Nao valeu a pena.")
    print()

    # 5) taxa de empate prevista vs real, por temporada.
    print("=== taxa de empate: prevista (media do modelo) vs real, por temporada ===")
    for val_season in HISTORICAL_SEASONS:
        rows = final_rows[val_season]
        p_home, p_draw, p_away = davidson_probs(
            rows["r_home_pre"].to_numpy(), rows["r_away_pre"].to_numpy(), best_hfa, nu_final
        )
        prevista = float(np.mean(p_draw))
        real = float((rows["outcome"] == "draw").mean())
        print(f"  {val_season}: prevista {100 * prevista:.1f}%  |  real {100 * real:.1f}%")
    print()

    # 6) grafico de calibracao (reliability diagram), pooled nos 3 splits.
    plot_calibration(final_pooled, best_hfa, nu_final)

    # 7) 2026 como conferencia final, uma unica vez.
    holdout_rows = walk_forward(by_season, list(HISTORICAL_SEASONS), HOLDOUT_SEASON, best_k, best_hfa)
    holdout_loss = score_with_nu(holdout_rows, best_hfa, nu_final)
    p_home, p_draw, p_away = davidson_probs(
        holdout_rows["r_home_pre"].to_numpy(), holdout_rows["r_away_pre"].to_numpy(), best_hfa, nu_final
    )
    print(f"=== conferencia final em 2026 ({len(holdout_rows)} partidas ja disputadas) ===")
    print(f"  log-loss: {holdout_loss:.4f}")
    print(f"  taxa de empate prevista {100 * float(np.mean(p_draw)):.1f}% "
          f"vs real {100 * float((holdout_rows['outcome'] == 'draw').mean()):.1f}%")
    print()

    print(f"PARAMETROS FINAIS: K={best_k}, HFA={best_hfa}, nu={nu_final:.4f}, rating_inicial={INITIAL_RATING:.0f}, escala={SCALE:.0f}")


def plot_heatmap(grid: np.ndarray) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(grid, aspect="auto", origin="lower", cmap="viridis_r")
    ax.set_xticks(range(len(HFA_GRID)))
    ax.set_xticklabels(HFA_GRID, rotation=45)
    ax.set_yticks(range(len(K_GRID)))
    ax.set_yticklabels(K_GRID)
    ax.set_xlabel("HFA")
    ax.set_ylabel("K")
    ax.set_title("log-loss medio (3 splits) por combinacao K/HFA")
    fig.colorbar(im, ax=ax, label="log-loss medio")

    i_min, j_min = np.unravel_index(np.argmin(grid), grid.shape)
    ax.scatter([j_min], [i_min], marker="*", s=200, color="red", label="minimo do grid")
    ax.legend(loc="upper right")
    fig.tight_layout()

    PROCESSED.mkdir(parents=True, exist_ok=True)
    out = PROCESSED / "grid_heatmap_elo.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"mapa de calor do grid salvo em {out}")
    print(f"  minimo: log-loss={grid.min():.4f} em K={K_GRID[i_min]}/HFA={HFA_GRID[j_min]}")
    print(f"  faixa do grid: min {grid.min():.4f}, max {grid.max():.4f}, desvio {grid.std():.4f}")
    print()


def plot_calibration(rows: pd.DataFrame, hfa: float, nu: float) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    p_home, p_draw, p_away = davidson_probs(
        rows["r_home_pre"].to_numpy(), rows["r_away_pre"].to_numpy(), hfa, nu
    )
    outcome = rows["outcome"].to_numpy()

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), sharey=True)
    specs = [("home", p_home, "vitoria mandante"), ("draw", p_draw, "empate"), ("away", p_away, "vitoria visitante")]
    bins = np.linspace(0, 1, 11)

    for ax, (label, probs, title) in zip(axes, specs):
        actual = (outcome == label).astype(float)
        bin_idx = np.digitize(probs, bins) - 1
        bin_idx = np.clip(bin_idx, 0, 9)
        xs, ys, ns = [], [], []
        for b in range(10):
            mask = bin_idx == b
            if mask.sum() == 0:
                continue
            xs.append(probs[mask].mean())
            ys.append(actual[mask].mean())
            ns.append(mask.sum())
        ax.plot([0, 1], [0, 1], "--", color="gray", linewidth=1, label="perfeito")
        ax.scatter(xs, ys, s=[20 + n for n in ns], alpha=0.8)
        ax.set_title(title)
        ax.set_xlabel("probabilidade prevista")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
    axes[0].set_ylabel("frequencia real")
    axes[0].legend(loc="upper left", fontsize=8)
    fig.suptitle("Calibracao do Elo+Davidson (out-of-sample, 2023-2025 pooled)")
    fig.tight_layout()

    PROCESSED.mkdir(parents=True, exist_ok=True)
    out = PROCESSED / "calibracao_elo.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"grafico de calibracao salvo em {out}")
    print()


if __name__ == "__main__":
    main()

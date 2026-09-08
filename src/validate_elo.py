"""Validacao permanente do motor de Elo -- rode de novo sempre que K,
HFA, nu ou o bootstrap de placar mudarem em config.py/elo.py.

Simula uma temporada fechada (2025) INTEIRA a partir da rodada 0,
aquecendo o rating so em temporadas anteriores (2023-2024, sem
vazamento -- 2025 nunca entra no aquecimento), e compara PONTOS e
saldo de gols simulados com os reais. Uma replicacao isolada e uma
amostra: aqui cada uma das ELO_N_SIMULATIONS (10 mil) e uma replicacao
independente da temporada inteira, e comparamos a MEDIA e o INTERVALO
entre replicacoes -- nao um unico sorteio -- contra o unico valor real
observado.

Resultado de 2026-09-07 (documentado no README, secao "Limitacoes
conhecidas"): pontos batem com as 3 temporadas fechadas (2023-2025),
sem compressao. Saldo de gols simulado saiu com ~62% da dispersao real
-- titulo/G4/Z4 vem de pontos (Davidson), nao sao afetados; so o
desempate por saldo em empates exatos de pontos e vitorias fica menos
preciso. Ver backlog do README para a correcao futura (bootstrap
ponderado por kernel na diferenca de Elo, nao implementada).

IMPORTANTE -- consumidores manuais deste output, que NAO atualizam
sozinhos quando ele muda (nenhum dos dois le este script, so o texto
que alguem copiou daqui uma vez):
  1. README.md, secao "Limitacoes conhecidas" (a tabela de desvio
     simulado vs real).
  2. fora-da-sumula-v3.html, pagina "Como funciona" -> passo 4 ->
     objeto `const V = {...}` (busque por "NUMEROS FIXOS" no arquivo).
Depois de rodar este script de novo (por causa de K/HFA/nu ou do
bootstrap de placar terem mudado), atualize os dois a mao.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import ELO_DRAW_NU, ELO_HFA, ELO_K, ELO_N_SIMULATIONS, RAW  # noqa: E402
from elo import build_score_pools, load_matches, simulate_remainder, warm_up_ratings  # noqa: E402

VALIDATION_SEASON = 2025
WARMUP_SEASONS = [2023, 2024]


def _replication_stats(arr: np.ndarray) -> dict[str, np.ndarray]:
    """arr tem shape (n_sims, n_teams): uma replicacao de temporada por
    linha. Calcula, PARA CADA replicacao, o desvio entre clubes, os
    percentis, o campeao e o lanterna -- depois devolve as series (uma
    por replicacao) pra quem chamar agregar (media + intervalo)."""
    return {
        "std": arr.std(axis=1),
        "pct": np.percentile(arr, [5, 25, 50, 75, 95], axis=1),  # shape (5, n_sims)
        "champion": arr.max(axis=1),
        "last": arr.min(axis=1),
    }


def _print_replication_comparison(label: str, sim_stats: dict[str, np.ndarray], real: np.ndarray) -> float:
    std_mean, std_p5, std_p95 = sim_stats["std"].mean(), *np.percentile(sim_stats["std"], [5, 95])
    pct_mean = sim_stats["pct"].mean(axis=1)  # media, por nivel de percentil, entre as replicacoes
    champ_mean, champ_p5, champ_p95 = sim_stats["champion"].mean(), *np.percentile(sim_stats["champion"], [5, 95])
    last_mean, last_p5, last_p95 = sim_stats["last"].mean(), *np.percentile(sim_stats["last"], [5, 95])

    print(f"=== {label}: media entre replicacoes (intervalo 5-95%) vs real (1 temporada) ===")
    print(
        f"  desvio entre os 20 clubes -- simulado: {std_mean:.1f} [{std_p5:.1f}, {std_p95:.1f}]  "
        f"|  real: {real.std():.1f}"
    )
    qs = [5, 25, 50, 75, 95]
    print(f"  percentis {qs} entre clubes -- media das replicacoes: {np.round(pct_mean, 1)}")
    print(f"  percentis {qs} entre clubes -- real:                  {np.round(np.percentile(real, qs), 1)}")
    print(f"  campeao -- simulado: {champ_mean:.1f} [{champ_p5:.1f}, {champ_p95:.1f}]  |  real: {real.max():.0f}")
    print(f"  lanterna -- simulado: {last_mean:.1f} [{last_p5:.1f}, {last_p95:.1f}]  |  real: {real.min():.0f}")
    ratio = std_mean / real.std()
    print(f"  razao desvio medio simulado/real: {ratio:.2f}")
    print()
    return ratio


def validate() -> None:
    df = load_matches()
    pools = build_score_pools(df)

    season_matches = df[df["season"] == VALIDATION_SEASON].copy()
    teams = sorted(
        int(t) for t in pd.unique(pd.concat([season_matches["home_team_id"], season_matches["away_team_id"]]))
    )

    base_ratings = warm_up_ratings(df, WARMUP_SEASONS, ELO_K, ELO_HFA)
    novos = [t for t in teams if t not in base_ratings]
    print(
        f"aquecimento: {WARMUP_SEASONS} | {len(base_ratings)} times com rating herdado, "
        f"{len(novos)} novos em {VALIDATION_SEASON} comecando em 1500 (promovidos)"
    )
    print()

    stats0 = {t: {"points": 0.0, "wins": 0.0, "goal_diff": 0.0, "goals_for": 0.0} for t in teams}
    remaining = season_matches.sort_values("date")

    _, final = simulate_remainder(
        base_ratings, stats0, remaining, teams, pools, ELO_K, ELO_HFA, ELO_DRAW_NU, ELO_N_SIMULATIONS
    )

    real = pd.read_parquet(RAW / "standings.parquet")
    real_season = real[real["season"] == VALIDATION_SEASON]

    print(f"({final['points'].shape[0]} replicacoes independentes de {VALIDATION_SEASON}, a partir da rodada 0)")
    print()
    _print_replication_comparison("PONTOS", _replication_stats(final["points"]), real_season["points"].to_numpy())
    ratio_gd = _print_replication_comparison(
        "SALDO DE GOLS", _replication_stats(final["goal_diff"]), real_season["goal_difference"].to_numpy()
    )
    if ratio_gd < 0.8:
        print(
            "AVISO (saldo de gols): desvio simulado bem menor que o real -- "
            "bootstrap pode estar achatando a variancia."
        )
        print()


if __name__ == "__main__":
    validate()

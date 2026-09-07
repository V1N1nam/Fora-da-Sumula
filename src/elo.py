"""Motor de Elo (rating cronologico, sem vazamento) + Monte Carlo do
restante da temporada corrente. Usa os parametros calibrados em
config.py -- ver src/calibrate_elo.py para o porque de K=20/HFA=65 em
vez do argmin do grid (K=32/HFA=110): o mapa de calor mostrou platou e o
ganho nao passava do ruido entre splits.

Sorteio de placar dentro da simulacao (decidido em 2026-09-07): o
Elo+Davidson so da resultado (casa/empate/fora), nao gols. Em vez de um
modelo de gols novo -- o que reintroduziria a complexidade tipo-xG que a
gente descartou de proposito ao trocar de fonte --, o placar de cada
partida simulada e sorteado por bootstrap dos placares REAIS de
temporadas FECHADAS (2023-2025; 2026 fica de fora do balde por estar
incompleta -- vies de calendario sem ganho, e 1140 jogos ja e amostra
suficiente), condicionado so ao resultado sorteado, com reposicao e
independente por jogo dentro de cada simulacao (nada de reamostrar o
mesmo placar em bloco). O vies existe -- uma goleada fica
descorrelacionada da forca real dos times, ja que o balde e global, nao
por faixa de diferenca de Elo -- mas e simetrico (nao favorece nenhum
lado) e se dilui em 10 mil simulacoes. Fatiar por faixa de Elo deixaria
baldes pequenos demais com so 1140 jogos historicos.

Validado em 2026-09-07 (ver src/validate_elo.py, rode de novo sempre
que o modelo mudar): pontos batem com as 3 temporadas fechadas
(2023-2025), sem compressao. Saldo de gols simulado tem ~62% da
dispersao real -- titulo/G4/Z4 vem de pontos (Davidson), nao sao
afetados; so o desempate por saldo em empates exatos de pontos e
vitorias fica menos preciso. Detalhes e numeros completos no README,
secao "Limitacoes conhecidas". Correcao futura registrada no backlog
do README (bootstrap ponderado por kernel na diferenca de Elo),
NAO implementada.

O update do rating usa Elo padrao (S=1/0.5/0 pelo resultado sorteado,
E=logistico). Davidson entra SO na previsao de 3 vias (sorteio do
resultado), nunca no update.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import (  # noqa: E402
    CURRENT_SEASON, ELO_DRAW_NU, ELO_HFA, ELO_INITIAL_RATING, ELO_K,
    ELO_N_SIMULATIONS, PROCESSED, RAW,
)
from calibrate_elo import davidson_probs, expected_home, update_ratings  # noqa: E402

RNG = np.random.default_rng(20260907)

TIEBREAK_SEASONS = (2023, 2024, 2025)  # temporadas fechadas p/ o balde de placares


# --------------------------------------------------------------- dados

def load_matches() -> pd.DataFrame:
    df = pd.read_parquet(RAW / "matches.parquet")
    df["date"] = pd.to_datetime(df["utc_date"])
    return df.sort_values("date").reset_index(drop=True)


def build_score_pools(df: pd.DataFrame) -> dict[str, np.ndarray]:
    closed = df[
        df["season"].isin(TIEBREAK_SEASONS) & df["home_goals"].notna() & df["away_goals"].notna()
    ]
    hg = closed["home_goals"].to_numpy(dtype=int)
    ag = closed["away_goals"].to_numpy(dtype=int)
    return {
        "home": np.column_stack([hg[hg > ag], ag[hg > ag]]),
        "draw": np.column_stack([hg[hg == ag], ag[hg == ag]]),
        "away": np.column_stack([hg[hg < ag], ag[hg < ag]]),
    }


def warm_up_ratings(df: pd.DataFrame, seasons: list[int], k: float, hfa: float) -> dict[int, float]:
    ratings: dict[int, float] = {}
    for s in sorted(seasons):
        g = df[(df["season"] == s) & df["home_goals"].notna() & df["away_goals"].notna()]
        for home, away, hg, ag in zip(g["home_team_id"], g["away_team_id"], g["home_goals"], g["away_goals"]):
            update_ratings(ratings, int(home), int(away), int(hg), int(ag), k, hfa)
    return ratings


def played_through_round(
    season_matches: pd.DataFrame, round_no: int, k: float, hfa: float, base_ratings: dict[int, float]
) -> tuple[dict[int, float], dict[int, dict[str, float]]]:
    """Ratings e estatisticas reais (pontos/vitorias/saldo/gols pro) apos
    as partidas jogadas com rodada <= round_no. Recomeca de base_ratings
    a cada chamada -- redundante, mas barato (algumas centenas de jogos)
    e evita ambiguidade de ordenacao entre rodada e data real."""
    ratings = dict(base_ratings)
    teams = pd.unique(pd.concat([season_matches["home_team_id"], season_matches["away_team_id"]]))
    stats = {int(t): {"points": 0.0, "wins": 0.0, "goal_diff": 0.0, "goals_for": 0.0} for t in teams}

    played = season_matches[
        (season_matches["matchday"] <= round_no)
        & season_matches["home_goals"].notna()
        & season_matches["away_goals"].notna()
    ].sort_values("date")

    for home, away, hg, ag in zip(
        played["home_team_id"], played["away_team_id"], played["home_goals"], played["away_goals"]
    ):
        home, away, hg, ag = int(home), int(away), int(hg), int(ag)
        update_ratings(ratings, home, away, hg, ag, k, hfa)
        stats[home]["goals_for"] += hg
        stats[home]["goal_diff"] += hg - ag
        stats[away]["goals_for"] += ag
        stats[away]["goal_diff"] += ag - hg
        if hg > ag:
            stats[home]["points"] += 3
            stats[home]["wins"] += 1
        elif hg < ag:
            stats[away]["points"] += 3
            stats[away]["wins"] += 1
        else:
            stats[home]["points"] += 1
            stats[away]["points"] += 1

    return ratings, stats


# ------------------------------------------------------- monte carlo

def simulate_remainder(
    ratings: dict[int, float],
    stats: dict[int, dict[str, float]],
    remaining: pd.DataFrame,
    teams: list[int],
    pools: dict[str, np.ndarray],
    k: float,
    hfa: float,
    nu: float,
    n_sims: int,
) -> tuple[dict[int, dict[str, float]], dict[str, np.ndarray]]:
    """Simula o restante da temporada n_sims vezes. Ratings e placar
    evoluem DENTRO de cada simulacao conforme os resultados sorteados
    (nada de rating estatico)."""
    n_teams = len(teams)
    idx = {t: i for i, t in enumerate(teams)}

    rating_mat = np.tile(np.array([ratings.get(t, ELO_INITIAL_RATING) for t in teams]), (n_sims, 1))
    points = np.tile(np.array([stats[t]["points"] for t in teams]), (n_sims, 1))
    wins = np.tile(np.array([stats[t]["wins"] for t in teams]), (n_sims, 1))
    goal_diff = np.tile(np.array([stats[t]["goal_diff"] for t in teams]), (n_sims, 1))
    goals_for = np.tile(np.array([stats[t]["goals_for"] for t in teams]), (n_sims, 1))

    fixtures = list(zip(remaining["home_team_id"].astype(int), remaining["away_team_id"].astype(int)))

    for home, away in fixtures:
        hi, ai = idx[home], idx[away]
        r_home, r_away = rating_mat[:, hi], rating_mat[:, ai]

        e_home = expected_home(r_home, r_away, hfa)
        p_home, p_draw, _ = davidson_probs(r_home, r_away, hfa, nu)

        u = RNG.random(n_sims)
        is_home = u < p_home
        is_draw = (~is_home) & (u < p_home + p_draw)
        is_away = ~(is_home | is_draw)

        hg = np.empty(n_sims, dtype=int)
        ag = np.empty(n_sims, dtype=int)
        for mask, pool in ((is_home, pools["home"]), (is_draw, pools["draw"]), (is_away, pools["away"])):
            n = int(mask.sum())
            if n == 0:
                continue
            picks = RNG.integers(0, len(pool), size=n)
            hg[mask] = pool[picks, 0]
            ag[mask] = pool[picks, 1]

        s_home = np.where(is_home, 1.0, np.where(is_draw, 0.5, 0.0))

        points[:, hi] += np.where(is_home, 3, np.where(is_draw, 1, 0))
        points[:, ai] += np.where(is_away, 3, np.where(is_draw, 1, 0))
        wins[:, hi] += is_home
        wins[:, ai] += is_away
        goal_diff[:, hi] += hg - ag
        goal_diff[:, ai] += ag - hg
        goals_for[:, hi] += hg
        goals_for[:, ai] += ag

        # update padrao de Elo (S=1/0.5/0, E=logistico) -- Davidson so
        # entrou acima, na hora de sortear o resultado.
        rating_mat[:, hi] += k * (s_home - e_home)
        rating_mat[:, ai] -= k * (s_home - e_home)

    # criterios de desempate do Brasileirao, nessa ordem: pontos,
    # vitorias, saldo de gols, gols pro. (Head-to-head e disciplina
    # ficam de fora -- nao da pra calcular com os dados disponiveis.)
    score = points * 1e9 + wins * 1e6 + (goal_diff + 1000) * 1e3 + goals_for
    order = np.argsort(-score, axis=1)
    rank = np.empty_like(order)
    rows = np.arange(n_sims)[:, None]
    rank[rows, order] = np.arange(1, n_teams + 1)[None, :]

    probs = {}
    for t in teams:
        i = idx[t]
        probs[t] = {
            "titulo_prob": float(np.mean(rank[:, i] == 1)),
            "g4_prob": float(np.mean(rank[:, i] <= 4)),
            "z4_prob": float(np.mean(rank[:, i] >= n_teams - 3)),
        }

    final = {"points": points, "wins": wins, "goal_diff": goal_diff, "goals_for": goals_for, "teams": teams}
    return probs, final


# ------------------------------------------------- temporada corrente

def run_current_season(df: pd.DataFrame, pools: dict[str, np.ndarray]) -> tuple[pd.DataFrame, pd.DataFrame]:
    season_matches = df[df["season"] == CURRENT_SEASON].copy()
    teams = sorted(
        int(t) for t in pd.unique(pd.concat([season_matches["home_team_id"], season_matches["away_team_id"]]))
    )
    team_names: dict[int, str] = {}
    for _, row in season_matches.iterrows():
        team_names[int(row["home_team_id"])] = row["home_team"]
        team_names[int(row["away_team_id"])] = row["away_team"]

    base_ratings = warm_up_ratings(df, [s for s in df["season"].unique() if s < CURRENT_SEASON], ELO_K, ELO_HFA)

    played_rounds = sorted(
        season_matches.loc[
            season_matches["home_goals"].notna() & season_matches["away_goals"].notna(), "matchday"
        ].unique()
    )

    prob_rows, rating_rows = [], []
    for r in played_rounds:
        r = int(r)
        ratings_r, stats_r = played_through_round(season_matches, r, ELO_K, ELO_HFA, base_ratings)

        played_ids = set(
            season_matches.loc[
                (season_matches["matchday"] <= r)
                & season_matches["home_goals"].notna()
                & season_matches["away_goals"].notna(),
                "match_id",
            ]
        )
        remaining = season_matches[~season_matches["match_id"].isin(played_ids)].sort_values("date")

        probs, _ = simulate_remainder(
            ratings_r, stats_r, remaining, teams, pools, ELO_K, ELO_HFA, ELO_DRAW_NU, ELO_N_SIMULATIONS
        )

        for t in teams:
            prob_rows.append(
                {"season": CURRENT_SEASON, "round": r, "team_id": t, "team": team_names[t], **probs[t]}
            )
            rating_rows.append(
                {
                    "season": CURRENT_SEASON,
                    "round": r,
                    "team_id": t,
                    "team": team_names[t],
                    "rating": ratings_r.get(t, ELO_INITIAL_RATING),
                }
            )
        print(f"  rodada {r}: simulado ({len(remaining)} jogos restantes)")

    return pd.DataFrame(prob_rows), pd.DataFrame(rating_rows)


# --------------------------------------------------------------- main

def main() -> None:
    df = load_matches()
    pools = build_score_pools(df)
    print(
        f"baldes de placar (2023-2025): casa {len(pools['home'])}, "
        f"empate {len(pools['draw'])}, fora {len(pools['away'])}"
    )
    print()

    print(f"simulando temporada {CURRENT_SEASON}, {ELO_N_SIMULATIONS} simulacoes por rodada...")
    probs_df, ratings_df = run_current_season(df, pools)

    PROCESSED.mkdir(parents=True, exist_ok=True)
    probs_df.to_parquet(PROCESSED / "elo_probabilidades.parquet", index=False)
    ratings_df.to_parquet(PROCESSED / "elo_ratings.parquet", index=False)
    print(f"\nelo_probabilidades: {len(probs_df)} linhas -> elo_probabilidades.parquet")
    print(f"elo_ratings: {len(ratings_df)} linhas -> elo_ratings.parquet")
    print("ok")


if __name__ == "__main__":
    main()

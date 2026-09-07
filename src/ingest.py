"""Ingestao via football-data.org, gravando parquet cru em data/raw.

FBref saiu do pipeline: a versao atual do soccerdata so raspa via
undetected-chromedriver (anti-bot), e nao vamos contornar isso. A API da
ESPN tambem saiu: site.api.espn.com bloqueia por faixa de IP de datacenter
(Akamai), 403 mesmo com header de navegador. football-data.org e API JSON
de verdade, com chave, feita pra ser chamada por servidor -- roda limpo em
qualquer runner do Actions.

Rate limit do plano gratis: 10 requests/minuto. Respeitamos com espaco
minimo entre chamadas e leitura do header de retry em caso de 429 --
nao adianta contornar isso, o token so tem essa cota mesmo.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import COMPETITION, CURRENT_SEASON, RAW, SEASONS  # noqa: E402

API_BASE = "https://api.football-data.org/v4"
REQUESTS_PER_MINUTE = 10
MIN_INTERVAL = 60 / REQUESTS_PER_MINUTE  # 6s entre chamadas
MAX_RETRIES = 3

_last_call = 0.0


def _token() -> str:
    token = os.environ.get("FOOTBALL_DATA_TOKEN")
    if not token:
        raise RuntimeError(
            "FOOTBALL_DATA_TOKEN nao definido. Gere uma chave gratuita em "
            "https://www.football-data.org/client/register e exporte a "
            "variavel de ambiente (ou grave em .env, veja .env.example). "
            "Sem isso a API responde 403 antes de qualquer coisa util."
        )
    return token


def _get(path: str, params: dict | None = None, attempt: int = 1) -> dict:
    """GET autenticado respeitando o limite de 10 req/min do plano gratis."""
    global _last_call
    elapsed = time.monotonic() - _last_call
    if elapsed < MIN_INTERVAL:
        time.sleep(MIN_INTERVAL - elapsed)

    resp = requests.get(
        f"{API_BASE}{path}",
        headers={"X-Auth-Token": _token()},
        params=params,
        timeout=30,
    )
    _last_call = time.monotonic()

    if resp.status_code == 429:
        if attempt > MAX_RETRIES:
            resp.raise_for_status()
        retry_after = resp.headers.get("Retry-After")
        wait = int(retry_after) if retry_after else 60
        print(f"  429 (rate limit). esperando {wait}s (tentativa {attempt}/{MAX_RETRIES})...")
        time.sleep(wait)
        return _get(path, params, attempt=attempt + 1)

    resp.raise_for_status()
    return resp.json()


def _merge_seasons(new: pd.DataFrame, path: Path, seasons_fetched: list[int]) -> pd.DataFrame:
    """Execucao incremental: substitui so as temporadas raspadas agora.

    As demais ficam como estavam no parquet existente. Sem isso, rodar so
    a temporada corrente apagaria o backfill historico.
    """
    if not path.exists():
        return new
    old = pd.read_parquet(path)
    keep = old[~old["season"].isin(seasons_fetched)]
    return pd.concat([keep, new], ignore_index=True)


def _save(df: pd.DataFrame, name: str, seasons_fetched: list[int]) -> Path:
    RAW.mkdir(parents=True, exist_ok=True)
    out = RAW / f"{name}.parquet"
    merged = _merge_seasons(df, out, seasons_fetched)
    merged.to_parquet(out, index=False)
    print(f"  {name}: {len(merged)} linhas, {len(merged.columns)} colunas -> {out.name}")
    return out


def fetch_matches(season: int) -> pd.DataFrame:
    data = _get(f"/competitions/{COMPETITION}/matches", params={"season": season})
    rows = []
    for m in data["matches"]:
        full_time = m["score"]["fullTime"]
        home_goals, away_goals = full_time["home"], full_time["away"]
        tem_placar = home_goals is not None and away_goals is not None
        rows.append(
            {
                "match_id": m["id"],
                "season": season,
                "matchday": m["matchday"],
                "utc_date": m["utcDate"],
                "home_team_id": m["homeTeam"]["id"],
                "home_team": m["homeTeam"]["name"],
                "away_team_id": m["awayTeam"]["id"],
                "away_team": m["awayTeam"]["name"],
                "home_goals": home_goals,
                "away_goals": away_goals,
                # A API da football-data.org tem um bug conhecido (achado em
                # 2026-09-07): parte das partidas com placar preenchido traz
                # status como uma string de data em vez de "FINISHED". Por
                # isso "partida encerrada" e decidido pelo placar (nao-nulo),
                # nunca por status -- e status cru fica guardado ao lado pra
                # dar pra auditar se a API corrigir isso um dia.
                "status_raw": m["status"],
                "status_anomalo": tem_placar and m["status"] != "FINISHED",
            }
        )
    return pd.DataFrame(rows)


def fetch_standings(season: int) -> pd.DataFrame:
    data = _get(f"/competitions/{COMPETITION}/standings", params={"season": season})
    total = next(s for s in data["standings"] if s["type"] == "TOTAL")
    rows = []
    for row in total["table"]:
        team = row["team"]
        rows.append(
            {
                "season": season,
                "position": row["position"],
                "team_id": team["id"],
                "team": team["name"],
                "short_name": team["shortName"],
                "tla": team["tla"],
                "played_games": row["playedGames"],
                "won": row["won"],
                "draw": row["draw"],
                "lost": row["lost"],
                "points": row["points"],
                "goals_for": row["goalsFor"],
                "goals_against": row["goalsAgainst"],
                "goal_difference": row["goalDifference"],
            }
        )
    return pd.DataFrame(rows)


def _report_anomalias(matches: pd.DataFrame) -> None:
    com_placar = matches[matches["home_goals"].notna() & matches["away_goals"].notna()]
    if len(com_placar) == 0:
        return
    anomalas = com_placar["status_anomalo"].sum()
    pct = 100 * anomalas / len(com_placar)
    print(f"\nstatus anomalo: {anomalas}/{len(com_placar)} partidas com placar ({pct:.1f}%)")
    if pct > 20:
        print(
            "AVISO: mais de 20% das partidas com placar tem status fora do "
            "esperado. Isso passou do que foi observado em 2026-09-07 -- "
            "pode ser sinal de que o comportamento da API mudou. Vale conferir "
            "status_raw manualmente antes de confiar no criterio de placar."
        )


def main(full: bool = False) -> None:
    seasons = SEASONS if full else [CURRENT_SEASON]
    print(f"ingest: football-data.org | competicao {COMPETITION} | temporadas {seasons}\n")

    match_frames, standings_frames = [], []
    for season in seasons:
        print(f"temporada {season}")
        match_frames.append(fetch_matches(season))
        standings_frames.append(fetch_standings(season))

    matches = pd.concat(match_frames, ignore_index=True)
    standings = pd.concat(standings_frames, ignore_index=True)

    _save(matches, "matches", seasons)
    _save(standings, "standings", seasons)
    _report_anomalias(matches)
    print("\nok")


if __name__ == "__main__":
    main(full="--full" in sys.argv)

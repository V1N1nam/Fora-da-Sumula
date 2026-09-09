"""Metricas derivadas em cima dos dados ja existentes (data/raw,
data/processed) -- nenhuma fonte nova, nenhum dado novo. Cada funcao
grava seu proprio Parquet em data/processed.

Reusa a mesma funcao de Davidson e os mesmos parametros calibrados
(K, HFA, NU) de config.py -- nada aqui recalibra ou inventa parametro.
Todo calculo que percorre jogos anda em ordem cronologica e usa o
rating VIGENTE na data do jogo, nunca o rating final (mesmo cuidado
que ja foi problema em elo.py).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import (  # noqa: E402
    CURRENT_SEASON, ELO_DRAW_NU, ELO_HFA, ELO_INITIAL_RATING, ELO_K, PROCESSED, RAW,
)
from calibrate_elo import HISTORICAL_SEASONS, davidson_probs, update_ratings  # noqa: E402
from elo import load_matches  # noqa: E402


# ------------------------------------------------------ rating pre-jogo

def build_match_ratings(df: pd.DataFrame) -> pd.DataFrame:
    """Caminha por TODAS as partidas com placar (2023-2026), em ordem
    cronologica continua e sem resetar entre temporadas -- mesma logica
    de warm_up_ratings em elo.py -- gravando o rating PRE-jogo de cada
    lado antes de atualiza-lo. E a base de qualquer metrica derivada
    que precise da forca dos times na hora do jogo, nao a atual."""
    played = df[df["home_goals"].notna() & df["away_goals"].notna()]

    ratings: dict[int, float] = {}
    rows = []
    for row in played.itertuples(index=False):
        home, away = int(row.home_team_id), int(row.away_team_id)
        hg, ag = int(row.home_goals), int(row.away_goals)
        rows.append(
            {
                "match_id": row.match_id,
                "season": row.season,
                "round": row.matchday,
                "date": row.date,
                "home_team_id": home,
                "home_team": row.home_team,
                "away_team_id": away,
                "away_team": row.away_team,
                "home_goals": hg,
                "away_goals": ag,
                "r_home_pre": ratings.get(home, ELO_INITIAL_RATING),
                "r_away_pre": ratings.get(away, ELO_INITIAL_RATING),
            }
        )
        update_ratings(ratings, home, away, hg, ag, ELO_K, ELO_HFA)

    return pd.DataFrame(rows)


# --------------------------------------------------------- 1. xpts_forca

def xpts_forca(match_ratings: pd.DataFrame) -> pd.DataFrame:
    """xPTS "de forca": pontos esperados pela probabilidade de
    resultado do Elo+Davidson (rating pre-jogo dos dois lados, sem
    vazamento), NAO por xG. Essa e a diferenca-chave em relacao ao xPTS
    classico: mede sorte/azar contra a FORCA do adversario (o resultado
    bateu com o que o rating prognosticava?), nao contra a QUALIDADE DAS
    CHANCES criadas no jogo -- isso exigiria dado de xG, que essa fonte
    (football-data.org) nao tem. Documentar essa distincao no site
    sempre que essa metrica aparecer."""
    r_home = match_ratings["r_home_pre"].to_numpy()
    r_away = match_ratings["r_away_pre"].to_numpy()
    p_home, p_draw, p_away = davidson_probs(r_home, r_away, ELO_HFA, ELO_DRAW_NU)

    hg = match_ratings["home_goals"].to_numpy()
    ag = match_ratings["away_goals"].to_numpy()
    real_home_pts = np.where(hg > ag, 3, np.where(hg == ag, 1, 0))
    real_away_pts = np.where(ag > hg, 3, np.where(hg == ag, 1, 0))
    xpts_home = 3 * p_home + p_draw
    xpts_away = 3 * p_away + p_draw

    long = pd.concat(
        [
            pd.DataFrame(
                {
                    "season": match_ratings["season"],
                    "team_id": match_ratings["home_team_id"],
                    "team": match_ratings["home_team"],
                    "real_pts": real_home_pts,
                    "xpts": xpts_home,
                }
            ),
            pd.DataFrame(
                {
                    "season": match_ratings["season"],
                    "team_id": match_ratings["away_team_id"],
                    "team": match_ratings["away_team"],
                    "real_pts": real_away_pts,
                    "xpts": xpts_away,
                }
            ),
        ],
        ignore_index=True,
    )

    agg = long.groupby(["season", "team_id", "team"], as_index=False).agg(
        jogos=("real_pts", "size"),
        pontos_reais=("real_pts", "sum"),
        xpts=("xpts", "sum"),
    )
    agg["xpts"] = agg["xpts"].round(2)
    agg["diferenca"] = (agg["pontos_reais"] - agg["xpts"]).round(2)
    return agg.sort_values(["season", "diferenca"], ascending=[True, False]).reset_index(drop=True)


# ------------------------------------------------------ 2. maiores_zebras

def maiores_zebras(match_ratings: pd.DataFrame) -> pd.DataFrame:
    """Para cada jogo disputado, a probabilidade que o Davidson (rating
    pre-jogo dos dois lados, sem vazamento) dava pro resultado que
    realmente aconteceu. Ordenado do menos provavel pro mais provavel --
    o topo da tabela e a lista de zebras."""
    r_home = match_ratings["r_home_pre"].to_numpy()
    r_away = match_ratings["r_away_pre"].to_numpy()
    p_home, p_draw, p_away = davidson_probs(r_home, r_away, ELO_HFA, ELO_DRAW_NU)

    hg = match_ratings["home_goals"].to_numpy()
    ag = match_ratings["away_goals"].to_numpy()
    resultado = np.where(hg > ag, "V", np.where(hg == ag, "E", "D"))
    probabilidade = np.where(hg > ag, p_home, np.where(hg == ag, p_draw, p_away))

    out = pd.DataFrame(
        {
            "date": match_ratings["date"],
            "season": match_ratings["season"],
            "mandante_id": match_ratings["home_team_id"],
            "mandante": match_ratings["home_team"],
            "visitante_id": match_ratings["away_team_id"],
            "visitante": match_ratings["away_team"],
            "placar": [f"{int(h)}-{int(a)}" for h, a in zip(hg, ag)],
            "resultado": resultado,
            "probabilidade": probabilidade.round(4),
        }
    )
    return out.sort_values("probabilidade", ascending=True).reset_index(drop=True)


# -------------------------------------------------------- 3. sequencias

def _streak_current_and_best(flags: pd.Series) -> tuple[int, int]:
    """(atual, recorde) de uma sequencia booleana, na ordem cronologica
    dada. 'Atual' e o trecho que termina no ultimo elemento da serie."""
    best = cur = 0
    for f in flags:
        cur = cur + 1 if f else 0
        best = max(best, cur)
    return cur, best


def _current_streak_start_date(dates: pd.Series, flags: pd.Series):
    """Data do primeiro jogo do trecho ATUAL (o que termina no ultimo
    jogo da serie) -- None se a sequencia atual for 0."""
    start = None
    for d, f in zip(dates, flags):
        if f:
            if start is None:
                start = d
        else:
            start = None
    return start


def sequencias(match_ratings: pd.DataFrame) -> pd.DataFrame:
    """Por clube, maior sequencia ATUAL (contando ate o ultimo jogo
    disputado) e HISTORICA (o maior trecho ja visto) de: invencibilidade,
    vitorias seguidas, jogos sem vencer, jogos sem sofrer gol. Em cima de
    todas as partidas disputadas (2023-2026) em ordem cronologica
    continua -- a sequencia nao reseta so porque a temporada acabou.

    Intencional, nao bug (confirmado com o caso real do Botafogo: 18
    jogos invicto em 2023, corretamente capturado): "sequencia atual"
    pode incluir jogos de temporadas anteriores. Quando essa metrica
    aparecer no site, o texto precisa deixar isso explicito -- ex.
    "invicto ha 8 jogos, desde outubro de 2025" -- nunca so o numero
    pelado, que sugeriria (errado) que a contagem e so da temporada
    corrente."""
    long = pd.concat(
        [
            pd.DataFrame(
                {
                    "date": match_ratings["date"],
                    "team_id": match_ratings["home_team_id"],
                    "team": match_ratings["home_team"],
                    "gf": match_ratings["home_goals"],
                    "ga": match_ratings["away_goals"],
                }
            ),
            pd.DataFrame(
                {
                    "date": match_ratings["date"],
                    "team_id": match_ratings["away_team_id"],
                    "team": match_ratings["away_team"],
                    "gf": match_ratings["away_goals"],
                    "ga": match_ratings["home_goals"],
                }
            ),
        ],
        ignore_index=True,
    )

    rows = []
    for tid, g in long.groupby("team_id", sort=False):
        g = g.sort_values("date")
        unbeaten_flags = g["gf"] >= g["ga"]
        win_flags = g["gf"] > g["ga"]
        cur_unb, best_unb = _streak_current_and_best(unbeaten_flags)
        cur_win, best_win = _streak_current_and_best(win_flags)
        cur_wl, best_wl = _streak_current_and_best(g["gf"] <= g["ga"])
        cur_cs, best_cs = _streak_current_and_best(g["ga"] == 0)
        unb_since = _current_streak_start_date(g["date"], unbeaten_flags)
        win_since = _current_streak_start_date(g["date"], win_flags)
        rows.append(
            {
                "team_id": int(tid),
                "team": g["team"].iloc[-1],
                "invencibilidade_atual": cur_unb,
                "invencibilidade_recorde": best_unb,
                "invencibilidade_atual_desde": unb_since,
                "vitorias_atual": cur_win,
                "vitorias_recorde": best_win,
                "vitorias_atual_desde": win_since,
                "sem_vencer_atual": cur_wl,
                "sem_vencer_recorde": best_wl,
                "sem_sofrer_atual": cur_cs,
                "sem_sofrer_recorde": best_cs,
            }
        )
    return pd.DataFrame(rows).sort_values("team").reset_index(drop=True)


# --------------------------------------------------- 4. mando_por_clube

def mando_por_clube(match_ratings: pd.DataFrame) -> pd.DataFrame:
    """Vantagem de mando ESPECIFICA de cada clube: pontos por jogo em
    casa vs fora, comparado com a media da liga inteira no mesmo
    universo de jogos (2023-2026, todas as partidas disputadas -- inclui
    clube promovido so em 2026, que ficaria de fora se o universo fosse
    restrito a 2023-2025). A media da liga usada aqui bate, dentro do
    esperado, com a estimativa de HFA feita em calibrate_elo.py (secao
    2c, so 2023-2025) -- ver conferencia impressa em main().

    jogos_casa e jogos_fora ficam como colunas separadas (nao um
    jogos_total unico) de proposito: numa temporada fechada os dois
    batem, mas com 2026 incompleto no meio do universo eles podem
    divergir por 1-2 jogos dependendo de onde o calendario esta --
    esconder isso atras de um total unico mascararia esse detalhe.
    temporadas conta quantas temporadas distintas o clube tem no
    universo -- clube com so 1 temporada (ex.: promovido em 2026) tem
    vantagem_relativa muito mais ruidosa que um com 4, e isso precisa
    ficar visivel pra quem for decidir o que exibir no site, sem
    precisar recalcular pra descobrir."""
    hg = match_ratings["home_goals"].to_numpy()
    ag = match_ratings["away_goals"].to_numpy()
    home_pts = np.where(hg > ag, 3, np.where(hg == ag, 1, 0))
    away_pts = np.where(ag > hg, 3, np.where(hg == ag, 1, 0))

    home_rows = pd.DataFrame(
        {"team_id": match_ratings["home_team_id"], "team": match_ratings["home_team"], "mando": "casa", "pts": home_pts}
    )
    away_rows = pd.DataFrame(
        {"team_id": match_ratings["away_team_id"], "team": match_ratings["away_team"], "mando": "fora", "pts": away_pts}
    )
    long = pd.concat([home_rows, away_rows], ignore_index=True)

    liga_ppg_casa = float(home_rows["pts"].mean())
    liga_ppg_fora = float(away_rows["pts"].mean())
    liga_vantagem = liga_ppg_casa - liga_ppg_fora

    piv = long.groupby(["team_id", "team", "mando"], as_index=False)["pts"].agg(["size", "sum"])
    piv = piv.pivot_table(index=["team_id", "team"], columns="mando", values=["size", "sum"])
    piv.columns = [f"{stat}_{mando}" for stat, mando in piv.columns]
    piv = piv.reset_index().rename(
        columns={"size_casa": "jogos_casa", "sum_casa": "pts_casa", "size_fora": "jogos_fora", "sum_fora": "pts_fora"}
    )

    for c in ("jogos_casa", "pts_casa", "jogos_fora", "pts_fora"):
        piv[c] = piv[c].fillna(0).astype(int)

    piv["ppg_casa"] = (piv["pts_casa"] / piv["jogos_casa"]).round(3)
    piv["ppg_fora"] = (piv["pts_fora"] / piv["jogos_fora"]).round(3)
    piv["vantagem_mando"] = (piv["ppg_casa"] - piv["ppg_fora"]).round(3)
    piv["liga_ppg_casa"] = round(liga_ppg_casa, 3)
    piv["liga_ppg_fora"] = round(liga_ppg_fora, 3)
    piv["liga_vantagem_mando"] = round(liga_vantagem, 3)
    piv["vantagem_relativa"] = (piv["vantagem_mando"] - liga_vantagem).round(3)

    temporadas = (
        pd.concat(
            [
                match_ratings[["home_team_id", "season"]].rename(columns={"home_team_id": "team_id"}),
                match_ratings[["away_team_id", "season"]].rename(columns={"away_team_id": "team_id"}),
            ],
            ignore_index=True,
        )
        .groupby("team_id")["season"]
        .nunique()
        .rename("temporadas")
    )
    piv = piv.merge(temporadas, on="team_id", how="left")

    cols = [
        "team_id", "team", "temporadas", "jogos_casa", "pts_casa", "ppg_casa", "jogos_fora", "pts_fora", "ppg_fora",
        "vantagem_mando", "liga_ppg_casa", "liga_ppg_fora", "liga_vantagem_mando", "vantagem_relativa",
    ]
    return piv[cols].sort_values("vantagem_relativa", ascending=False).reset_index(drop=True)


# ----------------------------------------------- 5. confronto_historico

def confronto_historico(match_ratings: pd.DataFrame) -> pd.DataFrame:
    """Historico de confronto direto entre cada par de clubes que ja se
    enfrentou em 2023-2026: jogos, vitorias de cada lado, empates, media
    de gols de cada lado. Uma linha por par (team_a_id < team_b_id) --
    nao duplicada como A-B e B-A. Independe de mando: 'vitorias_a' conta
    toda vitoria do clube A sobre o B, em casa ou fora."""
    home_id = match_ratings["home_team_id"].to_numpy()
    away_id = match_ratings["away_team_id"].to_numpy()
    home_team = match_ratings["home_team"].to_numpy()
    away_team = match_ratings["away_team"].to_numpy()
    hg = match_ratings["home_goals"].to_numpy()
    ag = match_ratings["away_goals"].to_numpy()

    a_is_home = home_id <= away_id
    a_id = np.where(a_is_home, home_id, away_id)
    b_id = np.where(a_is_home, away_id, home_id)
    team_a = np.where(a_is_home, home_team, away_team)
    team_b = np.where(a_is_home, away_team, home_team)
    gols_a = np.where(a_is_home, hg, ag)
    gols_b = np.where(a_is_home, ag, hg)
    vencedor = np.where(gols_a > gols_b, "a", np.where(gols_a == gols_b, "empate", "b"))

    df = pd.DataFrame(
        {
            "team_a_id": a_id, "team_b_id": b_id, "team_a": team_a, "team_b": team_b,
            "gols_a": gols_a, "gols_b": gols_b, "vencedor": vencedor,
        }
    )

    agg = df.groupby(["team_a_id", "team_b_id"]).agg(
        team_a=("team_a", "last"),
        team_b=("team_b", "last"),
        jogos=("vencedor", "size"),
        vitorias_a=("vencedor", lambda s: int((s == "a").sum())),
        empates=("vencedor", lambda s: int((s == "empate").sum())),
        vitorias_b=("vencedor", lambda s: int((s == "b").sum())),
        gols_a_total=("gols_a", "sum"),
        gols_b_total=("gols_b", "sum"),
    ).reset_index()

    agg["media_gols_a"] = (agg["gols_a_total"] / agg["jogos"]).round(2)
    agg["media_gols_b"] = (agg["gols_b_total"] / agg["jogos"]).round(2)

    cols = [
        "team_a_id", "team_b_id", "team_a", "team_b", "jogos", "vitorias_a", "empates", "vitorias_b",
        "gols_a_total", "gols_b_total", "media_gols_a", "media_gols_b",
    ]
    return agg[cols].sort_values("jogos", ascending=False).reset_index(drop=True)


# --------------------------------------------------- 6. ritmo_campeao

def ritmo_campeao(df: pd.DataFrame) -> pd.DataFrame:
    """Pontos acumulados do LIDER da tabela, rodada a rodada, pra cada
    temporada fechada (2023-2025, do inicio ao fim) e pra 2026 (ate a
    ultima rodada disputada -- nao precisa de caso especial, ja e o que
    sobra ao filtrar so partidas com placar). O lider pode trocar de
    time ao longo da temporada -- guarda quem era o lider em cada
    rodada, nao so os pontos. Desempate entre lideres empatados em
    pontos usa a mesma ordem do resto do pipeline: pontos, vitorias,
    saldo de gols, gols pro (ver simulate_remainder em elo.py)."""
    played = df[df["home_goals"].notna() & df["away_goals"].notna()]

    rows = []
    for season, g in played.groupby("season"):
        names: dict[int, str] = {}
        for row in g.itertuples(index=False):
            names[int(row.home_team_id)] = row.home_team
            names[int(row.away_team_id)] = row.away_team

        stats = {t: {"points": 0, "wins": 0, "goal_diff": 0, "goals_for": 0} for t in names}
        for rnd, rnd_games in g.groupby("matchday"):
            for m in rnd_games.itertuples(index=False):
                home, away = int(m.home_team_id), int(m.away_team_id)
                hg, ag = int(m.home_goals), int(m.away_goals)
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

            leader_id = max(
                stats,
                key=lambda t: (stats[t]["points"], stats[t]["wins"], stats[t]["goal_diff"], stats[t]["goals_for"]),
            )
            rows.append(
                {
                    "season": int(season),
                    "round": int(rnd),
                    "leader_team_id": leader_id,
                    "leader_team": names[leader_id],
                    "leader_points": stats[leader_id]["points"],
                }
            )

    return pd.DataFrame(rows).sort_values(["season", "round"]).reset_index(drop=True)


# --------------------------------------------------------------------- 7. cenarios

def cenarios() -> pd.DataFrame:
    """Pra cada clube, na rodada mais recente ja simulada: confirmado ou
    descartado MATEMATICAMENTE (dentro da resolucao de 10 mil
    simulacoes) de titulo, G4 e Z4 -- probabilidade virou exatamente 0%
    ou exatamente 100% no Monte Carlo de elo.py. Nao e combinatoria
    exata, e a aproximacao do Monte Carlo ja rodado (nenhuma simulacao
    nova aqui).

    Arquivo separado, nao coluna nova em elo_probabilidades.parquet:
    esse parquet e escrito por elo.py (dono do dado), nao por
    derived.py -- fazer merge nele aqui criaria uma dependencia de
    ordem de execucao (derived.py so poderia rodar depois de elo.py E
    reescrever o arquivo dele) e colunas quase todas vazias, ja que
    'confirmado/descartado' so faz sentido pra rodada mais recente, nao
    pro historico inteiro que o parquet carrega. Um arquivo pequeno,
    so com a rodada atual, e trivial de juntar por team_id quando
    precisar."""
    probs = pd.read_parquet(PROCESSED / "elo_probabilidades.parquet")
    current_round = int(probs["round"].max())
    now = probs[probs["round"] == current_round].copy()

    for col, prefix in (("titulo_prob", "titulo"), ("g4_prob", "g4"), ("z4_prob", "z4")):
        now[f"{prefix}_confirmado"] = now[col] >= 1.0
        now[f"{prefix}_descartado"] = now[col] <= 0.0

    cols = [
        "season", "round", "team_id", "team",
        "titulo_confirmado", "titulo_descartado",
        "g4_confirmado", "g4_descartado",
        "z4_confirmado", "z4_descartado",
    ]
    return now[cols].reset_index(drop=True)


# ---------------------------------------------------- 8. zebras_da_rodada

def zebras_da_rodada(match_ratings: pd.DataFrame, season: int, round_: int) -> pd.DataFrame:
    """Mesmo ranking de maiores_zebras, mas so com os jogos de UMA
    rodada fechada -- pra "zebras da rodada que fechou" no export/site,
    nao o top histórico de 2023-2026 inteiro. Reusa maiores_zebras() em
    cima de um recorte de match_ratings (sem duplicar a conta de
    probabilidade)."""
    subset = match_ratings[(match_ratings["season"] == season) & (match_ratings["round"] == round_)]
    out = maiores_zebras(subset)
    out.insert(1, "round", round_)
    return out


# ---------------------------------------------- 9. zebra_provavel_proxima_rodada

def zebra_provavel_proxima_rodada(matches: pd.DataFrame, ratings: pd.DataFrame) -> pd.DataFrame:
    """Para os jogos AGENDADOS (sem placar) da proxima rodada, a
    probabilidade de vitoria do lado mais fraco (por rating ATUAL, ja
    que o jogo ainda nao aconteceu -- nao existe rating pre-jogo aqui).
    Ordenado do jogo com maior chance de zebra pro menor. 'ratings' e o
    elo_ratings.parquet gerado por elo.py (rating por team_id a cada
    rodada); usa a rodada mais recente disponivel nele.

    'Proxima rodada' e definida por DATA, nao pelo menor numero de
    rodada ainda incompleta -- mesmo cuidado de build_form_and_next()
    em build_site.py: partida sem placar com data ANTERIOR ao ultimo
    jogo ja disputado esta adiada sem nova data, nao e "o proximo jogo"
    (confirmado com dado real: 3 jogos da rodada 21 carregam a data
    original de 29/07 sem nunca ter sido atualizada, enquanto o 4o jogo
    da mesma rodada [Botafogo x Gremio] foi remarcado pra 16/09 -- veio
    DEPOIS de rodadas seguintes por causa do adiamento). Sem esse
    corte, o card de previa mistura uma data de 2 meses atras com uma
    futura na mesma lista."""
    current = matches[matches["season"] == CURRENT_SEASON]
    cutoff = current.loc[current["home_goals"].notna(), "utc_date"].max()
    scheduled = current[current["home_goals"].isna() & (current["utc_date"] >= cutoff)]
    if scheduled.empty:
        return pd.DataFrame(columns=[
            "season", "round", "mandante_id", "mandante", "visitante_id", "visitante",
            "data", "favorito", "prob_zebra",
        ])
    next_round = int(scheduled.loc[scheduled["utc_date"].idxmin(), "matchday"])
    fixtures = scheduled[scheduled["matchday"] == next_round]

    current_round = int(ratings["round"].max())
    latest = ratings[ratings["round"] == current_round].set_index("team_id")["rating"]

    rows = []
    for row in fixtures.itertuples(index=False):
        home, away = int(row.home_team_id), int(row.away_team_id)
        if home not in latest.index or away not in latest.index:
            continue
        r_home, r_away = float(latest[home]), float(latest[away])
        p_home, p_draw, p_away = davidson_probs(
            np.array([r_home]), np.array([r_away]), ELO_HFA, ELO_DRAW_NU
        )
        p_home, p_away = float(p_home[0]), float(p_away[0])
        favorito = "mandante" if r_home >= r_away else "visitante"
        prob_zebra = p_away if favorito == "mandante" else p_home
        rows.append(
            {
                "season": int(row.season),
                "round": next_round,
                "mandante_id": home,
                "mandante": row.home_team,
                "visitante_id": away,
                "visitante": row.away_team,
                "data": row.utc_date,
                "favorito": favorito,
                "prob_zebra": round(prob_zebra, 4),
            }
        )
    return pd.DataFrame(rows).sort_values("prob_zebra", ascending=False).reset_index(drop=True)


# --------------------------------------------------------------- main

def main() -> None:
    df = load_matches()
    match_ratings = build_match_ratings(df)
    print(f"match_ratings: {len(match_ratings)} partidas com placar, rating pre-jogo em ordem cronologica continua")
    print()

    PROCESSED.mkdir(parents=True, exist_ok=True)

    xpts = xpts_forca(match_ratings)
    xpts.to_parquet(PROCESSED / "xpts_forca.parquet", index=False)
    print(f"xpts_forca: {len(xpts)} linhas -> xpts_forca.parquet")
    print(xpts.head(5).to_string(index=False))
    print()

    zebras = maiores_zebras(match_ratings)
    zebras.to_parquet(PROCESSED / "zebras.parquet", index=False)
    print(f"zebras: {len(zebras)} linhas -> zebras.parquet")
    print(zebras.head(10).to_string(index=False))
    print()

    seq = sequencias(match_ratings)
    seq.to_parquet(PROCESSED / "sequencias.parquet", index=False)
    print(f"sequencias: {len(seq)} linhas -> sequencias.parquet")
    print(seq.head(5).to_string(index=False))
    print()

    mando = mando_por_clube(match_ratings)
    mando.to_parquet(PROCESSED / "mando_clube.parquet", index=False)
    print(f"mando_clube: {len(mando)} linhas -> mando_clube.parquet")
    print(mando.head(5).to_string(index=False))
    print(mando.tail(5).to_string(index=False))
    print()

    # conferencia: a media da liga usada acima (2023-2026, todo o
    # universo de jogos) contra a estimativa de HFA feita so em
    # 2023-2025 em calibrate_elo.py -- devem ficar proximas.
    hist = match_ratings[match_ratings["season"].isin(HISTORICAL_SEASONS)]
    hg_h, ag_h = hist["home_goals"].to_numpy(), hist["away_goals"].to_numpy()
    ppg_casa_hist = float(np.where(hg_h > ag_h, 3, np.where(hg_h == ag_h, 1, 0)).mean())
    ppg_fora_hist = float(np.where(ag_h > hg_h, 3, np.where(hg_h == ag_h, 1, 0)).mean())
    print(f"conferencia -- liga 2023-2026 (usada acima): casa {mando['liga_ppg_casa'].iloc[0]:.3f} pts/jogo, "
          f"fora {mando['liga_ppg_fora'].iloc[0]:.3f}, vantagem {mando['liga_vantagem_mando'].iloc[0]:.3f}")
    print(f"conferencia -- liga so 2023-2025 (mesmo universo do HFA=65 calibrado): casa {ppg_casa_hist:.3f} pts/jogo, "
          f"fora {ppg_fora_hist:.3f}, vantagem {ppg_casa_hist - ppg_fora_hist:.3f}")
    print()

    h2h = confronto_historico(match_ratings)
    h2h.to_parquet(PROCESSED / "h2h.parquet", index=False)
    print(f"h2h: {len(h2h)} linhas -> h2h.parquet")
    print(h2h.head(5).to_string(index=False))
    print()

    ritmo = ritmo_campeao(df)
    ritmo.to_parquet(PROCESSED / "ritmo_campeao.parquet", index=False)
    print(f"ritmo_campeao: {len(ritmo)} linhas -> ritmo_campeao.parquet")
    print(ritmo.groupby("season").tail(1).to_string(index=False))
    print()

    # checagem: pontos do lider na ultima rodada de cada temporada
    # fechada tem que bater exatamente com o campeao real em standings.
    standings = pd.read_parquet(RAW / "standings.parquet")
    print("conferencia -- lider na ultima rodada vs campeao real (standings.parquet):")
    tudo_bateu = True
    for season in (2023, 2024, 2025):  # 2026 nao e temporada fechada, sem campeao pra comparar
        last = ritmo[ritmo["season"] == season].sort_values("round").iloc[-1]
        champ = standings[(standings["season"] == season) & (standings["position"] == 1)]
        if champ.empty:
            continue
        champ = champ.iloc[0]
        bate = int(last["leader_team_id"]) == int(champ["team_id"]) and int(last["leader_points"]) == int(champ["points"])
        tudo_bateu &= bate
        print(
            f"  {season}: lider simulado = {last['leader_team']} ({int(last['leader_points'])} pts) | "
            f"campeao real = {champ['team']} ({int(champ['points'])} pts) | bate: {bate}"
        )
    print(f"  TODAS BATERAM: {tudo_bateu}")
    print()

    cen = cenarios()
    cen.to_parquet(PROCESSED / "cenarios.parquet", index=False)
    print(f"cenarios: {len(cen)} linhas -> cenarios.parquet (rodada {int(cen['round'].iloc[0])})")
    print(cen.to_string(index=False))
    print()

    current_round = int(cen["round"].iloc[0])
    zeb_rodada = zebras_da_rodada(match_ratings, CURRENT_SEASON, current_round)
    zeb_rodada.to_parquet(PROCESSED / "zebras_rodada.parquet", index=False)
    print(f"zebras_rodada: {len(zeb_rodada)} linhas -> zebras_rodada.parquet (rodada {current_round})")
    print(zeb_rodada.to_string(index=False))
    print()

    elo_ratings = pd.read_parquet(PROCESSED / "elo_ratings.parquet")
    zeb_prox = zebra_provavel_proxima_rodada(df, elo_ratings)
    zeb_prox.to_parquet(PROCESSED / "zebra_provavel.parquet", index=False)
    prox_round = int(zeb_prox["round"].iloc[0]) if len(zeb_prox) else current_round + 1
    print(f"zebra_provavel: {len(zeb_prox)} linhas -> zebra_provavel.parquet (rodada {prox_round})")
    print(zeb_prox.to_string(index=False))


if __name__ == "__main__":
    main()

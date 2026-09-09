"""Le data/processed + data/raw, monta o objeto DATA e gera docs/index.html
a partir do template fora-da-sumula-v3.html.

Pasta e "docs" (nao "site") porque o GitHub Pages, no modo "Deploy from
a branch", so aceita "/ (root)" ou "/docs" como pasta de publicacao --
nao existe opcao de pasta arbitraria.

So o payload muda. O template tem exatamente DOIS pontos de injecao,
marcados com sentinela no proprio arquivo (nada de casar regex contra
HTML de layout, que quebrava a cada mudanca de design):

    const DATA = {...};                        /*__DATA__*/
    const HFA=..., NU=..., ELO_K=..., NSIM=...; /*__ELO__*/

Tudo que antes era escrito em HTML pelo build -- manchete da home,
numero da rodada -- agora entra como campo do payload e e renderizado
pelo template. O texto continua sendo gerado aqui (ver build_headline):
o gerador e dono do que a home *diz*, o template e dono de como ela
*parece*.

O template guarda uma copia do payload real da ultima geracao, entao
abrir fora-da-sumula-v3.html direto no navegador continua funcionando
pra iterar design sem rodar o pipeline.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import (  # noqa: E402
    CURRENT_SEASON, ELO_DRAW_NU, ELO_HFA, ELO_K, ELO_N_SIMULATIONS, PROCESSED, RAW, ROOT,
)

TEMPLATE = ROOT / "fora-da-sumula-v3.html"
OUTPUT = ROOT / "docs" / "index.html"

# Ferramenta interna de export de imagens pra X -- pagina separada, sem
# link em nenhum nav do site publico (ver <meta name="robots"> no
# proprio template). Mesmo mecanismo de sentinela do template
# principal, payload bem mais enxuto (so o que os 4 tipos de card
# precisam, nao o DATA inteiro).
EXPORT_TEMPLATE = ROOT / "fora-da-sumula-export.html"
EXPORT_OUTPUT = ROOT / "docs" / "export" / "index.html"

DECIDED_THRESHOLD = 0.85
N_ZEBRAS = 24
N_FORM = 5

# A football-data.org devolve "Mineiro" e "Paranaense" como short_name
# dos dois Atleticos -- nome tecnicamente correto (evita colisao entre
# eles), mas nao e como o torcedor chama o clube. Normaliza pro nome
# popular so nesses dois casos.
SHORT_NAME_OVERRIDES = {
    1766: "Atlético-MG",   # CA Mineiro
    1768: "Athletico-PR",  # CA Paranaense
}


def build_short_names(standings: pd.DataFrame) -> dict[int, str]:
    """Nome curto canonico por clube, mesmo pra clube fora da temporada
    corrente (usado em zebras/h2h, que cobrem 2023-2026 inteiro): pega o
    short_name da temporada mais recente em que o clube apareceu, com as
    mesmas correcoes de nome popular aplicadas na tabela atual."""
    latest = standings.sort_values("season").groupby("team_id").last()["short_name"]
    names = {int(tid): name for tid, name in latest.items()}
    names.update(SHORT_NAME_OVERRIDES)
    return names


def load_current_matches() -> pd.DataFrame:
    """Partidas da temporada corrente, ordenadas por data real de jogo.

    A ordem e por `utc_date`, nao por `matchday`: rodada adiada e
    remarcada fora de ordem existe (ha jogos da rodada 21 ainda sem
    placar depois da 26), e "ultimos 5 jogos" e uma leitura cronologica,
    nao de numero de rodada.
    """
    m = pd.read_parquet(RAW / "matches.parquet")
    m = m[m["season"] == CURRENT_SEASON].copy()
    return m.sort_values("utc_date").reset_index(drop=True)


def build_form_and_next(
    matches: pd.DataFrame, short_names: dict[int, str], cutoff: str
) -> tuple[dict[int, list[dict]], dict[int, dict | None]]:
    """Ultimos N jogos disputados (mais antigo primeiro) e proximo jogo
    agendado de cada clube. Fato bruto de data/raw -- nenhum modelo
    envolvido, nenhuma regra de classificacao.

    `cutoff` e a data do jogo mais recente ja disputado: partida sem
    placar ANTES dela e adiada sem nova data, nao "o proximo jogo".
    """
    played = matches[matches["home_goals"].notna() & matches["away_goals"].notna()]
    scheduled = matches[matches["home_goals"].isna() & (matches["utc_date"] >= cutoff)]

    form: dict[int, list[dict]] = {}
    for row in played.itertuples(index=False):
        for tid, opp, gf, ga, mando in (
            (int(row.home_team_id), int(row.away_team_id), int(row.home_goals), int(row.away_goals), "C"),
            (int(row.away_team_id), int(row.home_team_id), int(row.away_goals), int(row.home_goals), "F"),
        ):
            form.setdefault(tid, []).append({
                "r": "V" if gf > ga else "D" if gf < ga else "E",
                "c": mando,
                "o": short_names.get(opp, str(opp)),
                "p": f"{gf}-{ga}",
                "d": row.utc_date[:10],
                "md": int(row.matchday),
            })

    nxt: dict[int, dict | None] = {}
    for row in scheduled.itertuples(index=False):
        for tid, opp, mando in (
            (int(row.home_team_id), int(row.away_team_id), "C"),
            (int(row.away_team_id), int(row.home_team_id), "F"),
        ):
            if tid not in nxt:
                nxt[tid] = {
                    "o": short_names.get(opp, str(opp)),
                    "c": mando,
                    "d": row.utc_date[:10],
                    "md": int(row.matchday),
                }

    return {t: v[-N_FORM:] for t, v in form.items()}, nxt


def build_data() -> dict:
    probs = pd.read_parquet(PROCESSED / "elo_probabilidades.parquet")
    ratings = pd.read_parquet(PROCESSED / "elo_ratings.parquet")
    standings = pd.read_parquet(RAW / "standings.parquet")
    standings_now = standings[standings["season"] == CURRENT_SEASON].set_index("team_id")
    short_names = build_short_names(standings)

    matches = load_current_matches()
    updated = matches[matches["home_goals"].notna()]["utc_date"].max()
    form, next_match = build_form_and_next(matches, short_names, updated)

    xpts = pd.read_parquet(PROCESSED / "xpts_forca.parquet")
    xpts_now = xpts[xpts["season"] == CURRENT_SEASON].set_index("team_id")
    seq = pd.read_parquet(PROCESSED / "sequencias.parquet").set_index("team_id")
    mando = pd.read_parquet(PROCESSED / "mando_clube.parquet").set_index("team_id")
    cenarios = pd.read_parquet(PROCESSED / "cenarios.parquet").set_index("team_id")

    rounds = sorted(int(x) for x in probs["round"].unique())
    current_round = rounds[-1]

    # ordem canonica: rating atual, do maior pro menor
    last_rating = ratings[ratings["round"] == current_round].set_index("team_id")["rating"]
    team_order = [int(t) for t in last_rating.sort_values(ascending=False).index]

    teams = {}
    for tid in team_order:
        pg = probs[probs["team_id"] == tid].sort_values("round")
        rg = ratings[ratings["team_id"] == tid].sort_values("round")
        assert list(pg["round"]) == rounds and list(rg["round"]) == rounds

        default_short = standings_now.loc[tid, "short_name"] if tid in standings_now.index else pg["team"].iloc[0]
        row = {
            "team": pg["team"].iloc[0],
            "short_name": SHORT_NAME_OVERRIDES.get(tid, default_short),
            "titulo": [round(float(v), 4) for v in pg["titulo_prob"]],
            "g4": [round(float(v), 4) for v in pg["g4_prob"]],
            "z4": [round(float(v), 4) for v in pg["z4_prob"]],
            "rating": [round(float(v), 1) for v in rg["rating"]],
        }
        if tid in standings_now.index:
            s = standings_now.loc[tid]
            row["real_position"] = int(s["position"])
            row["real_points"] = int(s["points"])
            row["real_played"] = int(s["played_games"])
            row["real_won"] = int(s["won"])
            row["real_draw"] = int(s["draw"])
            row["real_lost"] = int(s["lost"])
            row["real_gf"] = int(s["goals_for"])
            row["real_ga"] = int(s["goals_against"])
            row["real_gd"] = int(s["goal_difference"])

        if tid in xpts_now.index:
            x = xpts_now.loc[tid]
            row["xpts_jogos"] = int(x["jogos"])
            row["xpts_pontos_reais"] = int(x["pontos_reais"])
            row["xpts_valor"] = round(float(x["xpts"]), 1)
            row["xpts_diferenca"] = round(float(x["diferenca"]), 1)

        if tid in seq.index:
            sq = seq.loc[tid]
            row["seq_invicto_atual"] = int(sq["invencibilidade_atual"])
            row["seq_invicto_recorde"] = int(sq["invencibilidade_recorde"])
            row["seq_invicto_desde"] = (
                sq["invencibilidade_atual_desde"].date().isoformat()
                if pd.notna(sq["invencibilidade_atual_desde"]) else None
            )
            row["seq_vitorias_atual"] = int(sq["vitorias_atual"])
            row["seq_vitorias_recorde"] = int(sq["vitorias_recorde"])
            row["seq_vitorias_desde"] = (
                sq["vitorias_atual_desde"].date().isoformat()
                if pd.notna(sq["vitorias_atual_desde"]) else None
            )

        if tid in mando.index:
            m = mando.loc[tid]
            row["mando_temporadas"] = int(m["temporadas"])
            row["mando_ppg_casa"] = round(float(m["ppg_casa"]), 2)
            row["mando_ppg_fora"] = round(float(m["ppg_fora"]), 2)
            row["mando_liga_ppg_casa"] = round(float(m["liga_ppg_casa"]), 2)
            row["mando_liga_ppg_fora"] = round(float(m["liga_ppg_fora"]), 2)
            row["mando_liga_vantagem"] = round(float(m["liga_vantagem_mando"]), 2)
            row["mando_vantagem"] = round(float(m["vantagem_mando"]), 2)
            row["mando_vantagem_relativa"] = round(float(m["vantagem_relativa"]), 2)

        if tid in cenarios.index:
            c = cenarios.loc[tid]
            for col in (
                "titulo_confirmado", "titulo_descartado", "g4_confirmado",
                "g4_descartado", "z4_confirmado", "z4_descartado",
            ):
                row[col] = bool(c[col])

        row["form"] = form.get(tid, [])
        row["next"] = next_match.get(tid)

        teams[str(tid)] = row

    zebras, zebras_universo = build_zebras(short_names)
    zebras_rodada = build_zebras_rodada(short_names)
    proxima_rodada = build_proxima_rodada(short_names)

    data = {
        "season": CURRENT_SEASON,
        "current_round": current_round,
        "total_rounds": int(matches["matchday"].max()),
        "updated": updated[:10],
        "rounds": rounds,
        "team_order": [str(t) for t in team_order],
        "teams": teams,
        "zebras": zebras,
        "zebras_universo": zebras_universo,
        "ritmo": build_ritmo(),
        "h2h": build_h2h(team_order, short_names),
        "zebras_rodada": zebras_rodada,
        "proxima_rodada": proxima_rodada,
    }
    data["headline"] = build_headline(data)
    return data


def build_zebras(short_names: dict[int, str], n: int = N_ZEBRAS) -> tuple[list[dict], int]:
    """As n partidas mais improvaveis de 2023-2026, mais recente primeiro
    em caso de empate na probabilidade. Devolve tambem o tamanho do
    universo -- o site diz "as n maiores de X partidas", e esse X nao
    pode ser chutado no template."""
    z = pd.read_parquet(PROCESSED / "zebras.parquet")
    universo = len(z)
    z = z.sort_values(["probabilidade", "date"], ascending=[True, False]).head(n)
    out = []
    for row in z.itertuples(index=False):
        out.append(
            {
                "date": row.date.date().isoformat(),
                "mandante": short_names.get(int(row.mandante_id), row.mandante),
                "visitante": short_names.get(int(row.visitante_id), row.visitante),
                "placar": row.placar,
                "probabilidade": round(float(row.probabilidade), 4),
            }
        )
    return out, universo


def build_zebras_rodada(short_names: dict[int, str]) -> list[dict]:
    """As zebras da rodada que acabou de fechar, mais improvavel primeiro
    -- gerado por derived.zebras_da_rodada() e gravado em
    zebras_rodada.parquet. Diferente de build_zebras(): aqui e so a
    rodada atual, nao o top historico de 2023-2026."""
    z = pd.read_parquet(PROCESSED / "zebras_rodada.parquet")
    out = []
    for row in z.itertuples(index=False):
        out.append(
            {
                "date": row.date.date().isoformat(),
                "round": int(row.round),
                "mandante": short_names.get(int(row.mandante_id), row.mandante),
                "visitante": short_names.get(int(row.visitante_id), row.visitante),
                "placar": row.placar,
                "probabilidade": round(float(row.probabilidade), 4),
            }
        )
    return out


def build_proxima_rodada(short_names: dict[int, str]) -> list[dict]:
    """Confrontos agendados da proxima rodada, com a probabilidade de
    zebra (lado mais fraco por rating atual vencer) de cada jogo --
    gerado por derived.zebra_provavel_proxima_rodada() e gravado em
    zebra_provavel.parquet, ja ordenado do jogo com mais chance de
    zebra pro com menos."""
    z = pd.read_parquet(PROCESSED / "zebra_provavel.parquet")
    out = []
    for row in z.itertuples(index=False):
        out.append(
            {
                "round": int(row.round),
                "data": row.data[:10],
                "mandante": short_names.get(int(row.mandante_id), row.mandante),
                "visitante": short_names.get(int(row.visitante_id), row.visitante),
                "favorito": row.favorito,
                "prob_zebra": round(float(row.prob_zebra), 4),
            }
        )
    return out


def build_ritmo() -> dict[str, list[int]]:
    """Pontos do lider por rodada, por temporada -- pra sobrepor o ritmo
    da temporada corrente ao das 3 anteriores no mesmo eixo de rodada."""
    r = pd.read_parquet(PROCESSED / "ritmo_campeao.parquet")
    out = {}
    for season, g in r.groupby("season"):
        g = g.sort_values("round")
        out[str(int(season))] = [int(v) for v in g["leader_points"]]
    return out


def build_h2h(team_order: list[int], short_names: dict[int, str]) -> dict[str, dict]:
    """Confronto direto, so entre pares dos 20 clubes da temporada
    corrente (o simulador de confronto na pagina Forca so oferece esses
    20) -- os outros 336 pares de h2h.parquet nao tem onde aparecer."""
    h = pd.read_parquet(PROCESSED / "h2h.parquet")
    current = set(team_order)
    h = h[h["team_a_id"].isin(current) & h["team_b_id"].isin(current)]
    out = {}
    for row in h.itertuples(index=False):
        key = f"{row.team_a_id}_{row.team_b_id}"
        out[key] = {
            "a": row.team_a_id,
            "b": row.team_b_id,
            "jogos": int(row.jogos),
            "va": int(row.vitorias_a),
            "empates": int(row.empates),
            "vb": int(row.vitorias_b),
            "ga": round(float(row.media_gols_a), 2),
            "gb": round(float(row.media_gols_b), 2),
        }
    return out


def build_headline(data: dict) -> dict:
    """Texto da manchete da home (quem lidera, se trocou de lider na
    ultima rodada, se a disputa ja esta decidida) gerado a partir dos
    dados, nao escrito a mao. Vai como campo do payload -- quem desenha
    o herói é o template."""
    teams = data["teams"]
    order = sorted(data["team_order"], key=lambda tid: teams[tid]["titulo"][-1], reverse=True)
    p1_id, p2_id = order[0], order[1]
    p1, p2 = teams[p1_id], teams[p2_id]
    p1_now = p1["titulo"][-1]

    prev_leader_id = max(data["team_order"], key=lambda tid: teams[tid]["titulo"][-2])
    lead_changed = prev_leader_id != p1_id

    p1_pct, p2_pct = round(p1_now * 100), round(p2["titulo"][-1] * 100)
    rest_pct = max(0, 100 - p1_pct - p2_pct)

    if p1_now > DECIDED_THRESHOLD:
        answer = (
            f'A disputa está praticamente decidida. O <b>{p1["short_name"]}</b> tem '
            f'<b>{p1_pct}%</b> de chance de título — o resto da tabela soma {100 - p1_pct}%.'
        )
    elif lead_changed:
        prev_name = teams[prev_leader_id]["short_name"]
        answer = (
            f'Sobraram dois. O <b>{p1["short_name"]}</b> passou o {prev_name} na última '
            f'rodada e assumiu a ponta. Os outros {len(order) - 2} clubes somam <b>{rest_pct}%</b>.'
        )
    else:
        answer = (
            f'Sobraram dois. O <b>{p1["short_name"]}</b> segue na frente do '
            f'{p2["short_name"]}. Os outros {len(order) - 2} clubes somam <b>{rest_pct}%</b>.'
        )

    return {"answer": answer, "p1": p1_id, "p2": p2_id}


def build_export_data(data: dict) -> dict:
    """Payload enxuto pro gerador de imagens (fora-da-sumula-export.html):
    so os 4 tipos de card usam, nao o DATA inteiro do site (que carrega
    historico de titulo/g4/z4/rating por clube, rodada a rodada -- peso
    morto pra quem so quer gerar uma imagem). Derivado do `data` que
    build_data() ja montou, sem reler parquet."""
    cenarios = []
    for tid in data["team_order"]:
        t = data["teams"][tid]
        flags = {
            k: t[k]
            for k in (
                "titulo_confirmado", "titulo_descartado", "g4_confirmado",
                "g4_descartado", "z4_confirmado", "z4_descartado",
            )
            if k in t
        }
        if any(flags.values()):
            cenarios.append({"team": t["short_name"], **flags})

    return {
        "season": data["season"],
        "current_round": data["current_round"],
        "total_rounds": data["total_rounds"],
        "updated": data["updated"],
        "zebras_rodada": data["zebras_rodada"],
        "proxima_rodada": data["proxima_rodada"],
        "ritmo": data["ritmo"],
        "cenarios": cenarios,
    }


def render_export(template: str, export_data: dict) -> str:
    """Mesma logica de sentinela de render(), so pra sentinela DATA --
    o template de export nao tem a segunda sentinela __ELO__ (nao mostra
    parametro de modelo, so os 4 cards)."""
    data_json = json.dumps(export_data, ensure_ascii=False, separators=(",", ":"))
    template, n = re.subn(
        r"^const DATA = .*?; /\*__DATA__\*/$",
        lambda _: f"const DATA = {data_json}; /*__DATA__*/",
        template, count=1, flags=re.M | re.S,
    )
    assert n == 1, "nao achei a sentinela /*__DATA__*/ no template de export"
    return template


def fmt_num(v: float) -> str:
    return str(int(v)) if float(v).is_integer() else repr(float(v))


def render(template: str, data: dict) -> str:
    """Injeta payload e parametros do modelo nas duas sentinelas do
    template. Falha alto se alguma sumir -- site com dado velho e pior
    que build quebrado.

    As regex nao precisam tolerar CRLF: main() ja le o template com
    Path.read_text() sem argumento de newline, que usa universal
    newlines por padrao e converte \r\n/\r pra \n na leitura inteira
    -- nao so nessas duas linhas. A string que chega aqui nunca tem
    \r, entao o `$` sozinho basta."""
    data_json = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
    template, n = re.subn(
        r"^const DATA = .*?; /\*__DATA__\*/$",
        lambda _: f"const DATA = {data_json}; /*__DATA__*/",
        template, count=1, flags=re.M | re.S,
    )
    assert n == 1, "nao achei a sentinela /*__DATA__*/ no template"

    elo_line = (
        f"const HFA={fmt_num(ELO_HFA)}, NU={fmt_num(ELO_DRAW_NU)}, "
        f"ELO_K={fmt_num(ELO_K)}, NSIM={ELO_N_SIMULATIONS}; /*__ELO__*/"
    )
    template, n = re.subn(
        r"^const HFA=.*?; /\*__ELO__\*/$", lambda _: elo_line, template, count=1, flags=re.M,
    )
    assert n == 1, "nao achei a sentinela /*__ELO__*/ no template"
    return template


def main() -> None:
    data = build_data()
    # Sem argumento `newline`: Path.read_text() so ganhou esse parametro
    # no Python 3.13 -- o workflow do Actions roda 3.12, e passar
    # newline="" aqui derruba o build la com TypeError (mesmo passando
    # limpo numa maquina com Python mais novo). O default de read_text()
    # (universal newlines) ja converte \r\n/\r pra \n sozinho, entao a
    # tolerancia a CRLF do template editado no Windows nao se perde --
    # so para de depender de um argumento que nem toda versao suportada
    # do Python tem. write_text() abaixo tem `newline` desde o Python
    # 3.10 -- essa API nao precisa mudar.
    html = render(TEMPLATE.read_text(encoding="utf-8"), data)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(html, encoding="utf-8", newline="\n")
    print(f"{OUTPUT} escrito ({OUTPUT.stat().st_size / 1024:.0f} KB), rodada {data['current_round']}, "
          f"{len(data['teams'])} clubes")

    export_data = build_export_data(data)
    export_html = render_export(EXPORT_TEMPLATE.read_text(encoding="utf-8"), export_data)
    EXPORT_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    EXPORT_OUTPUT.write_text(export_html, encoding="utf-8", newline="\n")
    print(f"{EXPORT_OUTPUT} escrito ({EXPORT_OUTPUT.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()

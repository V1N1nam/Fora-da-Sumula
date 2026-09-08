"""Le data/processed + data/raw, monta o objeto DATA e gera docs/index.html
a partir do template aprovado fora-da-sumula-v3.html.

Pasta e "docs" (nao "site") porque o GitHub Pages, no modo "Deploy from
a branch", so aceita "/ (root)" ou "/docs" como pasta de publicacao --
nao existe opcao de pasta arbitraria.

So o payload muda: o bloco `const DATA = {...}`, o texto da manchete da
home (gerado a partir dos dados -- ver build_headline) e a linha
`const HFA=..., NU=...;` do simulador de confronto (puxada de config.py,
pra nao divergir se o modelo for recalibrado). CSS, estrutura de paginas
e todo o resto do JS ficam intocados -- o template e a fonte da verdade
pro design.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import CURRENT_SEASON, ELO_DRAW_NU, ELO_HFA, PROCESSED, RAW, ROOT  # noqa: E402

TEMPLATE = ROOT / "fora-da-sumula-v3.html"
OUTPUT = ROOT / "docs" / "index.html"

DECIDED_THRESHOLD = 0.85

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


def build_data() -> dict:
    probs = pd.read_parquet(PROCESSED / "elo_probabilidades.parquet")
    ratings = pd.read_parquet(PROCESSED / "elo_ratings.parquet")
    standings = pd.read_parquet(RAW / "standings.parquet")
    standings_now = standings[standings["season"] == CURRENT_SEASON].set_index("team_id")
    short_names = build_short_names(standings)

    xpts = pd.read_parquet(PROCESSED / "xpts_forca.parquet")
    xpts_now = xpts[xpts["season"] == CURRENT_SEASON].set_index("team_id")
    seq = pd.read_parquet(PROCESSED / "sequencias.parquet").set_index("team_id")
    mando = pd.read_parquet(PROCESSED / "mando_clube.parquet").set_index("team_id")
    cenarios = pd.read_parquet(PROCESSED / "cenarios.parquet").set_index("team_id")

    rounds = sorted(int(x) for x in probs["round"].unique())
    current_round = rounds[-1]

    # ordem canonica: rating atual, do maior pro menor (igual ao protótipo aprovado)
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

        teams[str(tid)] = row

    return {
        "current_round": current_round,
        "rounds": rounds,
        "team_order": [str(t) for t in team_order],
        "teams": teams,
        "zebras": build_zebras(short_names),
        "ritmo": build_ritmo(),
        "h2h": build_h2h(team_order, short_names),
    }


def build_zebras(short_names: dict[int, str], n: int = 15) -> list[dict]:
    """15 partidas mais improvaveis de 2023-2026, mais recente primeiro
    em caso de empate na probabilidade."""
    z = pd.read_parquet(PROCESSED / "zebras.parquet")
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


def build_headline(data: dict) -> tuple[str, str]:
    """Gera o texto da manchete (quem lidera, se trocou de lider na ultima
    rodada, se a disputa ja esta decidida) a partir dos dados -- em vez de
    escrito a mao. Retorna (paragrafo .answer, bloco .duo)."""
    teams = data["teams"]
    order = sorted(data["team_order"], key=lambda tid: teams[tid]["titulo"][-1], reverse=True)
    p1_id, p2_id = order[0], order[1]
    p1, p2 = teams[p1_id], teams[p2_id]
    p1_now, p2_now = p1["titulo"][-1], p2["titulo"][-1]

    prev_leader_id = max(data["team_order"], key=lambda tid: teams[tid]["titulo"][-2])
    lead_changed = prev_leader_id != p1_id

    p1_pct, p2_pct = round(p1_now * 100), round(p2_now * 100)
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
            f'rodada e assumiu a ponta. Os outros 18 clubes somam <b>{rest_pct}%</b>.'
        )
    else:
        answer = (
            f'Sobraram dois. O <b>{p1["short_name"]}</b> segue na frente do '
            f'{p2["short_name"]}. Os outros 18 clubes somam <b>{rest_pct}%</b>.'
        )

    duo = (
        '<div class="duo">\n'
        f'    <div><div class="n" style="color:var(--good)">{p1_pct}%</div>'
        f'<div class="c">{p1["short_name"]}<br>{p1["real_points"]} pontos</div></div>\n'
        f'    <div><div class="n">{p2_pct}%</div>'
        f'<div class="c">{p2["short_name"]}<br>{p2["real_points"]} pontos</div></div>\n'
        '  </div>\n  '
    )
    return f'<p class="answer">{answer}</p>\n  ', duo


def fmt_num(v: float) -> str:
    return str(int(v)) if float(v).is_integer() else repr(float(v))


def main() -> None:
    data = build_data()
    template = TEMPLATE.read_text(encoding="utf-8", newline="")

    # 1. payload
    data_json = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
    template, n = re.subn(r"const DATA = \{.*?\};\nconst C=", f"const DATA = {data_json};\nconst C=", template, flags=re.S)
    assert n == 1, "nao achei o bloco 'const DATA = ...' no template"

    # 2. manchete da home (gerada, nao escrita a mao)
    answer_html, duo_html = build_headline(data)
    template, n = re.subn(
        r'<p class="answer">.*?(?=<div id="raceChart">)',
        answer_html + duo_html,
        template, count=1, flags=re.S,
    )
    assert n == 1, "nao achei o bloco de manchete (.answer + .duo) no template"

    # 3. parametros do simulador de confronto, puxados de config.py
    old_line = "const HFA=65, NU=0.7405;"
    new_line = f"const HFA={fmt_num(ELO_HFA)}, NU={fmt_num(ELO_DRAW_NU)};"
    assert old_line in template, "nao achei a linha 'const HFA=...' no template"
    template = template.replace(old_line, new_line, 1)

    # 4. numero da rodada no cabecalho -- ficava fixo no template
    # (bug latente: nunca era substituido, entao o cabecalho ia
    # congelar na rodada do momento em que o template foi escrito).
    template, n = re.subn(
        r'<span class="round">rodada \d+</span>',
        f'<span class="round">rodada {data["current_round"]}</span>',
        template, count=1,
    )
    assert n == 1, "nao achei o span.round no template"

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(template, encoding="utf-8", newline="\n")
    print(f"{OUTPUT} escrito ({OUTPUT.stat().st_size / 1024:.0f} KB), rodada {data['current_round']}, "
          f"{len(data['teams'])} clubes")


if __name__ == "__main__":
    main()

"""Le data/processed + data/raw, monta o objeto DATA e gera site/index.html
a partir do template aprovado fora-da-sumula-v3.html.

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
OUTPUT = ROOT / "site" / "index.html"

DECIDED_THRESHOLD = 0.85


def build_data() -> dict:
    probs = pd.read_parquet(PROCESSED / "elo_probabilidades.parquet")
    ratings = pd.read_parquet(PROCESSED / "elo_ratings.parquet")
    standings = pd.read_parquet(RAW / "standings.parquet")
    standings_now = standings[standings["season"] == CURRENT_SEASON].set_index("team_id")

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

        row = {
            "team": pg["team"].iloc[0],
            "short_name": standings_now.loc[tid, "short_name"] if tid in standings_now.index else pg["team"].iloc[0],
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
        teams[str(tid)] = row

    return {
        "current_round": current_round,
        "rounds": rounds,
        "team_order": [str(t) for t in team_order],
        "teams": teams,
    }


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

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(template, encoding="utf-8", newline="\n")
    print(f"{OUTPUT} escrito ({OUTPUT.stat().st_size / 1024:.0f} KB), rodada {data['current_round']}, "
          f"{len(data['teams'])} clubes")


if __name__ == "__main__":
    main()

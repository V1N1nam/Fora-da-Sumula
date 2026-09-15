"""Card de Status Semanal em PNG: fundo gerado por IA, resto desenhado em Pillow.

Mesma divisao de trabalho do `card_confronto.py` (dado -> template Python ->
fundo atmosferico pela IA -> texto/numero desenhado pelo Pillow, nunca pela
IA). A diferenca e o que guia a cor do fundo: o Confronto tem dois clubes
naturais (casa/fora), o Status Semanal nao -- aqui o fundo usa a cor de
marca do clube da "maior alta de título da semana" (o dado que abre o
card), com um unico feixe de luz subindo (em vez dos dois feixes que se
encontram no card de Confronto), sugerindo a subida na tabela.

Formato: so `x` (1600x900, 16:9, o mesmo aspecto do card SVG do gerador de
imagens) por enquanto. Os formatos em pe (feed/story) do Confronto pedem
layout empilhado -- fora do escopo deste primeiro teste; ver ali se algum
dia for extendido pra ca.

Economia de credito: mesmo esquema do Confronto -- fundo cacheado em disco
por (versao do prompt, cor, tamanho, qualidade), `--sem-ia` desenha um
fundo procedural sem chamar a API, default `--qualidade low`.

De onde vem o dado: mesmo payload de `docs/export/index.html`
(`mudancas_semana`, `cenarios`, `ritmo`), lido com o mesmo helper de
`card_confronto.py` (`load_payload`). Se o formato do payload de export
mudar, este modulo precisa acompanhar.

Uso:
    python src/card_status.py --sem-ia
    python src/card_status.py --qualidade low
"""

from __future__ import annotations

import argparse
import hashlib
import math
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageEnhance, ImageFilter

sys.path.insert(0, str(Path(__file__).resolve().parent))
from card_confronto import (  # noqa: E402
    BG_CACHE,
    CARDS,
    MODEL,
    OPENAI_IMAGES,
    OUT_DIR,
    C,
    _scrim,
    carrega_env,
    cor_marca,
    fonte,
    hexa,
    largura_texto,
    load_payload,
    load_teams_meta,
    painel,
    texto,
)

import base64
import json
import os

import requests

# --- formato (so "x" por enquanto) -------------------------------------

W, H = 1600, 900
AI_SIZE = "1536x1024"
M, DIR = 75, 1600 - 75

# --- layout: escalado 4/3 a partir do card SVG (1200x675) do gerador de
# imagens (renderStatus em fora-da-sumula-export.html) -- mesma proporcao
# 16:9, entao os dois layouts sao o mesmo desenho em escalas diferentes.
CAB_Y, CAB_H = 58, 44
LABEL_Y = 152
# BOX_H folgado (250, nao os 160*4/3=213 que a conta ingenua do SVG dava):
# o SVG mede glifo exato antes de renderizar, o Pillow nao -- o numero
# grande do delta (fonte 45) mais a barra de magnitude embaixo precisam
# de mais respiro vertical do que a mesma proporcao no SVG, senao a
# barra corta o texto (visto no primeiro teste renderizado).
BOX_Y, BOX_H, BOX_GAP = 195, 250, 32
BOX_W = (W - 2 * M - BOX_GAP) // 2
BOX1_X, BOX2_X = M, M + BOX_W + BOX_GAP
DIV1_Y = BOX_Y + BOX_H + 29
CHIP_LABEL_Y = DIV1_Y + 40
CHIP_Y0 = CHIP_LABEL_Y + 19
CHIP_H = 35
DIV2_Y = CHIP_Y0 + CHIP_H + 24
HEADER_Y = DIV2_Y + 40
ROWS_Y0 = HEADER_Y + 45
ROW_H = 48
MAX_ROWS = 3
COL_DIV_X = 784
CHART_X0 = 827
CHART_X1 = W - M
LEGEND_Y = HEADER_Y + 34
CHART_TOP = LEGEND_Y + 20
CHART_H = ROWS_Y0 + MAX_ROWS * ROW_H - CHART_TOP - 5
RODAPE_Y = 848

G4_THRESH = 15


def dec1(v: float) -> str:
    return f"{v:.1f}".replace(".", ",")


# ======================================================================
# fundo: um unico feixe de luz na cor do clube da maior alta da semana
# ======================================================================

PROMPT_VERSION = 1


def bg_prompt(cor: str) -> str:
    """Mesmas restricoes do prompt do Confronto (o que importa e a parte
    negativa -- "no text" -- entao segue em ingles), so a composicao muda:
    um feixe so, subindo, em vez de dois se encontrando no meio."""
    return (
        "Cinematic abstract background for a weekly sports-stats recap "
        "card, wide establishing shot inside a large empty football "
        "stadium late at night, moments before kickoff. "
        "A single tall column of light rises from the pitch near the "
        "lower-left third of the frame, climbing up toward the top edge "
        "like a spotlight tracking upward momentum, thin drifting haze "
        f"catching the beam, soft defined edges, faint halo, burning {cor}. "
        "The rest of the frame stays near-black. Floodlight rigs sit "
        "above the top edge, out of frame. "
        "Mown grass with visible mower stripes fades into shadow beneath "
        "the light column, a wet sheen catching the color, a thin drift "
        "of smoke or mist hanging just above the turf. Suspended dust "
        "motes and fine particles float through the beam. Far behind, "
        "the stands read only as dim out-of-focus texture and scattered "
        "pinpricks of light, deep in shadow. "
        "The top quarter of the frame must stay almost black and empty "
        "-- no lamp, no flare, no bright object up there. "
        "Shot on a 35mm anamorphic lens at a wide aperture, shallow "
        "depth of field, deep saturated color inside the beam and "
        "crushed blacks everywhere else, gentle falloff, subtle lens "
        "bloom, fine 35mm film grain. Moody, atmospheric and restrained "
        "-- editorial, not a glossy video-game splash screen. "
        "Absolutely no text, no letters, no numbers, no logos, no crests, "
        "no badges, no jerseys, no players, no faces, no watermarks, "
        "no scoreboard, no advertising boards, no goalposts in the "
        "center of the frame."
    )


def bg_procedural(cor: str) -> Image.Image:
    """Fundo sem IA: feixe vertical procedural na cor do clube, pra iterar
    layout de graca (mesma logica de `bg_procedural` do Confronto)."""
    rgb = tuple(int(cor.lstrip("#")[i:i + 2], 16) for i in (0, 2, 4))
    img = Image.new("RGB", (W, H), hexa("bg")[:3])
    px = img.load()
    cx = W * 0.30
    for x in range(W):
        dist = abs(x - cx) / (W * 0.22)
        ganho = math.exp(-dist * dist)
        for y in range(0, H, 3):
            t = y / H
            escuro = 0.15 + 0.55 * t  # mais claro embaixo, como o prompt pede
            g = ganho * escuro
            r, g_, b = px[x, y]
            px[x, y] = (
                int(r + (rgb[0] - r) * g),
                int(g_ + (rgb[1] - g_) * g),
                int(b + (rgb[2] - b) * g),
            )
    return img.resize((W, H)).convert("RGBA") if img.size != (W, H) else img.convert("RGBA")


def bg_ia(cor: str, qualidade: str, forcar: bool) -> Image.Image:
    prompt = bg_prompt(cor)
    chave = hashlib.sha1(
        f"status|{PROMPT_VERSION}|{MODEL}|{AI_SIZE}|{qualidade}|{prompt}".encode()
    ).hexdigest()[:16]
    cache = BG_CACHE / f"{chave}.png"

    if cache.exists() and not forcar:
        print(f"  fundo: cache {cache.name} (sem custo)")
        return Image.open(cache).convert("RGBA")

    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        raise SystemExit(
            "OPENAI_API_KEY nao definido (veja .env.example). "
            "Pra desenhar o card sem chamar a API, use --sem-ia."
        )

    print(f"  fundo: gerando na API ({MODEL}, {AI_SIZE}, qualidade {qualidade})...")
    resp = requests.post(
        OPENAI_IMAGES,
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        json={
            "model": MODEL,
            "prompt": prompt,
            "size": AI_SIZE,
            "quality": qualidade,
            "n": 1,
            "output_format": "png",
        },
        timeout=180,
    )
    if not resp.ok:
        raise SystemExit(f"API de imagem respondeu {resp.status_code}: {resp.text[:500]}")

    item = resp.json()["data"][0]
    if item.get("b64_json"):
        bruto = base64.b64decode(item["b64_json"])
    else:
        bruto = requests.get(item["url"], timeout=60).content

    BG_CACHE.mkdir(parents=True, exist_ok=True)
    cache.write_bytes(bruto)
    return Image.open(cache).convert("RGBA")


def preparar_fundo(img: Image.Image) -> Image.Image:
    """Corta pra 16:9, realca a atmosfera e protege as zonas de texto --
    mesmos passos do Confronto (crop, blur minimo, brilho/cor, veu global,
    vinheta), so sem o ganho por lado (aqui so tem um feixe, nao dois)."""
    alvo = W / H
    if abs(img.width / img.height - alvo) > 0.01:
        nova_h = int(img.width / alvo)
        if nova_h <= img.height:
            topo = (img.height - nova_h) // 2
            img = img.crop((0, topo, img.width, topo + nova_h))
        else:
            nova_w = int(img.height * alvo)
            esq = (img.width - nova_w) // 2
            img = img.crop((esq, 0, esq + nova_w, img.height))
    img = img.resize((W, H), Image.LANCZOS).convert("RGBA")

    img = img.filter(ImageFilter.GaussianBlur(0.5))
    img = ImageEnhance.Brightness(img).enhance(1.30)
    img = ImageEnhance.Color(img).enhance(1.15)

    img = Image.alpha_composite(img, Image.new("RGBA", (W, H), hexa("bg", 55)))

    vinheta = Image.new("L", (W // 8, H // 8), 0)
    d = ImageDraw.Draw(vinheta)
    d.ellipse((-W // 16, -H // 16, W // 8 + W // 16, H // 8 + H // 16), fill=255)
    vinheta = vinheta.resize((W, H), Image.BICUBIC).filter(ImageFilter.GaussianBlur(70))
    escuro = Image.new("RGBA", (W, H), hexa("bg", 255))
    escuro.putalpha(Image.eval(vinheta, lambda v: int((255 - v) * 0.42)))
    img = Image.alpha_composite(img, escuro)

    # topo: wordmark + tag. base: tudo que a partir da secao de chips pra
    # baixo senta direto no fundo (sem painel proprio), entao precisa de
    # mais contraste que os dois cartoes-KPI (que tem painel opaco).
    img = Image.alpha_composite(img, _scrim(SimpleLayout, (0, 0, W, CAB_Y + CAB_H + 40), 120, 0))
    img = Image.alpha_composite(img, _scrim(SimpleLayout, (0, DIV1_Y - 20, W, H), 40, 150))
    return img


class SimpleLayout:
    """`_scrim()` do card_confronto so usa L.W/L.H -- em vez de importar a
    classe Layout (acoplada aos 3 formatos do Confronto), um objeto
    minimo com os 2 atributos que a funcao realmente le."""
    W, H = W, H


# ======================================================================
# o card
# ======================================================================

def kpi_box(img: Image.Image, d: ImageDraw.ImageDraw, box_x: int, label: str,
            item: dict | None, max_abs: float) -> None:
    pad = 37
    painel(img, (box_x, BOX_Y, box_x + BOX_W, BOX_Y + BOX_H), 18, hexa("surface", 225))
    d.rounded_rectangle((box_x, BOX_Y, box_x + BOX_W, BOX_Y + BOX_H), radius=18,
                        outline=hexa("line2"), width=1)
    texto(d, (box_x + pad, BOX_Y + 40), label, fonte(16), hexa("ink3"), tracking=1.3)

    if not item:
        texto(d, (box_x + pad, BOX_Y + BOX_H / 2 + 14), "Sem dado da rodada anterior",
              fonte(20, bold=False), hexa("ink4"))
        return

    up = item["delta_pp"] >= 0
    cor = hexa("accent") if up else hexa("#F5786A")
    nome = item["team"]
    tam = 35 if len(nome) > 12 else 39 if len(nome) > 8 else 43
    texto(d, (box_x + pad, BOX_Y + 88), nome, fonte(tam), hexa("ink"))

    seta = "↑ +" if up else "↓ "
    texto(d, (box_x + pad, BOX_Y + 148), f"{seta}{dec1(item['delta_pp'])} p.p.", fonte(45), cor)

    x_sub = box_x + BOX_W - pad
    texto(d, (x_sub, BOX_Y + 148), "agora em", fonte(17, bold=False), hexa("ink3"), anchor="ra")
    texto(d, (x_sub, BOX_Y + 171), f"{dec1(item['atual_pct'])}%", fonte(20), hexa("ink2"),
          anchor="ra")

    bar_y, bar_w = BOX_Y + BOX_H - 34, BOX_W - 2 * pad
    d.rounded_rectangle((box_x + pad, bar_y, box_x + pad + bar_w, bar_y + 11), radius=5,
                        fill=hexa("surface2"))
    fill_w = max(8, bar_w * min(1.0, abs(item["delta_pp"]) / max_abs))
    d.rounded_rectangle((box_x + pad, bar_y, box_x + pad + fill_w, bar_y + 11), radius=5, fill=cor)


def chip_row(img: Image.Image, d: ImageDraw.ImageDraw, items: list[dict],
             x0: int, y0: int) -> None:
    """Uma linha so (o card fica denso demais com mais), com medida real
    de texto do Pillow em vez do estimador por caractere que o SVG usa
    (svg estatico nao tem `textlength` disponivel antes de renderizar;
    aqui tem, entao a largura da pilula e exata)."""
    f = fonte(17)
    x, shown = x0, 0
    max_x = W - M
    for it in items:
        w = largura_texto(d, it["text"], f) + 46
        if x + w > max_x:
            break
        painel(img, (x, y0, x + w, y0 + CHIP_H), CHIP_H / 2, hexa("surface", 225))
        d.rounded_rectangle((x, y0, x + w, y0 + CHIP_H), radius=CHIP_H / 2,
                            outline=hexa("line2"), width=1)
        d.ellipse((x + 16, y0 + CHIP_H / 2 - 5, x + 26, y0 + CHIP_H / 2 + 5), fill=it["color"])
        texto(d, (x + 34, y0 + CHIP_H / 2), it["text"], f, hexa("ink2"), anchor="lm")
        x += w + 14
        shown += 1
    rest = len(items) - shown
    if rest > 0:
        texto_rest = f"+{rest} mudança{'s' if rest > 1 else ''}"
        w = largura_texto(d, texto_rest, f) + 46
        painel(img, (x, y0, x + w, y0 + CHIP_H), CHIP_H / 2, hexa("surface2", 225))
        texto(d, (x + w / 2, y0 + CHIP_H / 2), texto_rest, f, hexa("ink3"), anchor="mm")


def montar_card(payload: dict, meta: dict, fundo: Image.Image) -> Image.Image:
    m = payload["mudancas_semana"]
    max_abs = max(
        abs((m.get("titulo_alta") or {}).get("delta_pp", 0)),
        abs((m.get("titulo_queda") or {}).get("delta_pp", 0)),
        1,
    )

    img = preparar_fundo(fundo)
    d = ImageDraw.Draw(img)
    HALO = (img, 7, 235)

    # --- casco: faixa de topo, wordmark, tag de rodada ---
    d.rectangle((0, 0, W, 8), fill=hexa("accent"))
    painel(img, (M, CAB_Y, M + CAB_H, CAB_Y + CAB_H), 10, hexa("accent"))
    texto(d, (M + CAB_H / 2, CAB_Y + CAB_H / 2), "FS", fonte(21), hexa("onAccent"), anchor="mm")
    texto(d, (M + CAB_H + 16, CAB_Y + CAB_H / 2), "FORA DA SÚMULA", fonte(23), hexa("ink"),
          anchor="lm", halo=HALO)

    tag = f"BRASILEIRÃO {payload['season']} · RODADA {payload['current_round']}"
    f_tag = fonte(17, bold=False)
    tag_w = largura_texto(d, tag, f_tag) + 40
    tag_x0 = DIR - tag_w
    painel(img, (tag_x0, CAB_Y, tag_x0 + tag_w, CAB_Y + CAB_H), CAB_H // 2, hexa("surface2", 225))
    d.rounded_rectangle((tag_x0, CAB_Y, tag_x0 + tag_w, CAB_Y + CAB_H), radius=CAB_H // 2,
                        outline=hexa("line2"), width=1)
    texto(d, (tag_x0 + tag_w / 2, CAB_Y + CAB_H / 2), tag, f_tag, hexa("ink2"), anchor="mm")

    prev_round = payload.get("prev_round")
    prev_label = (f"RODADA {prev_round} → {payload['current_round']}" if prev_round is not None
                 else f"RODADA {payload['current_round']}")
    texto(d, (M, LABEL_Y), f"O QUE MUDOU — {prev_label}", fonte(20), hexa("blue"),
          tracking=2.0, halo=HALO)

    # --- 1. cartoes-KPI (maior alta / maior queda) ---
    kpi_box(img, d, BOX1_X, "MAIOR ALTA DE TÍTULO NA SEMANA", m.get("titulo_alta"), max_abs)
    kpi_box(img, d, BOX2_X, "MAIOR QUEDA DE TÍTULO NA SEMANA", m.get("titulo_queda"), max_abs)

    d.line((M, DIV1_Y, W - M, DIV1_Y), fill=hexa("line2"), width=1)

    # --- 2. mudancas de cenario (chips) ---
    tipo_cor = {
        "título confirmado": hexa("accent"), "título descartado": hexa("ink3"),
        "G4 confirmado": hexa("accent"), "G4 descartado": hexa("ink3"),
        "rebaixamento confirmado": hexa("#F5786A"), "fora do Z4": hexa("accent"),
    }
    chips = []
    for x in m.get("g4_entrou", []):
        chips.append({"text": f"{x['team']} entrou no G4 (+{dec1(x['delta_pp'])}pp)", "color": hexa("accent")})
    for x in m.get("g4_saiu", []):
        chips.append({"text": f"{x['team']} saiu do G4 ({dec1(x['delta_pp'])}pp)", "color": hexa("#F5786A")})
    for n in m.get("novidades", []):
        chips.append({"text": f"{n['team']}: {n['tipo']}", "color": tipo_cor.get(n["tipo"], hexa("ink3"))})
    if not chips:
        chips = [{"text": "Nenhum cenário virou nesta rodada", "color": hexa("ink4")}]

    texto(d, (M, CHIP_LABEL_Y),
          f"MUDANÇAS DE CENÁRIO NESTA RODADA (G4 ±{G4_THRESH}p.p. ou confirmação/descarte)",
          fonte(16), hexa("ink3"), tracking=1.3, halo=HALO)
    chip_row(img, d, chips, M, CHIP_Y0)

    d.line((M, DIV2_Y, W - M, DIV2_Y), fill=hexa("line2"), width=1)

    # --- 3a. cenarios acumulados (esquerda) ---
    cen = payload["cenarios"]
    grupos = [
        ("Título confirmado", "titulo_confirmado", hexa("accent")),
        ("Título descartado", "titulo_descartado", hexa("ink3")),
        ("G4 confirmado", "g4_confirmado", hexa("accent")),
        ("G4 descartado", "g4_descartado", hexa("ink3")),
        ("Z4 confirmado (rebaixado)", "z4_confirmado", hexa("#F5786A")),
        ("Fora do Z4 (salvo)", "z4_descartado", hexa("accent")),
    ]
    grupos = [
        (label, cor, [c["team"] for c in cen if c.get(key)])
        for label, key, cor in grupos
    ]
    grupos = [g for g in grupos if g[2]]

    texto(d, (M, HEADER_Y), f"CENÁRIOS ACUMULADOS — RODADA {payload['current_round']}",
          fonte(16), hexa("ink3"), tracking=1.3, halo=HALO)
    if len(grupos) > MAX_ROWS:
        extra = len(grupos) - MAX_ROWS
        texto(d, (COL_DIV_X - 40, HEADER_Y), f"+{extra} categoria{'s' if extra > 1 else ''}",
              fonte(15), hexa("ink4"), anchor="ra", halo=HALO)

    for i, (label, cor, times) in enumerate(grupos[:MAX_ROWS]):
        y = ROWS_Y0 + i * ROW_H
        d.rounded_rectangle((M, y, M + 8, y + 37), radius=4, fill=cor)
        nomes = ", ".join(times[:3]) + (f" +{len(times) - 3}" if len(times) > 3 else "")
        texto(d, (M + 22, y + 2), f"{label.upper()} ({len(times)})", fonte(15), cor, halo=HALO)
        texto(d, (M + 22, y + 22), nomes, fonte(19, bold=False), hexa("ink2"), halo=HALO)

    # --- 3b. ritmo do lider (direita) ---
    seasons = sorted(payload["ritmo"].keys())
    cur_season = str(payload["season"])
    cur = payload["ritmo"].get(cur_season, [])
    others = [s for s in seasons if s != cur_season]
    max_n = max(len(payload["ritmo"][s]) for s in seasons)
    max_v = max(v for s in seasons for v in payload["ritmo"][s])

    def px(i):
        return CHART_X0 + i / (max_n - 1) * (CHART_X1 - CHART_X0)

    def pv(v):
        return CHART_TOP + CHART_H - v / max_v * CHART_H

    texto(d, (CHART_X0, HEADER_Y), "RITMO DO LÍDER", fonte(16), hexa("ink3"),
          tracking=1.3, halo=HALO)
    last_i = len(cur) - 1
    prev_season = others[-1] if others else None
    prev_at_same = None
    if prev_season and last_i < len(payload["ritmo"][prev_season]):
        prev_at_same = payload["ritmo"][prev_season][last_i]
    if prev_at_same is not None:
        texto(d, (W - M, HEADER_Y), f"{prev_season} na rodada {last_i + 1}: {prev_at_same} pts",
              fonte(15), hexa("ink4"), anchor="ra", halo=HALO)

    leg_x = CHART_X0
    for yr, cor, bold in [(cur_season, hexa("accent"), True)] + [(s, hexa("ink4"), False) for s in others]:
        d.line((leg_x, LEGEND_Y - 5, leg_x + 18, LEGEND_Y - 5), fill=cor, width=4)
        texto(d, (leg_x + 26, LEGEND_Y), yr, fonte(15, bold=bold), cor, anchor="lm")
        leg_x += 26 + largura_texto(d, yr, fonte(15, bold=bold)) + 20

    overlay = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    od = ImageDraw.Draw(overlay)
    od.line((CHART_X0, CHART_TOP + CHART_H, CHART_X1, CHART_TOP + CHART_H),
            fill=hexa("line2"), width=1)
    for s in others:
        pts = [(px(i), pv(v)) for i, v in enumerate(payload["ritmo"][s])]
        od.line(pts, fill=hexa("ink4", 140), width=3, joint="curve")
    pts_cur = [(px(i), pv(v)) for i, v in enumerate(cur)]
    od.line(pts_cur, fill=hexa("accent"), width=4, joint="curve")
    img.alpha_composite(overlay)
    d = ImageDraw.Draw(img)

    if cur:
        cx_last, cy_last = px(last_i), pv(cur[last_i])
        d.ellipse((cx_last - 7, cy_last - 7, cx_last + 7, cy_last + 7), fill=hexa("accent"))
        acima = cy_last - CHART_TOP > 26
        texto(d, (cx_last, cy_last - 18 if acima else cy_last + 28), f"{cur[last_i]} pts",
              fonte(17), hexa("accent"), anchor="mm", halo=HALO)

    texto(d, (DIR, RODAPE_Y), "foradasumula.com.br", fonte(17), hexa("ink3"),
          anchor="rs", halo=HALO)

    return img.convert("RGB")


def find_team_id(payload: dict, nome: str) -> str | None:
    for tid, t in payload["teams_confronto"].items():
        if t["short_name"] == nome:
            return tid
    return None


def gerar(args) -> Path:
    carrega_env()
    payload = load_payload()
    meta = load_teams_meta()

    m = payload["mudancas_semana"]
    alta = m.get("titulo_alta")
    tid = find_team_id(payload, alta["team"]) if alta else payload["team_order"][0]
    cor = cor_marca(meta.get(tid, {}))

    if args.sem_ia:
        print(f"fundo: procedural (--sem-ia, sem custo) -- cor {cor}")
        fundo = bg_procedural(cor)
    else:
        print(f"fundo: cor {cor} (clube da maior alta da semana)")
        fundo = bg_ia(cor, args.qualidade, args.forcar_fundo)

    card = montar_card(payload, meta, fundo)
    saida = Path(args.saida) / f"status-semanal-rodada-{payload['current_round']}.png"
    saida.parent.mkdir(parents=True, exist_ok=True)
    card.save(saida, "PNG", optimize=True)
    print(f"-> {saida} ({W}x{H}, {saida.stat().st_size / 1024:.0f} KB)")
    return saida


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Card de Status Semanal em PNG (fundo por IA, dado e texto em Pillow)."
    )
    ap.add_argument("--sem-ia", action="store_true",
                    help="fundo procedural, sem chamar a API de imagem")
    ap.add_argument("--qualidade", default="low", choices=["low", "medium", "high"],
                    help="qualidade da imagem na API (default low)")
    ap.add_argument("--forcar-fundo", action="store_true",
                    help="ignora o cache e gera o fundo de novo (custa credito)")
    ap.add_argument("--saida", default=str(OUT_DIR), help=f"pasta de saida (default {OUT_DIR})")
    args = ap.parse_args()
    gerar(args)


if __name__ == "__main__":
    main()

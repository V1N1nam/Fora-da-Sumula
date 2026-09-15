"""Card de Confronto em PNG: fundo gerado por IA, resto desenhado em Pillow.

Divisao de trabalho (e o ponto todo do modulo):

  dado do payload -> template em Python -> fundo/atmosfera pela IA
  -> escudo, nome, numero e legenda desenhados pelo Pillow

A IA **nunca** escreve texto nem numero no card. Ela so pinta um fundo
abstrato (luz de estadio, degrade nas cores dos dois clubes), que ainda
leva escurecimento + vinheta por cima antes de qualquer coisa ser
escrita. Isso vale pelos dois lados da regra do projeto: todo card sai
do dado, nunca da mao (nem da alucinacao) -- e modelo de imagem erra
texto, acento e escudo com uma facilidade que nao compensa o risco.

Economia de credito (o pedido era "mandar o maximo possivel de uma vez"):

  * o fundo e cacheado em disco por chave = (versao do prompt, cores dos
    dois clubes, tamanho, qualidade). Gerar o mesmo confronto de novo, ou
    um confronto entre dois clubes de cores equivalentes, nao gasta nada.
  * `--todos-proxima` monta os 10 jogos da rodada em uma execucao, ja
    reaproveitando o cache entre eles.
  * `--sem-ia` desenha um fundo procedural (degrade nas cores dos clubes)
    e nao chama a API -- da pra iterar layout de graca e so no fim ligar
    a IA.
  * o default e `--qualidade low`. Testado contra `high` e `medium` no
    mesmo par de cores: o que fazia o fundo parecer ruim no comeco era o
    pos-processamento, nao o tier. Depois de arrumar o pos, `low` nao
    ficou atras -- e o pos achata detalhe fino de qualquer jeito.
  * os dois formatos em pe (`feed` e `story`) pedem o mesmo tamanho a
    API (1024x1536) e o mesmo prompt, entao compartilham fundo: gerar os
    dois custa uma imagem, nao duas.

Como so existem 7 cores de marca entre os 20 clubes (`cor_marca` mapeia
o clubColors da API num palete fixo), o universo inteiro de fundos e
28 pares por orientacao. Na pratica o custo converge pra zero depois
das primeiras semanas.

Formatos de saida (`--formato`): `x` 1600x900, `feed` 1080x1350 e
`story` 1080x1920. Aceita mais de um na mesma chamada.

De onde vem o dado: do payload que `build_site.py` ja gravou em
`docs/export/index.html`, lido pela **mesma sentinela** `/*__DATA__*/`.
Nada de numero e recalculado aqui -- forca, posicao, pontos e h2h saem
prontos, e a formula de probabilidade e a transcricao literal de
`confrontoProb()` do template. Se o card e o site divergirem, e bug de
transcricao, nao de fonte de dado.

Fora do pipeline do Actions (como `calibrate_elo.py` e `validate_elo.py`):
roda a mao, quando se quer a arte pra postar.

Uso:
    python src/card_confronto.py --casa Flamengo --fora Palmeiras
    python src/card_confronto.py --casa FLA --fora PAL --formato x feed story
    python src/card_confronto.py --todos-proxima
    python src/card_confronto.py --casa FLA --fora PAL --sem-ia
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import math
import os
import random
import re
import sys
import unicodedata
from pathlib import Path

import numpy as np
import requests
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import COMPETITION, CURRENT_SEASON, ROOT  # noqa: E402

# --- caminhos ---------------------------------------------------------

EXPORT_HTML = ROOT / "docs" / "export" / "index.html"
ASSETS = ROOT / "assets"
CRESTS = ASSETS / "crests"
TEAMS_JSON = ASSETS / "teams.json"
CARDS = ROOT / "cards"
BG_CACHE = CARDS / "bg"
OUT_DIR = CARDS / "out"

# --- formatos de saida ------------------------------------------------
#
# Tres destinos, cada um com o tamanho que a rede recomenda:
#   x     1600x900  (16:9) feed do X. Mesmo aspecto do card SVG do
#                   gerador (1200x675) -- os dois sao irmaos visuais.
#   feed  1080x1350 (4:5)  post estatico do Instagram, o que mais ocupa
#                   tela no feed.
#   story 1080x1920 (9:16) story/reels.
#
# `ai` e o tamanho pedido a API, que so oferece 1536x1024 (paisagem),
# 1024x1536 (retrato) e 1024x1024. Pegamos o de orientacao mais proxima
# e cortamos o miolo -- por isso o prompt pede espaco limpo no centro.
# Cortar 16:9 de uma paisagem pra virar 9:16 destruiria a composicao,
# entao formato em pe pede geracao em pe: o tamanho entra na chave do
# cache, logo cada formato tem o fundo dele.
#
# `empilhado` decide o arranjo: no 16:9 a barra fica a esquerda e o
# comparativo a direita; nos formatos em pe tudo desce em blocos.
FORMATOS = {
    "x":     {"w": 1600, "h": 900,  "ai": "1536x1024", "empilhado": False},
    "feed":  {"w": 1080, "h": 1350, "ai": "1024x1536", "empilhado": True},
    "story": {"w": 1080, "h": 1920, "ai": "1024x1536", "empilhado": True},
}


class Layout:
    """Coordenadas do card para um formato.

    Os numeros sao explicitos por formato em vez de derivados de uma
    regra generica: sao tres layouts que precisam ser olhados a olho de
    qualquer jeito, e regra generica aqui daria a ilusao de que mudar
    uma proporcao e seguro.
    """

    def __init__(self, nome: str):
        f = FORMATOS[nome]
        self.nome = nome
        self.W, self.H = f["w"], f["h"]
        self.ai_size = f["ai"]
        self.empilhado = f["empilhado"]

        if nome == "x":
            self.M = 75
            self.cab_y, self.cab_h = 58, 44
            self.label_y = 152
            self.clube_y, self.creste = 196, 108
            self.bar = (75, 396, 780, 168, 20)
            self.col_x0, self.col_x1 = 1075, self.W - 75
            self.div_x, self.stats_y, self.linha_h = 1015, 396, 100
            self.nota_y, self.rodape_y = 792, 848
        elif nome == "feed":
            self.M = 60
            self.cab_y, self.cab_h = 50, 44
            self.label_y = 150
            self.clube_y, self.creste = 200, 96
            self.bar = (60, 420, 960, 190, 22)
            self.col_x0, self.col_x1 = 60, self.W - 60
            self.div_x, self.stats_y, self.linha_h = None, 710, 115
            self.nota_y, self.rodape_y = 1170, 1295
        else:  # story
            # area segura: a interface do Instagram cobre ~160px no topo
            # (foto e nome) e ~250px na base (campo de resposta), entao o
            # conteudo vive entre ~200 e ~1660. Sem isso o wordmark e a
            # atribuicao saem escondidos atras da UI no story publicado.
            self.M = 60
            self.cab_y, self.cab_h = 210, 48
            self.label_y = 350
            self.clube_y, self.creste = 420, 120
            self.bar = (60, 720, 960, 230, 26)
            self.col_x0, self.col_x1 = 60, self.W - 60
            self.div_x, self.stats_y, self.linha_h = None, 1060, 132
            self.nota_y, self.rodape_y = 1500, 1620

    @property
    def DIR(self) -> int:
        return self.W - self.M

# --- paleta (a mesma de fora-da-sumula-export.html) --------------------

C = {
    "bg": "#0B0E10", "surface": "#13171A", "surface2": "#181D20",
    "line": "#232A2C", "line2": "#333B3E",
    "ink": "#ECEFEE", "ink2": "#A5AFB3", "ink3": "#7B888D", "ink4": "#7C858B",
    "accent": "#3FD49C", "accentSoft": "#10302A",
    "blue": "#7DB0F0", "blueSoft": "#152537",
    "amber": "#E4B54C",
    "onAccent": "#08130F",
}


def hexa(name: str, alpha: int = 255) -> tuple[int, int, int, int]:
    """Cor da paleta como RGBA (Pillow nao le '#rrggbb' com alpha)."""
    h = C.get(name, name).lstrip("#")
    return (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16), alpha)


# ======================================================================
# payload
# ======================================================================

SENTINEL = re.compile(r"^const DATA = (.*?); /\*__DATA__\*/$", re.M | re.S)


def load_payload() -> dict:
    if not EXPORT_HTML.exists():
        raise SystemExit(
            f"{EXPORT_HTML} nao existe. Rode o pipeline antes "
            "(ingest -> elo -> derived -> build_site)."
        )
    m = SENTINEL.search(EXPORT_HTML.read_text(encoding="utf-8"))
    assert m, (
        "nao achei a sentinela /*__DATA__*/ em docs/export/index.html. "
        "Se o formato do payload de export mudou, este modulo precisa "
        "acompanhar -- ele le o mesmo JSON que o gerador de imagens."
    )
    return json.loads(m.group(1))


def _fold(s: str) -> str:
    """Normaliza pra busca: sem acento, sem hifen, minusculo."""
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]", "", s.lower())


def resolve_team(payload: dict, meta: dict, termo: str) -> str:
    """Aceita nome, apelido ou TLA e devolve o team_id (string) do payload."""
    alvo = _fold(termo)
    cands: dict[str, list[str]] = {}
    for tid, t in payload["teams_confronto"].items():
        chaves = [t["short_name"]]
        if tid in meta:
            chaves += [meta[tid]["name"], meta[tid]["tla"], meta[tid]["short_name"]]
        cands[tid] = [_fold(k) for k in chaves]

    for tid, chaves in cands.items():
        if alvo in chaves:
            return tid
    parciais = [tid for tid, chaves in cands.items() if any(alvo in k for k in chaves)]
    if len(parciais) == 1:
        return parciais[0]
    if parciais:
        nomes = ", ".join(payload["teams_confronto"][t]["short_name"] for t in parciais)
        raise SystemExit(f"'{termo}' e ambiguo: {nomes}")
    nomes = ", ".join(sorted(t["short_name"] for t in payload["teams_confronto"].values()))
    raise SystemExit(f"clube '{termo}' nao encontrado. Opcoes: {nomes}")


# ======================================================================
# modelo -- transcricao literal do template de export
# ======================================================================

def confronto_prob(rh: float, ra: float, hfa: float, nu: float) -> tuple[float, float, float]:
    """Elo + Davidson, igual a confrontoProb() do gerador de imagens.

    Nao reimplementa nada: HFA e nu vem do payload (que por sua vez vem
    do config.py). Recalibrar o modelo nao pede toque aqui.
    """
    th = 10 ** ((rh + hfa) / 400)
    ta = 10 ** (ra / 400)
    dr = nu * math.sqrt(th * ta)
    s = th + ta + dr
    return th / s, dr / s, ta / s


def plural(n: int, s: str, p: str | None = None) -> str:
    return s if n == 1 else (p or s + "s")


def h2h_note(payload: dict, hk: str, ak: str) -> tuple[str, bool]:
    """(texto, amostra_pequena) -- igual a h2hNoteConfronto()."""
    T = payload["teams_confronto"]
    a, b = min(int(hk), int(ak)), max(int(hk), int(ak))
    rec = payload["h2h"].get(f"{a}_{b}")
    if not rec:
        return "Nunca se enfrentaram na Série A desde 2023.", False
    if rec["jogos"] < 4:
        return (
            f"Poucos confrontos recentes entre os dois ({rec['jogos']} "
            f"{plural(rec['jogos'], 'jogo')} desde 2023) — amostra pequena "
            "demais pra detalhar retrospecto.",
            True,
        )
    h_is_a = int(hk) == rec["a"]
    vh = rec["va"] if h_is_a else rec["vb"]
    vv = rec["vb"] if h_is_a else rec["va"]
    return (
        f"Retrospecto desde 2023 ({rec['jogos']} jogos): "
        f"{vh} {plural(vh, 'vitória', 'vitórias')} do {T[hk]['short_name']}, "
        f"{rec['empates']} {plural(rec['empates'], 'empate')}, "
        f"{vv} {plural(vv, 'vitória', 'vitórias')} do {T[ak]['short_name']}.",
        False,
    )


DISCLAIMERS = [
    "Isso é probabilidade do modelo (Elo + 10 mil simulações), não é garantia de resultado.",
    "Número de simulação, não fato consumado — o jogo ainda decide sozinho.",
    "Modelo estatístico, não bola de cristal: trate como chance, não certeza.",
    "Probabilidade calculada antes da bola rolar, não uma previsão fechada do placar.",
]

CAP_CONFRONTO = {
    "parelho": [
        lambda h, a, pct, fav: f"{h} e {a} chegam parelhos nas contas de força — o modelo não vê favorito claro nesse confronto.",
        lambda h, a, pct, fav: f"Duelo equilibrado entre {h} e {a}: a força atual dos dois times está bem próxima.",
    ],
    "favorito": [
        lambda h, a, pct, fav: f"{fav} entra como favorito pelo modelo de força, mas {pct} de chance pro outro lado mantém o jogo em aberto.",
        lambda h, a, pct, fav: f"Força atual favorece o {fav} nesse confronto entre {h} e {a}, com {pct} reservado pro resultado inverso.",
    ],
    "claro": [
        lambda h, a, pct, fav: f"{fav} é favorito com folga pelo modelo de força — o outro lado fica com só {pct} de chance.",
        lambda h, a, pct, fav: f"Diferença grande de força entre {h} e {a}: o modelo dá só {pct} de chance pro azarão.",
    ],
}


def pct1(v: float) -> str:
    return f"{v * 100:.1f}".replace(".", ",") + "%"


def pct0(v: float) -> str:
    return f"{round(v * 100)}%"


def caption(payload: dict, hk: str, ak: str, h: float, a: float) -> str:
    """Legenda de rede social: template local com variacao, sem LLM --
    mesma regra do gerador de imagens (o texto sai do dado, nao da IA)."""
    T = payload["teams_confronto"]
    nh, na = T[hk]["short_name"], T[ak]["short_name"]
    fav_casa = h >= a
    fav_pct, azar_pct = (h, a) if fav_casa else (a, h)
    fav = nh if fav_casa else na
    diff = fav_pct - azar_pct
    tier = "parelho" if diff < 0.06 else "favorito" if diff < 0.25 else "claro"
    linha = random.choice(CAP_CONFRONTO[tier])(nh, na, pct1(azar_pct), fav)
    return f"{linha} {random.choice(DISCLAIMERS)}"


# ======================================================================
# escudos e cores dos clubes (football-data.org, 1 request por refresh)
# ======================================================================

# clubColors vem da API como texto ("Maroon / Green / White"). Mapa so
# das palavras que aparecem na Serie A -- palavra fora do mapa cai no
# cinza neutro e o fundo sai generico, nao quebra.
COLOR_WORDS = {
    "white": "#F2F2F2", "black": "#1A1A1A", "red": "#D32F2F",
    "blue": "#1E5AA8", "green": "#1E8E4E", "yellow": "#F2C438",
    "maroon": "#7B1F2B", "gold": "#D4A62A", "orange": "#E4762B",
    "purple": "#6B3FA0", "grey": "#8A8A8A", "gray": "#8A8A8A",
    "crimson": "#B3153A", "navy": "#152A5E", "sky": "#6FB2E8",
    "claret": "#7B1F2B", "silver": "#C0C0C0", "brown": "#6B4423",
    "pink": "#E06A9C",
}


def carrega_env() -> None:
    """Le .env pro ambiente, sem sobrescrever o que ja veio de fora.

    O resto do pipeline roda no Actions, onde as chaves chegam por
    secret -- este modulo roda na mao, onde elas moram no .env. Sem
    dependencia nova: o formato e CHAVE=valor por linha.
    """
    env = ROOT / ".env"
    if not env.exists():
        return
    for linha in env.read_text(encoding="utf-8").splitlines():
        linha = linha.strip()
        if not linha or linha.startswith("#") or "=" not in linha:
            continue
        chave, valor = linha.split("=", 1)
        os.environ.setdefault(chave.strip(), valor.strip().strip("'\""))


def _token() -> str:
    tok = os.environ.get("FOOTBALL_DATA_TOKEN")
    if not tok:
        raise SystemExit(
            "FOOTBALL_DATA_TOKEN nao definido (veja .env.example). "
            "So e preciso na primeira vez, pra baixar escudo e cores "
            "dos 20 clubes -- depois tudo fica em assets/."
        )
    return tok


def load_teams_meta(refresh: bool = False) -> dict:
    """Escudo (PNG) e cores de cada clube, cacheados em assets/.

    Uma unica chamada a /competitions/BSA/teams cobre os 20 clubes; o
    resultado e estavel ao longo da temporada, entao so se refaz com
    --refresh-times (rebranding, clube promovido no ano seguinte).
    """
    if TEAMS_JSON.exists() and not refresh:
        return json.loads(TEAMS_JSON.read_text(encoding="utf-8"))

    print("baixando escudos e cores dos clubes (1 request)...")
    resp = requests.get(
        f"https://api.football-data.org/v4/competitions/{COMPETITION}/teams",
        headers={"X-Auth-Token": _token()},
        params={"season": CURRENT_SEASON},
        timeout=30,
    )
    resp.raise_for_status()

    CRESTS.mkdir(parents=True, exist_ok=True)
    meta = {}
    for t in resp.json()["teams"]:
        tid = str(t["id"])
        crest_path = CRESTS / f"{tid}.png"
        if not crest_path.exists() and t.get("crest"):
            img = requests.get(t["crest"], timeout=30)
            if img.ok:
                crest_path.write_bytes(img.content)
        meta[tid] = {
            "name": t["name"],
            "short_name": t["shortName"],
            "tla": t["tla"],
            "colors": parse_colors(t.get("clubColors") or ""),
        }
    TEAMS_JSON.parent.mkdir(parents=True, exist_ok=True)
    TEAMS_JSON.write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8", newline="\n"
    )
    print(f"  {len(meta)} clubes -> {TEAMS_JSON.relative_to(ROOT)}")
    return meta


def parse_colors(s: str) -> list[str]:
    """'Maroon / Green / White' -> ['#7B1F2B', '#1E8E4E', '#F2F2F2']."""
    out = []
    for parte in s.split("/"):
        for palavra in parte.strip().lower().split():
            if palavra in COLOR_WORDS:
                out.append(COLOR_WORDS[palavra])
                break
    return out or ["#8A8A8A"]


def cor_marca(meta_team: dict) -> str:
    """Cor de assinatura do clube: a primeira que nao e preto nem branco
    (senao Corinthians, Botafogo e Athletico sairiam todos iguais). Se o
    clube so tem preto e branco mesmo, fica a primeira."""
    cores = meta_team.get("colors") or ["#8A8A8A"]
    for c in cores:
        if c not in ("#F2F2F2", "#1A1A1A", "#C0C0C0"):
            return c
    return cores[0]



def load_crest(tid: str, size: int) -> Image.Image | None:
    p = CRESTS / f"{tid}.png"
    if not p.exists():
        return None
    img = Image.open(p).convert("RGBA")
    img.thumbnail((size, size), Image.LANCZOS)
    quadro = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    quadro.paste(img, ((size - img.width) // 2, (size - img.height) // 2), img)
    return quadro


# ======================================================================
# fundo: IA (cacheada) ou procedural
# ======================================================================

# Versao do prompt: entra na chave do cache. Mexeu no texto do prompt,
# suba isso -- senao o cache antigo continua sendo servido e voce nao ve
# a mudanca (e acha que a API ignorou).
PROMPT_VERSION = 4

OPENAI_IMAGES = "https://api.openai.com/v1/images/generations"
MODEL = "gpt-image-1-mini"


def bg_prompt(cor_casa: str, cor_fora: str, L: "Layout") -> str:
    """Prompt em ingles de proposito: modelo de imagem obedece melhor a
    restricao negativa ("no text") em ingles, e aqui a restricao e o que
    mais importa -- todo texto do card e escrito pelo Pillow.

    O prompt descreve a composicao **do card**, nao uma foto bonita. A
    v2 pedia os refletores "nos cantos superiores" e eles nasceram
    exatamente em cima do wordmark, da tag de rodada e dos escudos --
    a parte de cima e a mais ocupada do layout. Aqui a luz e empurrada
    pra metade de baixo, que e onde o card tem espaco vazio, e o topo
    e pedido quase preto. Mexer nessa divisao de zonas sem olhar o
    layout quebra a legibilidade de novo.
    """
    enquadramento = (
        "wide establishing shot" if not L.empilhado
        else "tall vertical portrait-orientation shot"
    )
    # Onde a luz do gramado pode pousar depende do formato, porque as
    # zonas vazias do card mudam. No 16:9 a metade de baixo esta livre.
    # No 9:16 e no 4:5 ela e ocupada pelo comparativo (posicao, pontos,
    # forca): pedir "lower half" ali lavou a linha da FORCA no primeiro
    # story gerado. Em pe, a faixa clara desce pro rodape.
    zona_luz = (
        "The glow pools low: the lower half is the brightest part of the "
        "image," if not L.empilhado else
        "The frame stays dark through the middle; only the bottom sixth "
        "of the image lifts into light, a low bright band of"
    )
    return (
        "Cinematic abstract background for a sports statistics card, "
        f"{enquadramento} inside a large empty football stadium "
        "late at night, moments before kickoff. "
        "The floodlight rigs sit above the top edge, out of frame: only "
        "their volumetric beams enter, raking down at a steep angle "
        "through thin drifting haze, each beam with soft defined edges "
        f"and a faint halo. The beams on the left burn {cor_casa}, the "
        f"beams on the right burn {cor_fora}; where they meet, a deep "
        "near-black neutral wedge runs down the middle of the frame and "
        "the two colors bleed into it without mixing into mud. "
        f"{zona_luz} light washing across mown grass with visible mower "
        "stripes fading into shadow, a wet sheen catching the color, and "
        "a thin drift of smoke or mist hanging just above the turf. "
        "Suspended dust motes and fine particles float through the "
        "beams. Far behind, the stands read only as dim out-of-focus "
        "texture and scattered pinpricks of light, deep in shadow. "
        "The top quarter of the frame must stay almost black and empty "
        "-- no lamp, no flare, no bright object up there. "
        + ("The entire middle of the frame must stay dark, smooth and "
           "featureless. " if L.empilhado else "") +
        "Shot on a 35mm anamorphic lens at a wide aperture, shallow "
        "depth of field, deep saturated color inside the beams and "
        "crushed blacks everywhere else, gentle falloff, subtle lens "
        "bloom, fine 35mm film grain. Moody, atmospheric and restrained "
        "-- editorial, not a glossy video-game splash screen. "
        "Absolutely no text, no letters, no numbers, no logos, no crests, "
        "no badges, no jerseys, no players, no faces, no watermarks, "
        "no scoreboard, no advertising boards, no goalposts in the "
        "center of the frame."
    )


def bg_procedural(cor_casa: str, cor_fora: str, L: "Layout") -> Image.Image:
    """Fundo sem IA: degrade horizontal entre as cores dos dois clubes.

    Nao e placeholder feio -- e o fundo de fallback de verdade, pra
    iterar layout sem gastar credito e pra o card ainda sair se a API
    estiver fora do ar na hora de postar.
    """
    a = tuple(int(cor_casa.lstrip("#")[i:i + 2], 16) for i in (0, 2, 4))
    b = tuple(int(cor_fora.lstrip("#")[i:i + 2], 16) for i in (0, 2, 4))
    faixa = Image.new("RGB", (L.W, 1))
    px = faixa.load()
    for x in range(L.W):
        # curva em S: as cores ficam nas pontas e o meio escurece,
        # deixando o centro limpo -- mesma intencao do prompt da IA.
        t = x / (L.W - 1)
        mistura = t * t * (3 - 2 * t)
        escuro = 1 - 0.55 * math.sin(math.pi * t)
        px[x, 0] = tuple(
            int((a[i] + (b[i] - a[i]) * mistura) * escuro * 0.5) for i in range(3)
        )
    return faixa.resize((L.W, L.H), Image.BICUBIC).convert("RGBA")


def bg_ia(cor_casa: str, cor_fora: str, qualidade: str, forcar: bool,
          L: "Layout") -> Image.Image:
    """Fundo da API de imagem, com cache em disco por (prompt, cores).

    Confronto repetido -- ou par de clubes com as mesmas cores -- nao
    gera request nenhum.
    """
    prompt = bg_prompt(cor_casa, cor_fora, L)
    chave = hashlib.sha1(
        f"{PROMPT_VERSION}|{MODEL}|{L.ai_size}|{qualidade}|{prompt}".encode()
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

    print(f"  fundo: gerando na API ({MODEL}, {L.ai_size}, qualidade {qualidade})...")
    resp = requests.post(
        OPENAI_IMAGES,
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        json={
            "model": MODEL,
            "prompt": prompt,
            "size": L.ai_size,
            "quality": qualidade,
            "n": 1,
            "output_format": "png",
        },
        timeout=180,
    )
    if not resp.ok:
        # O corpo do erro da API diz o motivo real (modelo sem acesso,
        # organizacao nao verificada, parametro invalido). Sem imprimir
        # isso, o traceback nao ajuda em nada.
        raise SystemExit(f"API de imagem respondeu {resp.status_code}: {resp.text[:500]}")

    item = resp.json()["data"][0]
    if item.get("b64_json"):
        bruto = base64.b64decode(item["b64_json"])
    else:
        bruto = requests.get(item["url"], timeout=60).content

    BG_CACHE.mkdir(parents=True, exist_ok=True)
    cache.write_bytes(bruto)
    return Image.open(cache).convert("RGBA")


def _scrim(L: "Layout", caixa, a0: int, a1: int) -> Image.Image:
    """Faixa de escurecimento com alfa variando de a0 (topo) a a1 (base).

    Escurecer o card inteiro pra garantir leitura apaga justamente o
    que se pagou pra gerar: a primeira versao usava veu global de
    alpha 165 sobre um fundo que ja vinha com media RGB 20, e o
    refletor sumia. Aqui o escurecimento e local -- so a faixa onde
    vai cair texto solto (topo do wordmark, base do retrospecto). O
    miolo, onde a barra e o comparativo tem painel proprio, fica
    intacto.
    """
    x0, y0, x1, y1 = caixa
    rampa = Image.new("L", (1, max(y1 - y0, 1)))
    px = rampa.load()
    for i in range(rampa.height):
        px[0, i] = int(a0 + (a1 - a0) * i / max(rampa.height - 1, 1))
    camada = Image.new("RGBA", (L.W, L.H), (0, 0, 0, 0))
    faixa = Image.new("RGBA", (x1 - x0, y1 - y0), hexa("bg", 255))
    faixa.putalpha(rampa.resize((x1 - x0, y1 - y0)))
    camada.paste(faixa, (x0, y0))
    return camada


def _ganho_por_lado(p_casa: float, p_fora: float, largura: int) -> np.ndarray:
    """Rampa horizontal de ganho de brilho: favorito acende mais.

    A luz do card passa a carregar dado. O empate e ignorado de
    proposito -- ele nao tem lado no quadro --, entao a divisao e a
    forca relativa dos dois times: p_casa / (p_casa + p_fora).

    Por que aqui e nao no prompt: a IA nao obedece proporcao numerica
    ("62% contra 38%" ela nao faz), e botar a probabilidade no prompt
    poria ela na chave do cache -- cada confronto viraria uma geracao
    nova. Feito no Python, o mesmo fundo cacheado serve pra qualquer
    par de probabilidades e o resultado e exato.
    """
    total = p_casa + p_fora
    fatia = p_casa / total if total else 0.5

    # expoente < 1 amortece: em 70/30 a diferenca de brilho fica visivel
    # sem um lado estourar e o outro apagar. Clamp pelo mesmo motivo.
    ganho_esq = min(max((2 * fatia) ** 0.7, 0.58), 1.42)
    ganho_dir = min(max((2 * (1 - fatia)) ** 0.7, 0.58), 1.42)

    # transicao suave (smoothstep) no terco central, pra nao aparecer
    # uma emenda vertical no meio do card
    x = np.linspace(0.0, 1.0, largura, dtype=np.float32)
    t = np.clip((x - 0.33) / 0.34, 0.0, 1.0)
    t = t * t * (3 - 2 * t)
    return ganho_esq + (ganho_dir - ganho_esq) * t


def preparar_fundo(img: Image.Image, p_casa: float, p_fora: float,
                   L: "Layout") -> Image.Image:
    """Corta pra 16:9, realca a atmosfera e protege as zonas de texto.

    Este passo e o que torna seguro pedir qualidade baixa: o que
    sobrevive dele e atmosfera, nao detalhe. Tambem e o que garante
    contraste de leitura -- o texto nunca depende do que a IA decidiu
    pintar naquela regiao.
    """
    alvo = L.W / L.H
    if abs(img.width / img.height - alvo) > 0.01:
        nova_h = int(img.width / alvo)
        if nova_h <= img.height:
            topo = (img.height - nova_h) // 2
            img = img.crop((0, topo, img.width, topo + nova_h))
        else:
            nova_w = int(img.height * alvo)
            esq = (img.width - nova_w) // 2
            img = img.crop((esq, 0, esq + nova_w, img.height))
    img = img.resize((L.W, L.H), Image.LANCZOS).convert("RGBA")

    # desfoque minimo, so pra assentar o upscale de 1536 pra 1600. Era
    # 1.2 quando o default era qualidade low e nao havia detalhe a
    # perder; com qualidade alta, borrar o grao e a particula joga fora
    # exatamente o que se pagou a mais.
    img = img.filter(ImageFilter.GaussianBlur(0.5))

    # o modelo entrega escuro demais pra 1600x900 (media RGB ~20): em vez
    # de escurecer de novo, realca -- e o realce que faz o refletor
    # aparecer no card, nao no arquivo bruto
    img = ImageEnhance.Brightness(img).enhance(1.30)
    img = ImageEnhance.Color(img).enhance(1.15)

    # o favoritismo do modelo vira brilho: lado do favorito mais aceso.
    # Antes do veu e da vinheta, pra que os dois ainda normalizem o
    # resultado e o lado forte nao estoure.
    arr = np.asarray(img, dtype=np.float32)
    arr[..., :3] *= _ganho_por_lado(p_casa, p_fora, L.W)[None, :, None]
    img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), "RGBA")

    # veu global minimo: baixa o pico do refletor sem apagar a cena
    img = Image.alpha_composite(img, Image.new("RGBA", (L.W, L.H), hexa("bg", 55)))

    # vinheta suave nas bordas (era 170 chapado, apagava os cantos)
    vinheta = Image.new("L", (L.W // 8, L.H // 8), 0)
    d = ImageDraw.Draw(vinheta)
    d.ellipse((-L.W // 16, -L.H // 16, L.W // 8 + L.W // 16, L.H // 8 + L.H // 16), fill=255)
    vinheta = vinheta.resize((L.W, L.H), Image.BICUBIC).filter(ImageFilter.GaussianBlur(70))
    escuro = Image.new("RGBA", (L.W, L.H), hexa("bg", 255))
    escuro.putalpha(Image.eval(vinheta, lambda v: int((255 - v) * 0.42)))
    img = Image.alpha_composite(img, escuro)

    # zonas de texto sem painel proprio: topo (wordmark + rotulo da
    # secao) e base (retrospecto + atribuicao). Ancorado no layout, nao
    # em numero fixo -- em 9:16 a mesma constante de 1600x900 cairia no
    # meio do card.
    img = Image.alpha_composite(img, _scrim(L, (0, 0, L.W, L.label_y + 40), 105, 0))
    # nos formatos em pe a faixa clara do gramado cai justamente na
    # altura do retrospecto, entao a base escurece mais e comeca antes
    base_a0, base_a1 = (30, 150) if L.empilhado else (0, 105)
    img = Image.alpha_composite(
        img, _scrim(L, (0, L.nota_y - 90, L.W, L.H), base_a0, base_a1))
    return img


# ======================================================================
# tipografia
# ======================================================================

# O site usa Archivo. Se ela estiver em assets/fonts/, usa; senao cai
# pra uma equivalente do sistema e, por ultimo, pra DejaVu (que vem com
# o matplotlib, ja dependencia do projeto) -- assim o card sai em
# qualquer maquina, mesmo que nao identico ao site.
FONT_DIRS = [ASSETS / "fonts", Path("C:/Windows/Fonts"), Path("/usr/share/fonts")]
FONT_BOLD = ["Archivo-Bold.ttf", "Archivo_Bold.ttf", "Inter-Bold.ttf",
             "segoeuib.ttf", "arialbd.ttf", "DejaVuSans-Bold.ttf"]
FONT_REG = ["Archivo-Regular.ttf", "Archivo_Regular.ttf", "Inter-Regular.ttf",
            "segoeui.ttf", "arial.ttf", "DejaVuSans.ttf"]
_font_cache: dict[tuple[str, int], ImageFont.FreeTypeFont] = {}
_font_avisado = False


def _achar_fonte(nomes: list[str]) -> Path:
    for nome in nomes:
        for d in FONT_DIRS:
            p = d / nome
            if p.exists():
                return p
    import matplotlib
    sufixo = "DejaVuSans-Bold.ttf" if nomes is FONT_BOLD else "DejaVuSans.ttf"
    return Path(matplotlib.get_data_path()) / "fonts" / "ttf" / sufixo


def fonte(tam: int, bold: bool = True) -> ImageFont.FreeTypeFont:
    global _font_avisado
    chave = ("b" if bold else "r", tam)
    if chave not in _font_cache:
        p = _achar_fonte(FONT_BOLD if bold else FONT_REG)
        if not _font_avisado and "Archivo" not in p.name:
            print(
                f"  fonte: Archivo nao encontrada, usando {p.name}. "
                "Pra bater com o site, ponha Archivo-Bold.ttf e "
                "Archivo-Regular.ttf em assets/fonts/."
            )
            _font_avisado = True
        _font_cache[chave] = ImageFont.truetype(str(p), tam)
    return _font_cache[chave]


def texto(d, xy, s, f, cor, anchor="la", tracking=0.0, halo=None, largura_max=None):
    """Desenha texto; com tracking > 0, letra a letra (o Pillow nao tem
    letter-spacing, e os rotulos em caixa alta do layout dependem dele).

    `halo=(imagem, raio, alfa)` pinta antes uma copia escura desfocada
    atras do texto. E o que mantem legivel o texto que cai direto sobre
    o fundo da IA (wordmark, rotulo da secao, comparativo, retrospecto)
    sem precisar escurecer a cena inteira -- escurecer apagaria o
    refletor, que e justamente o que se pagou pra gerar.

    `largura_max` quebra em linhas. So o retrospecto usa: em 1600 de
    largura ele cabe sempre numa linha, em 1080 nao -- e sem quebra o
    texto sairia pela borda do card em vez de falhar alto.
    """
    if largura_max and d.textlength(s, font=f) > largura_max:
        linhas, atual = [], ""
        for palavra in s.split(" "):
            tenta = f"{atual} {palavra}".strip()
            if atual and d.textlength(tenta, font=f) > largura_max:
                linhas.append(atual)
                atual = palavra
            else:
                atual = tenta
        linhas.append(atual)
        alt = f.size * 1.35
        for i, linha in enumerate(linhas):
            texto(d, (xy[0], xy[1] + i * alt), linha, f, cor, anchor, tracking, halo)
        return

    if halo:
        base, raio, alfa = halo
        camada = Image.new("RGBA", base.size, (0, 0, 0, 0))
        texto(ImageDraw.Draw(camada), xy, s, f, hexa("bg", alfa), anchor, tracking)
        base.alpha_composite(camada.filter(ImageFilter.GaussianBlur(raio)))
    if not tracking:
        d.text(xy, s, font=f, fill=cor, anchor=anchor)
        return
    largura = sum(d.textlength(ch, font=f) + tracking for ch in s) - tracking
    x, y = xy
    if anchor[0] == "m":
        x -= largura / 2
    elif anchor[0] == "r":
        x -= largura
    for ch in s:
        d.text((x, y), ch, font=f, fill=cor, anchor="l" + anchor[1])
        x += d.textlength(ch, font=f) + tracking


def largura_texto(d, s, f, tracking=0.0) -> float:
    if not tracking:
        return d.textlength(s, font=f)
    return sum(d.textlength(ch, font=f) + tracking for ch in s) - tracking


def painel(base: Image.Image, caixa, raio: int, cor) -> None:
    """Retangulo arredondado semi-opaco sobre o fundo da IA. E o que
    garante que o numero seja legivel em cima de qualquer imagem."""
    camada = Image.new("RGBA", base.size, (0, 0, 0, 0))
    ImageDraw.Draw(camada).rounded_rectangle(caixa, radius=raio, fill=cor)
    base.alpha_composite(camada)


# ======================================================================
# o card
# ======================================================================

def montar_card(payload: dict, meta: dict, hk: str, ak: str,
                fundo: Image.Image, L: "Layout") -> Image.Image:
    T = payload["teams_confronto"]
    th, ta = T[hk], T[ak]
    h, e, a = confronto_prob(
        th["forca"], ta["forca"], payload["elo_hfa"], payload["elo_nu"]
    )

    img = preparar_fundo(fundo, h, a, L)
    d = ImageDraw.Draw(img)
    # halo padrao pros textos que caem direto sobre o fundo da IA
    HALO = (img, 7, 235)

    M, DIR = L.M, L.DIR

    # --- casco: faixa de topo, wordmark, tag de rodada, atribuicao ---
    d.rectangle((0, 0, L.W, 8), fill=hexa("accent"))
    cy, ch = L.cab_y, L.cab_h
    painel(img, (M, cy, M + ch, cy + ch), 10, hexa("accent"))
    texto(d, (M + ch / 2, cy + ch / 2), "FS", fonte(21), hexa("onAccent"), anchor="mm")
    texto(d, (M + ch + 16, cy + ch / 2), "FORA DA SÚMULA", fonte(23), hexa("ink"),
          anchor="lm", halo=HALO)

    tag = f"BRASILEIRÃO {payload['season']} · RODADA {payload['current_round']}"
    f_tag = fonte(17, bold=False)
    tag_w = largura_texto(d, tag, f_tag) + 40
    # no formato em pe a largura nao comporta wordmark e tag na mesma
    # linha: a tag desce pra propria linha, alinhada a esquerda
    if tag_w > L.W - M - (M + ch + 16 + largura_texto(d, "FORA DA SÚMULA", fonte(23)) + 30):
        tag_x0, tag_y0 = M, cy + ch + 16
    else:
        tag_x0, tag_y0 = DIR - tag_w, cy
    painel(img, (tag_x0, tag_y0, tag_x0 + tag_w, tag_y0 + ch), ch // 2,
           hexa("surface2", 225))
    d.rounded_rectangle((tag_x0, tag_y0, tag_x0 + tag_w, tag_y0 + ch), radius=ch // 2,
                        outline=hexa("line2"), width=1)
    texto(d, (tag_x0 + tag_w / 2, tag_y0 + ch / 2), tag, f_tag, hexa("ink2"), anchor="mm")

    texto(d, (M, L.label_y), "CONFRONTO — QUEM GANHARIA", fonte(19), hexa("blue"),
          tracking=2.0, halo=HALO)

    # --- clubes: escudo, nome, sublinhado na cor do clube, badge ---
    CRESTE = L.creste
    topo_clube = L.clube_y
    # em 1080 de largura os dois nomes + escudos brigam por espaco, entao
    # o corpo cai um degrau antes do que cairia em 1600
    limite_g, limite_m = (13, 9) if not L.empilhado else (10, 7)
    for lado, (tid, t, badge, cor_badge, fundo_badge) in enumerate([
        (hk, th, "CASA", "accent", "accentSoft"),
        (ak, ta, "FORA", "blue", "blueSoft"),
    ]):
        esquerda = lado == 0
        crest = load_crest(tid, CRESTE)
        x_crest = M if esquerda else DIR - CRESTE
        if crest:
            img.alpha_composite(crest, (x_crest, topo_clube))

        folga = 26 if not L.empilhado else 18
        x_txt = M + CRESTE + folga if esquerda else DIR - CRESTE - folga
        anchor_x = "l" if esquerda else "r"
        nome = t["short_name"]
        tam = 34 if len(nome) > limite_g else 40 if len(nome) > limite_m else 48
        f_nome = fonte(tam)
        texto(d, (x_txt, topo_clube + CRESTE * 0.41), nome, f_nome, hexa("ink"),
              anchor=anchor_x + "m", halo=(img, 9, 245))

        # sublinhado curto na cor de marca do clube: identidade visual do
        # time sem repintar a paleta do card inteiro
        larg = largura_texto(d, nome, f_nome)
        y_lin = topo_clube + CRESTE * 0.65
        x0 = x_txt if esquerda else x_txt - larg
        d.rectangle((x0, y_lin, x0 + larg, y_lin + 4), fill=cor_marca(meta.get(tid, {})))

        f_badge = fonte(14)
        bw = largura_texto(d, badge, f_badge, tracking=1.4) + 34
        bx0 = x_txt if esquerda else x_txt - bw
        painel(img, (bx0, y_lin + 18, bx0 + bw, y_lin + 52), 17, hexa(fundo_badge, 235))
        texto(d, (bx0 + bw / 2, y_lin + 36), badge, f_badge, hexa(cor_badge),
              anchor="mm", tracking=1.4)

    texto(d, (L.W / 2, topo_clube + CRESTE * 0.45), "vs", fonte(22, bold=False),
          hexa("ink4"), anchor="mm", halo=HALO)

    # --- barra de probabilidade (3 segmentos, cantos arredondados) ---
    bar_x, bar_y, bar_w, bar_h, raio = L.bar
    w_h = round(bar_w * h)
    w_e = round(bar_w * e)
    w_a = bar_w - w_h - w_e

    barra = Image.new("RGBA", (bar_w, bar_h), (0, 0, 0, 0))
    bd = ImageDraw.Draw(barra)
    segs = [(0, w_h, "accent", "CASA", pct0(h)),
            (w_h, w_e, "ink2", "EMPATE", pct0(e)),
            (w_h + w_e, w_a, "blue", "FORA", pct0(a))]
    for x, w, cor, rotulo, valor in segs:
        bd.rectangle((x, 0, x + w, bar_h), fill=hexa(cor))
        cx = x + w / 2
        if w >= 112:
            texto(bd, (cx, bar_h * 0.32), rotulo, fonte(15), hexa("onAccent"),
                  anchor="mm", tracking=1.6)
            texto(bd, (cx, bar_h * 0.66), valor, fonte(46), hexa("onAccent"), anchor="mm")
        elif w >= 46:
            texto(bd, (cx, bar_h / 2), valor, fonte(26), hexa("onAccent"), anchor="mm")
    mascara = Image.new("L", (bar_w, bar_h), 0)
    ImageDraw.Draw(mascara).rounded_rectangle((0, 0, bar_w - 1, bar_h - 1), radius=raio, fill=255)
    barra.putalpha(Image.composite(barra.getchannel("A"), mascara, mascara))
    img.alpha_composite(barra, (bar_x, bar_y))

    texto(d, (M, bar_y + bar_h + 26),
          "Chance de cada resultado hoje, pela força atual e o mando de campo.",
          fonte(17, bold=False), hexa("ink3"), halo=HALO)

    # --- comparativo numerico: ao lado da barra no 16:9, abaixo dela
    #     nos formatos em pe (onde nao ha largura pra duas colunas) ---
    col_x0, col_x1 = L.col_x0, L.col_x1
    col_cx = (col_x0 + col_x1) / 2
    linha_h = L.linha_h
    if L.div_x:
        d.line((L.div_x, L.stats_y, L.div_x, L.stats_y + 3 * linha_h),
               fill=hexa("line2"), width=1)
    linhas = [
        ("POSIÇÃO", f"{th['real_position']}º" if th.get("real_position") else "—",
         f"{ta['real_position']}º" if ta.get("real_position") else "—"),
        ("PONTOS", th.get("real_points", "—"), ta.get("real_points", "—")),
        ("FORÇA", round(th["forca"]), round(ta["forca"])),
    ]
    for i, (rotulo, vh, va) in enumerate(linhas):
        y = L.stats_y + i * linha_h
        if i:
            d.line((col_x0, y, col_x1, y), fill=hexa("line2"), width=1)
        texto(d, (col_cx, y + 26), rotulo, fonte(13), hexa("ink3"),
              anchor="mm", tracking=1.7, halo=HALO)
        texto(d, (col_x0, y + linha_h * 0.68), str(vh), fonte(36), hexa("ink"),
              anchor="lm", halo=HALO)
        texto(d, (col_cx, y + linha_h * 0.68), "–", fonte(19, bold=False),
              hexa("ink4"), anchor="mm")
        texto(d, (col_x1, y + linha_h * 0.68), str(va), fonte(36), hexa("ink"),
              anchor="rm", halo=HALO)

    # --- retrospecto + atribuicao ---
    nota, pequena = h2h_note(payload, hk, ak)
    texto(d, (M, L.nota_y), nota, fonte(17, bold=False),
          hexa("amber" if pequena else "ink3"), halo=HALO,
          largura_max=L.W - 2 * M)
    texto(d, (DIR, L.rodape_y), "foradasumula.com.br", fonte(17), hexa("ink3"),
          anchor="rs", halo=HALO)

    return img.convert("RGB")


def gerar(payload, meta, hk, ak, args, L: "Layout") -> Path:
    T = payload["teams_confronto"]
    nome_arq = (
        f"confronto-{T[hk]['short_name']}-{T[ak]['short_name']}-{L.nome}.png"
        .lower().replace(" ", "-")
    )

    cor_casa = cor_marca(meta.get(hk, {}))
    cor_fora = cor_marca(meta.get(ak, {}))
    if args.sem_ia:
        print(f"  [{L.nome}] fundo: procedural (--sem-ia, sem custo)")
        fundo = bg_procedural(cor_casa, cor_fora, L)
    else:
        print(f"  [{L.nome}] {L.W}x{L.H}")
        fundo = bg_ia(cor_casa, cor_fora, args.qualidade, args.forcar_fundo, L)

    card = montar_card(payload, meta, hk, ak, fundo, L)
    saida = Path(args.saida) / nome_arq
    saida.parent.mkdir(parents=True, exist_ok=True)
    card.save(saida, "PNG", optimize=True)
    print(f"  -> {saida.relative_to(ROOT)} ({L.W}x{L.H}, "
          f"{saida.stat().st_size / 1024:.0f} KB)")
    return saida


def gerar_jogo(payload, meta, hk, ak, args) -> None:
    """Um confronto em todos os formatos pedidos.

    A legenda e uma so pro jogo (nao muda com o formato), entao e escrita
    uma vez ao lado do primeiro arquivo, com o nome do confronto.
    """
    T = payload["teams_confronto"]
    print(f"{T[hk]['short_name']} x {T[ak]['short_name']}")
    for nome in args.formato:
        gerar(payload, meta, hk, ak, args, Layout(nome))

    h, _e, a = confronto_prob(T[hk]["forca"], T[ak]["forca"],
                              payload["elo_hfa"], payload["elo_nu"])
    legenda = caption(payload, hk, ak, h, a)
    txt = Path(args.saida) / (
        f"confronto-{T[hk]['short_name']}-{T[ak]['short_name']}.txt"
        .lower().replace(" ", "-")
    )
    txt.write_text(legenda, encoding="utf-8", newline="\n")
    print(f"  legenda: {legenda}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Card de Confronto em PNG (fundo por IA, dado e texto em Pillow)."
    )
    ap.add_argument("--casa", help="clube mandante (nome, apelido ou TLA)")
    ap.add_argument("--fora", help="clube visitante (nome, apelido ou TLA)")
    ap.add_argument("--todos-proxima", action="store_true",
                    help="gera os jogos da proxima rodada (do payload) de uma vez")
    ap.add_argument("--formato", nargs="+", default=["x"], choices=list(FORMATOS),
                    metavar="FORMATO",
                    help="destino(s): x (1600x900, feed do X), feed (1080x1350, "
                         "post do Instagram), story (1080x1920, story/reels). "
                         "Aceita mais de um. Default: x")
    ap.add_argument("--sem-ia", action="store_true",
                    help="fundo procedural, sem chamar a API de imagem")
    ap.add_argument("--qualidade", default="low", choices=["low", "medium", "high"],
                    help="qualidade da imagem na API (default low: o pos-"
                         "processamento achata detalhe fino de qualquer jeito, "
                         "e nos testes low nao ficou atras de high)")
    ap.add_argument("--forcar-fundo", action="store_true",
                    help="ignora o cache e gera o fundo de novo (custa credito)")
    ap.add_argument("--refresh-times", action="store_true",
                    help="rebaixa escudos e cores da football-data.org")
    ap.add_argument("--saida", default=str(OUT_DIR), help=f"pasta de saida (default {OUT_DIR})")
    args = ap.parse_args()

    carrega_env()
    payload = load_payload()
    meta = load_teams_meta(refresh=args.refresh_times)

    if args.todos_proxima:
        jogos = payload["proxima_rodada"]
        if not jogos:
            raise SystemExit("payload sem proxima_rodada -- temporada encerrada?")
        print(f"proxima rodada: {len(jogos)} jogos\n")
        for j in jogos:
            gerar_jogo(payload, meta, j["mandante_id"], j["visitante_id"], args)
            print()
        return

    if not args.casa or not args.fora:
        raise SystemExit("informe --casa e --fora, ou use --todos-proxima")
    hk = resolve_team(payload, meta, args.casa)
    ak = resolve_team(payload, meta, args.fora)
    if hk == ak:
        raise SystemExit("mandante e visitante sao o mesmo clube")
    gerar_jogo(payload, meta, hk, ak, args)


if __name__ == "__main__":
    main()

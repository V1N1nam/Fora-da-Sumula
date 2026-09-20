"""Scratch (fora do pipeline): com que frequencia o resultado de maior
probabilidade do Elo+Davidson bateu com o resultado real.

E o complemento do que src/validate_elo.py ja mede: la o criterio e
log-loss (qualidade da probabilidade inteira), aqui e acerto bruto do
"favorito" (so o argmax das 3 vias). Nada aqui recalibra nem grava
parquet: le, conta e imprime.

Rating usado em cada jogo e SEMPRE o pre-jogo, vindo de
data/processed/match_ratings.parquet (derived.build_match_ratings),
que ja caminha em ordem cronologica continua sem resetar entre
temporadas. Nao recalculamos trajetoria de rating aqui de proposito:
CLAUDE.md manda metrica nova que percorre jogos consumir esse parquet,
justamente pra nao abrir um segundo lugar onde a mesma conta mora.

Criterio de "jogo encerrado": placar preenchido (home_goals/away_goals),
nunca status_raw -- mesma regra do ingest.py.

Nota que explica o 0% de acerto em empate, e nao e artefato da amostra:
no Davidson, p_draw/p_home = nu*sqrt(ta/th) e p_draw/p_away =
nu*sqrt(th/ta), cujo produto e nu^2. Com nu=0.74 (nu < 1) o produto e
0.55, entao pelo menos uma das duas razoes e sempre menor que 1: o
empate NUNCA e o maior dos tres, pra nenhum par de ratings. O teto do
p_draw e nu/(2+nu) = 27.0% (times iguais, sem mando), abaixo dos 36.5%
que cada lado tem ali. Ou seja, "acerto do favorito" aqui mede so
quanto o modelo separa mandante de visitante; empate ele nunca crava,
e por construcao, nao por calibracao ruim. Log-loss (validate_elo.py)
continua sendo o criterio que enxerga o empate.

Ressalva de leitura, que vale pra 2023: o match_ratings comeca flat em
1500 em 2023-04, entao o primeiro terco daquela temporada e previsto
com rating ainda nao convergido (mesmo viés de aquecimento discutido no
CLAUDE.md sobre o teste do openfootball). 2024-2026 herdam rating ja
rodado e nao tem esse problema.

Uso: python check_aproveitamento.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from config import ELO_DRAW_NU, ELO_HFA, ELO_K, PROCESSED, RAW  # noqa: E402
from calibrate_elo import davidson_probs  # noqa: E402

SEASONS = (2023, 2024, 2025, 2026)
LABELS = {"home": "vitoria mandante", "draw": "empate", "away": "vitoria visitante"}


# --------------------------------------------------------------- dados

def load_jogos() -> pd.DataFrame:
    """Jogos encerrados (placar preenchido) com o rating pre-jogo dos dois
    lados. O parquet de rating manda no conteudo; o raw entra so pra
    conferir que o match_ratings nao esta velho."""
    raw = pd.read_parquet(RAW / "matches.parquet")
    jogados = raw[raw["home_goals"].notna() & raw["away_goals"].notna()]

    mr = pd.read_parquet(PROCESSED / "match_ratings.parquet")

    faltando = set(jogados["match_id"]) - set(mr["match_id"])
    if faltando:
        raise SystemExit(
            f"match_ratings.parquet esta atrasado: {len(faltando)} jogos com placar "
            "no raw nao tem rating pre-jogo. Rode src/derived.py antes deste script."
        )

    df = mr[mr["match_id"].isin(set(jogados["match_id"]))].copy()
    return df.sort_values("date").reset_index(drop=True)


# ------------------------------------------------------------- contagem

def aproveitamento(df: pd.DataFrame) -> dict:
    """Favorito = argmax das 3 probabilidades de Davidson no rating
    pre-jogo. Compara com o resultado real e conta acerto."""
    p_home, p_draw, p_away = davidson_probs(
        df["r_home_pre"].to_numpy(), df["r_away_pre"].to_numpy(), ELO_HFA, ELO_DRAW_NU
    )
    probs = np.column_stack([p_home, p_draw, p_away])
    vias = np.array(["home", "draw", "away"])

    favorito = vias[probs.argmax(axis=1)]
    confianca = probs.max(axis=1)

    hg = df["home_goals"].to_numpy()
    ag = df["away_goals"].to_numpy()
    real = np.where(hg > ag, "home", np.where(hg == ag, "draw", "away"))

    acerto = favorito == real

    por_resultado = {}
    for via in vias:
        mask = real == via
        n = int(mask.sum())
        por_resultado[via] = {
            "jogos": n,
            "acertos": int(acerto[mask].sum()),
            "taxa": float(acerto[mask].mean()) if n else float("nan"),
        }

    # matriz real x favorito, pra ver pra onde vai o erro.
    matriz = {r: {f: int(((real == r) & (favorito == f)).sum()) for f in vias} for r in vias}

    return {
        "jogos": len(df),
        "acertos": int(acerto.sum()),
        "taxa": float(acerto.mean()),
        "por_resultado": por_resultado,
        "matriz": matriz,
        "favorito_dist": {f: int((favorito == f).sum()) for f in vias},
        "confianca_media": float(confianca.mean()),
        "acerto_chutando_mandante": float((real == "home").mean()),
    }


# -------------------------------------------------------------- relatorio

def imprime(season: int, res: dict) -> None:
    print(f"=== temporada {season} ===")
    print(f"  jogos encerrados na conta: {res['jogos']}")
    print(f"  acerto geral: {res['acertos']}/{res['jogos']} = {100 * res['taxa']:.1f}%")
    print(f"  confianca media do favorito: {100 * res['confianca_media']:.1f}%")
    print(
        f"  referencia, chutar mandante sempre: "
        f"{100 * res['acerto_chutando_mandante']:.1f}%"
    )

    print("  acerto por resultado REAL do jogo:")
    for via, label in LABELS.items():
        d = res["por_resultado"][via]
        taxa = "n/a" if d["jogos"] == 0 else f"{100 * d['taxa']:.1f}%"
        print(f"    {label:<20} {d['acertos']:>3}/{d['jogos']:<3} = {taxa}")

    fav = res["favorito_dist"]
    print(
        f"  favorito apontado pelo modelo: mandante {fav['home']}, "
        f"empate {fav['draw']}, visitante {fav['away']}"
    )

    print("  matriz real (linha) x favorito (coluna):")
    print(f"    {'':<20}{'fav.mandante':>14}{'fav.empate':>12}{'fav.visitante':>15}")
    for via, label in LABELS.items():
        linha = res["matriz"][via]
        print(f"    {label:<20}{linha['home']:>14}{linha['draw']:>12}{linha['away']:>15}")
    print()


def main() -> None:
    df = load_jogos()
    print(
        f"parametros em uso (config.py, nada recalibrado aqui): "
        f"K={ELO_K:.0f}, HFA={ELO_HFA:.0f}, nu={ELO_DRAW_NU:.4f}"
    )
    print(f"base: {len(df)} jogos com placar preenchido, {df['date'].min().date()} a {df['date'].max().date()}")
    print()

    resultados = {}
    for season in SEASONS:
        g = df[df["season"] == season]
        if g.empty:
            continue
        resultados[season] = aproveitamento(g)
        imprime(season, resultados[season])

    print("=== resumo, taxa de acerto geral por temporada ===")
    for season, res in resultados.items():
        print(f"  {season}: {100 * res['taxa']:.1f}%  ({res['acertos']}/{res['jogos']})")

    fechadas = [res for s, res in resultados.items() if s != 2026]
    if fechadas and 2026 in resultados:
        jogos_f = sum(r["jogos"] for r in fechadas)
        acertos_f = sum(r["acertos"] for r in fechadas)
        taxa_f = acertos_f / jogos_f
        print(
            f"  2023-2025 juntas: {100 * taxa_f:.2f}%  ({acertos_f}/{jogos_f})  |  "
            f"2026 ({100 * resultados[2026]['taxa']:.2f}%) esta "
            f"{100 * (resultados[2026]['taxa'] - taxa_f):+.2f} p.p. em relacao a elas"
        )
    print()
    print(
        "nota: empate nunca aparece como favorito porque nu < 1 faz o p_draw "
        "do Davidson nunca ser o maior dos tres (ver docstring). O 0% na linha "
        "de empate e estrutural, nao sinal de calibracao ruim."
    )


if __name__ == "__main__":
    main()

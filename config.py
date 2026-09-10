"""Configuracao central do pipeline."""

from pathlib import Path

# Codigo da competicao na football-data.org (nao e mais soccerdata/FBref:
# ver README para o motivo da troca de fonte).
COMPETITION = "BSA"

# Temporadas disponiveis no plano gratis da football-data.org. Confirmado
# por chamada real em 2026-09-07: 2020-2022 respondem 403 (fora do plano),
# 2023-2026 respondem 200. Ajuste aqui se o plano mudar.
SEASONS = [2023, 2024, 2025, 2026]

# Temporada corrente: a unica que e re-raspada a cada execucao.
# As anteriores ficam congeladas e nao geram request.
CURRENT_SEASON = 2026

ROOT = Path(__file__).parent
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"

# --- Elo (calibrado em 2026-09-07, ver src/calibrate_elo.py) ---
#
# O grid search (K x HFA, log-loss cronologico, 3 splits em 2023-2025)
# apontou K=32/HFA=110 como argmin, mas o mapa de calor mostrou platou
# (desvio do grid inteiro menor que o desvio entre splits do proprio
# vencedor) e o ganho de log-loss do argmin sobre K=20/HFA=65 (0.0076)
# ficou abaixo do desvio entre splits (0.0184) -- ganho nao distinguivel
# de ruido de amostra. A estimativa de HFA direto dos dados (pontos em
# casa vs fora, sem passar pelo Elo) deu 79, na mesma regiao de 65 e
# longe de 110, reforcando que 110 estava compensando K alto. Optou-se
# por K=20/HFA=65: parametro redondo e defensavel, nao argmin de ruido.
ELO_INITIAL_RATING = 1500.0
ELO_SCALE = 400.0
ELO_K = 20.0
ELO_HFA = 65.0

# Forca do empate no modelo de Davidson (1970), que converte a P(vitoria)
# binaria do Elo em 3 resultados. Estimado por maxima verossimilhanca
# sobre 2023-2025 usando a trajetoria em ELO_K/ELO_HFA (nao faz parte do
# grid search -- ver calibrate_elo.py sobre por que tratar como nuisance
# e valido aqui: reestimar no ponto final mudou o valor em so 4.5%).
ELO_DRAW_NU = 0.7404522613065327

# Simulacoes de Monte Carlo do restante da temporada corrente.
ELO_N_SIMULATIONS = 10_000

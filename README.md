# brasileirao-hub

Pipeline de métricas derivadas do Brasileirão. Busca a API do
football-data.org a cada rodada, mantém um rating de Elo com Monte
Carlo do restante da temporada (probabilidade de título, G4 e Z4 por
clube), e grava Parquet versionado no próprio repositório. Sem
servidor, sem banco, sem custo fixo.

## Ordem de execução

Na primeira vez, **na sua máquina**, não no Actions:

```bash
python -m venv .venv
.venv/Scripts/activate    # Linux/Mac: source .venv/bin/activate
pip install -r requirements.txt

# 1. Token do football-data.org (gratuito, registro em
#    https://www.football-data.org/client/register). Nunca comite
#    o valor -- copie .env.example para .env e preencha.
cp .env.example .env

# 2. Backfill historico. O plano gratis libera 2023 em diante
#    (temporadas mais antigas respondem 403). Respeita sozinho o
#    limite de 10 requests/minuto.
python src/ingest.py --full

# 3. Calibre K e vantagem de mando do Elo contra o historico.
#    One-off -- so precisa rodar de novo se voce mudar de proposito
#    os parametros ou o metodo de calibracao.
python src/calibrate_elo.py

# 4. Rating + Monte Carlo do restante da temporada corrente
python src/elo.py

# 5. Gera docs/index.html a partir do template fora-da-sumula-v3.html
python src/build_site.py
```

Depois disso, commite `data/` e `docs/`, e o Actions assume: segunda e
quinta ele busca só a temporada corrente, recalcula e regera o site.

## GitHub Pages precisa ser configurado manualmente uma vez

O workflow gera `docs/index.html` e commita, mas não liga o Pages
sozinho. Depois do primeiro push: Settings → Pages → Build and
deployment → Source: "Deploy from a branch" → Branch: `main`, pasta
`/docs`. **Não existe opção de pasta arbitrária aqui** -- o dropdown do
GitHub só aceita `/ (root)` ou `/docs`, por isso o gerador escreve em
`docs/` e não em `site/`. Sem essa configuração o arquivo fica no
repositório mas não é servido em lugar nenhum.

## O secret do GitHub Actions precisa existir antes do primeiro run

O workflow lê o token de `secrets.FOOTBALL_DATA_TOKEN`. Configure em
Settings → Secrets and variables → Actions → New repository secret,
com o mesmo valor do seu `.env` local, **antes** de deixar o cron
rodar sozinho. Sem isso `ingest.py` falha com uma mensagem clara de
token ausente -- não com um 403 confuso vindo da API.

## Validação do modelo

`src/validate_elo.py` simula uma temporada fechada inteira a partir da
rodada 0 (aquecendo o rating só nas temporadas anteriores, sem
vazamento) e compara pontos e saldo de gols simulados com os reais.
Não está no workflow do Actions -- é pesado (~10s, 10 mil réplicas) e
só importa rodar de novo quando K, HFA, ν ou o bootstrap de placar em
`elo.py` mudarem. Rode manualmente:

```bash
python src/validate_elo.py
```

Resultado mais recente na seção "Limitações conhecidas" abaixo.

## Métricas derivadas (`src/derived.py`)

Camada extra em cima de `data/raw` e `data/processed`, sem fonte nova e
sem dado novo. Reusa a mesma função de Davidson e os mesmos parâmetros
calibrados (K, HFA, ν) de `config.py` -- nada aqui recalibra nada.

- **`xpts_forca.parquet`** -- xPTS "de força": pontos esperados pela
  probabilidade de resultado do Elo+Davidson (rating pré-jogo, sem
  vazamento), não por xG. Mede sorte contra a força do adversário, não
  contra a qualidade das chances criadas -- essa fonte não tem xG.
- **`zebras.parquet`** -- por partida disputada, a probabilidade que o
  modelo dava pro resultado que de fato ocorreu (rating pré-jogo),
  ordenado do menos provável pro mais provável.
- **`sequencias.parquet`** -- por clube, sequência atual e recorde de
  invencibilidade, vitórias seguidas, jogos sem vencer e jogos sem
  sofrer gol. **As sequências cruzam temporadas sem resetar** (a
  "sequência atual" de um clube pode incluir jogos do ano anterior --
  confirmado com o Botafogo, que tem `invencibilidade_recorde=18`
  batendo com a sequência invicta real de 2023). Intencional, não bug
  -- mas qualquer texto do site que mostrar essa métrica precisa deixar
  isso explícito (ex: "invicto há 8 jogos, desde outubro de 2025"),
  nunca só o número pelado.
- **`mando_clube.parquet`** -- vantagem de mando específica de cada
  clube (pontos por jogo em casa vs fora), comparada com a média da
  liga inteira. Traz `temporadas` (nº de temporadas com dados do
  clube) e `jogos_casa`/`jogos_fora` explícitos e separados -- clube
  com só 1-2 temporadas de amostra (ex.: promovido recente) tem
  `vantagem_relativa` bem mais ruidosa que um com 4, e isso precisa
  estar visível sem recalcular nada.
- **`h2h.parquet`** -- confronto direto entre cada par de clubes que já
  se enfrentou em 2023-2026 (jogos, vitórias de cada lado, empates,
  média de gols), uma linha por par (não duplicada A-B/B-A).
- **`ritmo_campeao.parquet`** -- pontos acumulados do líder da tabela,
  rodada a rodada, por temporada (líder pode trocar de time ao longo
  da campanha -- a série guarda quem era o líder em cada rodada, não
  só os pontos). Verificado: o líder na última rodada de cada
  temporada fechada (2023-2025) bate exatamente com o campeão real em
  `standings.parquet`.
- **`cenarios.parquet`** -- por clube, na rodada mais recente: título,
  G4 e Z4 confirmados ou descartados matematicamente (probabilidade
  virou exatamente 0% ou 100% nas 10 mil simulações do Monte Carlo já
  rodado por `elo.py` -- nenhuma simulação nova aqui). Arquivo
  separado do `elo_probabilidades.parquet` de propósito: aquele é dono
  do `elo.py`, e "confirmado/descartado" só faz sentido pra rodada
  atual, não pro histórico inteiro que ele carrega.

## Limitações conhecidas

### Saldo de gols simulado tem variância menor que a real

O Monte Carlo de `src/elo.py` sorteia o resultado de cada partida
simulada (casa/empate/fora) via Elo+Davidson, mas não tem modelo de
gols: o placar de cada partida simulada é bootstrap de placares reais
de temporadas fechadas (2023-2025), tirado de um balde global por tipo
de resultado, sem correlação com o tamanho da diferença de força entre
os times.

Validado em 2026-09-07 (`src/validate_elo.py`, 10.000 réplicas
independentes da temporada 2025 inteira a partir da rodada 0, com
rating herdado de 2023-2024 -- não de 1500 flat):

| métrica | desvio simulado (média, intervalo 5-95% entre réplicas) | desvio real |
| --- | --- | --- |
| pontos | 11.68 `[8.86, 14.57]` | 2023: 12.14 · 2024: 13.23 · 2025: 14.34 — **as 3 dentro do intervalo simulado** |
| saldo de gols | 13.7 `[10.3, 17.3]` | 2025: 22.0 — **fora do intervalo simulado** |

Razão do desvio simulado sobre o real: **0.81 para pontos** (validado
contra as 3 temporadas fechadas, sem compressão) e **0.62 para saldo de
gols** (comprimido). Extremos batem o mesmo padrão: campeão/lanterna de
pontos reais caem perto ou dentro do intervalo simulado; campeão/lanterna
de saldo de gols reais (51 / -47 em 2025) ficam bem fora do intervalo
simulado (26.8 / -26.1).

**Efeito prático:** título, G4 e Z4 vêm de pontos (via Davidson), não do
bootstrap de placar -- não são afetados pela compressão. O desempate por
saldo de gols em casos raros de empate exato em pontos e vitórias fica
menos preciso (a extremidade das goleadas fica sub-representada no
saldo simulado).

### `status_raw` da football-data.org é inconsistente

Parte das partidas com placar preenchido vem com `status` como uma
string de data em vez de `"FINISHED"` (achado em 2026-09-07, ~7-8% das
partidas de uma rodada chegaram a vir assim). `ingest.py` grava o valor
cru em `status_raw` e marca `status_anomalo=True` nesses casos, mas
decide "partida encerrada" pelo placar preenchido, nunca pelo texto do
status. Se `status_anomalo` passar de 20% numa rodada, `ingest.py`
avisa no log -- pode ser sinal de mudança de comportamento da API.

## Backlog

- **Bootstrap de placar ponderado por kernel na diferença de Elo** (não
  implementado). Em vez do balde global por resultado -- que comprime o
  saldo de gols simulado, ver Limitações conhecidas acima --, ponderar
  os 1.140 jogos históricos de 2023-2025 por proximidade da diferença
  de Elo do confronto real ao confronto sendo simulado (kernel gaussiano
  ou similar) em vez de fatiar em baldes fixos por faixa de força.
  Recupera a correlação entre desequilíbrio de força e tamanho da
  goleada sem esvaziar a amostra em baldes pequenos.

## Sobre a fonte

Os dados vêm da API do [football-data.org](https://www.football-data.org),
competição BSA (Brasileirão Série A). O plano grátis é **só para uso
não comercial** e cobre a temporada corrente e as 3 anteriores (2023
em diante -- confirmado por chamada real em 2026-09-07; temporadas mais
antigas respondem 403). Publique métrica derivada e gráfico com
crédito à fonte; não republique os dados brutos como produto nem sirva
anúncio em cima disso -- é o que separa uso tolerado de uso comercial.

O limite de 10 requests/minuto existe por um motivo. Não o contorne.

## Estrutura

```text
config.py                  competicao, temporadas, parametros do Elo
src/ingest.py              football-data.org -> data/raw/*.parquet (incremental)
src/probe_schema.py        dump do schema real
src/calibrate_elo.py       grid search de K/HFA + Davidson nu (one-off)
src/elo.py                 rating + Monte Carlo -> data/processed/*.parquet
src/derived.py              metricas derivadas -> data/processed/*.parquet
src/validate_elo.py        validacao do modelo (rode quando ele mudar)
src/build_site.py          data/processed + data/raw -> docs/index.html
fora-da-sumula-v3.html     template do site (design aprovado; so o payload muda)
docs/index.html            gerado -- e o que o GitHub Pages serve (pasta fixa do Pages)
.github/workflows/         agendamento, commit e build do site automáticos
```

# CLAUDE.md — Fora da Súmula

## O que é
Site de probabilidade e estatística do Brasileirão. Elo + Davidson
para empate + Monte Carlo, calibrado e validado contra 2023-2025.
NÃO é hub de estatística avançada (xG, posse, passe) — essa ideia
foi descartada por falta de fonte gratuita estável. Ver
"Fontes descartadas" abaixo antes de sugerir qualquer API nova.

## Regras que já foram violadas uma vez — não repetir

- **Nunca contornar proteção anti-bot.** FBref e ESPN bloquearam
  o projeto por scraping/IP de datacenter. Não sugerir Selenium,
  undetected-chromedriver, proxy, ou qualquer bypass.
- **Rating usado em cada jogo é sempre o vigente NA DATA do jogo**,
  nunca o rating final/atual. Já causou bug real no xpts_forca e
  no zebras — sempre processar em ordem cronológica. Hoje isso é
  garantido pelo `match_ratings` de `derived.py` (rating pré-jogo,
  ordem cronológica contínua, sem resetar entre temporadas):
  métrica nova que percorra jogos deve consumir ele, não recalcular.
- **"Próxima rodada" nunca é `matchday.min()` entre agendados.**
  Jogos adiados (POSTPONED) sem nova data quebram isso. Sempre
  filtrar por `utc_date >= cutoff`, nunca por número de rodada
  cru. Bug real, corrigido em dois lugares — e continuam sendo só
  dois: `build_form_and_next` (build_site.py:100) e
  `zebra_provavel_proxima_rodada` (derived.py:502). O terceiro
  consumidor, `build_proxima_rodada` (build_site.py:387), lê o
  parquet que o derived já gerou e herda o corte correto — não
  duplique a lógica lá. Checar se apareceu um quarto lugar antes
  de adicionar feature nova de "próximo jogo".
- **`status_raw` da API é inconsistente** (já veio com timestamp
  no lugar do enum). Nunca confiar só nele para decisão de "jogo
  encerrado" — usar `score.fullTime` preenchido como critério.
  O `ingest.py` já faz isso, guarda `status_raw` ao lado só para
  auditoria, e marca `status_anomalo`; `_report_anomalias()` avisa
  se a taxa passar de 20% (sinal de que a API mudou de comportamento).
- **`Path.read_text()` não aceita `newline=`** — já causou falha
  real no Actions. É argumento de `open()`, não de `read_text()`;
  o parâmetro só existe em `read_text()` a partir do Python 3.13 e
  o workflow roda **3.12**. `write_text(..., newline="\n")` é
  válido (existe desde 3.10) e é o que o build usa na escrita.

## Fonte de dados

- **football-data.org** (plano grátis) é a fonte principal.
  Cobre só placar, rodada, tabela — SEM xG, posse, passe,
  escalação. Uso "não comercial" no free tier — se o site algum
  dia tiver anúncio, assinatura, ou patrocínio pago vinculado ao
  conteúdo dele, reavaliar com eles antes. O rodapé do site
  ("Uso não comercial") depende disso continuar verdade.
- Plano grátis cobre 2023 em diante; 2020-2022 respondem 403.
  Só a temporada corrente (`CURRENT_SEASON`) é re-raspada; as
  anteriores ficam congeladas e não geram request.
- **dadosfutebol.com.br** (R$99/mês, ainda não assinado) tem
  estatística avançada real (posse, xG, escalação, eventos),
  fonte SofaScore por trás. Cogitado, não contratado — critério
  de quando assinar: 10 mil seguidores no TikTok com views
  consistentes (é a única das redes com RPM documentado e
  alcançável no Brasil).

## Fontes descartadas — não sugerir de novo

FBref (Cloudflare), ESPN API interna (Akamai bloqueia
datacenter), Sportmonks (€48+/mês), API-Football (sem xG),
SofaScore direto (ToS proíbe scraping), Stathead (ferramenta de
busca, não API), Big Balls Data (não cobre Brasil), StatsHub
(widget da Sportradar, não API), Opta/Stats Perform direto
(enterprise, sem self-service).

**Resíduo da era FBref:** `config.py` teve `SHOT_SCRAPE_DELAY`,
`BIG_CHANCE_XG` e `MAX_GOALS` — sobras da ideia de xG, nunca lidas
por módulo nenhum. Removidos. Não recriar: não existe pipeline de
chute/xG, e constante órfã em `config.py` só faz parecer que existe.

## Calibração do modelo (não recalibrar sem motivo)

K=20, HFA=65, ν(Davidson)=0.7404522613065327 — escolhidos por
robustez (platô no grid search), não por argmin. O argmin era
K=32/HFA=110, mas o ganho de log-loss (0.0076) ficou abaixo do
desvio entre splits (0.0184), e o HFA estimado direto dos dados
deu 79 — perto de 65, longe de 110.

Se recalibrar, o que **não** precisa de trabalho manual: os
números do painel "Como funciona" e do simulador de confronto no
JS. Eles são injetados no build a partir de `config.py`, na linha
`const HFA=..., NU=..., ELO_K=..., NSIM=...; /*__ELO__*/` do
template (`fora-da-sumula-v3.html`), substituída por
`build_site.render()` — que dá `assert` se a sentinela sumir.
Não hardcode esses valores no template.

O que **continua manual** depois de recalibrar: propagar o
resultado de `validate_elo.py` — ver a regra abaixo.

## Resultado de validate_elo.py mora em dois lugares — atualizar os dois no mesmo commit

`src/validate_elo.py` é manual (não está no Actions) e roda quando
K, HFA, ν ou o bootstrap de placar de `elo.py` mudarem. O número
que ele imprime é copiado à mão para **dois** lugares, e nada no
build compara um com o outro:

1. `const V = {sim, lo, hi, reais}` em
   `fora-da-sumula-v3.html:1901` — desenha a faixa de validação da
   ficha. É a única coisa daquela seção que não vem do payload.
2. A tabela da seção "Limitações conhecidas" do `README.md`.

**Sempre que `validate_elo.py` rodar de novo com resultado
diferente, os dois vão no mesmo commit.** Não existe sentinela,
assert ou teste que pegue divergência aqui — ao contrário de
`/*__ELO__*/` e `/*__DATA__*/`, que quebram o build alto se
sumirem. Se só um for atualizado, README e site passam a afirmar
validações diferentes do mesmo modelo, em silêncio, e a única
forma de descobrir é alguém comparar na mão. Atualizar um sem o
outro é pior que não atualizar nenhum.

## Pipeline (ordem importa)

ingest.py → elo.py → derived.py → build_site.py
(derived.py lê elo_probabilidades.parquet, por isso vem depois)

`calibrate_elo.py` e `validate_elo.py` são one-off manuais, fora do
workflow. O Actions roda o pipeline segunda e quinta às 09:00 UTC e
commita `data/` e `docs/`.

`build_site.py` gera **duas** saídas do mesmo payload:
`docs/index.html` (site) e `docs/export/index.html` (ferramenta
interna de export de imagem pra redes). Mudança no formato do
payload pode quebrar a segunda em silêncio — conferir as duas.

## Convenções

- Comentários e nomes de coluna em português informal, código em
  inglês/padrão.
- Todo card/manchete no site é gerado do dado, nunca escrito à
  mão — mesmo texto de legenda de rede social usa template com
  variação, sem LLM.
- Regra de amostra pequena: métrica com menos de 3-4
  temporadas/jogos de base não aparece com a mesma confiança
  visual que uma com histórico completo (ver o ramo
  `mando_temporadas < 3` no template).

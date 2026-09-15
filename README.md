# brasileirao-hub

Pipeline de métricas derivadas do Brasileirão. Busca a API do
football-data.org a cada rodada, mantém um rating de Elo com Monte
Carlo do restante da temporada (probabilidade de título, G4, Z4 e
faixas de vaga continental por clube), e grava Parquet versionado no
próprio repositório. Sem servidor, sem banco, sem custo fixo.

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

# 5. Metricas derivadas (xPTS, zebras, sequencias, mando, h2h, ritmo,
#    cenarios). build_site.py le todas elas -- rode antes do site.
python src/derived.py

# 6. Gera docs/index.html a partir do template fora-da-sumula-v3.html
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
  G4, Z4, pré-Libertadores e Sul-Americana confirmados ou descartados
  matematicamente (probabilidade virou exatamente 0% ou 100% nas 10 mil
  simulações do Monte Carlo já rodado por `elo.py` -- nenhuma simulação
  nova aqui). Arquivo separado do `elo_probabilidades.parquet` de
  propósito: aquele é dono do `elo.py`, e "confirmado/descartado" só
  faz sentido pra rodada atual, não pro histórico inteiro que ele
  carrega. "Descartado" do Z4 vira o selo "salvo" na tabela do site
  quando o clube chegou a ter risco real (≥10% de Z4 em alguma rodada) --
  time que nunca teve risco não ganha selo, seria ruído.

### `elo_probabilidades.parquet` -- faixas de classificação (`src/elo.py`)

Por clube, por rodada: `titulo_prob`, `g4_prob`, `z4_prob` (já
existiam) e três colunas de vaga continental adicionadas em
2026-09-13, contadas no mesmo Monte Carlo de 10 mil simulações --
nenhum dado novo, nenhuma simulação nova:

- `libertadores_prob` -- terminou entre 1º e 4º. **Idêntica a
  `g4_prob`** de propósito: `g4_prob` já é consumida por vários
  lugares que não tem nada a ver com vaga de copa (badge "G4
  garantido" da tabela, `cenarios.parquet`, "o que mudou" da semana) e
  renomear quebraria esses consumidores ou forçaria o aviso de vaga de
  copa (abaixo) numa UI que já existia sem ele. `libertadores_prob`
  existe separada só pra quem for consumir o dado pensando
  especificamente em "vaga de Libertadores" -- ver `src/elo.py` para o
  raciocínio completo.
- `pre_libertadores_prob` -- terminou em 5º ou 6º.
- `sulamericana_prob` -- terminou entre 7º e 12º.

**Limitação, válida pras três colunas acima**: contam só posição final
na tabela do Brasileirão. Não descontam a vaga extra de Libertadores
que o campeão da Copa do Brasil ganha, nem a vaga extra de Sul-Americana
que o campeão da própria Libertadores ganha, quando esses clubes não
terminam dentro da faixa por tabela -- o pipeline não tem dado da Copa
do Brasil nem da Libertadores (só `football-data.org`/Brasileirão Série
A). Um campeão de copa fora da faixa "empurra" pra baixo quem ficaria
na vaga só pela tabela, deslocando a distribuição inteira. Não há
correção estimada pra isso. O site mostra um aviso equivalente nas
abas de Pré-Libertadores e Sul-Americana da seção "O que está em jogo".

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
| pontos | 11.68 `[8.86, 14.57]` | 2023: 12.14 · 2024: 13.23 · 2025: 14.34, **as 3 dentro do intervalo simulado** |
| saldo de gols | 13.7 `[10.3, 17.3]` | 2025: 22.0, **fora do intervalo simulado** |

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

### `xpts_forca` (Elo) pode divergir forte de xP baseado em xG real

`xpts_forca.parquet` mede sorte contra **força** (Elo+Davidson,
rating pré-jogo), não contra **qualidade de chance** (xG) -- ver
descrição do arquivo acima. Comparação pontual contra o xP público do
DataFutebol/Opta (baseado em xG real), rodada 26/2026, a partir de um
print de terceiros (viz "Data via BeGriffis | Viz by @DataFutebol"),
líder com 53-54 pontos batendo com nossa rodada 26:

| clube | pontos reais | xPTS Elo (nosso) | diferença Elo | xP público (xG, aprox.) | diferença xG |
| --- | --- | --- | --- | --- | --- |
| Athletico-PR | 45 | 32.35 | **+12.65** | ~44 | **+1** |
| Coritiba | 37 | 28.73 | +8.27 | ~31 | +6 |

Coritiba: os dois modelos concordam na direção e na magnitude (time
supera o esperado nos dois). Athletico-PR: divergência forte -- o
xG público vê o time quase no ponto certo (+1), enquanto o Elo vê uma
sobreperformance grande (+12.65). Leitura mais provável: o rating de
Elo do Athletico-PR ainda carrega força de temporadas/rodadas
anteriores e não capturou uma melhora recente do time -- Elo reage com
atraso a mudanças de patamar, xG por jogo não.

**Não é bug** -- é a limitação já documentada acima (Elo mede força,
não qualidade de chance) se manifestando de forma extrema num caso
concreto. Guardado aqui como referência caso a métrica seja
questionada publicamente (ex.: alguém comparar os dois números e achar
que o site "errou" a conta do Athletico-PR).

### `status_raw` da football-data.org é inconsistente

Parte das partidas com placar preenchido vem com `status` como uma
string de data em vez de `"FINISHED"` (achado em 2026-09-07, ~7-8% das
partidas de uma rodada chegaram a vir assim). `ingest.py` grava o valor
cru em `status_raw` e marca `status_anomalo=True` nesses casos, mas
decide "partida encerrada" pelo placar preenchido, nunca pelo texto do
status. Se `status_anomalo` passar de 20% numa rodada, `ingest.py`
avisa no log -- pode ser sinal de mudança de comportamento da API.

## Card de Confronto em imagem (`src/card_confronto.py`)

Gera o PNG 1600x900 (tamanho de feed do X) do confronto entre dois
clubes: **fundo pela API de imagem da OpenAI, todo o resto pelo
Pillow**. A IA não escreve texto, número nem escudo. Ela só pinta
atmosfera abstrata (luz de estádio, degradê nas cores dos dois
clubes), que ainda leva escurecimento, desfoque e vinheta antes de
qualquer coisa ser desenhada por cima. Nome, probabilidade, posição,
pontos, força, retrospecto e legenda saem do dado, como todo o resto
do site.

O dado vem do payload que `build_site.py` já gravou em
`docs/export/index.html`, lido pela mesma sentinela `/*__DATA__*/` do
gerador de imagens. Nada é recalculado aqui. A fórmula de
probabilidade é transcrição literal de `confrontoProb()` do template.
Rode o pipeline antes.

```bash
# fundo procedural, não chama a API (iterar layout de graça)
python src/card_confronto.py --casa Flamengo --fora Palmeiras --sem-ia

# com a IA (gpt-image-1-mini, qualidade low por padrão)
python src/card_confronto.py --casa FLA --fora PAL

# os três formatos de uma vez
python src/card_confronto.py --casa FLA --fora PAL --formato x feed story

# os 10 jogos da próxima rodada de uma vez, reaproveitando o cache
python src/card_confronto.py --todos-proxima
```

`--formato` escolhe o destino: `x` 1600x900 (16:9, feed do X, o
padrão), `feed` 1080x1350 (4:5, post do Instagram) e `story`
1080x1920 (9:16). No 16:9 a barra fica à esquerda e o comparativo à
direita; nos dois formatos em pé tudo desce em blocos. O `story`
mantém o conteúdo entre ~200 e ~1660 de altura, porque a interface do
Instagram cobre o topo e a base. Sem isso o wordmark e a atribuição
saem escondidos atrás da UI.

Os dois formatos em pé pedem o mesmo tamanho à API (1024x1536, a
orientação retrato que ela oferece) e o mesmo prompt, então
**compartilham o fundo**: gerar os dois custa uma imagem, não duas. O
16:9 tem o fundo dele, porque recortar 9:16 de uma paisagem destruiria a
composição. O prompt também muda por orientação: no 16:9 a luz do
gramado pode ocupar a metade de baixo, que está vazia; em pé essa
área é o comparativo, então a faixa clara desce pro rodapé.

**A luz do fundo carrega dado.** O lado do favorito fica mais aceso,
na proporção da força relativa dos dois times, `p_casa / (p_casa +
p_fora)`, ignorando o empate, que não tem lado no quadro. Isso é feito
no Python (`_ganho_por_lado`), depois da IA, e não no prompt: modelo de
imagem não obedece proporção numérica, e pôr a probabilidade no prompt
a colocaria na chave do cache, e cada confronto viraria uma geração
nova. Assim o mesmo fundo cacheado serve pra qualquer par de
probabilidades, e a proporção sai exata. O expoente e o clamp em
`_ganho_por_lado` amortecem o efeito, pra que 70/30 não estoure um
lado e apague o outro.

Sai em `cards/out/` (PNG + `.txt` com a legenda pronta pra postar),
pasta ignorada pelo git. `assets/crests/` e `assets/teams.json` vêm de
uma única chamada a `/competitions/BSA/teams` e ficam versionados.

**O que segura o custo**, já que o modelo cobra por imagem gerada:

- o fundo é cacheado em `cards/bg/` por chave `(versão do prompt,
  cores dos dois clubes, tamanho, qualidade)`. Gerar o mesmo confronto
  de novo não gasta nada, e dois confrontos entre clubes de cores
  equivalentes compartilham o fundo.
- o padrão é `--qualidade low`. Testado contra `high` e `medium` no
  mesmo par de cores (vermelho × verde): o que fazia o fundo parecer
  ruim no começo era o pós-processamento, não o tier. Depois de
  arrumar o pós, o `low` não ficou atrás, e o pós achata detalhe fino
  de qualquer jeito. O `medium` ainda desobedeceu a composição pedida
  (pôs refletor em cima do texto), mas com uma geração de cada não dá
  pra dizer se é o tier ou sorteio.
- como só existem **7 cores de marca** entre os 20 clubes (`cor_marca`
  mapeia o `clubColors` da API num palete fixo), o universo inteiro de
  fundos é 28 pares por orientação. O custo converge pra zero depois
  das primeiras semanas. Em compensação, dois confrontos com a mesma
  combinação de cores saem com fundo idêntico.
- `--sem-ia` desenha um degradê procedural nas cores dos clubes e não
  chama a API. É o fallback de verdade, não placeholder: o card sai
  igual se a API estiver fora do ar na hora de postar.

Se mudar o texto do prompt, suba `PROMPT_VERSION` no módulo. Senão o
cache antigo continua sendo servido e parece que a API ignorou a
mudança.

O prompt descreve a composição **do card**, não uma foto bonita: a
luz é empurrada pra metade de baixo e o topo é pedido quase preto,
porque a parte de cima do layout é a mais ocupada (wordmark, tag de
rodada, escudos, nomes). Uma versão anterior pedia os refletores "nos
cantos superiores" e eles nasceram em cima do rótulo da seção. Mexer
nessa divisão de zonas sem olhar o layout quebra a legibilidade de
novo.

Fora do pipeline do Actions, como `calibrate_elo.py` e
`validate_elo.py`: roda à mão, quando se quer a arte pra postar.

## Backlog

- **Bootstrap de placar ponderado por kernel na diferença de Elo** (não
  implementado). Em vez do balde global por resultado -- que comprime o
  saldo de gols simulado, ver Limitações conhecidas acima --, ponderar
  os 1.140 jogos históricos de 2023-2025 por proximidade da diferença
  de Elo do confronto real ao confronto sendo simulado (kernel gaussiano
  ou similar) em vez de fatiar em baldes fixos por faixa de força.
  Recupera a correlação entre desequilíbrio de força e tamanho da
  goleada sem esvaziar a amostra em baldes pequenos.
- **Histórico pré-2023 via `openfootball/south-america`** (não
  implementado, avaliado em 2026-09-11, ver `CLAUDE.md`). Testes
  de log-loss mostraram que o modelo generaliza bem pra 2020, mas o
  trabalho real (parser do `.txt` + reconciliação manual de
  nome→`team_id` sem erro silencioso) só compensa se o site crescer
  a ponto de precisar de calibração mais robusta.

## Sobre a fonte

Os dados vêm da API do [football-data.org](https://www.football-data.org),
competição BSA (Brasileirão Série A). O plano grátis é **só para uso
não comercial** e cobre a temporada corrente e as 3 anteriores (2023
em diante -- confirmado por chamada real em 2026-09-07; temporadas mais
antigas respondem 403). Publique métrica derivada e gráfico com
crédito à fonte; não republique os dados brutos como produto nem sirva
anúncio em cima disso -- é o que separa uso tolerado de uso comercial.

O limite de 10 requests/minuto existe por um motivo. Não o contorne.

## Contribuindo

Licença MIT (ver `LICENSE`) -- fique à vontade pra abrir issue ou PR.

Pra rodar local, siga "Ordem de execução" acima (passos 1-6). Depois
de mudar algo, rode o trecho relevante do pipeline (`ingest.py` →
`elo.py` → `derived.py` → `build_site.py`, nessa ordem -- `derived.py`
lê o parquet que `elo.py` gera) e confira o `docs/index.html` resultante
antes de abrir o PR. Leia o `CLAUDE.md` antes: ele lista decisões que já
foram tentadas e revertidas (bypass de anti-bot, fontes de dados
descartadas, bugs reais já corrigidos) pra não repetir o mesmo caminho.

K=20, HFA=65 e ν=0.74 (Davidson) não são valores definitivos -- foram
escolhidos por robustez num grid search, documentados com o raciocínio
em `CLAUDE.md`. Proponha recalibração à vontade, mas venha com
justificativa: rode `src/calibrate_elo.py` e `src/validate_elo.py` e
mostre o antes/depois no PR. Se `validate_elo.py` mudar de resultado,
atualize os dois lugares que citam o número (ver a regra em
`CLAUDE.md`) no mesmo commit.

## Estrutura

```text
config.py                  competicao, temporadas, parametros do Elo
src/ingest.py              football-data.org -> data/raw/*.parquet (incremental)
src/probe_schema.py        dump do schema real
src/calibrate_elo.py       grid search de K/HFA + Davidson nu (one-off)
src/elo.py                 rating + Monte Carlo -> data/processed/*.parquet
src/derived.py              metricas derivadas -> data/processed/*.parquet
src/validate_elo.py        validacao do modelo (rode quando ele mudar)
src/card_confronto.py      card PNG 1600x900 do confronto (fundo por IA, resto Pillow)
src/build_site.py          data/processed + data/raw -> docs/index.html
fora-da-sumula-v3.html     template do site: CSS, paginas e JS. O build so troca
                           as duas linhas com sentinela (/*__DATA__*/ e /*__ELO__*/),
                           e o arquivo guarda o payload da ultima geracao -- da pra
                           abrir direto no navegador pra iterar design sem pipeline
docs/index.html            gerado -- e o que o GitHub Pages serve (pasta fixa do Pages)
assets/crests, teams.json  escudo e cores dos 20 clubes (1 request, versionado)
cards/                     PNG gerado e cache de fundo da IA (ignorado pelo git)
.github/workflows/         agendamento, commit e build do site automáticos
```

# WhatsRecap: contexto completo para o projeto de benchmark de modelos

> Documento de passagem de contexto. Quem lê é **outra sessão** que vai começar um projeto novo,
> separado deste repositório, para **testar e escolher os modelos de linguagem** que o WhatsRecap
> vai usar. Ele resume tudo o que existe hoje (estado em 2026-10-03, commit `ef9aa32`) e tudo
> o que o projeto de benchmark precisa saber para medir o que importa.
>
> - Repositório do app: `/home/zelph/Documentos/whats-recap` (branch `main`)
> - Planejamento original (fora do repo): `/home/zelph/Downloads/planejamento-analisador-whatsapp.md`
> - Brief de design do frontend: `docs/design-brief.md`

---

## 1. O que é o WhatsRecap

App desktop que analisa o export de uma conversa do WhatsApp **entre duas pessoas** e mostra
estatísticas, análises semânticas (desculpas, brigas) e um chat de perguntas sobre a conversa.

- **Tauri v2** (Rust + React 19/TS/Vite), **100% local**: nenhuma mensagem sai da máquina.
  Isso vale também para o LLM: tudo roda num `llama-server` local. **Nada de APIs na nuvem**
  para dados de conversa.
- Interface e respostas em **português do Brasil**.
- Projeto pessoal / portfólio. Os resultados do benchmark devem ser documentados no README.

### Princípio central

> **Código conta, LLM classifica.** Toda contagem e agregação é determinística (Rust + SQLite).
> O LLM só (a) classifica trechos, (b) escolhe uma ferramenta e preenche parâmetros, ou
> (c) redige uma frase a partir de números já calculados. Ele **nunca** produz os números finais.

Consequência para o benchmark: o que se mede é **fidelidade a schema, acerto de classificação e
de parâmetros, e ausência de alucinação numérica**, não "conhecimento" do modelo.

---

## 2. Hardware alvo (onde o benchmark de verdade precisa rodar)

Existem duas máquinas:

| | Máquina de desenvolvimento (esta) | Máquina final (alvo) |
|---|---|---|
| SO | Linux (Arch, KDE Wayland) | — |
| CPU | — | Ryzen 7 5700X (8C/16T, Zen 3, AVX2, sem iGPU) |
| GPU | — | **Radeon RX 7600, 8 GB GDDR6** (RDNA 3, `gfx1102`) |
| RAM | — | 16 GB DDR4 |

**Números de velocidade e VRAM só valem se medidos na máquina final.** Na de desenvolvimento dá
para montar o harness, rodar testes de qualidade com modelos pequenos e validar o pipeline.

### Runtime decidido

- **llama.cpp `llama-server` com backend Vulkan** (ROCm na RX 7600 não é oficial e exige
  `HSA_OVERRIDE_GFX_VERSION`; Vulkan funciona sem gambiarra).
- Será empacotado como **sidecar do Tauri** (o app inicia e encerra o servidor).
- API compatível com OpenAI (`/v1/chat/completions`), então o cliente pode trocar de runtime.
- **Saída estruturada obrigatória** via `response_format` com `json_schema` (ou gramática GBNF).
  Modelos pequenos sem restrição de saída não são confiáveis.

### Orçamento de VRAM (8 GB)

| Item | Estimativa |
|---|---|
| Modelo 7–8B em `Q4_K_M` | ~4,5–5 GB |
| KV cache 8k–16k tokens | ~1–2 GB (menos com KV em `q8_0`) |
| Sobra para sistema/desktop | ~1 GB |

Limite prático: **~7–9B quantizado, contexto de ~8k–16k**.

### Modelos candidatos (do planejamento; atualizar com o que existir hoje)

- Família **Qwen** na faixa de 7–8B (costuma ir bem em português).
- Família **Gemma** na faixa de ~9B (mais pesado, apertado em 8 GB).
- Vale incluir também modelos menores (1–4B) como baseline de velocidade, e checar
  lançamentos recentes nessas famílias e em outras com bom português.
- **Embeddings** (etapa separada, pode rodar na CPU): `multilingual-e5-small` (rápido) ou
  `bge-m3` (melhor, mais pesado).

> A medição de **processamento de prompt (tok/s)** e **geração (tok/s)** decide se o
> "modo completo" das análises (passar a conversa inteira pelo LLM) é viável ou se fica só o
> "modo rápido" com pré-filtro.

---

## 3. Estado do app

| Fase | Entrega | Status |
|---|---|---|
| 0 — Setup | Workspace Rust, Tauri + React, CI | ✅ |
| 1 — Parser e storage | Android/iOS, dia/mês, SQLite + FTS5, sessões, fixtures | ✅ |
| 2 — Estatísticas | Features 1–16 | ✅ |
| 3 — Dashboard | Gráficos SVG, visualizador virtualizado | ✅ |
| 4 — Mídia | Figurinhas (SHA-256 + dHash), áudios (duração Ogg) | ✅ |
| 8 (parcial) — Chat tipo 1 | Roteador + interpretação **heurística** + ferramentas fechadas | ✅ sem LLM |
| **5 — Infra de LLM** | Sidecar, **benchmark de modelos na RX 7600**, cliente com JSON schema | ⏳ **é o que o projeto novo destrava** |
| 6 — Desculpas | Pré-filtro + pipeline LLM | ⏳ |
| 7 — Brigas | Expansão de contexto, merge de intervalos | ⏳ |
| 9 — Chat tipo 2 | Indexação (embeddings + FTS5), busca híbrida, citações | ⏳ |

Ainda **não existe cliente HTTP de LLM** no código. O contrato (schemas e prompts) já existe
em `crates/core-chat/src/llm.rs` (seção 6.1 abaixo).

### Estrutura do repositório

```
crates/
  core-parser    .txt/.zip → mensagens normalizadas (streaming, 2 passadas)
  core-storage   SQLite por conversa: messages, sessions, FTS5, media, stats_cache, tabelas LLM
  core-stats     features 1–18 (rayon); também normalização de texto/stopwords
  core-media     figurinhas e áudios lidos do .zip
  core-chat      chat tipo 1: router.rs, intent.rs (heurística), tools.rs, answer.rs, llm.rs
  synth          gerador de conversas sintéticas + bin `anonymize`
src-tauri/       commands Tauri (chat_ask, chat_examples, get_stats, search_messages…)
src/             React: Home, Dashboard, Viewer, Settings, Chat
```

Rust edition 2024, workspace em `Cargo.toml`. Rust instalado via rustup em `~/.cargo`
(o PATH precisa de `$HOME/.cargo/bin`). Frontend com pnpm, Node 22+.

---

## 4. Dados disponíveis para o benchmark

### 4.1 Formato das mensagens no banco

Cada conversa importada vira um SQLite (`~/.local/share/dev.whatsrecap.app/conversations/<id>.db`):

```sql
messages(id INTEGER PRIMARY KEY,  -- sequencial a partir de 1; é a referência que o LLM cita
         ts INTEGER NOT NULL,     -- horário local do export, em segundos, sem fuso
         author TEXT,             -- NULL = sistema
         kind TEXT NOT NULL,      -- text|sticker|audio|image|video|doc|deleted|system|media_hidden
         text TEXT, media_file TEXT, session_id INTEGER);
sessions(id, start_id, end_id, start_ts, end_ts);       -- nova sessão após gap > 6h (configurável)
llm_events(id, type, start_id, end_id, payload_json);  -- já criada, vazia (desculpas, brigas)
llm_jobs(id, type, chunk_index, status, updated_at);   -- já criada, vazia (checkpoint/retomada)
chunks(id, session_id, start_id, end_id, text, embedding BLOB);  -- já criada, vazia (RAG)
messages_fts  -- FTS5, tokenize 'unicode61 remove_diacritics 2'
meta(key, value)  -- summary (autores, plataforma…), settings
```

O projeto de benchmark pode ler esses `.db` diretamente (somente leitura) ou depender dos crates
por `path` (`core-storage`, `core-chat`), para reaproveitar `Db::open`, `Db::sessions()`,
`Db::messages_from(id, n)`, `Db::search_any(...)` e `Conversation::execute(&ToolCall)`.

### 4.2 Conversas sintéticas (sem dados reais)

```sh
cd /home/zelph/Documentos/whats-recap
cargo run --release -p synth -- --days 365 --per-day 60 --names "Ana,Bruno" --seed 42 -o /tmp/conversa.txt
cargo run --release -p synth -- --days 1825 --per-day 120 --platform ios -o /tmp/conversa.txt
cargo run --release -p synth -- --days 1825 --per-day 120 --media -o /tmp/conversa.zip
# importa e mostra estatísticas; o banco fica em $TMPDIR/whats-recap-analyze.db
# (sobrescrito a cada execução: copie-o se quiser guardar)
cargo run --release -p core-stats --example analyze -- /tmp/conversa.txt --json /tmp/stats.json
```

Flags do `synth`: `--platform android|ios`, `--days`, `--start AAAA-MM-DD`, `--names "A,B"`,
`--per-day`, `--seed` (determinístico), `--media` (exige `-o .zip`), `-o`.

**Limitação importante:** o texto sintético é gerado por templates. Ele serve para perguntas
quantitativas (chat tipo 1), mas **não tem brigas, desculpas sinceras nem assuntos coerentes**.
Para avaliar as fases 6, 7 e 9 o projeto de benchmark vai precisar de um **conjunto rotulado
próprio**: trechos escritos à mão (ou gerados e revisados) com o gabarito de desculpas,
brigas (intervalos de ids) e perguntas pontuais com as mensagens que respondem.

### 4.3 Exports reais

- Fixtures do parser em `crates/core-parser/tests/fixtures/` (escritas à mão, pequenas).
- Anonimizador para exports reais:
  `cargo run -p synth --bin anonymize -- "Conversa.txt" -o saida.txt` (troca nomes, palavras,
  números, URLs). **Conversas reais nunca devem ir para nenhum serviço externo nem para o git.**

### 4.4 Perguntas com gabarito já existentes (chat tipo 1)

`crates/core-chat/src/tests.rs` tem uma conversa mínima com valores contados à mão
(ex.: "amor": Ana 1, Bruno 3; madrugada: Ana 2, Bruno 1; janeiro de 2024 = mês com mais
mensagens; medianas de resposta Ana 240s, Bruno 90s). `crates/core-chat/src/intent.rs` tem
testes com a `ToolCall` esperada para cada pergunta — servem de semente para o dataset de
avaliação da seleção de ferramenta. Para rodar perguntas contra um `.db`:

```sh
cargo run --release -p core-chat --example ask -- --json conversa.db "Quem responde mais rápido?" "Qual mês teve mais mensagens?"
```

---

## 5. Tarefas que o LLM vai executar (o que precisa ser testado)

### Tarefa A — Roteador do chat (curta, muito frequente)

Classificar a pergunta em `quantitative` | `retrieval` | `global`. Uma heurística por regex
(`crates/core-chat/src/router.rs`) já resolve os casos óbvios; o LLM só recebe os `Undecided`.
Schema em `llm::router_schema()`. Exemplos:

- quantitative: "Quantas vezes o Bruno falou 'amor'?", "Quem responde mais rápido?",
  "Qual mês teve mais mensagens?"
- retrieval: "quando a gente combinou a viagem pra Floripa?", "o que ela disse sobre o emprego novo?"
- global (fora do escopo, resposta fixa): "Como evoluiu a nossa relação?", "faz um resumo"

**Métricas:** acurácia, latência (deve ser < ~1 s).

### Tarefa B — Seleção de ferramenta e parâmetros (chat tipo 1)

O modelo **não lê a conversa**. Recebe o prompt de `llm::tool_selection_prompt(authors,
first_date, last_date)` e devolve JSON no schema `llm::tool_schema(authors)` (os nomes dos
autores entram como `enum`, então ele não pode inventar uma pessoa). Ferramentas fechadas:

| Ferramenta | Args |
|---|---|
| `count_word` | `word`, `author?`, `from?`, `to?` |
| `messages_by_hour` | `author?`, `from?`, `to?` |
| `messages_by_period` | `granularity` (`day`/`week`/`month`/`year`/`weekday`), `author?`, `from?`, `to?` |
| `first_occurrence` | `word`, `author?` |
| `response_time` | `author?`, `from?`, `to?` |

Formato: `{"tool":"count_word","args":{"word":"amor","author":"Bruno","from":"2024-01-01"}}`.
Datas `AAAA-MM-DD`, inclusivas. Regras: omitir o que não foi mencionado; em "quem…?" comparar
as duas pessoas (sem `author`). A saída desserializa direto em `core_chat::ToolCall`.

O foco do LLM aqui são as perguntas que a heurística **não** entende, por exemplo
"quantas vezes a gente brigou?" (fora das ferramentas → deveria falhar com elegância),
datas relativas ("no último verão", "ano passado"), apelidos, sinônimos, erros de digitação.

**Métricas:** JSON válido (com schema forçado deve ser 100%), acerto exato da ferramenta,
acerto de cada argumento (palavra, autor, datas, granularidade), latência.

### Tarefa C — Redação da resposta a partir do resultado

Prompt `llm::answer_prompt(question, tool_output_json)`: uma ou duas frases em português,
**somente com os números do resultado**. Hoje isso é feito por templates em `answer.rs`;
o LLM é opcional aqui.

**Métricas:** todo número citado existe no JSON de entrada (checável automaticamente por regex),
fluência em pt-BR, nenhum número inventado ou arredondado de forma enganosa.

### Tarefa D — Detecção de pedidos de desculpa (fase 6, feature 19)

Entrada: janela de mensagens formatada com ids:

```
[1234] Ana: você nunca me escuta
[1235] Bruno: nossa, lá vem
```

Pré-filtro em Rust (modo rápido, padrão) seleciona janelas em torno de "desculpa", "desculpe",
"foi mal", "perdão", "me perdoa", "sorry" e variações. Saída:
`{"apologies":[{"msg_id":1234,"from":"Ana","to":"Bruno","sincere":true}]}`.
Definição: pedido de desculpas direcionado à outra pessoa; "desculpa a demora" conta, mas com
`sincere: false` (cortesia) vs. `true` (após conflito). O prompt deve ter few-shot.

**Métricas:** precisão/recall por `msg_id`, acerto de `sincere`, consistência entre execuções.

### Tarefa E — Detecção de brigas (fase 7, feature 20) — a mais difícil

Mesma entrada em janelas. Pré-filtro: CAPS LOCK, excesso de `?!`, troca rápida de mensagens
curtas, palavrões, respostas secas ("ok.", "tanto faz", "blz"), apagadas em sequência. Saída:

```json
{
  "conflicts": [{ "start_id": 1200, "end_id": 1260, "participants": ["Ana","Bruno"],
                  "intensity": 3, "resolved": true, "summary": "Discussão sobre atraso no jantar" }],
  "need_context": { "direction": "before", "reason": "a discussão já começa no início do trecho" }
}
```

`intensity` 1–5; `direction` = `"before"` | `"after"` | `null`. Definição: desacordo com tom
negativo e escalada envolvendo as duas pessoas; **ironia e provocação de brincadeira não contam**.
O backend expande a janela por regra determinística (conflito encostado na borda do chunk) e
também atende `need_context`, com no máximo 3 expansões por evento. Conflitos sobrepostos entre
chunks são unidos; a contagem final é feita pelo código a partir de `llm_events`.

**Métricas:** recall/precisão de conflitos (sobreposição de intervalo, ex.: IoU ≥ 0,5),
erro de borda (start/end), falsos positivos em brincadeiras, uso correto de `need_context`,
erro médio de `intensity`, qualidade do `summary` em pt-BR.

### Tarefa F — Chat tipo 2: RAG com citações (fase 9, feature 22)

Indexação: janelas de mensagens (sessões, subdivididas se longas) → embeddings (via
`sqlite-vec`) + FTS5. Consulta: busca híbrida (vetorial + full-text, combinadas por
**Reciprocal Rank Fusion**), 5–10 janelas com ids vão ao LLM, que responde **citando os ids**
(`#1234`), que viram links no visualizador.

**Métricas (duas partes):**
- *Embeddings:* recall@k das janelas corretas, tempo de indexação na CPU para conversas grandes
  (~190 mil mensagens), comparação só-vetorial vs. só-FTS vs. híbrida.
- *LLM:* resposta correta, citações válidas (ids existem e estão no contexto) e pertinentes,
  recusa quando a resposta não está no contexto.

### Throughput (decide o modo completo)

Uma conversa de 5 anos tem ~190 mil mensagens. Medir, por modelo/quantização/contexto:
tok/s de prompt, tok/s de geração, VRAM usada, tempo por chunk de ~8k tokens, e extrapolar
o tempo total do modo completo e do modo rápido.

---

## 6. Contratos existentes no código (copiar, não reinventar)

### 6.1 `crates/core-chat/src/llm.rs`

```rust
pub fn router_schema() -> Value  // {"type": enum["quantitative","retrieval","global"]}, additionalProperties false
pub fn tool_schema(authors: &[String]) -> Value  // oneOf das 5 ferramentas; author = enum(authors);
                                                 // date = pattern ^\d{4}-\d{2}-\d{2}$; word 1..60 chars
pub fn tool_selection_prompt(authors: &[String], first_date: &str, last_date: &str) -> String
pub fn answer_prompt(question: &str, tool_output: &Value) -> String
```

Prompt de seleção de ferramenta (texto atual):

```
Você escolhe UMA ferramenta para responder a uma pergunta sobre uma conversa de WhatsApp entre
{A} e {B} (de {first_date} a {last_date}). Você não lê a conversa: só escolhe a ferramenta e
preenche os parâmetros.

Ferramentas:
- count_word(word, author?, from?, to?): quantas vezes uma palavra ou expressão curta aparece.
- messages_by_hour(author?, from?, to?): mensagens por hora do dia (0–23), por pessoa. Use para
  "de madrugada", "que horas", "quem manda mais".
- messages_by_period(granularity, author?, from?, to?): mensagens por dia, semana, mês, ano ou dia da semana.
- first_occurrence(word, author?): a primeira mensagem com a palavra ou expressão.
- response_time(author?, from?, to?): tempo de resposta (média e mediana) por pessoa.

Regras: datas no formato AAAA-MM-DD, inclusivas; omita parâmetros não mencionados; "author" só
quando a pergunta for sobre uma pessoa específica (em "quem...?", compare as duas e omita).
```

Prompt de resposta:

```
Pergunta: {question}

Resultado calculado (JSON):
{tool_output}

Responda em português, em uma ou duas frases, usando SOMENTE os números do resultado.
Não invente números nem arredonde de forma que mude o sentido.
```

Os prompts **podem e devem ser iterados** no projeto de benchmark; o que vencer volta para
`llm.rs`.

### 6.2 Resultado das ferramentas (`ToolResult`, JSON com tag `tool`, campos camelCase)

- `count_word`: `{word, total, byAuthor[], messages, examples[ids]}`
- `messages_by_hour`: `{byAuthor: [[24 números], [24 números]]}`
- `messages_by_period`: `{granularity, periods[{period, byAuthor[], total}], top[], bottom[]}`
- `first_occurrence`: `{word, message: MessageRow | null}`
- `response_time`: `{byAuthor[{count, meanSecs, medianSecs, buckets[4]}], fastest: 0|1|null}`

`byAuthor` segue a ordem dos autores da conversa. Erros de parâmetro (autor desconhecido,
palavra vazia, datas invertidas) viram resposta textual, não falha.

---

## 7. Restrições e preferências que o projeto novo deve respeitar

- **Privacidade:** nenhuma conversa (nem trecho) vai para API externa. Modelos de nuvem só podem
  ser usados, se for o caso, para gerar ou rotular **dados sintéticos**, nunca dados reais.
- **pt-BR em tudo:** prompts, datasets e qualidade de texto avaliados em português brasileiro
  informal (gírias, "kkk", abreviações como "vc", "blz", erros de digitação, áudios e figurinhas
  intercalados).
- **Saída estruturada sempre** (`json_schema`), e o resultado tem de desserializar nos tipos Rust
  existentes.
- **Reprodutibilidade:** registrar modelo, arquivo GGUF, quantização, versão/commit do llama.cpp,
  flags (`-ngl`, `-c`, `--cache-type-k/v`, `-fa`), temperatura e seed em cada resultado.
- **Commits sem linha `Co-Authored-By` do Claude** (preferência do usuário).
- O app é em Rust; o harness pode ser em Rust (reaproveita os crates via `path`) ou em
  Python/outra linguagem chamando o `llama-server` por HTTP — o que importa é o contrato JSON.

---

## 8. Entregáveis esperados do projeto de benchmark

1. Harness que sobe/usa um `llama-server` e roda os datasets das tarefas A–F contra vários
   modelos/quantizações, gravando resultados em arquivo (JSON/CSV).
2. Datasets rotulados em pt-BR para cada tarefa (com gabarito), sem dados reais.
3. Medição de throughput e VRAM **na RX 7600** (Vulkan).
4. Tabela comparativa e recomendação: modelo de chat/classificação, quantização, tamanho de
   contexto, modelo de embedding, e se o "modo completo" é viável.
5. Prompts finais (para voltar a `crates/core-chat/src/llm.rs` e para as fases 6, 7 e 9).

---

## 9. Riscos já conhecidos

- Modelos de 7–8B erram em **ironia e contexto** (brigas). Mitigação prevista: few-shot,
  pré-filtro, expansão determinística de contexto e eventos clicáveis para o usuário conferir.
- **Tempo do modo completo** em conversas de anos pode ser de horas → modo rápido como padrão,
  jobs retomáveis (`llm_jobs`).
- Limites ainda a calibrar: gap de sessão (6h), sobreposição entre chunks (20–30 mensagens),
  tamanho de janela do pré-filtro, top-k do RAG.
- Os formatos de export ainda não foram validados com exports reais (não afeta o benchmark
  de LLM, mas afeta dados de teste vindos de exports reais).

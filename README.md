# whatsrecap-evals

Benchmark dos modelos de linguagem locais que o [WhatsRecap](../whats-recap) vai usar.
Sobe um `llama-server` (llama.cpp, Vulkan) por modelo, roda as tarefas do app com **saída estruturada
obrigatória** (`json_schema`) e grava cada resultado com o hardware, o modelo, a versão do llama.cpp
e as flags usadas.

O crate depende do `core-chat` do app por `path`: os prompts e schemas atuais (`core_chat::llm`), a
heurística do roteador e a desserialização de `ToolCall` são **os do app**, não cópias. A variante de
prompt `v1` é sempre o que o app envia hoje; as outras são experimentos.

> Velocidade e VRAM só valem na **máquina alvo** (Ryzen 7 5700X + RX 7600 8 GB). Na máquina de
> desenvolvimento o harness serve para validar o pipeline e medir qualidade com modelos pequenos.
> Cada execução registra `is_target` e o relatório separa os dois casos.

## Preparação

```sh
scripts/setup-llama-cpp.sh            # Linux: baixa o llama.cpp pré-compilado (Vulkan) para vendor/
scripts/setup-llama-cpp.ps1           # Windows: idem (zip win-vulkan-x64; -Backend rocm-10.0 para comparar)
cargo build --release                 # binário em target/release/evals
./target/release/evals selftest       # valida datasets e métricas sem modelo (gabarito → nota máxima)
./target/release/evals list           # modelos, embedders, perfis e o que já foi baixado
./target/release/evals download qwen3.5-4b-q4km e5-small-q8   # ou --all (~45 GB)
```

O `Cargo.toml` aponta para `../whats-recap/crates/core-chat` (os dois repositórios lado a lado). Se o
app estiver em outro lugar, ajuste esse caminho.

## Uso

```sh
# Qualidade (tarefas A–F). Perfil padrão: dev (CPU). Na máquina alvo: -p rx7600.
evals run qwen3.5-4b-q4km -p rx7600
evals run qwen3.5-4b-q4km qwen3.5-9b-iq4xs gemma4-e4b-q4_0 -p rx7600 -n 3   # 3 repetições: consistência
evals run qwen3.5-4b-q4km -t tools,answer -v v2      # variante de prompt experimental
evals run qwen3.5-4b-q4km -t tools --ids t15,t16     # só alguns itens
evals run qwen3.5-4b-q4km --server-args "-c 16384 --reasoning on"

# Throughput, VRAM e extrapolação do modo completo (padrão: perfil rx7600, chunk de 8k)
evals bench qwen3.5-4b-q4km qwen3.5-9b-iq4xs
evals bench qwen3.5-9b-q4km -p rx7600-8k --chunk-tokens 8192 --fast-hit-rate 0.02

# Embeddings do chat tipo 2: vetorial × FTS5 × híbrida (RRF)
evals retrieval e5-small-q8 bge-m3-q8 -p r5700x-cpu

# Tabela comparativa de tudo o que já rodou
evals report        # results/REPORT.md e results/summary.csv

# Depurar um prompt
evals prompt tools t15 -v v2
```

Cada execução cria `results/runs/<data>_<tipo>_<modelo>_<perfil>_<host>/` com `meta.json`
(modelo, GGUF, quantização, versão do llama.cpp, argv completo do servidor, hardware, temperatura,
seed), `items.jsonl` (cada chamada: saída, erro, latência, `timings` do servidor e pontuação),
`summary.json` e o log do `llama-server` (inclui a tabela de memória por dispositivo).

### Fluxo sugerido na RX 7600

1. `evals hw` deve mostrar `"is_target": true`.
2. `evals bench <todos os candidatos>`: descarta o que não cabe ou é lento demais.
3. `evals run <sobreviventes> -p rx7600 -n 3`, depois `-v v2` para comparar prompts.
4. `evals retrieval e5-small-q8 bge-m3-q8 -p r5700x-cpu` (embeddings na CPU, como previsto).
5. `evals report` e levar a tabela para o README do app.

## Tarefas e métricas

| Tarefa | Dataset | Métrica principal | Outras |
|---|---|---|---|
| A — roteador | `router.json` (44) | acurácia | acurácia nas perguntas que a heurística do app não decide / erra |
| B — ferramenta | `tools.json` (40) | acerto exato (fora as "nenhuma ferramenta") | acerto em perguntas difíceis, por argumento (palavra, autor, datas, granularidade), recusa |
| C — redação | `answer.json` (14) | ok = fatos certos + nenhum número inventado + ≤ 2 frases | números inventados, fatos obrigatórios |
| D — desculpas | `apologies.json` (14 janelas) | F1 por `msg_id` | precisão, recall, `sincere`, direção, falsos positivos em janelas sem desculpa |
| E — brigas | `conflicts.json` (12 janelas) | F1 (IoU ≥ 0,5) | IoU médio, erro de borda, erro de intensidade, `resolved`, `need_context`, falsos positivos em zoeira |
| F — RAG | `rag.json` (18 perguntas, 20 janelas) | ok | resposta certa, cita o gabarito, citações válidas, recusa quando não há resposta |
| Embeddings | `rag.json` + 150 janelas de ruído | recall@k, MRR | vetorial × FTS5 × híbrida, tempo de indexação extrapolado |
| Throughput | trecho de ~8k tokens | tok/s de prompt e de geração | VRAM, s/chunk, horas do modo completo, minutos do modo rápido |

Todas as métricas são determinísticas (nenhum LLM avalia outro). Todo `items.jsonl` guarda as
saídas para revisão manual da fluência em pt-BR e dos `summary` das brigas.

Todos os datasets são fictícios e escritos à mão, sem dados reais; cada arquivo explica o próprio
formato no campo `_doc`. Para acrescentar itens, edite o JSON e rode `evals selftest`.

## Achados já visíveis no teste com o Qwen3.5-2B (CPU, poucos itens)

- **C, prompt `v1`:** o `ToolResult` não diz de quem é cada posição de `byAuthor` e traz `ts` como
  número. O modelo responde "os autores 1 e 3". A variante `v2` acrescenta `authors` e uma data
  legível (`contracts::enrich_tool_output`); se vencer, a mudança vai para o app.
- **B, prompt `v1`:** sem a data de hoje no prompt, datas relativas ("ano passado") não têm como
  dar certo, e o schema não tem saída para "nenhuma ferramenta" (`none_accuracy` só existe na `v2`).
- **Embeddings:** no corpus atual, a busca híbrida (RRF) ficou abaixo da só-vetorial com o
  e5-small, porque o FTS com `OR` traz muito ruído. Vale testar FTS com `AND`/frases antes de fixar a
  fusão.

## Estrutura

```
models.toml            modelos candidatos, embedders e perfis (dev, dev-vulkan, rx7600, rx7600-8k, r5700x-cpu)
datasets/              conjuntos rotulados em pt-BR
scripts/setup-llama-cpp.sh
src/
  main.rs              CLI (clap)
  contracts.rs         v1 = core_chat::llm; variantes e propostas para D/E/F
  tasks/               uma tarefa por arquivo: monta o prompt, pontua, agrega, gabarito do selftest
  scoring.rs           métricas compartilhadas (ToolCall, números em pt-BR, IoU, P/R/F1)
  runner.rs            execução e gravação dos resultados
  bench.rs             llama-bench + chunk real + extrapolação
  retrieval.rs         embeddings, FTS5 (mesmo tokenizer do app) e RRF
  server.rs client.rs  llama-server como subprocesso e cliente HTTP
  hardware.rs          impressão digital do hardware e VRAM (sysfs no Linux, typeperf no Windows)
  report.rs            REPORT.md e summary.csv
```

A leitura de VRAM usa o sysfs do `amdgpu` no Linux e o contador "GPU Adapter Memory" (`typeperf`) no
Windows; nos dois casos é a VRAM total em uso (inclui o desktop), por isso cada execução grava a leitura
antes e depois de carregar o modelo. Os buffers que o llama.cpp reporta no log valem como estimativa
independente. No Windows cada leitura leva ~1 s, então o pico é amostrado a cada ~1 s.

# Comparação de modelos — RX 7600 (2026-10-29)

Primeira rodada completa do benchmark na máquina alvo. Os resultados brutos ficam em
`results/runs/` (fora do git); este arquivo guarda os números e as decisões.

**Decisão:** família **Qwen3.5**. Modelo recomendado para o app: **Qwen3.5 9B `IQ4_XS`**, sem flash
attention. O **Qwen3.5 4B `Q4_K_M`** é a alternativa se a VRAM apertar.

O foco do app são **perguntas sobre a conversa** (roteador, seleção de ferramenta, redação e RAG).
Desculpas e brigas são secundárias, e nelas o Qwen ainda perde para o Gemma (ver "Pendências").

## Ambiente

| | |
|---|---|
| Máquina | Ryzen 7 5700X · Radeon RX 7600 8 GB · 32 GB RAM · Windows 11 Pro 10.0.26200 |
| Driver | AMD 32.0.31041.1004 |
| llama.cpp | `b11938` (commit `19cce6919`), binário oficial `win-vulkan-x64` |
| Saída | `json_schema` obrigatório, temperatura 0, seed 42, `--reasoning off` |
| Repetições | 3 por item (consistência entre execuções medida em cada tarefa) |

Perfis usados (`models.toml`):

- `rx7600`: `-ngl 99 -c 16384 -fa on -ctk q8_0 -ctv q8_0`
- `rx7600-nofa`: `-ngl 99 -c 16384 -fa off` (KV em f16; criado nesta rodada)

## Candidatos

| id | arquivo | tamanho |
|---|---|---|
| `qwen3.5-4b-q4km` | `Qwen3.5-4B-Q4_K_M.gguf` | 2,6 GiB |
| `qwen3.5-9b-iq4xs` | `Qwen3.5-9B-IQ4_XS.gguf` | 4,8 GiB |
| `gemma4-e4b-q4_0` | `gemma-4-E4B-it-Q4_0.gguf` | 4,3 GiB |
| `e5-small-q8` (embedder) | `multilingual-e5-small-q8_0.gguf` | 126 MiB |

## Throughput e VRAM

Chunk real de ~7.340 tokens com o prompt e o schema de brigas, pelo `llama-server`. "Modo completo"
= 190 mil mensagens em chunks de 8k com sobreposição de 25 mensagens, 2 passadas (desculpas + brigas).

| modelo | perfil | prompt tok/s | geração tok/s | s/chunk | modo completo | pico VRAM¹ | Δ ao carregar |
|---|---|---|---|---|---|---|---|
| Qwen3.5 4B | com FA | 468 | 31,1 | 27,5 | 5,7 h | 4,4 GB | 2,9 GB |
| Qwen3.5 4B | **sem FA** | **1.382** | **53,0** | **11,3** | **2,35 h** | 5,5 GB | 3,8 GB |
| Qwen3.5 9B | com FA | 368 | 24,6 | 29,5 | 6,1 h | 6,3 GB | 4,7 GB |
| Qwen3.5 9B | **sem FA** | **874** | **37,6** | **13,7** | **2,85 h** | **7,2 GB** | 5,5 GB |
| Gemma 4 E4B | **com FA** | 526 | **32,7** | 25,2 | 5,25 h | 4,4 GB | 3,0 GB |
| Gemma 4 E4B | sem FA | 788 | 19,3 | 28,2 | 5,9 h | 4,9 GB | 3,3 GB |

¹ VRAM total da placa (contador "GPU Adapter Memory"), incluindo ~1,4–1,7 GB do desktop.

`llama-bench` (pp512 / pp8192 / tg128, tok/s), para referência: Qwen 4B 1.602 / 1.429 / 33;
Qwen 9B 927 / 886 / 26; Gemma 1.696 / 1.251 / 52. O `llama-bench` não mostra a perda do FA que o
`llama-server` tem (ver achado 3).

**Modo rápido** (só janelas do pré-filtro; 1% das mensagens → 1.900 janelas × 2 tarefas), pela
latência real das janelas de desculpas e brigas: **Qwen 4B ~50 min**, Qwen 9B ~65 min, Gemma
~105 min. A extrapolação impressa pelo `evals bench` superestima esse número (ver "Pendências").

## Qualidade

Métrica principal de cada tarefa (ver README). Melhor de cada coluna em negrito.

### v1 (prompts atuais do app)

| modelo | A roteador | B ferramenta | C redação | D desculpas (F1) | E brigas (F1) | F RAG |
|---|---|---|---|---|---|---|
| Qwen3.5 4B | 86% | 41% | 57% | **0,96** | 0,46 | **94%** |
| Qwen3.5 9B | **95%** | 38% | 57% | 0,88 | 0,31 | **94%** |
| Gemma 4 E4B | 77% | **59%** | 50% | 0,89 | **0,86** | 59% |

### v2 (prompts experimentais de A–C)

| modelo | A roteador | B ferramenta | C redação |
|---|---|---|---|
| Qwen3.5 4B | 91% | 38% | 71% |
| Qwen3.5 9B | **98%** | 14% ² | **86%** |
| Gemma 4 E4B | 73% | **59%** | 57% |

² Colapso causado pela ordem das chaves no schema (achado 1): o modelo escolhe `messages_by_hour`
para quase tudo.

### B com o schema corrigido (`"tool"` antes de `"args"`)

| modelo | v1 exato | v1 normalizado ³ | v2 exato | v2 normalizado ³ | ferramenta certa (v2) |
|---|---|---|---|---|---|
| Qwen3.5 4B | 51% | 65% | 41% | 57% | 100% |
| Qwen3.5 9B | 41% | 68% | 49% | **76%** | 100% |
| Gemma 4 E4B | 65% | 73% | **77%** | **84%** | 97% |

³ Descartando `from`/`to` iguais aos limites da conversa (equivalem a omitir o filtro). Calculado
fora do harness sobre os `items.jsonl`.

### Latência (p50) e consistência

| modelo | roteador | ferramenta | redação | RAG | consistência entre repetições |
|---|---|---|---|---|---|
| Qwen3.5 4B | 0,3 s | 0,7–1,2 s | 0,9 s | 1,5 s | 100% em todas as tarefas |
| Qwen3.5 9B | 0,4 s | 1,0–1,8 s | 1,4–1,5 s | 2,0 s | 100% em todas as tarefas |
| Gemma 4 E4B | 0,4–0,5 s | 1,1–1,7 s | 1,0–1,6 s | 2,5 s (até 6 s) | 79–100% |

### Embeddings (chat tipo 2)

`e5-small-q8` na CPU (`r5700x-cpu`), 170 janelas, 14 perguntas:

| busca | MRR | recall@1 | recall@5 | recall@10 |
|---|---|---|---|---|
| **só vetorial** | **0,84** | **0,79** | **0,86** | **0,93** |
| híbrida (RRF) | 0,70 | 0,64 | 0,79 | 0,79 |
| só FTS5 (`OR`) | 0,63 | 0,57 | 0,64 | 0,71 |

Indexação: ~2.400 tok/s → ~20 min para 190 mil mensagens.

## Achados

1. **Ordem das chaves no `tool_schema` (afeta o app).** O workspace serializa JSON com as chaves
   em ordem alfabética (`serde_json` sem `preserve_order`), então cada ramo do `oneOf` chega ao
   llama.cpp com `"args"` antes de `"tool"`. A gramática segue essa ordem e obriga o modelo a
   preencher os argumentos antes de escolher a ferramenta. Teste A/B nas 40 perguntas, só trocando
   a ordem: Qwen 9B v2 **14 → 39/40** ferramentas certas; os demais sobem para 37–39/40.
2. **Datas do período inteiro.** Os Qwen preenchem `from`/`to` com a conversa toda (15 das 20
   falhas do 9B). É inofensivo, e o código pode descartar.
3. **Flash attention no Vulkan/Windows.** No `llama-server`, `-fa on` deixa o processamento de
   prompt do Qwen ~3× mais lento (o kernel parece percorrer o KV de 16k inteiro). Sem FA, os Qwen
   ficam 2–2,5× mais rápidos de ponta a ponta. O Gemma é o contrário: sem FA a geração cai de 33 para
   19 tok/s. As flags do sidecar precisam ser por modelo.
4. **O nível de log (`-lv 4`) não afeta a velocidade** (testado: ~500 tok/s com e sem).
5. **O Gemma é fraco em RAG** (59%, com respostas de até 6 s) e menos consistente entre repetições.
6. **Brigas no Qwen:** pede `need_context: "after"` quase sempre, junta duas brigas em uma, marca
   provocação de brincadeira como briga e, às vezes, cita um id que não está na janela.

## Recomendação

**Qwen3.5 9B `IQ4_XS`, perfil sem FA**, para o chat:

- melhor roteador (98%) e melhor redação (86%, menos números inventados);
- empata com o 4B no RAG (94%), muito acima do Gemma;
- com o schema corrigido, chega a 76% de acerto exato em B (o Gemma, a 84%);
- latência de 1–2 s por resposta, imperceptível num chat;
- 100% consistente entre repetições.

Custos:

- **VRAM:** 5,5 GB ao carregar, pico de 7,2 GB na placa com o desktop. Sobra ~0,8 GB; jogos ou
  outros usos pesados da GPU com o app aberto podem não caber. Opções a testar: `-c 8192` ou
  `-ctk q8_0` sem FA (só o K quantizado não exige FA).
- **Velocidade nas análises:** modo completo de 2,85 h (contra 2,35 h do 4B); modo rápido ~65 min.

**Alternativa: Qwen3.5 4B `Q4_K_M`, perfil sem FA**, se a VRAM ou o tempo de análise pesarem mais:
3,8 GB ao carregar, o mais rápido em tudo, e o melhor em desculpas. Perde em roteador (91%) e redação
(71%).

O Gemma 4 E4B fica de fora: é o melhor em brigas e em seleção de ferramenta, mas o pior em RAG,
roteador e redação, que são o foco.

## Pendências

- [ ] Corrigir a ordem das chaves no `tool_schema` do app (`crates/core-chat/src/llm.rs`) e no
      evals, e normalizar `from`/`to` iguais aos limites da conversa; rodar B de novo
- [ ] Iterar o prompt de brigas (E) para o Qwen: `need_context`, brigas juntas, zoeira, ids
- [ ] Testar `qwen3.5-9b-q4km` (5,7 GB; pode não caber com 16k sem FA) e as opções de VRAM acima
- [ ] `evals retrieval bge-m3-q8` e FTS com `AND`/frases
- [ ] Corrigir a extrapolação do modo rápido em `src/bench.rs`, que aplica a saída de um chunk
      inteiro (~300 tokens) a cada janela pequena
- [ ] Ampliar os datasets (40 perguntas em B e 18 em F: diferenças de poucos pontos são ruído)
- [ ] Opcional: build ROCm do llama.cpp (`scripts/setup-llama-cpp.ps1 -Backend rocm-10.0`)

## Reprodução

```powershell
.\target\release\evals.exe bench qwen3.5-4b-q4km gemma4-e4b-q4_0 qwen3.5-9b-iq4xs              # com FA
.\target\release\evals.exe bench qwen3.5-4b-q4km gemma4-e4b-q4_0 qwen3.5-9b-iq4xs -p rx7600-nofa
.\target\release\evals.exe run qwen3.5-4b-q4km qwen3.5-9b-iq4xs -p rx7600-nofa -n 3
.\target\release\evals.exe run gemma4-e4b-q4_0 -p rx7600 -n 3
.\target\release\evals.exe run <modelos> -t router,tools,answer -v v2 -n 3
.\target\release\evals.exe retrieval e5-small-q8 -p r5700x-cpu
.\target\release\evals.exe report
```

A tabela "B com o schema corrigido" usou um build temporário com `serde_json/preserve_order`
(execuções `20261029-2220*` a `20261029-2232*`).

//! Contratos com o LLM.
//!
//! As variantes `v1` de roteador (schema), seleção de ferramenta e redação vêm direto de
//! `core_chat::llm`: o benchmark testa exatamente o que o app envia. As outras variantes são
//! experimentos; a que vencer volta para o `llm.rs`. As tarefas D, E e F ainda não têm contrato no
//! app: os schemas e prompts daqui são a proposta.

use core_chat::llm;
use serde_json::{Value, json};

// ---------------------------------------------------------------- A: roteador

pub fn router_schema() -> Value {
    llm::router_schema()
}

const ROUTER_V1: &str = "\
Você classifica perguntas sobre uma conversa de WhatsApp entre duas pessoas. Responda só com o tipo:
- quantitative: contagem, frequência, comparação ou estatística que dá para calcular (quantas vezes, quem manda mais, que horas, qual mês, tempo de resposta, primeira vez que alguém disse algo).
- retrieval: pergunta pontual sobre o conteúdo de algum momento da conversa (o que alguém disse, quando combinaram algo, onde foram).
- global: pede uma análise ou resumo da conversa inteira ou da relação (como evoluiu, resumo, melhor fase).";

const ROUTER_V2_EXAMPLES: &str = "

Exemplos:
\"quantas vezes ela me chamou de chato?\" → quantitative
\"quem puxa mais assunto de manhã?\" → quantitative
\"o que a gente decidiu sobre o aluguel?\" → retrieval
\"qual era o nome do restaurante que ele indicou?\" → retrieval
\"a gente brigava mais no começo?\" → global
\"me conta como foi nosso 2023\" → global";

pub fn router_prompt(variant: &str) -> String {
    match variant {
        "v2" => format!("{ROUTER_V1}{ROUTER_V2_EXAMPLES}"),
        _ => ROUTER_V1.to_string(),
    }
}

// ---------------------------------------------------------------- B: ferramenta

/// Variantes que acrescentam a saída `{"tool":"none"}` ao schema (o `llm.rs` não tem saída para
/// perguntas fora das ferramentas).
pub fn tool_allows_none(variant: &str) -> bool {
    variant == "v2"
}

pub fn tool_schema(authors: &[String], variant: &str) -> Value {
    let mut s = llm::tool_schema(authors);
    if tool_allows_none(variant) {
        s["oneOf"].as_array_mut().unwrap().push(json!({
            "type": "object",
            "properties": {
                "tool": { "const": "none" },
                "args": { "type": "object", "properties": {}, "additionalProperties": false }
            },
            "required": ["tool", "args"],
            "additionalProperties": false
        }));
    }
    s
}

pub fn tool_prompt(
    variant: &str,
    authors: &[String],
    first: &str,
    last: &str,
    today: &str,
) -> String {
    let base = llm::tool_selection_prompt(authors, first, last);
    match variant {
        // v1 + data de hoje (datas relativas), apelidos e a saída "none".
        "v2" => format!(
            "{base}\n\nHoje é {today}. Converta datas relativas (\"ano passado\", \"mês passado\", \"no verão\") em \
intervalos AAAA-MM-DD a partir de hoje; o verão vai de 21 de dezembro a 20 de março. Apelidos e pronomes (\"ela\", \
\"ele\", \"a Aninha\") se referem a uma das duas pessoas pelo nome. Em count_word e first_occurrence, \"word\" é só a \
palavra ou expressão procurada, sem aspas, como a pessoa escreveria. Se nenhuma ferramenta responde (ex.: brigas, \
sentimentos, assuntos), use {{\"tool\":\"none\",\"args\":{{}}}}."
        ),
        _ => base,
    }
}

// ---------------------------------------------------------------- C: redação

/// O app pede texto livre; aqui a resposta vai num campo para manter "saída estruturada sempre".
pub fn answer_schema() -> Value {
    json!({
        "type": "object",
        "properties": { "answer": { "type": "string", "minLength": 1, "maxLength": 400 } },
        "required": ["answer"],
        "additionalProperties": false
    })
}

pub fn answer_prompt(variant: &str, question: &str, output: &Value, authors: &[String]) -> String {
    match variant {
        "v2" => format!(
            "{} Em \"byAuthor\" a ordem é a das pessoas em \"authors\". Tempos estão em segundos: pode convertê-los \
para minutos. Dias da semana: 0 = segunda, 1 = terça, 2 = quarta, 3 = quinta, 4 = sexta, 5 = sábado, 6 = domingo. \
Não calcule porcentagens e não cite ids de mensagens.",
            llm::answer_prompt(question, &enrich_tool_output(output, authors))
        ),
        _ => llm::answer_prompt(question, output),
    }
}

/// Proposta para o app: o `ToolResult` hoje não diz de quem é cada posição de `byAuthor` nem traz
/// `ts` legível. A variante v2 acrescenta `authors` e `message.date`.
pub fn enrich_tool_output(output: &Value, authors: &[String]) -> Value {
    let mut out = output.clone();
    out["authors"] = json!(authors);
    if let Some(ts) = out["message"]["ts"].as_i64()
        && let Some(t) = chrono::DateTime::from_timestamp(ts, 0)
    {
        out["message"]["date"] = json!(t.format("%Y-%m-%d %H:%M").to_string());
    }
    out
}

// ---------------------------------------------------------------- D: desculpas

pub fn apologies_schema(authors: &[String]) -> Value {
    json!({
        "type": "object",
        "properties": {
            "apologies": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "msg_id": { "type": "integer" },
                        "from": { "enum": authors },
                        "to": { "enum": authors },
                        "sincere": { "type": "boolean" }
                    },
                    "required": ["msg_id", "from", "to", "sincere"],
                    "additionalProperties": false
                }
            }
        },
        "required": ["apologies"],
        "additionalProperties": false
    })
}

pub fn apologies_prompt(_variant: &str, a: &str, b: &str) -> String {
    format!(
        "Você lê um trecho de conversa de WhatsApp entre {a} e {b}. Cada linha é \"[id] Pessoa: mensagem\".
Encontre os pedidos de desculpa que uma pessoa faz à OUTRA. Para cada um, informe o id da mensagem, quem pede (\"from\"), para quem (\"to\") e \"sincere\":
- sincere = true: pede desculpas por algo que magoou, irritou ou prejudicou a outra pessoa (normalmente depois de um conflito).
- sincere = false: desculpa de cortesia (\"desculpa a demora\", \"foi mal, tava no banho\"), sem conflito.
Não conte: desculpa dirigida a terceiros, desculpa citada ou contada (\"ele pediu desculpa pro chefe\"), ironia (\"ah, desculpa por existir\"), \"perdão\" como interjeição de surpresa, nem pedir desculpas em nome de outra pessoa.
Se não houver nenhum, devolva uma lista vazia.

Exemplo:
[10] {a}: vc esqueceu de novo de me buscar
[11] {b}: putz amor desculpa, de verdade, vacilei feio
[12] {a}: tá
[13] {b}: foi mal a demora pra responder, tava em reunião
Saída: {{\"apologies\":[{{\"msg_id\":11,\"from\":\"{b}\",\"to\":\"{a}\",\"sincere\":true}},{{\"msg_id\":13,\"from\":\"{b}\",\"to\":\"{a}\",\"sincere\":false}}]}}"
    )
}

// ---------------------------------------------------------------- E: brigas

pub fn conflicts_schema(authors: &[String]) -> Value {
    json!({
        "type": "object",
        "properties": {
            "conflicts": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "start_id": { "type": "integer" },
                        "end_id": { "type": "integer" },
                        "participants": { "type": "array", "items": { "enum": authors }, "minItems": 1, "maxItems": 2 },
                        "intensity": { "type": "integer", "minimum": 1, "maximum": 5 },
                        "resolved": { "type": "boolean" },
                        "summary": { "type": "string", "maxLength": 160 }
                    },
                    "required": ["start_id", "end_id", "participants", "intensity", "resolved", "summary"],
                    "additionalProperties": false
                }
            },
            "need_context": {
                "type": "object",
                "properties": {
                    "direction": { "enum": ["before", "after", null] },
                    "reason": { "type": "string", "maxLength": 160 }
                },
                "required": ["direction", "reason"],
                "additionalProperties": false
            }
        },
        "required": ["conflicts", "need_context"],
        "additionalProperties": false
    })
}

pub fn conflicts_prompt(_variant: &str, a: &str, b: &str) -> String {
    format!(
        "Você lê um trecho de conversa de WhatsApp entre {a} e {b}. Cada linha é \"[id] Pessoa: mensagem\".
Encontre as BRIGAS: desacordo com tom negativo e escalada envolvendo as duas pessoas (acusações, ironia ácida, respostas secas, CAPS LOCK, ameaça de parar de falar). Provocação de brincadeira, zoeira com \"kkk\", reclamar de terceiros juntos ou discordar com calma NÃO são brigas.
Para cada briga: start_id (primeira mensagem do desacordo), end_id (última mensagem dele, incluindo a reconciliação se houver), participants, intensity de 1 (alfinetada) a 5 (briga muito pesada), resolved (terminou em reconciliação ou acordo dentro do trecho) e summary (uma frase em português dizendo o motivo).
need_context: \"before\" se uma briga já está acontecendo na primeira linha do trecho, \"after\" se uma briga continua depois da última linha, null caso contrário; reason explica em poucas palavras (ou \"\" se null).
Se não houver brigas, devolva \"conflicts\": [].

Exemplo:
[50] {b}: chego 20h
[51] {a}: vc falou isso ontem e chegou 22h
[52] {b}: lá vem
[53] {a}: LÁ VEM O QUE? eu fiquei esperando feito idiota
[54] {b}: tá bom, errei. hoje eu chego 20h, prometo
[55] {a}: ok
[56] {b}: kkkk olha esse meme
Saída: {{\"conflicts\":[{{\"start_id\":51,\"end_id\":55,\"participants\":[\"{a}\",\"{b}\"],\"intensity\":3,\"resolved\":true,\"summary\":\"Discussão por causa do atraso de ontem\"}}],\"need_context\":{{\"direction\":null,\"reason\":\"\"}}}}"
    )
}

// ---------------------------------------------------------------- F: RAG

pub fn rag_schema() -> Value {
    json!({
        "type": "object",
        "properties": {
            "found": { "type": "boolean" },
            "answer": { "type": "string", "maxLength": 400 },
            "citations": { "type": "array", "items": { "type": "integer" }, "maxItems": 6 }
        },
        "required": ["found", "answer", "citations"],
        "additionalProperties": false
    })
}

pub fn rag_prompt(_variant: &str, a: &str, b: &str, context: &str) -> String {
    format!(
        "Você responde perguntas sobre uma conversa de WhatsApp entre {a} e {b} usando SOMENTE os trechos abaixo. Cada linha é \"[id] data Pessoa: mensagem\".
- Se a resposta estiver nos trechos: found = true, answer em português (uma ou duas frases, cite os ids no texto como #123) e citations com os ids das mensagens que sustentam a resposta.
- Se não estiver: found = false, answer explica que não encontrou na conversa, citations = [].
Nunca use conhecimento de fora nem invente detalhes.

Trechos:
{context}"
    )
}

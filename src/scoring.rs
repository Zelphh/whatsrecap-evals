//! Funções de pontuação compartilhadas. Tudo determinístico: nenhum LLM avalia outro LLM aqui.

use chrono::{DateTime, Datelike, Timelike};
use core_chat::ToolCall;
use regex::Regex;
use serde_json::{Value, json};
use std::collections::BTreeSet;
use std::sync::LazyLock;

/// Minúsculas e sem acentos: a mesma normalização das heurísticas do app.
pub fn fold(s: &str) -> String {
    core_chat::router::fold(s)
}

pub fn norm_word(s: &str) -> String {
    static EDGES: LazyLock<Regex> =
        LazyLock::new(|| Regex::new(r#"^["'“”‘’«»\s]+|["'“”‘’«»\s.,!?]+$"#).unwrap());
    let f = fold(s);
    let f = EDGES.replace_all(f.trim(), "");
    f.split_whitespace().collect::<Vec<_>>().join(" ")
}

pub fn round(x: f64, d: i32) -> f64 {
    let p = 10f64.powi(d);
    (x * p).round() / p
}

pub fn pct(xs: &[f64], p: f64) -> Option<f64> {
    let mut v: Vec<f64> = xs.iter().copied().filter(|x| x.is_finite()).collect();
    if v.is_empty() {
        return None;
    }
    v.sort_by(f64::total_cmp);
    let k = (v.len() - 1) as f64 * p;
    let (lo, hi) = (
        k.floor() as usize,
        (k.floor() as usize + 1).min(v.len() - 1),
    );
    Some(round(v[lo] + (v[hi] - v[lo]) * (k - lo as f64), 1))
}

pub fn mean(xs: impl IntoIterator<Item = f64>) -> Option<f64> {
    let v: Vec<f64> = xs.into_iter().filter(|x| x.is_finite()).collect();
    (!v.is_empty()).then(|| round(v.iter().sum::<f64>() / v.len() as f64, 4))
}

pub fn rate(xs: impl IntoIterator<Item = bool>) -> Option<f64> {
    let v: Vec<bool> = xs.into_iter().collect();
    (!v.is_empty()).then(|| round(v.iter().filter(|x| **x).count() as f64 / v.len() as f64, 4))
}

pub fn prf(tp: usize, fp: usize, fn_: usize) -> Value {
    let p = if tp + fp > 0 {
        tp as f64 / (tp + fp) as f64
    } else if fn_ == 0 {
        1.0
    } else {
        0.0
    };
    let r = if tp + fn_ > 0 {
        tp as f64 / (tp + fn_) as f64
    } else {
        1.0
    };
    let f = if p + r > 0.0 {
        2.0 * p * r / (p + r)
    } else {
        0.0
    };
    json!({"precision": round(p, 4), "recall": round(r, 4), "f1": round(f, 4)})
}

// ---------------------------------------------------------------- ToolCall

/// Saída do modelo na tarefa B: uma `ToolCall` do app ou "nenhuma ferramenta".
pub enum Selected {
    Tool(ToolCall),
    None,
}

/// Desserializa como o app faria (`serde_json::from_value::<ToolCall>`) e aplica as checagens que
/// fariam `Conversation::execute` falhar: autor desconhecido, palavra vazia, datas invertidas.
pub fn parse_tool_call(
    out: &Value,
    authors: &[String],
    allow_none: bool,
) -> Result<Selected, String> {
    if allow_none && out["tool"] == "none" {
        return Ok(Selected::None);
    }
    let call: ToolCall =
        serde_json::from_value(out.clone()).map_err(|e| format!("não desserializa: {e}"))?;
    if let Some(a) = call.author()
        && !authors.iter().any(|x| x == a)
    {
        return Err(format!("autor desconhecido: {a}"));
    }
    let v = serde_json::to_value(&call).unwrap();
    if let Some(w) = v["args"]["word"].as_str()
        && w.trim().is_empty()
    {
        return Err("word vazia".into());
    }
    if let (Some(f), Some(t)) = call.range()
        && f > t
    {
        return Err("datas invertidas".into());
    }
    Ok(Selected::Tool(call))
}

/// `{"tool": ..., "args": {...}}` normalizado (parâmetros ausentes somem, datas em AAAA-MM-DD).
pub fn selected_value(s: &Selected) -> Value {
    match s {
        Selected::Tool(c) => serde_json::to_value(c).unwrap(),
        Selected::None => json!({"tool": "none", "args": {}}),
    }
}

fn arg_matches(key: &str, expected: &Value, got: Option<&Value>) -> bool {
    let alts: Vec<&Value> = match expected {
        Value::Array(a) => a.iter().collect(),
        v => vec![v],
    };
    let Some(got) = got else {
        return alts.iter().any(|a| a.is_null());
    };
    alts.iter().filter(|a| !a.is_null()).any(|a| {
        if key == "word"
            && let (Some(x), Some(y)) = (a.as_str(), got.as_str())
        {
            return norm_word(x) == norm_word(y);
        }
        *a == got
    })
}

#[derive(Debug, Clone)]
pub struct ToolMatch {
    pub tool_ok: bool,
    pub exact: bool,
    /// Acerto por parâmetro (só quando a ferramenta está certa). Parâmetro inventado = false.
    pub args: Vec<(String, bool)>,
}

fn compare_one(expected: &Value, got: &Value) -> ToolMatch {
    let tool_ok = expected["tool"] == got["tool"];
    let empty = serde_json::Map::new();
    let exp = expected["args"].as_object().unwrap_or(&empty);
    let out = got["args"].as_object().unwrap_or(&empty);
    let keys: BTreeSet<&String> = exp.keys().chain(out.keys()).collect();
    let args: Vec<(String, bool)> = keys
        .into_iter()
        .map(|k| {
            (
                k.clone(),
                exp.get(k).is_some_and(|e| arg_matches(k, e, out.get(k))),
            )
        })
        .collect();
    ToolMatch {
        tool_ok,
        exact: tool_ok && args.iter().all(|(_, ok)| *ok),
        args: if tool_ok { args } else { vec![] },
    }
}

/// Compara com a melhor alternativa do gabarito (`expected` é uma chamada ou uma lista delas).
pub fn best_tool_match(expected: &Value, got: &Value) -> ToolMatch {
    let alts: Vec<&Value> = match expected {
        Value::Array(a) => a.iter().collect(),
        v => vec![v],
    };
    alts.into_iter()
        .map(|e| compare_one(e, got))
        .max_by_key(|m| {
            (
                m.exact,
                m.tool_ok,
                m.args.iter().filter(|(_, ok)| *ok).count(),
            )
        })
        .unwrap()
}

// ---------------------------------------------------------------- números na resposta (tarefa C)

static NUM_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?i)(?:^|[^\w#])(\d{1,3}(?:\.\d{3})+|\d+)(?:,(\d+))?(?:\s*(mil|milhões|milhão))?")
        .unwrap()
});

/// Números em pt-BR: "18.234" = 18234, "25,5" = 25.5, "18 mil" = 18000. Devolve (valor, aproximado),
/// aproximado quando escrito com "mil"/"milhões".
pub fn extract_numbers(text: &str) -> Vec<(f64, bool)> {
    NUM_RE
        .captures_iter(text)
        .map(|c| {
            let whole = c[1].replace('.', "");
            let mut v: f64 = match c.get(2) {
                Some(d) => format!("{whole}.{}", d.as_str()).parse().unwrap_or(0.0),
                None => whole.parse().unwrap_or(0.0),
            };
            let approx = c.get(3).is_some();
            if let Some(m) = c.get(3) {
                v *= if m.as_str().to_lowercase().starts_with("milh") {
                    1e6
                } else {
                    1e3
                };
            }
            (v, approx)
        })
        .collect()
}

/// Números que uma resposta fiel pode citar: valores do resultado e conversões diretas
/// (segundos → minutos/horas, `ts` → data/hora, "AAAA-MM-DD" → partes, índice de hora), números da
/// pergunta e os extras do item.
pub fn allowed_numbers(output: &Value, question: &str, extra: &[f64]) -> Vec<f64> {
    let mut allowed: Vec<f64> = extra.to_vec();
    allowed.extend(extract_numbers(question).into_iter().map(|(v, _)| v));
    fn walk(node: &Value, key: &str, out: &mut Vec<f64>) {
        match node {
            Value::Number(n) => {
                let v = n.as_f64().unwrap_or(f64::NAN);
                out.push(v);
                if key.to_lowercase().ends_with("secs") {
                    for d in [60.0, 3600.0] {
                        out.push(round(v / d, 1));
                        out.push((v / d).round());
                    }
                }
                if key == "ts"
                    && let Some(t) = DateTime::from_timestamp(v as i64, 0)
                {
                    out.extend([
                        t.year() as f64,
                        t.month() as f64,
                        t.day() as f64,
                        t.hour() as f64,
                        t.minute() as f64,
                    ]);
                }
            }
            Value::String(s) => {
                static DATE: LazyLock<Regex> =
                    LazyLock::new(|| Regex::new(r"^(\d{4})(?:-(\d{2})(?:-(\d{2}))?)?$").unwrap());
                if let Some(c) = DATE.captures(s) {
                    out.extend(
                        c.iter()
                            .skip(1)
                            .flatten()
                            .filter_map(|m| m.as_str().parse::<f64>().ok()),
                    );
                }
            }
            Value::Array(a) => {
                if a.len() == 24 && a.iter().all(Value::is_u64) {
                    out.extend((0..24).map(f64::from)); // índice = hora do dia
                }
                a.iter().for_each(|x| walk(x, key, out));
            }
            Value::Object(o) => o.iter().for_each(|(k, x)| walk(x, k, out)),
            _ => {}
        }
    }
    walk(output, "", &mut allowed);
    allowed
}

pub fn unsupported_numbers(text: &str, allowed: &[f64]) -> Vec<f64> {
    extract_numbers(text)
        .into_iter()
        .filter(|(v, approx)| {
            !allowed.contains(v)
                // "18 mil" para 18.234 é arredondamento aceitável
                && !(*approx && allowed.iter().any(|a| *a != 0.0 && ((v - a) / a).abs() <= 0.05))
        })
        .map(|(v, _)| v)
        .collect()
}

pub fn contains_any(text: &str, group: &[String]) -> bool {
    let t = fold(text);
    group.iter().any(|g| t.contains(&fold(g)))
}

pub fn sentence_count(text: &str) -> usize {
    static SPLIT: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"[.!?]+(\s+|$)").unwrap());
    SPLIT
        .split(text.trim())
        .filter(|p| !p.trim().is_empty())
        .count()
}

// ---------------------------------------------------------------- intervalos (tarefa E)

pub fn iou(a: (i64, i64), b: (i64, i64)) -> f64 {
    let inter = (a.1.min(b.1) - a.0.max(b.0) + 1).max(0);
    let union = (a.1 - a.0 + 1) + (b.1 - b.0 + 1) - inter;
    if union > 0 {
        inter as f64 / union as f64
    } else {
        0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn numbers() {
        assert_eq!(
            extract_numbers("foram 18.234 mensagens, 25,5 min e 18 mil; #123 não conta"),
            vec![(18234.0, false), (25.5, false), (18000.0, true)]
        );
        let allowed = allowed_numbers(
            &json!({"medianSecs": 420.0, "byAuthor": [3, 1]}),
            "em 2024?",
            &[],
        );
        assert!(unsupported_numbers("7 minutos, 3 vezes, em 2024", &allowed).is_empty());
        assert_eq!(unsupported_numbers("foram 5 vezes", &allowed), vec![5.0]);
    }

    #[test]
    fn tool_calls() {
        let authors = vec!["Ana".to_string(), "Bruno".to_string()];
        let bad_date = json!({"tool": "count_word", "args": {"word": "x", "from": "2024-02-30"}});
        assert!(parse_tool_call(&bad_date, &authors, false).is_err());
        assert!(parse_tool_call(&json!({"tool": "none", "args": {}}), &authors, false).is_err());
        assert!(parse_tool_call(&json!({"tool": "none", "args": {}}), &authors, true).is_ok());
        let got = selected_value(
            &parse_tool_call(
                &json!({"tool": "count_word", "args": {"word": "“Amor”"}}),
                &authors,
                false,
            )
            .unwrap(),
        );
        let m = best_tool_match(
            &json!({"tool": "count_word", "args": {"word": "amor", "to": [null, "2024-01-01"]}}),
            &got,
        );
        assert!(m.exact);
    }

    #[test]
    fn intervals_and_sentences() {
        assert_eq!(iou((1, 10), (6, 15)), 5.0 / 15.0);
        assert_eq!(
            sentence_count("Foi o Bruno. Ele mandou 1.850 mensagens!"),
            2
        );
    }
}

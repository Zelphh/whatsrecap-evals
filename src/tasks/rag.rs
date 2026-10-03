//! F (parte LLM) — resposta com citações a partir de 5 janelas: acerto, citações válidas e
//! pertinentes, e recusa quando a resposta não está no contexto.

use super::{
    ConversationMeta, Msg, Request, Rng, Row, Task, failed_ids, fmt_window, load, scores,
    system_user,
};
use crate::contracts;
use crate::scoring as S;
use anyhow::Result;
use regex::Regex;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::collections::BTreeSet;
use std::sync::LazyLock;

pub const CONTEXT_WINDOWS: usize = 5;

#[derive(Debug, Clone, Deserialize)]
pub struct Window {
    pub id: String,
    pub date: String,
    pub messages: Vec<Msg>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Question {
    pub id: String,
    pub q: String,
    pub answerable: bool,
    #[serde(default)]
    pub gold_windows: Vec<String>,
    #[serde(default)]
    pub gold_ids: Vec<i64>,
    #[serde(default)]
    pub must: Vec<Vec<String>>,
    /// Janelas que entram no contexto mesmo sem responder (armadilhas).
    #[serde(default)]
    pub context_extra: Vec<String>,
}

#[derive(Deserialize)]
struct File {
    windows: Vec<Window>,
    questions: Vec<Question>,
}

#[derive(Serialize, Deserialize)]
struct Score {
    pass: bool,
    answerable: bool,
    found_ok: bool,
    citations_valid: bool,
    answer: String,
    content_ok: Option<bool>,
    cites_gold: Option<bool>,
    cite_precision: Option<f64>,
    refusal_ok: Option<bool>,
}

pub struct Rag {
    conv: ConversationMeta,
    pub windows: Vec<Window>,
    pub questions: Vec<Question>,
}

impl Rag {
    pub fn load(conv: ConversationMeta) -> Result<Self> {
        let f: File = load("rag.json")?;
        Ok(Rag {
            conv,
            windows: f.windows,
            questions: f.questions,
        })
    }

    fn window(&self, id: &str) -> &Window {
        self.windows
            .iter()
            .find(|w| w.id == id)
            .expect("janela do gabarito existe")
    }

    /// Janelas que vão ao LLM: gabarito + armadilhas + ruído determinístico até completar 5, em ordem de data.
    fn context(&self, q: &Question) -> Vec<&Window> {
        let mut chosen: Vec<String> = vec![];
        for w in q.gold_windows.iter().chain(&q.context_extra) {
            if !chosen.contains(w) {
                chosen.push(w.clone());
            }
        }
        let mut rest: Vec<String> = self
            .windows
            .iter()
            .map(|w| w.id.clone())
            .filter(|w| !chosen.contains(w))
            .collect();
        Rng::from_str(&q.id).shuffle(&mut rest);
        let missing = CONTEXT_WINDOWS.saturating_sub(chosen.len());
        chosen.extend(rest.into_iter().take(missing));
        let mut ws: Vec<&Window> = chosen.iter().map(|id| self.window(id)).collect();
        ws.sort_by(|a, b| a.date.cmp(&b.date));
        ws
    }
}

impl Task for Rag {
    fn name(&self) -> &'static str {
        "rag"
    }
    fn title(&self) -> &'static str {
        "F — RAG com citações"
    }
    fn headline(&self) -> &'static str {
        "ok"
    }
    fn ids(&self) -> Vec<String> {
        self.questions.iter().map(|q| q.id.clone()).collect()
    }

    fn build(&self, i: usize, variant: &str) -> Request {
        let q = &self.questions[i];
        let ctx = self
            .context(q)
            .iter()
            .map(|w| fmt_window(&w.messages, Some(&w.date)))
            .collect::<Vec<_>>()
            .join("\n\n");
        let a = &self.conv.authors;
        Request {
            messages: system_user(
                contracts::rag_prompt(variant, &a[0], &a[1], &ctx),
                q.q.clone(),
            ),
            schema: contracts::rag_schema(),
            max_tokens: 300,
        }
    }

    fn score(&self, i: usize, out: Option<&Value>, _: &str) -> Value {
        static INLINE: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"#(\d+)").unwrap());
        let q = &self.questions[i];
        let null = Value::Null;
        let out = out.unwrap_or(&null);
        let found = out["found"].as_bool().unwrap_or(false);
        let answer = out["answer"].as_str().unwrap_or_default().to_string();
        let cites: BTreeSet<i64> = out["citations"]
            .as_array()
            .map(|a| a.iter().filter_map(Value::as_i64).collect())
            .unwrap_or_default();
        let inline: BTreeSet<i64> = INLINE
            .captures_iter(&answer)
            .filter_map(|c| c[1].parse().ok())
            .collect();
        let ctx_ids: BTreeSet<i64> = self
            .context(q)
            .iter()
            .flat_map(|w| w.messages.iter().map(|m| m.0))
            .collect();
        let citations_valid = cites.union(&inline).all(|c| ctx_ids.contains(c));
        let found_ok = found == q.answerable;
        let mut s = Score {
            pass: false,
            answerable: q.answerable,
            found_ok,
            citations_valid,
            answer: answer.clone(),
            content_ok: None,
            cites_gold: None,
            cite_precision: None,
            refusal_ok: None,
        };
        if q.answerable {
            let gold: BTreeSet<i64> = q.gold_ids.iter().copied().collect();
            s.content_ok = Some(found && q.must.iter().all(|g| S::contains_any(&answer, g)));
            s.cites_gold = Some(cites.intersection(&gold).next().is_some());
            s.cite_precision = (!cites.is_empty())
                .then(|| cites.intersection(&gold).count() as f64 / cites.len() as f64);
            s.pass = found_ok
                && citations_valid
                && s.content_ok == Some(true)
                && s.cites_gold == Some(true);
        } else {
            s.refusal_ok = Some(!found && cites.is_empty());
            s.pass = found_ok && citations_valid && s.refusal_ok == Some(true);
        }
        serde_json::to_value(s).unwrap()
    }

    fn gold(&self, i: usize, _: &str) -> Value {
        let q = &self.questions[i];
        if !q.answerable {
            return json!({"found": false, "answer": "Não encontrei isso na conversa.", "citations": []});
        }
        let parts: Vec<&str> = q.must.iter().map(|g| g[0].as_str()).collect();
        json!({"found": true, "answer": format!("{} (#{})", parts.join(" "), q.gold_ids[0]), "citations": q.gold_ids})
    }

    fn aggregate(&self, rows: &[Row]) -> Value {
        let s = scores::<Score>(rows);
        let pos = || s.iter().filter(|(_, x)| x.answerable);
        json!({
            "ok": S::rate(s.iter().map(|(_, x)| x.pass)),
            "answer_correct": S::rate(pos().filter_map(|(_, x)| x.content_ok)),
            "cites_gold": S::rate(pos().filter_map(|(_, x)| x.cites_gold)),
            "cite_precision": S::mean(pos().filter_map(|(_, x)| x.cite_precision)),
            "citations_valid": S::rate(s.iter().map(|(_, x)| x.citations_valid)),
            "refusal_acc": S::rate(s.iter().filter_map(|(_, x)| x.refusal_ok)),
            "failed_ids": failed_ids(rows),
        })
    }
}

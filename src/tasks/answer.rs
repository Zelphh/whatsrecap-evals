//! C — redação da resposta a partir do resultado da ferramenta. Checa que todo número citado sai
//! do resultado (ou de uma conversão direta dele) e que os fatos esperados aparecem.

use super::{ConversationMeta, Request, Row, Task, failed_ids, load, scores};
use crate::contracts;
use crate::scoring as S;
use anyhow::Result;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

#[derive(Deserialize)]
struct Item {
    id: String,
    q: String,
    /// `ToolResult` como o app serializa.
    output: Value,
    /// Cada grupo precisa de pelo menos uma das formas.
    #[serde(default)]
    must: Vec<Vec<String>>,
    #[serde(default)]
    forbid: Vec<String>,
    #[serde(default)]
    allowed_extra: Vec<f64>,
}

#[derive(Deserialize)]
struct File {
    items: Vec<Item>,
}

#[derive(Serialize, Deserialize)]
struct Score {
    pass: bool,
    text: String,
    must_ok: bool,
    no_invented_numbers: bool,
    invented: Vec<f64>,
    sentences: usize,
}

pub struct Answer {
    conv: ConversationMeta,
    items: Vec<Item>,
}

impl Answer {
    pub fn load(conv: ConversationMeta) -> Result<Self> {
        Ok(Answer {
            conv,
            items: load::<File>("answer.json")?.items,
        })
    }
}

impl Task for Answer {
    fn name(&self) -> &'static str {
        "answer"
    }
    fn title(&self) -> &'static str {
        "C — redação da resposta"
    }
    fn headline(&self) -> &'static str {
        "ok"
    }
    fn variants(&self) -> &'static [&'static str] {
        &["v1", "v2"]
    }
    fn ids(&self) -> Vec<String> {
        self.items.iter().map(|i| i.id.clone()).collect()
    }

    fn build(&self, i: usize, variant: &str) -> Request {
        let it = &self.items[i];
        // Como no app: um único turno de usuário, sem system.
        let prompt = contracts::answer_prompt(variant, &it.q, &it.output, &self.conv.authors);
        Request {
            messages: vec![json!({"role": "user", "content": prompt})],
            schema: contracts::answer_schema(),
            max_tokens: 200,
        }
    }

    fn score(&self, i: usize, out: Option<&Value>, _: &str) -> Value {
        let it = &self.items[i];
        let text = out
            .and_then(|o| o["answer"].as_str())
            .unwrap_or_default()
            .to_string();
        let allowed = S::allowed_numbers(&it.output, &it.q, &it.allowed_extra);
        let invented = S::unsupported_numbers(&text, &allowed);
        let must_ok = it.must.iter().all(|g| S::contains_any(&text, g))
            && !it
                .forbid
                .iter()
                .any(|f| S::contains_any(&text, std::slice::from_ref(f)));
        let sentences = if text.is_empty() {
            0
        } else {
            S::sentence_count(&text)
        };
        serde_json::to_value(Score {
            pass: must_ok && invented.is_empty() && (1..=2).contains(&sentences),
            no_invented_numbers: invented.is_empty(),
            text,
            must_ok,
            invented,
            sentences,
        })
        .unwrap()
    }

    fn gold(&self, i: usize, _: &str) -> Value {
        let parts: Vec<&str> = self.items[i].must.iter().map(|g| g[0].as_str()).collect();
        json!({"answer": format!("{}.", parts.join(" "))})
    }

    fn aggregate(&self, rows: &[Row]) -> Value {
        let s = scores::<Score>(rows);
        json!({
            "ok": S::rate(s.iter().map(|(_, x)| x.pass)),
            "must_ok": S::rate(s.iter().map(|(_, x)| x.must_ok)),
            "no_invented_numbers": S::rate(s.iter().map(|(_, x)| x.no_invented_numbers)),
            "within_2_sentences": S::rate(s.iter().map(|(_, x)| (1..=2).contains(&x.sentences))),
            "failed_ids": failed_ids(rows),
        })
    }
}

//! B — seleção de ferramenta e parâmetros (chat tipo 1). A saída é desserializada como
//! `core_chat::ToolCall`, exatamente como o app fará.

use super::{ConversationMeta, Request, Row, Task, failed_ids, load, scores, system_user};
use crate::contracts;
use crate::scoring::{self as S, Selected};
use anyhow::Result;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value, json};

#[derive(Deserialize)]
struct Item {
    id: String,
    q: String,
    /// Uma ToolCall ou lista de alternativas. Em args, lista = "qualquer um destes"; null = pode omitir.
    expected: Value,
    #[serde(default)]
    source: Option<String>,
}

#[derive(Deserialize)]
struct File {
    items: Vec<Item>,
}

#[derive(Serialize, Deserialize)]
struct Score {
    pass: bool,
    deserializes: bool,
    deserialize_error: Option<String>,
    tool_ok: bool,
    args: Vec<(String, bool)>,
    expects_none: bool,
    /// Fora dos casos que a heurística (`intent.rs`) já resolve.
    hard: bool,
}

pub struct Tools {
    conv: ConversationMeta,
    items: Vec<Item>,
}

impl Tools {
    pub fn load(conv: ConversationMeta) -> Result<Self> {
        Ok(Tools {
            conv,
            items: load::<File>("tools.json")?.items,
        })
    }
}

fn alternatives(v: &Value) -> Vec<&Value> {
    match v {
        Value::Array(a) => a.iter().collect(),
        v => vec![v],
    }
}

impl Task for Tools {
    fn name(&self) -> &'static str {
        "tools"
    }
    fn title(&self) -> &'static str {
        "B — seleção de ferramenta"
    }
    fn headline(&self) -> &'static str {
        "exact_in_scope"
    }
    fn variants(&self) -> &'static [&'static str] {
        &["v1", "v2"]
    }
    fn ids(&self) -> Vec<String> {
        self.items.iter().map(|i| i.id.clone()).collect()
    }

    fn build(&self, i: usize, variant: &str) -> Request {
        let c = &self.conv;
        Request {
            messages: system_user(
                contracts::tool_prompt(variant, &c.authors, &c.first_date, &c.last_date, &c.today),
                self.items[i].q.clone(),
            ),
            schema: contracts::tool_schema(&c.authors, variant),
            max_tokens: 120,
        }
    }

    fn score(&self, i: usize, out: Option<&Value>, variant: &str) -> Value {
        let it = &self.items[i];
        let expects_none = alternatives(&it.expected)
            .iter()
            .all(|e| e["tool"] == "none");
        let hard = it.source.as_deref() != Some("intent.rs");
        let parsed = match out {
            Some(o) => {
                S::parse_tool_call(o, &self.conv.authors, contracts::tool_allows_none(variant))
            }
            None => Err("sem JSON".into()),
        };
        let score = match parsed {
            Err(e) => Score {
                pass: false,
                deserializes: false,
                deserialize_error: Some(e),
                tool_ok: false,
                args: vec![],
                expects_none,
                hard,
            },
            Ok(sel) => {
                let m = S::best_tool_match(&it.expected, &S::selected_value(&sel));
                Score {
                    pass: m.exact,
                    deserializes: true,
                    deserialize_error: None,
                    tool_ok: m.tool_ok,
                    args: m.args,
                    expects_none,
                    hard,
                }
            }
        };
        serde_json::to_value(score).unwrap()
    }

    fn gold(&self, i: usize, variant: &str) -> Value {
        let e = alternatives(&self.items[i].expected)[0];
        if e["tool"] == "none" && !contracts::tool_allows_none(variant) {
            return json!({"tool": "none", "args": {}}); // impossível no schema v1: conta como erro
        }
        let mut args = Map::new();
        for (k, v) in e["args"].as_object().unwrap() {
            let v = match v {
                Value::Array(a) => a.iter().find(|x| !x.is_null()).cloned(),
                v => Some(v.clone()),
            };
            if let Some(v) = v {
                args.insert(k.clone(), v);
            }
        }
        // Passa pelo mesmo caminho da saída do modelo, para o gabarito também ser uma ToolCall válida.
        let raw = json!({"tool": e["tool"], "args": args});
        match S::parse_tool_call(
            &raw,
            &self.conv.authors,
            contracts::tool_allows_none(variant),
        ) {
            Ok(Selected::Tool(c)) => serde_json::to_value(c).unwrap(),
            _ => raw,
        }
    }

    fn aggregate(&self, rows: &[Row]) -> Value {
        let s = scores::<Score>(rows);
        let arg_rate = |keys: &[&str]| {
            S::rate(s.iter().flat_map(|(_, x)| {
                x.args
                    .iter()
                    .filter(|(k, _)| keys.contains(&k.as_str()))
                    .map(|(_, ok)| *ok)
            }))
        };
        let in_scope = || s.iter().filter(|(_, x)| !x.expects_none);
        json!({
            "exact": S::rate(s.iter().map(|(_, x)| x.pass)),
            "exact_in_scope": S::rate(in_scope().map(|(_, x)| x.pass)),
            "exact_hard": S::rate(in_scope().filter(|(_, x)| x.hard).map(|(_, x)| x.pass)),
            "tool_accuracy": S::rate(in_scope().map(|(_, x)| x.tool_ok)),
            "deserializes": S::rate(s.iter().map(|(_, x)| x.deserializes)),
            "none_accuracy": S::rate(s.iter().filter(|(_, x)| x.expects_none).map(|(_, x)| x.pass)),
            "word_acc": arg_rate(&["word"]),
            "author_acc": arg_rate(&["author"]),
            "date_acc": arg_rate(&["from", "to"]),
            "granularity_acc": arg_rate(&["granularity"]),
            "failed_ids": failed_ids(rows),
        })
    }
}

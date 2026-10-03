//! A — roteador: classifica a pergunta em quantitative | retrieval | global.
//! A heurística real do app (`core_chat::router::route`) marca o que ela já resolve: no app, o
//! LLM só recebe as perguntas `Undecided`.

use super::{Request, Row, Task, failed_ids, load, scores, system_user};
use crate::contracts;
use crate::scoring as S;
use anyhow::Result;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::collections::BTreeMap;

#[derive(Deserialize)]
struct Item {
    id: String,
    q: String,
    /// Uma rota ou uma lista de rotas aceitas.
    expected: Value,
}

impl Item {
    fn accepted(&self) -> Vec<String> {
        match &self.expected {
            Value::Array(a) => a
                .iter()
                .filter_map(|x| x.as_str().map(String::from))
                .collect(),
            v => vec![v.as_str().unwrap_or_default().to_string()],
        }
    }
}

#[derive(Serialize, Deserialize)]
struct Score {
    pass: bool,
    got: Option<String>,
    expected: String,
    heuristic: String,
    heuristic_correct: bool,
}

pub struct Router {
    items: Vec<Item>,
}

impl Router {
    pub fn load() -> Result<Self> {
        Ok(Router {
            items: load("router.json")?,
        })
    }
}

fn heuristic(q: &str) -> String {
    serde_json::to_value(core_chat::router::route(q))
        .unwrap()
        .as_str()
        .unwrap()
        .to_string()
}

impl Task for Router {
    fn name(&self) -> &'static str {
        "router"
    }
    fn title(&self) -> &'static str {
        "A — roteador"
    }
    fn headline(&self) -> &'static str {
        "accuracy"
    }
    fn variants(&self) -> &'static [&'static str] {
        &["v1", "v2"]
    }
    fn ids(&self) -> Vec<String> {
        self.items.iter().map(|i| i.id.clone()).collect()
    }

    fn build(&self, i: usize, variant: &str) -> Request {
        Request {
            messages: system_user(contracts::router_prompt(variant), self.items[i].q.clone()),
            schema: contracts::router_schema(),
            max_tokens: 16,
        }
    }

    fn score(&self, i: usize, out: Option<&Value>, _: &str) -> Value {
        let it = &self.items[i];
        let accepted = it.accepted();
        let got = out.and_then(|o| o["type"].as_str()).map(String::from);
        let h = heuristic(&it.q);
        serde_json::to_value(Score {
            pass: got.as_ref().is_some_and(|g| accepted.contains(g)),
            got,
            expected: accepted[0].clone(),
            heuristic_correct: accepted.contains(&h),
            heuristic: h,
        })
        .unwrap()
    }

    fn gold(&self, i: usize, _: &str) -> Value {
        json!({"type": self.items[i].accepted()[0]})
    }

    fn aggregate(&self, rows: &[Row]) -> Value {
        let s = scores::<Score>(rows);
        let mut errors: BTreeMap<String, usize> = BTreeMap::new();
        for (_, x) in s.iter().filter(|(_, x)| !x.pass) {
            *errors
                .entry(format!(
                    "{}→{}",
                    x.expected,
                    x.got.as_deref().unwrap_or("?")
                ))
                .or_default() += 1;
        }
        json!({
            "accuracy": S::rate(s.iter().map(|(_, x)| x.pass)),
            "accuracy_undecided": S::rate(s.iter().filter(|(_, x)| x.heuristic == "undecided").map(|(_, x)| x.pass)),
            "n_undecided": s.iter().filter(|(_, x)| x.heuristic == "undecided").count(),
            "accuracy_where_heuristic_wrong": S::rate(s.iter().filter(|(_, x)| !x.heuristic_correct).map(|(_, x)| x.pass)),
            "heuristic_accuracy": S::rate(s.iter().map(|(_, x)| x.heuristic_correct)),
            "errors": errors,
            "failed_ids": failed_ids(rows),
        })
    }
}

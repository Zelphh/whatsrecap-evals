//! D — pedidos de desculpa numa janela de mensagens: precisão/recall por `msg_id`, direção e `sincere`.

use super::{
    ConversationMeta, Msg, Request, Row, Task, failed_ids, fmt_window, load, scores, system_user,
};
use crate::contracts;
use crate::scoring as S;
use anyhow::Result;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::collections::BTreeMap;

#[derive(Debug, Clone, Serialize, Deserialize)]
struct Apology {
    msg_id: i64,
    from: String,
    to: String,
    sincere: bool,
}

#[derive(Deserialize)]
struct Item {
    id: String,
    messages: Vec<Msg>,
    gold: Vec<Apology>,
    /// Ids cujo `sincere` é discutível: contam para P/R, não para a acurácia de sincere.
    #[serde(default)]
    ambiguous_sincere: Vec<i64>,
}

#[derive(Deserialize)]
struct File {
    items: Vec<Item>,
}

#[derive(Serialize, Deserialize)]
struct Score {
    pass: bool,
    tp: usize,
    fp: usize,
    fn_: usize,
    invalid_ids: usize,
    direction_ok: Vec<bool>,
    sincere_ok: Vec<bool>,
    negative_window: bool,
}

pub struct Apologies {
    conv: ConversationMeta,
    items: Vec<Item>,
}

impl Apologies {
    pub fn load(conv: ConversationMeta) -> Result<Self> {
        Ok(Apologies {
            conv,
            items: load::<File>("apologies.json")?.items,
        })
    }
}

impl Task for Apologies {
    fn name(&self) -> &'static str {
        "apologies"
    }
    fn title(&self) -> &'static str {
        "D — desculpas"
    }
    fn headline(&self) -> &'static str {
        "f1"
    }
    fn ids(&self) -> Vec<String> {
        self.items.iter().map(|i| i.id.clone()).collect()
    }

    fn build(&self, i: usize, variant: &str) -> Request {
        let a = &self.conv.authors;
        Request {
            messages: system_user(
                contracts::apologies_prompt(variant, &a[0], &a[1]),
                fmt_window(&self.items[i].messages, None),
            ),
            schema: contracts::apologies_schema(a),
            max_tokens: 500,
        }
    }

    fn score(&self, i: usize, out: Option<&Value>, _: &str) -> Value {
        let it = &self.items[i];
        let preds: Vec<Apology> = out
            .and_then(|o| serde_json::from_value(o["apologies"].clone()).ok())
            .unwrap_or_default();
        let mut by_id: BTreeMap<i64, Apology> = BTreeMap::new();
        for p in preds {
            by_id.entry(p.msg_id).or_insert(p); // duplicadas contam uma vez
        }
        let gold: BTreeMap<i64, &Apology> = it.gold.iter().map(|g| (g.msg_id, g)).collect();
        let tp: Vec<i64> = by_id
            .keys()
            .copied()
            .filter(|k| gold.contains_key(k))
            .collect();
        let fp = by_id.len() - tp.len();
        let fn_ = gold.len() - tp.len();
        serde_json::to_value(Score {
            pass: fp == 0 && fn_ == 0,
            tp: tp.len(),
            fp,
            fn_,
            invalid_ids: by_id
                .keys()
                .filter(|k| !it.messages.iter().any(|m| m.0 == **k))
                .count(),
            direction_ok: tp
                .iter()
                .map(|k| by_id[k].from == gold[k].from && by_id[k].to == gold[k].to)
                .collect(),
            sincere_ok: tp
                .iter()
                .filter(|k| !it.ambiguous_sincere.contains(k))
                .map(|k| by_id[k].sincere == gold[k].sincere)
                .collect(),
            negative_window: gold.is_empty(),
        })
        .unwrap()
    }

    fn gold(&self, i: usize, _: &str) -> Value {
        json!({"apologies": self.items[i].gold})
    }

    fn aggregate(&self, rows: &[Row]) -> Value {
        let s = scores::<Score>(rows);
        let sum = |f: fn(&Score) -> usize| s.iter().map(|(_, x)| f(x)).sum::<usize>();
        let mut out = S::prf(sum(|x| x.tp), sum(|x| x.fp), sum(|x| x.fn_));
        let o = out.as_object_mut().unwrap();
        o.insert(
            "sincere_acc".into(),
            json!(S::rate(s.iter().flat_map(|(_, x)| x.sincere_ok.clone()))),
        );
        o.insert(
            "direction_acc".into(),
            json!(S::rate(s.iter().flat_map(|(_, x)| x.direction_ok.clone()))),
        );
        o.insert(
            "window_exact".into(),
            json!(S::rate(s.iter().map(|(_, x)| x.pass))),
        );
        o.insert(
            "fp_on_negative_windows".into(),
            json!(
                s.iter()
                    .filter(|(_, x)| x.negative_window)
                    .map(|(_, x)| x.fp)
                    .sum::<usize>()
            ),
        );
        o.insert("invalid_ids".into(), json!(sum(|x| x.invalid_ids)));
        o.insert("failed_ids".into(), json!(failed_ids(rows)));
        out
    }
}

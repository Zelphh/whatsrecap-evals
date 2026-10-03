//! E — brigas numa janela: casamento por IoU de intervalo de ids, erro de borda, intensidade,
//! `resolved`, `need_context` e falsos positivos em brincadeira.

use super::{
    ConversationMeta, Msg, Request, Row, Task, failed_ids, fmt_window, load, scores, system_user,
};
use crate::contracts;
use crate::scoring as S;
use anyhow::Result;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

pub const IOU_THRESHOLD: f64 = 0.5;

#[derive(Debug, Clone, Serialize, Deserialize)]
struct Conflict {
    start_id: i64,
    end_id: i64,
    intensity: i64,
    resolved: bool,
}

#[derive(Deserialize)]
struct Item {
    id: String,
    messages: Vec<Msg>,
    gold: Vec<Conflict>,
    /// Limítrofes: não contam como acerto nem como falso positivo.
    #[serde(default)]
    optional: Vec<Conflict>,
    need_context: Option<String>,
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
    invalid: usize,
    iou: Vec<f64>,
    start_err: Vec<f64>,
    end_err: Vec<f64>,
    intensity_err: Vec<f64>,
    resolved_ok: Vec<bool>,
    need_context_ok: bool,
    need_expected: bool,
    negative_window: bool,
    summaries: Vec<String>,
}

pub struct Conflicts {
    conv: ConversationMeta,
    items: Vec<Item>,
}

impl Conflicts {
    pub fn load(conv: ConversationMeta) -> Result<Self> {
        Ok(Conflicts {
            conv,
            items: load::<File>("conflicts.json")?.items,
        })
    }
}

impl Task for Conflicts {
    fn name(&self) -> &'static str {
        "conflicts"
    }
    fn title(&self) -> &'static str {
        "E — brigas"
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
                contracts::conflicts_prompt(variant, &a[0], &a[1]),
                fmt_window(&self.items[i].messages, None),
            ),
            schema: contracts::conflicts_schema(a),
            max_tokens: 700,
        }
    }

    fn score(&self, i: usize, out: Option<&Value>, _: &str) -> Value {
        let it = &self.items[i];
        let preds: Vec<Value> = out
            .and_then(|o| o["conflicts"].as_array().cloned())
            .unwrap_or_default();
        let need = out
            .and_then(|o| o["need_context"]["direction"].as_str())
            .map(String::from);
        let ids: Vec<i64> = it.messages.iter().map(|m| m.0).collect();
        let mut valid: Vec<(i64, i64, &Value)> = preds
            .iter()
            .filter_map(|p| Some((p["start_id"].as_i64()?, p["end_id"].as_i64()?, p)))
            .filter(|(s, e, _)| ids.contains(s) && ids.contains(e) && s <= e)
            .collect();
        valid.sort_by_key(|v| v.0);
        let invalid = preds.len() - valid.len();

        let mut used = vec![false; it.gold.len()];
        let mut matches: Vec<(&Value, &Conflict, f64)> = vec![];
        let mut fp = invalid;
        for (s, e, p) in &valid {
            let best = it
                .gold
                .iter()
                .enumerate()
                .filter(|(gi, _)| !used[*gi])
                .map(|(gi, g)| (S::iou((*s, *e), (g.start_id, g.end_id)), gi))
                .max_by(|a, b| a.0.total_cmp(&b.0));
            match best {
                Some((iou, gi)) if iou >= IOU_THRESHOLD => {
                    used[gi] = true;
                    matches.push((p, &it.gold[gi], iou));
                }
                _ if it
                    .optional
                    .iter()
                    .any(|o| S::iou((*s, *e), (o.start_id, o.end_id)) >= IOU_THRESHOLD) => {}
                _ => fp += 1,
            }
        }
        let fn_ = it.gold.len() - matches.len();
        let need_ok = need == it.need_context;
        serde_json::to_value(Score {
            pass: fp == 0 && fn_ == 0 && need_ok,
            tp: matches.len(),
            fp,
            fn_,
            invalid,
            iou: matches.iter().map(|m| S::round(m.2, 3)).collect(),
            start_err: matches
                .iter()
                .map(|(p, g, _)| (p["start_id"].as_i64().unwrap() - g.start_id).abs() as f64)
                .collect(),
            end_err: matches
                .iter()
                .map(|(p, g, _)| (p["end_id"].as_i64().unwrap() - g.end_id).abs() as f64)
                .collect(),
            intensity_err: matches
                .iter()
                .map(|(p, g, _)| (p["intensity"].as_i64().unwrap_or(0) - g.intensity).abs() as f64)
                .collect(),
            resolved_ok: matches
                .iter()
                .map(|(p, g, _)| p["resolved"].as_bool() == Some(g.resolved))
                .collect(),
            need_context_ok: need_ok,
            need_expected: it.need_context.is_some(),
            negative_window: it.gold.is_empty(),
            summaries: valid
                .iter()
                .map(|(_, _, p)| p["summary"].as_str().unwrap_or_default().to_string())
                .collect(),
        })
        .unwrap()
    }

    fn gold(&self, i: usize, _: &str) -> Value {
        let it = &self.items[i];
        let conflicts: Vec<Value> = it
            .gold
            .iter()
            .map(|g| json!({"start_id": g.start_id, "end_id": g.end_id, "participants": self.conv.authors,
                            "intensity": g.intensity, "resolved": g.resolved, "summary": "gabarito"}))
            .collect();
        json!({"conflicts": conflicts, "need_context": {"direction": it.need_context, "reason": ""}})
    }

    fn aggregate(&self, rows: &[Row]) -> Value {
        let s = scores::<Score>(rows);
        let sum = |f: fn(&Score) -> usize| s.iter().map(|(_, x)| f(x)).sum::<usize>();
        let flat = |f: fn(&Score) -> &Vec<f64>| S::mean(s.iter().flat_map(|(_, x)| f(x).clone()));
        let mut out = S::prf(sum(|x| x.tp), sum(|x| x.fp), sum(|x| x.fn_));
        let o = out.as_object_mut().unwrap();
        o.insert("mean_iou".into(), json!(flat(|x| &x.iou)));
        o.insert("start_err".into(), json!(flat(|x| &x.start_err)));
        o.insert("end_err".into(), json!(flat(|x| &x.end_err)));
        o.insert("intensity_mae".into(), json!(flat(|x| &x.intensity_err)));
        o.insert(
            "resolved_acc".into(),
            json!(S::rate(s.iter().flat_map(|(_, x)| x.resolved_ok.clone()))),
        );
        o.insert(
            "need_context_acc".into(),
            json!(S::rate(s.iter().map(|(_, x)| x.need_context_ok))),
        );
        o.insert(
            "need_context_recall".into(),
            json!(S::rate(
                s.iter()
                    .filter(|(_, x)| x.need_expected)
                    .map(|(_, x)| x.need_context_ok)
            )),
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
        o.insert("invalid_intervals".into(), json!(sum(|x| x.invalid)));
        o.insert(
            "window_exact".into(),
            json!(S::rate(s.iter().map(|(_, x)| x.pass))),
        );
        o.insert("failed_ids".into(), json!(failed_ids(rows)));
        out
    }
}

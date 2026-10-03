//! Tarefas A–F. Cada tarefa monta a requisição de um item, pontua a saída e agrega.
//!
//! `gold` devolve a saída perfeita de um item: o autoteste (`evals selftest`) passa o gabarito pela
//! pontuação e exige nota máxima, o que valida datasets e métricas sem modelo.

mod answer;
mod apologies;
mod conflicts;
mod rag;
mod router;
mod tools;

pub use rag::{Rag, Window};

use crate::config::root;
use anyhow::{Context, Result};
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

/// Mensagem num dataset: `[id, autor, texto]`.
pub type Msg = (i64, String, String);

#[derive(Debug, Clone, Deserialize)]
pub struct ConversationMeta {
    pub authors: Vec<String>,
    pub first_date: String,
    pub last_date: String,
    pub today: String,
}

pub fn load<T: DeserializeOwned>(name: &str) -> Result<T> {
    let path = root().join("datasets").join(name);
    let text =
        std::fs::read_to_string(&path).with_context(|| format!("lendo {}", path.display()))?;
    serde_json::from_str(&text).with_context(|| format!("interpretando {}", path.display()))
}

/// `[id] Pessoa: texto`, uma mensagem por linha (com a data quando dada).
pub fn fmt_window(messages: &[Msg], date: Option<&str>) -> String {
    let prefix = date.map(|d| format!("{d} ")).unwrap_or_default();
    messages
        .iter()
        .map(|(i, a, t)| format!("[{i}] {prefix}{a}: {t}"))
        .collect::<Vec<_>>()
        .join("\n")
}

pub fn system_user(system: String, user: String) -> Vec<Value> {
    vec![
        json!({"role": "system", "content": system}),
        json!({"role": "user", "content": user}),
    ]
}

pub struct Request {
    pub messages: Vec<Value>,
    pub schema: Value,
    pub max_tokens: u32,
}

/// Uma chamada ao modelo, como fica gravada em `items.jsonl`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Row {
    pub task: String,
    pub variant: String,
    pub item_id: String,
    pub rep: u32,
    pub output: Option<Value>,
    pub raw: Option<String>,
    pub error: Option<String>,
    pub latency_ms: f64,
    pub timings: Value,
    /// Métricas do item (formato próprio de cada tarefa) + `pass`.
    pub score: Value,
}

impl Row {
    pub fn passed(&self) -> bool {
        self.score["pass"].as_bool().unwrap_or(false)
    }
}

pub trait Task {
    fn name(&self) -> &'static str;
    fn title(&self) -> &'static str;
    /// Métrica principal (resumo e relatório).
    fn headline(&self) -> &'static str;
    fn variants(&self) -> &'static [&'static str] {
        &["v1"]
    }
    fn ids(&self) -> Vec<String>;
    fn build(&self, i: usize, variant: &str) -> Request;
    /// Pontuação do item; sempre tem `pass`.
    fn score(&self, i: usize, out: Option<&Value>, variant: &str) -> Value;
    fn gold(&self, i: usize, variant: &str) -> Value;
    fn aggregate(&self, rows: &[Row]) -> Value;
}

/// Desserializa o `score` das linhas no tipo de pontuação da tarefa.
pub fn scores<T: DeserializeOwned>(rows: &[Row]) -> Vec<(String, T)> {
    rows.iter()
        .filter_map(|r| {
            serde_json::from_value(r.score.clone())
                .ok()
                .map(|s| (r.item_id.clone(), s))
        })
        .collect()
}

pub fn failed_ids(rows: &[Row]) -> Vec<String> {
    let mut ids: Vec<String> = rows
        .iter()
        .filter(|r| !r.passed())
        .map(|r| r.item_id.clone())
        .collect();
    ids.dedup();
    ids
}

pub fn all() -> Result<Vec<Box<dyn Task>>> {
    let conv: ConversationMeta = load("conversation.json")?;
    Ok(vec![
        Box::new(router::Router::load()?),
        Box::new(tools::Tools::load(conv.clone())?),
        Box::new(answer::Answer::load(conv.clone())?),
        Box::new(apologies::Apologies::load(conv.clone())?),
        Box::new(conflicts::Conflicts::load(conv.clone())?),
        Box::new(rag::Rag::load(conv)?),
    ])
}

/// Gerador pseudoaleatório determinístico (SplitMix64), para contextos e ruído reprodutíveis.
pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Self {
        Rng(seed)
    }
    pub fn from_str(s: &str) -> Self {
        // FNV-1a
        Rng(s.bytes().fold(0xcbf29ce484222325u64, |h, b| {
            (h ^ b as u64).wrapping_mul(0x100000001b3)
        }))
    }
    pub fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e3779b97f4a7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
        z ^ (z >> 31)
    }
    pub fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }
    pub fn unit(&mut self) -> f64 {
        (self.next() >> 11) as f64 / (1u64 << 53) as f64
    }
    pub fn shuffle<T>(&mut self, v: &mut [T]) {
        for i in (1..v.len()).rev() {
            v.swap(i, self.below(i + 1));
        }
    }
}

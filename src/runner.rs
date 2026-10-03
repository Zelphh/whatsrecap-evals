//! Executa tarefas contra um `llama-server` e grava cada execução em
//! `results/runs/<data>_<tipo>_<modelo>_<perfil>_<host>/` (meta.json, items.jsonl, summary.json).

use crate::client::Client;
use crate::config::{Config, Model, Profile};
use crate::hardware::{self, Fingerprint};
use crate::scoring as S;
use crate::server::{Server, ServerInfo, llama_version};
use crate::tasks::{Row, Task};
use anyhow::Result;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value, json};
use std::collections::{BTreeMap, BTreeSet};
use std::io::Write;
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Meta {
    pub kind: String,
    pub started: String,
    pub finished: Option<String>,
    pub model: Model,
    pub model_path: String,
    pub profile: Profile,
    pub hardware: Fingerprint,
    pub llama_cpp: String,
    pub temperature: f64,
    pub seed: u64,
    /// Específico de cada tipo de execução (repetições, variante, tarefas…).
    pub params: Value,
    pub server: Option<Value>,
}

impl Meta {
    pub fn new(cfg: &Config, kind: &str, model: &Model, profile: &Profile, params: Value) -> Self {
        Meta {
            kind: kind.into(),
            started: now(),
            finished: None,
            model: model.clone(),
            model_path: cfg.path_of(model).display().to_string(),
            profile: profile.clone(),
            hardware: hardware::fingerprint(),
            llama_cpp: llama_version(cfg),
            temperature: cfg.temperature,
            seed: cfg.seed,
            params,
            server: None,
        }
    }

    pub fn set_server(&mut self, info: Option<ServerInfo>, external: Option<&str>) {
        self.server = match (info, external) {
            (Some(i), _) => Some(serde_json::to_value(i).unwrap()),
            (None, Some(u)) => Some(json!({"external_url": u})),
            _ => None,
        };
    }
}

pub fn now() -> String {
    chrono::Local::now().format("%Y-%m-%dT%H:%M:%S").to_string()
}

pub fn run_dir(cfg: &Config, kind: &str, model: &str, profile: &str) -> Result<PathBuf> {
    let stamp = chrono::Local::now().format("%Y%m%d-%H%M%S");
    let host: String = hardware::fingerprint()
        .host
        .chars()
        .filter(|c| c.is_alphanumeric() || *c == '-')
        .take(20)
        .collect();
    let d = cfg
        .results_dir
        .join("runs")
        .join(format!("{stamp}_{kind}_{model}_{profile}_{host}"));
    std::fs::create_dir_all(&d)?;
    Ok(d)
}

pub fn write_json(path: &Path, v: &impl Serialize) -> Result<()> {
    std::fs::write(path, serde_json::to_string_pretty(v)? + "\n")?;
    Ok(())
}

/// Fração de itens em que todas as repetições deram exatamente a mesma saída.
fn consistency(rows: &[Row]) -> Option<f64> {
    let mut by_item: BTreeMap<&str, (usize, BTreeSet<String>)> = BTreeMap::new();
    for r in rows {
        let e = by_item.entry(&r.item_id).or_default();
        e.0 += 1;
        e.1.insert(serde_json::to_string(&r.output).unwrap());
    }
    S::rate(
        by_item
            .values()
            .filter(|(n, _)| *n > 1)
            .map(|(_, outs)| outs.len() == 1),
    )
}

pub fn summarize(task: &dyn Task, variant: &str, rows: &[Row]) -> Value {
    let lat: Vec<f64> = rows.iter().map(|r| r.latency_ms).collect();
    let timing = |k: &str| {
        S::mean(
            rows.iter()
                .filter_map(|r| r.timings.get(k).and_then(Value::as_f64)),
        )
    };
    let mut out = json!({
        "variant": variant,
        "n": rows.len(),
        "json_valid": S::rate(rows.iter().map(|r| r.output.is_some())),
        "errors": rows.iter().filter(|r| r.error.is_some()).count(),
        "latency_ms_p50": S::pct(&lat, 0.5),
        "latency_ms_p95": S::pct(&lat, 0.95),
        "prompt_tps": timing("prompt_per_second"),
        "gen_tps": timing("predicted_per_second"),
        "consistency": consistency(rows),
        "headline": task.headline(),
    });
    if let Value::Object(agg) = task.aggregate(rows) {
        out.as_object_mut().unwrap().extend(agg);
    }
    out
}

pub struct RunOptions<'a> {
    pub variant: Option<&'a str>,
    pub repeats: u32,
    pub limit: Option<usize>,
    pub ids: Option<BTreeSet<String>>,
}

pub fn run_tasks(
    client: &Client,
    tasks: &[Box<dyn Task>],
    opt: &RunOptions,
    out_dir: &Path,
) -> Result<Map<String, Value>> {
    let mut fh = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(out_dir.join("items.jsonl"))?;
    let mut summaries = Map::new();
    for task in tasks {
        let variant = opt
            .variant
            .filter(|v| task.variants().contains(v))
            .unwrap_or(task.variants()[0]);
        let selected: Vec<usize> = task
            .ids()
            .iter()
            .enumerate()
            .filter(|(_, id)| opt.ids.as_ref().is_none_or(|s| s.contains(*id)))
            .map(|(i, _)| i)
            .take(opt.limit.unwrap_or(usize::MAX))
            .collect();
        let ids = task.ids();
        let mut rows = vec![];
        for (n, &i) in selected.iter().enumerate() {
            let req = task.build(i, variant);
            for rep in 0..opt.repeats {
                let res = client.chat(
                    &req.messages,
                    &req.schema,
                    req.max_tokens,
                    client.seed + rep as u64,
                    true,
                );
                let score = task.score(i, res.json.as_ref(), variant);
                let row = Row {
                    task: task.name().into(),
                    variant: variant.into(),
                    item_id: ids[i].clone(),
                    rep,
                    raw: if res.json.is_none() {
                        res.content.clone()
                    } else {
                        None
                    },
                    output: res.json,
                    error: res.error,
                    latency_ms: res.latency_ms,
                    timings: res.timings,
                    score,
                };
                writeln!(fh, "{}", serde_json::to_string(&row)?)?;
                rows.push(row);
            }
            let last = rows.last().unwrap();
            eprint!(
                "\r  {} [{variant}] {}/{} {} {:.0}ms   ",
                task.title(),
                n + 1,
                selected.len(),
                if last.passed() { "✓" } else { "✗" },
                last.latency_ms
            );
        }
        eprintln!();
        let s = summarize(task.as_ref(), variant, &rows);
        eprintln!(
            "  → {} = {}  (p50 {} ms)",
            task.headline(),
            s[task.headline()],
            s["latency_ms_p50"]
        );
        summaries.insert(task.name().into(), s);
    }
    Ok(summaries)
}

#[allow(clippy::too_many_arguments)]
pub fn eval_model(
    cfg: &Config,
    model: &Model,
    profile: &Profile,
    tasks: &[Box<dyn Task>],
    opt: &RunOptions,
    server_url: Option<&str>,
    extra_args: &[String],
) -> Result<PathBuf> {
    let out = run_dir(cfg, "eval", &model.id, &profile.id)?;
    let names: Vec<&str> = tasks.iter().map(|t| t.name()).collect();
    let mut meta = Meta::new(
        cfg,
        "eval",
        model,
        profile,
        json!({"variant": opt.variant, "repeats": opt.repeats, "limit": opt.limit, "tasks": names, "extra_args": extra_args}),
    );
    eprintln!(
        "→ {} · perfil {} · {}",
        model.id,
        profile.id,
        out.file_name().unwrap().to_string_lossy()
    );
    let summaries = match server_url {
        Some(url) => {
            meta.set_server(None, Some(url));
            run_tasks(
                &Client::new(url, cfg.temperature, cfg.seed, model.request.clone()),
                tasks,
                opt,
                &out,
            )?
        }
        None => {
            let srv = Server::start(cfg, model, profile, extra_args, &out, false)?;
            let s = run_tasks(
                &Client::new(&srv.url, cfg.temperature, cfg.seed, model.request.clone()),
                tasks,
                opt,
                &out,
            )?;
            meta.set_server(Some(srv.info()), None);
            s
        }
    };
    meta.finished = Some(now());
    write_json(&out.join("meta.json"), &meta)?;
    write_json(&out.join("summary.json"), &summaries)?;
    Ok(out)
}

//! Junta os `summary.json` de `results/runs/` em tabelas comparativas (`REPORT.md` + `summary.csv`).
//! Execuções fora da máquina alvo aparecem marcadas.

use crate::config::Config;
use crate::runner::Meta;
use crate::tasks::Task;
use anyhow::Result;
use serde_json::Value;
use std::fmt::Write as _;
use std::path::PathBuf;

/// Colunas extras por tarefa, além da métrica principal.
fn columns(task: &str) -> &'static [&'static str] {
    match task {
        "router" => &["accuracy_undecided", "accuracy_where_heuristic_wrong"],
        "tools" => &[
            "exact_hard",
            "tool_accuracy",
            "word_acc",
            "author_acc",
            "date_acc",
            "none_accuracy",
        ],
        "answer" => &["must_ok", "no_invented_numbers", "within_2_sentences"],
        "apologies" => &[
            "precision",
            "recall",
            "sincere_acc",
            "direction_acc",
            "fp_on_negative_windows",
        ],
        "conflicts" => &[
            "precision",
            "recall",
            "mean_iou",
            "intensity_mae",
            "need_context_acc",
            "fp_on_negative_windows",
        ],
        "rag" => &[
            "answer_correct",
            "cites_gold",
            "citations_valid",
            "refusal_acc",
        ],
        _ => &[],
    }
}

struct Run {
    dir: String,
    meta: Meta,
    summary: Value,
}

fn load_runs(cfg: &Config) -> Vec<Run> {
    let Ok(rd) = std::fs::read_dir(cfg.results_dir.join("runs")) else {
        return vec![];
    };
    let mut dirs: Vec<PathBuf> = rd.filter_map(Result::ok).map(|e| e.path()).collect();
    dirs.sort();
    dirs.into_iter()
        .filter_map(|d| {
            // Execuções interrompidas não têm meta/summary e ficam de fora.
            let meta =
                serde_json::from_str(&std::fs::read_to_string(d.join("meta.json")).ok()?).ok()?;
            let summary =
                serde_json::from_str(&std::fs::read_to_string(d.join("summary.json")).ok()?)
                    .ok()?;
            Some(Run {
                dir: d.file_name()?.to_string_lossy().into(),
                meta,
                summary,
            })
        })
        .collect()
}

fn fmt(v: &Value) -> String {
    match v {
        Value::Null => "—".into(),
        Value::Number(n) => match n.as_f64() {
            Some(x) if n.is_f64() && x.abs() <= 1.0 => format!("{x:.2}"),
            Some(x) if n.is_f64() => format!("{x:.1}"),
            _ => n.to_string(),
        },
        Value::String(s) => s.clone(),
        v => v.to_string(),
    }
}

fn hw(m: &Meta) -> String {
    if m.hardware.is_target {
        "**alvo**".into()
    } else {
        format!(
            "dev ({})",
            m.hardware
                .gpus
                .first()
                .map(|g| g.chars().take(24).collect::<String>())
                .unwrap_or("?".into())
        )
    }
}

fn csv_cell(v: &Value) -> String {
    let s = match v {
        Value::Null => String::new(),
        Value::String(s) => s.clone(),
        v => v.to_string(),
    };
    if s.contains([',', '"', '\n']) {
        format!("\"{}\"", s.replace('"', "\"\""))
    } else {
        s
    }
}

pub fn write(cfg: &Config, tasks: &[Box<dyn Task>]) -> Result<PathBuf> {
    let runs = load_runs(cfg);
    let mut md = String::from(
        "# Resultados do benchmark\n\nGerado por `evals report`. Cada linha é uma execução em `results/runs/`. \
Velocidade e VRAM só valem nas linhas marcadas **alvo** (RX 7600).\n\n",
    );
    let mut csv = String::from("task,run,model,quant,profile,target_hw,variant,metric,value\n");

    for task in tasks {
        let mut rows: Vec<(&Run, &Value)> = runs
            .iter()
            .filter(|r| r.meta.kind == "eval")
            .filter_map(|r| r.summary.get(task.name()).map(|s| (r, s)))
            .collect();
        if rows.is_empty() {
            continue;
        }
        let head = task.headline();
        let mut cols = vec![head];
        cols.extend(columns(task.name()));
        cols.extend([
            "json_valid",
            "consistency",
            "latency_ms_p50",
            "latency_ms_p95",
        ]);
        rows.sort_by(|a, b| {
            b.1[head]
                .as_f64()
                .unwrap_or(0.0)
                .total_cmp(&a.1[head].as_f64().unwrap_or(0.0))
        });
        writeln!(md, "## {}\n", task.title())?;
        writeln!(
            md,
            "| modelo | quant | perfil | hw | variante | {} | execução |",
            cols.join(" | ")
        )?;
        writeln!(md, "|{}", "---|".repeat(cols.len() + 6))?;
        for (r, s) in rows {
            let m = &r.meta;
            let cells: Vec<String> = cols.iter().map(|c| fmt(&s[*c])).collect();
            writeln!(
                md,
                "| {} | {} | {} | {} | {} | {} | `{}` |",
                m.model.id,
                m.model.quant,
                m.profile.id,
                hw(m),
                fmt(&s["variant"]),
                cells.join(" | "),
                r.dir
            )?;
            for c in &cols {
                writeln!(
                    csv,
                    "{},{},{},{},{},{},{},{},{}",
                    task.name(),
                    r.dir,
                    m.model.id,
                    m.model.quant,
                    m.profile.id,
                    m.hardware.is_target,
                    csv_cell(&s["variant"]),
                    c,
                    csv_cell(&s[*c])
                )?;
            }
        }
        md.push('\n');
    }

    let benches: Vec<&Run> = runs.iter().filter(|r| r.meta.kind == "bench").collect();
    if !benches.is_empty() {
        md.push_str(
            "## Throughput e VRAM\n\n| modelo | perfil | hw | n_ctx | pp tok/s (chunk) | tg tok/s | s/chunk | modo completo (h) | modo rápido (min) | pico VRAM (MiB) | buffers llama.cpp (MiB) | llama-bench | execução |\n|---|---|---|---|---|---|---|---|---|---|---|---|---|\n",
        );
        for r in benches {
            let b = &r.summary["bench"];
            let e = &b["extrapolation"];
            let buffers: f64 = b["vram"]["buffers_mib"]
                .as_object()
                .map(|o| o.values().filter_map(Value::as_f64).sum())
                .unwrap_or(0.0);
            let lb: Vec<String> = b["llama_bench"]
                .as_array()
                .map(|a| {
                    a.iter()
                        .map(|x| match x["n_prompt"].as_u64() {
                            Some(p) if p > 0 => format!("pp{p}={}", fmt(&x["tps"])),
                            _ => format!("tg{}={}", x["n_gen"], fmt(&x["tps"])),
                        })
                        .collect()
                })
                .unwrap_or_default();
            writeln!(
                md,
                "| {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {:.0} | {} | `{}` |",
                r.meta.model.id,
                r.meta.profile.id,
                hw(&r.meta),
                fmt(&b["chunk"]["n_ctx"]),
                fmt(&b["chunk"]["prompt_tps"]),
                fmt(&b["chunk"]["gen_tps"]),
                fmt(&e["secs_per_chunk"]),
                fmt(&e["full_mode_hours"]),
                fmt(&e["fast_mode_minutes"]),
                fmt(&b["vram"]["peak_used_mib"]),
                buffers,
                lb.join(", "),
                r.dir
            )?;
        }
        md.push('\n');
    }

    let rets: Vec<&Run> = runs.iter().filter(|r| r.meta.kind == "retrieval").collect();
    if !rets.is_empty() {
        md.push_str(
            "## Embeddings (chat tipo 2)\n\n| embedder | hw | modo | recall@1 | recall@3 | recall@5 | recall@10 | MRR | indexação tok/s | indexação extrapolada (min) | execução |\n|---|---|---|---|---|---|---|---|---|---|---|\n",
        );
        for r in rets {
            let s = &r.summary["retrieval"];
            for mode in ["vector", "fts", "hybrid"] {
                let x = &s[mode];
                let cells: Vec<String> = ["recall@1", "recall@3", "recall@5", "recall@10", "mrr"]
                    .iter()
                    .map(|k| fmt(&x[*k]))
                    .collect();
                writeln!(
                    md,
                    "| {} | {} | {mode} | {} | {} | {} | `{}` |",
                    r.meta.model.id,
                    hw(&r.meta),
                    cells.join(" | "),
                    fmt(&s["index_tokens_per_sec"]),
                    fmt(&s["extrapolated_index_minutes"]),
                    r.dir
                )?;
            }
        }
        md.push('\n');
    }

    std::fs::create_dir_all(&cfg.results_dir)?;
    let path = cfg.results_dir.join("REPORT.md");
    std::fs::write(&path, md)?;
    std::fs::write(cfg.results_dir.join("summary.csv"), csv)?;
    Ok(path)
}

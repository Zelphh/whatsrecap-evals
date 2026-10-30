//! Benchmark de modelos de linguagem locais para o WhatsRecap.
//!
//! Sobe um `llama-server` (llama.cpp) por modelo, roda as tarefas A–F com saída estruturada e
//! grava os resultados com o hardware em que rodaram. Veja `evals --help` e o README.

mod bench;
mod client;
mod config;
mod contracts;
mod fillers;
mod hardware;
mod report;
mod retrieval;
mod runner;
mod scoring;
mod server;
mod tasks;

use anyhow::{Result, bail};
use clap::{Args, Parser, Subcommand};
use config::Config;
use std::io::{Read, Write};
use std::path::PathBuf;
use tasks::Task;

#[derive(Parser)]
#[command(
    name = "evals",
    about = "Benchmark de modelos locais para o WhatsRecap"
)]
struct Cli {
    /// Caminho de models.toml (padrão: o da raiz do projeto).
    #[arg(long, global = true)]
    config: Option<PathBuf>,
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Args)]
struct Common {
    /// Ids de models.toml.
    #[arg(required = true)]
    models: Vec<String>,
    /// Perfil de hardware (models.toml).
    #[arg(long, short)]
    profile: Option<String>,
    /// Flags extras do llama-server, ex.: "-c 16384 -ctk q8_0".
    #[arg(long, allow_hyphen_values = true)]
    server_args: Option<String>,
}

impl Common {
    fn extra(&self) -> Vec<String> {
        self.server_args
            .as_deref()
            .unwrap_or("")
            .split_whitespace()
            .map(String::from)
            .collect()
    }
}

#[derive(Subcommand)]
enum Cmd {
    /// Mostra o hardware detectado.
    Hw,
    /// Lista modelos, embedders, perfis e tarefas.
    List,
    /// Baixa os GGUF do Hugging Face para models/.
    Download {
        models: Vec<String>,
        #[arg(long)]
        all: bool,
    },
    /// Valida datasets e métricas sem modelo: o gabarito tem de tirar nota máxima.
    Selftest,
    /// Mostra o prompt e o schema de um item.
    Prompt {
        task: String,
        item: Option<String>,
        #[arg(long, short)]
        variant: Option<String>,
    },
    /// Roda as tarefas de qualidade (A–F).
    Run {
        #[command(flatten)]
        common: Common,
        /// router,tools,answer,apologies,conflicts,rag ou all.
        #[arg(long, short, default_value = "all")]
        tasks: String,
        /// Variante de prompt (v1 = o que o app usa hoje).
        #[arg(long, short)]
        variant: Option<String>,
        /// Repetições por item (mede consistência).
        #[arg(long, short = 'n', default_value_t = 1)]
        repeats: u32,
        #[arg(long)]
        limit: Option<usize>,
        /// Só estes itens, separados por vírgula.
        #[arg(long)]
        ids: Option<String>,
        /// Usa um llama-server já rodando em vez de subir um.
        #[arg(long)]
        server_url: Option<String>,
    },
    /// Throughput, VRAM e extrapolação do modo completo.
    Bench {
        #[command(flatten)]
        common: Common,
        #[arg(long, default_value_t = 8192)]
        chunk_tokens: usize,
        #[arg(long, default_value_t = 3)]
        reps: u32,
        #[arg(long, default_value_t = 190_000)]
        total_messages: usize,
        /// Mensagens de sobreposição entre chunks.
        #[arg(long, default_value_t = 25)]
        overlap: usize,
        /// Janelas do pré-filtro por mensagem (modo rápido).
        #[arg(long, default_value_t = 0.01)]
        fast_hit_rate: f64,
        /// Mensagens por janela do pré-filtro.
        #[arg(long, default_value_t = 40)]
        fast_window: usize,
        #[arg(long)]
        skip_llama_bench: bool,
    },
    /// Avalia embedders: vetorial × FTS5 × híbrida (RRF).
    Retrieval {
        #[command(flatten)]
        common: Common,
        /// Janelas de ruído no corpus.
        #[arg(long, default_value_t = 150)]
        fillers: usize,
        #[arg(long, default_value_t = 190_000)]
        total_messages: usize,
        #[arg(long)]
        server_url: Option<String>,
    },
    /// Gera results/REPORT.md e results/summary.csv.
    Report,
}

fn select_tasks(all: Vec<Box<dyn Task>>, spec: &str) -> Result<Vec<Box<dyn Task>>> {
    if spec == "all" {
        return Ok(all);
    }
    let names: Vec<&str> = spec.split(',').collect();
    for n in &names {
        if !all.iter().any(|t| t.name() == *n) {
            bail!(
                "tarefa desconhecida: {n} (opções: {}, all)",
                all.iter().map(|t| t.name()).collect::<Vec<_>>().join(", ")
            );
        }
    }
    Ok(all
        .into_iter()
        .filter(|t| names.contains(&t.name()))
        .collect())
}

fn download(url: &str, dest: &std::path::Path) -> Result<()> {
    std::fs::create_dir_all(dest.parent().unwrap())?;
    let part = dest.with_extension("gguf.part");
    let have = std::fs::metadata(&part).map(|m| m.len()).unwrap_or(0);
    let agent = client::agent(std::time::Duration::from_secs(24 * 3600));
    let mut req = agent.get(url);
    if have > 0 {
        req = req.header("Range", format!("bytes={have}-"));
    }
    let mut resp = req.call()?;
    if !resp.status().is_success() {
        bail!("HTTP {} em {url}", resp.status());
    }
    let resumed = resp.status().as_u16() == 206;
    let total = resp
        .headers()
        .get("content-length")
        .and_then(|v| v.to_str().ok()?.parse::<u64>().ok())
        .unwrap_or(0)
        + if resumed { have } else { 0 };
    let mut file = std::fs::OpenOptions::new()
        .create(true)
        .append(resumed)
        .write(true)
        .truncate(!resumed)
        .open(&part)?;
    let mut done = if resumed { have } else { 0 };
    let mut reader = resp.body_mut().with_config().limit(u64::MAX).reader();
    let mut buf = vec![0u8; 1 << 20];
    loop {
        let n = reader.read(&mut buf)?;
        if n == 0 {
            break;
        }
        file.write_all(&buf[..n])?;
        let before = done;
        done += n as u64;
        if before >> 26 == done >> 26 {
            continue; // progresso a cada 64 MiB
        }
        eprint!(
            "\r  {}: {:.2}/{:.2} GiB",
            dest.file_name().unwrap().to_string_lossy(),
            done as f64 / 1073741824.0,
            total as f64 / 1073741824.0
        );
    }
    eprintln!();
    std::fs::rename(part, dest)?;
    Ok(())
}

fn selftest() -> Result<()> {
    let mut failed = 0;
    for task in tasks::all()? {
        for v in task.variants() {
            let rows: Vec<tasks::Row> = (0..task.ids().len())
                .map(|i| {
                    let req = task.build(i, v); // garante que todos os prompts montam
                    assert!(!req.messages.is_empty());
                    let gold = task.gold(i, v);
                    tasks::Row {
                        task: task.name().into(),
                        variant: v.to_string(),
                        item_id: task.ids()[i].clone(),
                        rep: 0,
                        score: task.score(i, Some(&gold), v),
                        output: Some(gold),
                        raw: None,
                        error: None,
                        latency_ms: 0.0,
                        timings: serde_json::Value::Null,
                    }
                })
                .collect();
            let s = runner::summarize(task.as_ref(), v, &rows);
            let ok = s[task.headline()].as_f64() == Some(1.0);
            failed += !ok as usize;
            println!(
                "  {} {:28} {v}: {} = {}{}",
                if ok { "✓" } else { "✗" },
                task.title(),
                task.headline(),
                s[task.headline()],
                if ok {
                    String::new()
                } else {
                    format!("  falhas: {}", s["failed_ids"])
                }
            );
        }
    }
    if failed > 0 {
        bail!("{failed} verificação(ões) falharam");
    }
    Ok(())
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    let cfg = Config::load(cli.config.as_deref())?;
    let profile_or = |p: &Option<String>, default: &str| -> Result<config::Profile> {
        Ok(cfg.profile(p.as_deref().unwrap_or(default))?.clone())
    };
    match cli.cmd {
        Cmd::Hw => {
            println!(
                "{}",
                serde_json::to_string_pretty(&hardware::fingerprint())?
            );
            println!("VRAM usada: {:?} MiB", hardware::vram_used_mib());
        }
        Cmd::List => {
            println!("Modelos:");
            for m in &cfg.models {
                let ok = if cfg.path_of(m).exists() { "✓" } else { " " };
                println!(
                    "  [{ok}] {:22} {:8} {:>4}B  {}",
                    m.id, m.quant, m.params_b, m.notes
                );
            }
            println!("Embedders:");
            for m in &cfg.embedders {
                let ok = if cfg.path_of(m).exists() { "✓" } else { " " };
                println!("  [{ok}] {:22} {:8}        {}", m.id, m.quant, m.notes);
            }
            println!("Perfis:");
            for p in cfg.profiles.values() {
                println!("  {:12} {}", p.id, p.description);
            }
            println!(
                "Tarefas: {}",
                tasks::all()?
                    .iter()
                    .map(|t| t.name())
                    .collect::<Vec<_>>()
                    .join(", ")
            );
        }
        Cmd::Download { models, all } => {
            let ids: Vec<String> = if all {
                cfg.models
                    .iter()
                    .chain(&cfg.embedders)
                    .map(|m| m.id.clone())
                    .collect()
            } else {
                models
            };
            if ids.is_empty() {
                bail!("informe os ids (veja `evals list`) ou --all");
            }
            for id in ids {
                let m = cfg.model(&id)?;
                let dest = cfg.path_of(m);
                if dest.exists() {
                    eprintln!("  {id}: já existe");
                    continue;
                }
                eprintln!("→ {id} ← {}", m.url());
                download(&m.url(), &dest)?;
            }
        }
        Cmd::Selftest => selftest()?,
        Cmd::Prompt {
            task,
            item,
            variant,
        } => {
            let t = select_tasks(tasks::all()?, &task)?.remove(0);
            let v = variant.as_deref().unwrap_or(t.variants().last().unwrap());
            let i = match item {
                Some(id) => t
                    .ids()
                    .iter()
                    .position(|x| *x == id)
                    .ok_or_else(|| anyhow::anyhow!("item {id} não existe"))?,
                None => 0,
            };
            let req = t.build(i, v);
            for m in &req.messages {
                println!(
                    "--- {} ---\n{}",
                    m["role"].as_str().unwrap_or("?"),
                    m["content"].as_str().unwrap_or("")
                );
            }
            println!("--- schema ---\n{}", req.schema);
        }
        Cmd::Run {
            common,
            tasks: spec,
            variant,
            repeats,
            limit,
            ids,
            server_url,
        } => {
            let profile = profile_or(&common.profile, "dev")?;
            let selected = select_tasks(tasks::all()?, &spec)?;
            let opt = runner::RunOptions {
                variant: variant.as_deref(),
                repeats,
                limit,
                ids: ids.map(|s| s.split(',').map(String::from).collect()),
            };
            for id in &common.models {
                let out = runner::eval_model(
                    &cfg,
                    cfg.model(id)?,
                    &profile,
                    &selected,
                    &opt,
                    server_url.as_deref(),
                    &common.extra(),
                )?;
                eprintln!("  resultados em {}", out.display());
            }
            report::write(&cfg, &tasks::all()?)?;
        }
        Cmd::Bench {
            common,
            chunk_tokens,
            reps,
            total_messages,
            overlap,
            fast_hit_rate,
            fast_window,
            skip_llama_bench,
        } => {
            let profile = profile_or(&common.profile, "rx7600")?;
            let o = bench::BenchOptions {
                chunk_tokens,
                reps,
                total_messages,
                overlap,
                fast_hit_rate,
                fast_window,
                skip_llama_bench,
            };
            for id in &common.models {
                bench::bench_model(&cfg, cfg.model(id)?, &profile, &o, &common.extra())?;
            }
            report::write(&cfg, &tasks::all()?)?;
        }
        Cmd::Retrieval {
            common,
            fillers,
            total_messages,
            server_url,
        } => {
            let profile = profile_or(&common.profile, "dev")?;
            for id in &common.models {
                retrieval::eval_embedder(
                    &cfg,
                    cfg.model(id)?,
                    &profile,
                    fillers,
                    total_messages,
                    server_url.as_deref(),
                    &common.extra(),
                )?;
            }
            report::write(&cfg, &tasks::all()?)?;
        }
        Cmd::Report => println!("{}", report::write(&cfg, &tasks::all()?)?.display()),
    }
    Ok(())
}

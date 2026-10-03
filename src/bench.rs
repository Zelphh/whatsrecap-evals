//! Throughput: decide se o "modo completo" (a conversa inteira pelo LLM) é viável.
//!
//! 1. `llama-bench`: tok/s de prompt (pp) e de geração (tg) em tamanhos fixos.
//! 2. Chunk real pelo `llama-server`: ~N tokens com o prompt de brigas e o schema, medindo o tempo de
//!    ponta a ponta e o pico de VRAM (amdgpu).
//! 3. Extrapolação para uma conversa de `total_messages` mensagens nos dois modos.
//!
//! Só tem valor de decisão na máquina alvo; em outra máquina valida o harness.

use crate::client::Client;
use crate::config::{Config, Model, Profile};
use crate::contracts;
use crate::fillers::filler_windows;
use crate::hardware::VramSampler;
use crate::runner::{Meta, now, run_dir, write_json};
use crate::scoring as S;
use crate::server::Server;
use crate::tasks::{ConversationMeta, Msg, Rag, fmt_window, load, system_user};
use anyhow::Result;
use serde::Serialize;
use serde_json::{Value, json};
use std::process::Command;

pub struct BenchOptions {
    pub chunk_tokens: usize,
    pub reps: u32,
    pub total_messages: usize,
    pub overlap: usize,
    pub fast_hit_rate: f64,
    pub fast_window: usize,
    pub skip_llama_bench: bool,
}

fn llama_bench(cfg: &Config, model: &Model, profile: &Profile, prompt_sizes: &[usize]) -> Value {
    let mut prompt_sizes = prompt_sizes.to_vec();
    prompt_sizes.sort_unstable();
    prompt_sizes.dedup();
    let sizes = prompt_sizes
        .iter()
        .map(|s| s.to_string())
        .collect::<Vec<_>>()
        .join(",");
    let path = cfg.path_of(model).display().to_string();
    let mut args: Vec<String> = [
        "-m", &path, "-p", &sizes, "-n", "128", "-r", "3", "-o", "json",
    ]
    .map(String::from)
    .to_vec();
    args.extend(profile.bench_args.iter().cloned());
    eprintln!("  llama-bench {}", args.join(" "));
    let out = match Command::new(cfg.bin("llama-bench"))
        .args(&args)
        .envs(&profile.env)
        .output()
    {
        Ok(o) if o.status.success() => o,
        Ok(o) => {
            eprintln!(
                "  llama-bench falhou: {}",
                String::from_utf8_lossy(&o.stderr)
                    .chars()
                    .rev()
                    .take(1500)
                    .collect::<String>()
                    .chars()
                    .rev()
                    .collect::<String>()
            );
            return Value::Null;
        }
        Err(e) => {
            eprintln!("  llama-bench não rodou: {e}");
            return Value::Null;
        }
    };
    let rows: Vec<Value> = serde_json::from_slice(&out.stdout).unwrap_or_default();
    Value::Array(
        rows.iter()
            .map(|r| {
                json!({
                    "n_prompt": r["n_prompt"], "n_gen": r["n_gen"],
                    "tps": r["avg_ts"].as_f64().map(|x| S::round(x, 2)),
                    "tps_stddev": r["stddev_ts"].as_f64().map(|x| S::round(x, 2)),
                    "backend": r.get("backends").or(r.get("backend")),
                    "n_gpu_layers": r["n_gpu_layers"], "flash_attn": r["flash_attn"],
                    "type_k": r["type_k"], "type_v": r["type_v"],
                })
            })
            .collect(),
    )
}

/// Mensagens no formato do app (janelas rotuladas + ruído), renumeradas em sequência.
fn realistic_lines(conv: &ConversationMeta) -> Result<Vec<Msg>> {
    let mut lines: Vec<Msg> = Rag::load(conv.clone())?
        .windows
        .into_iter()
        .flat_map(|w| w.messages)
        .collect();
    for f in ["conflicts.json", "apologies.json"] {
        let v: Value = load(f)?;
        for it in v["items"].as_array().unwrap() {
            lines.extend(serde_json::from_value::<Vec<Msg>>(it["messages"].clone())?);
        }
    }
    lines.extend(
        filler_windows(300, &conv.authors, 11, 0)
            .into_iter()
            .flat_map(|w| w.messages),
    );
    Ok(lines
        .into_iter()
        .enumerate()
        .map(|(i, (_, a, t))| (i as i64 + 1, a, t))
        .collect())
}

/// Trecho com ~`target` tokens. Devolve (texto, mensagens, tokens por mensagem).
fn build_chunk(
    client: &Client,
    conv: &ConversationMeta,
    target: usize,
) -> Result<(String, usize, f64)> {
    let base = realistic_lines(conv)?;
    let tok_per_msg = client.tokenize(&fmt_window(&base[..400], None))? as f64 / 400.0;
    let n = (target as f64 / tok_per_msg) as usize;
    // Repete a conversa se precisar de mais texto.
    let lines: Vec<Msg> = base
        .iter()
        .cycle()
        .take(n)
        .enumerate()
        .map(|(i, (_, a, t))| (i as i64 + 1, a.clone(), t.clone()))
        .collect();
    Ok((fmt_window(&lines, None), n, tok_per_msg))
}

#[derive(Serialize)]
struct Extrapolation {
    msgs_per_chunk: usize,
    chunks: usize,
    secs_per_chunk: f64,
    full_mode_hours: f64,
    fast_windows: usize,
    secs_per_fast_window: f64,
    fast_mode_minutes: f64,
    assumptions: Value,
}

/// Modo completo: chunks de `chunk_tokens` com sobreposição, 2 passadas (desculpas + brigas).
/// Modo rápido: uma janela de `fast_window` mensagens por acerto do pré-filtro.
#[allow(clippy::too_many_arguments)]
fn extrapolate(
    pp: f64,
    tg: f64,
    tok_per_msg: f64,
    overhead: f64,
    out_tokens: f64,
    o: &BenchOptions,
) -> Extrapolation {
    let passes = 2.0;
    let secs = |prompt: f64| prompt / pp + out_tokens / tg;
    let msgs_per_chunk = (((o.chunk_tokens as f64 - overhead) / tok_per_msg) as usize).max(1);
    let step = msgs_per_chunk.saturating_sub(o.overlap).max(1);
    let chunks = o.total_messages.div_ceil(step);
    let t_chunk = secs(overhead + msgs_per_chunk as f64 * tok_per_msg);
    let windows = (o.total_messages as f64 * o.fast_hit_rate).ceil() as usize;
    let t_window = secs(overhead + o.fast_window as f64 * tok_per_msg);
    Extrapolation {
        msgs_per_chunk,
        chunks,
        secs_per_chunk: S::round(t_chunk, 2),
        full_mode_hours: S::round(chunks as f64 * t_chunk * passes / 3600.0, 2),
        fast_windows: windows,
        secs_per_fast_window: S::round(t_window, 2),
        fast_mode_minutes: S::round(windows as f64 * t_window * passes / 60.0, 1),
        assumptions: json!({
            "total_messages": o.total_messages, "passes": passes, "chunk_tokens": o.chunk_tokens,
            "overlap_msgs": o.overlap, "out_tokens_per_call": out_tokens.round(),
            "fast_hit_rate": o.fast_hit_rate, "fast_window_msgs": o.fast_window,
        }),
    }
}

pub fn bench_model(
    cfg: &Config,
    model: &Model,
    profile: &Profile,
    o: &BenchOptions,
    extra: &[String],
) -> Result<()> {
    let conv: ConversationMeta = load("conversation.json")?;
    let out = run_dir(cfg, "bench", &model.id, &profile.id)?;
    eprintln!(
        "→ throughput {} · perfil {} · {}",
        model.id,
        profile.id,
        out.file_name().unwrap().to_string_lossy()
    );
    let mut meta = Meta::new(
        cfg,
        "bench",
        model,
        profile,
        json!({"chunk_tokens": o.chunk_tokens, "reps": o.reps, "extra_args": extra}),
    );
    let mut summary = serde_json::Map::new();
    if !o.skip_llama_bench {
        summary.insert(
            "llama_bench".into(),
            llama_bench(cfg, model, profile, &[512, 2048, o.chunk_tokens]),
        );
    }

    let vram = VramSampler::start();
    let srv = Server::start(cfg, model, profile, extra, &out, false)?;
    let client = Client::new(&srv.url, cfg.temperature, cfg.seed, model.request.clone());
    let n_ctx = client.props()["default_generation_settings"]["n_ctx"].clone();
    // Folga de 1200 tokens para o prompt de sistema e a saída.
    let (text, n_msgs, tok_per_msg) =
        build_chunk(&client, &conv, o.chunk_tokens.saturating_sub(1200))?;
    let system = contracts::conflicts_prompt("v1", &conv.authors[0], &conv.authors[1]);
    let overhead = client.tokenize(&system)? as f64 + 30.0;
    let mut runs = vec![];
    for r in 0..o.reps {
        let res = client.chat(
            &system_user(system.clone(), text.clone()),
            &contracts::conflicts_schema(&conv.authors),
            700,
            cfg.seed + r as u64,
            false,
        );
        eprintln!(
            "  chunk {}/{}: {} tok de prompt a {:.1} tok/s, {} gerados a {:.1} tok/s, {:.1}s{}",
            r + 1,
            o.reps,
            res.timing("prompt_n").unwrap_or(0.0),
            res.timing("prompt_per_second").unwrap_or(0.0),
            res.timing("predicted_n").unwrap_or(0.0),
            res.timing("predicted_per_second").unwrap_or(0.0),
            res.latency_ms / 1000.0,
            res.error
                .as_deref()
                .map(|e| format!(" (erro: {e})"))
                .unwrap_or_default()
        );
        runs.push(res);
    }
    let info = srv.info();
    drop(srv);
    let peak = vram.finish();

    let pp = S::mean(runs.iter().filter_map(|r| r.timing("prompt_per_second")));
    let tg = S::mean(runs.iter().filter_map(|r| r.timing("predicted_per_second")));
    let gen_n = S::mean(runs.iter().filter_map(|r| r.timing("predicted_n"))).unwrap_or(200.0);
    summary.insert(
        "chunk".into(),
        json!({
            "n_ctx": n_ctx, "messages": n_msgs, "tokens_per_message": S::round(tok_per_msg, 2),
            "prompt_tps": pp, "gen_tps": tg,
            "latency_ms_mean": S::mean(runs.iter().map(|r| r.latency_ms)),
            "json_valid": S::rate(runs.iter().map(|r| r.json.is_some())),
            "runs": runs.iter().map(|r| json!({"latency_ms": r.latency_ms, "timings": r.timings, "error": r.error})).collect::<Vec<_>>(),
        }),
    );
    summary.insert(
        "vram".into(),
        json!({
            "peak_used_mib": peak, "before_mib": info.vram_before_mib, "loaded_mib": info.vram_loaded_mib,
            "peak_delta_mib": peak.zip(info.vram_before_mib).map(|(p, b)| S::round(p - b, 1)),
            "buffers_mib": info.buffers_mib,
        }),
    );
    if let (Some(pp), Some(tg)) = (pp, tg) {
        let e = extrapolate(pp, tg, tok_per_msg, overhead, gen_n, o);
        eprintln!(
            "  modo completo: {} chunks × 2 tarefas ≈ {} h · modo rápido: {} janelas × 2 ≈ {} min",
            e.chunks, e.full_mode_hours, e.fast_windows, e.fast_mode_minutes
        );
        summary.insert("extrapolation".into(), serde_json::to_value(e)?);
    }
    if !meta.hardware.is_target {
        eprintln!(
            "  ⚠ esta não é a máquina alvo (RX 7600): velocidade e VRAM não valem para a decisão."
        );
    }
    meta.set_server(Some(info), None);
    meta.finished = Some(now());
    write_json(&out.join("meta.json"), &meta)?;
    write_json(&out.join("summary.json"), &json!({"bench": summary}))?;
    Ok(())
}

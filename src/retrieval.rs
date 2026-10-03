//! Embeddings do chat tipo 2: recall@k e MRR para busca só-vetorial, só-FTS5 (mesmo tokenizer do
//! app) e híbrida (Reciprocal Rank Fusion), e tempo de indexação extrapolado para uma conversa grande.

use crate::client::Client;
use crate::config::{Config, Model, Profile};
use crate::fillers::filler_windows;
use crate::runner::{Meta, now, run_dir, write_json};
use crate::scoring as S;
use crate::server::Server;
use crate::tasks::{ConversationMeta, Rag, Window, fmt_window, load};
use anyhow::Result;
use rusqlite::Connection;
use serde_json::{Value, json};
use std::collections::HashMap;
use std::time::Instant;

const KS: [usize; 4] = [1, 3, 5, 10];
const RRF_K: f64 = 60.0;
const STOP: &[&str] = &[
    "a", "o", "as", "os", "de", "da", "do", "das", "dos", "e", "em", "no", "na", "nos", "nas",
    "um", "uma", "uns", "umas", "pra", "para", "por", "que", "qual", "quais", "quando", "como",
    "onde", "quem", "se", "me", "te", "vc", "voce", "ele", "ela", "gente", "ja", "foi", "era",
    "com", "sobre", "isso", "essa", "esse",
];

type Ranking = HashMap<String, Vec<String>>;

fn fts_query(q: &str) -> String {
    S::fold(q)
        .split(|c: char| !c.is_alphanumeric())
        .filter(|t| t.chars().count() > 1 && !STOP.contains(t))
        .map(|t| format!("\"{t}\""))
        .collect::<Vec<_>>()
        .join(" OR ")
}

fn fts_rank(windows: &[Window], questions: &[(String, String)]) -> Result<Ranking> {
    let db = Connection::open_in_memory()?;
    db.execute_batch("CREATE VIRTUAL TABLE w USING fts5(wid UNINDEXED, body, tokenize='unicode61 remove_diacritics 2')")?;
    for w in windows {
        let body: Vec<&str> = w.messages.iter().map(|m| m.2.as_str()).collect();
        db.execute("INSERT INTO w VALUES (?1, ?2)", (&w.id, body.join(" ")))?;
    }
    let mut stmt = db.prepare("SELECT wid FROM w WHERE w MATCH ?1 ORDER BY bm25(w) LIMIT 50")?;
    let mut out = HashMap::new();
    for (id, q) in questions {
        let fq = fts_query(q);
        let ranked = if fq.is_empty() {
            vec![]
        } else {
            stmt.query_map([fq], |r| r.get::<_, String>(0))?
                .collect::<Result<Vec<_>, _>>()?
        };
        out.insert(id.clone(), ranked);
    }
    Ok(out)
}

fn cos(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let (na, nb) = (
        a.iter().map(|x| x * x).sum::<f32>().sqrt(),
        b.iter().map(|x| x * x).sum::<f32>().sqrt(),
    );
    if na > 0.0 && nb > 0.0 {
        dot / (na * nb)
    } else {
        0.0
    }
}

fn rrf(rankings: &[&Vec<String>]) -> Vec<String> {
    let mut score: HashMap<&String, f64> = HashMap::new();
    for r in rankings {
        for (i, w) in r.iter().enumerate() {
            *score.entry(w).or_default() += 1.0 / (RRF_K + i as f64 + 1.0);
        }
    }
    let mut v: Vec<(&String, f64)> = score.into_iter().collect();
    v.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(b.0)));
    v.into_iter().map(|(w, _)| w.clone()).collect()
}

fn metrics(r: &Ranking, gold: &[(String, Vec<String>)]) -> Value {
    let mut out = serde_json::Map::new();
    for k in KS {
        let hits = gold
            .iter()
            .map(|(id, g)| r[id].iter().take(k).any(|w| g.contains(w)));
        out.insert(format!("recall@{k}"), json!(S::rate(hits)));
    }
    let rr = gold.iter().map(|(id, g)| {
        r[id]
            .iter()
            .position(|w| g.contains(w))
            .map_or(0.0, |p| 1.0 / (p as f64 + 1.0))
    });
    out.insert("mrr".into(), json!(S::mean(rr)));
    Value::Object(out)
}

pub fn eval_embedder(
    cfg: &Config,
    model: &Model,
    profile: &Profile,
    n_fillers: usize,
    total_messages: usize,
    server_url: Option<&str>,
    extra: &[String],
) -> Result<()> {
    let conv: ConversationMeta = load("conversation.json")?;
    let rag = Rag::load(conv.clone())?;
    let mut windows = rag.windows.clone();
    windows.extend(filler_windows(n_fillers, &conv.authors, 7, 100_000));
    let qs: Vec<_> = rag.questions.iter().filter(|q| q.answerable).collect();
    let out = run_dir(cfg, "retrieval", &model.id, &profile.id)?;
    eprintln!(
        "→ {} · {} janelas · {} perguntas · {}",
        model.id,
        windows.len(),
        qs.len(),
        out.file_name().unwrap().to_string_lossy()
    );

    let go = |url: &str| -> Result<Value> {
        let client = Client::new(url, 0.0, 0, Default::default());
        let passages: Vec<String> = windows
            .iter()
            .map(|w| {
                format!(
                    "{}{}",
                    model.passage_prefix,
                    fmt_window(&w.messages, Some(&w.date))
                )
            })
            .collect();
        let n_tokens: usize = passages
            .iter()
            .map(|p| client.tokenize(p))
            .sum::<Result<usize>>()?;
        let t0 = Instant::now();
        let mut vecs = vec![];
        for chunk in passages.chunks(16) {
            vecs.extend(client.embed(chunk)?);
        }
        let index_secs = t0.elapsed().as_secs_f64();
        let t0 = Instant::now();
        let qvecs = client.embed(
            &qs.iter()
                .map(|q| format!("{}{}", model.query_prefix, q.q))
                .collect::<Vec<_>>(),
        )?;
        let query_ms = t0.elapsed().as_secs_f64() * 1000.0 / qs.len() as f64;

        let mut vec_rank: Ranking = HashMap::new();
        for (q, qv) in qs.iter().zip(&qvecs) {
            let mut sims: Vec<(f32, &String)> = vecs
                .iter()
                .zip(&windows)
                .map(|(v, w)| (cos(qv, v), &w.id))
                .collect();
            sims.sort_by(|a, b| b.0.total_cmp(&a.0));
            vec_rank.insert(
                q.id.clone(),
                sims.into_iter().take(50).map(|(_, w)| w.clone()).collect(),
            );
        }
        let fts = fts_rank(
            &windows,
            &qs.iter()
                .map(|q| (q.id.clone(), q.q.clone()))
                .collect::<Vec<_>>(),
        )?;
        let hybrid: Ranking = qs
            .iter()
            .map(|q| (q.id.clone(), rrf(&[&vec_rank[&q.id], &fts[&q.id]])))
            .collect();
        let gold: Vec<(String, Vec<String>)> = qs
            .iter()
            .map(|q| (q.id.clone(), q.gold_windows.clone()))
            .collect();
        let msgs: usize = windows.iter().map(|w| w.messages.len()).sum();
        Ok(json!({
            "corpus_windows": windows.len(),
            "questions": qs.len(),
            "dim": vecs.first().map(Vec::len),
            "vector": metrics(&vec_rank, &gold),
            "fts": metrics(&fts, &gold),
            "hybrid": metrics(&hybrid, &gold),
            "index_secs": S::round(index_secs, 2),
            "index_tokens_per_sec": S::round(n_tokens as f64 / index_secs, 1),
            "query_ms": S::round(query_ms, 1),
            "extrapolated_index_minutes": S::round(index_secs / msgs as f64 * total_messages as f64 / 60.0, 1),
            "extrapolated_for_messages": total_messages,
            "misses_hybrid@5": gold.iter().filter(|(id, g)| !hybrid[id].iter().take(5).any(|w| g.contains(w))).map(|(id, _)| id).collect::<Vec<_>>(),
        }))
    };

    let mut meta = Meta::new(
        cfg,
        "retrieval",
        model,
        profile,
        json!({"fillers": n_fillers, "total_messages": total_messages}),
    );
    let summary = match server_url {
        Some(u) => {
            meta.set_server(None, Some(u));
            go(u)?
        }
        None => {
            let srv = Server::start(cfg, model, profile, extra, &out, true)?;
            let s = go(&srv.url)?;
            meta.set_server(Some(srv.info()), None);
            s
        }
    };
    meta.finished = Some(now());
    write_json(&out.join("meta.json"), &meta)?;
    write_json(&out.join("summary.json"), &json!({"retrieval": summary}))?;
    for mode in ["vector", "fts", "hybrid"] {
        eprintln!("  {mode:7} {}", summary[mode]);
    }
    eprintln!(
        "  indexação: {} tok/s → ~{} min para {total_messages} mensagens",
        summary["index_tokens_per_sec"], summary["extrapolated_index_minutes"]
    );
    Ok(())
}

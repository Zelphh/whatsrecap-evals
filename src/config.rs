//! Leitura de `models.toml`: modelos candidatos, embedders e perfis de hardware.

use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

/// Raiz do projeto (onde está o `Cargo.toml`), para os caminhos relativos funcionarem de qualquer cwd.
pub fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Model {
    pub id: String,
    pub repo: String,
    pub file: String,
    #[serde(default)]
    pub family: String,
    #[serde(default)]
    pub params_b: f64,
    #[serde(default)]
    pub quant: String,
    /// Flags extras do llama-server para este modelo.
    #[serde(default)]
    pub args: Vec<String>,
    /// Campos extras em cada requisição.
    #[serde(default)]
    pub request: serde_json::Map<String, serde_json::Value>,
    /// Só embedders.
    #[serde(default)]
    pub query_prefix: String,
    #[serde(default)]
    pub passage_prefix: String,
    #[serde(default)]
    pub notes: String,
}

impl Model {
    pub fn url(&self) -> String {
        format!(
            "https://huggingface.co/{}/resolve/main/{}",
            self.repo, self.file
        )
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Profile {
    #[serde(default)]
    pub id: String,
    /// Flags do llama-server.
    pub args: Vec<String>,
    /// Flags equivalentes do llama-bench.
    #[serde(default)]
    pub bench_args: Vec<String>,
    #[serde(default)]
    pub env: BTreeMap<String, String>,
    #[serde(default)]
    pub description: String,
}

#[derive(Debug, Deserialize)]
struct Defaults {
    llama_dir: Option<String>,
    models_dir: Option<String>,
    results_dir: Option<String>,
    port: Option<u16>,
    temperature: Option<f64>,
    seed: Option<u64>,
}

#[derive(Debug, Deserialize)]
struct Raw {
    defaults: Option<Defaults>,
    #[serde(default)]
    models: Vec<Model>,
    #[serde(default)]
    embedders: Vec<Model>,
    #[serde(default)]
    profiles: BTreeMap<String, Profile>,
}

#[derive(Debug, Clone)]
pub struct Config {
    pub llama_dir: PathBuf,
    pub models_dir: PathBuf,
    pub results_dir: PathBuf,
    pub port: u16,
    pub temperature: f64,
    pub seed: u64,
    pub models: Vec<Model>,
    pub embedders: Vec<Model>,
    pub profiles: BTreeMap<String, Profile>,
}

impl Config {
    pub fn load(path: Option<&Path>) -> Result<Self> {
        let path = path
            .map(Path::to_path_buf)
            .unwrap_or_else(|| root().join("models.toml"));
        let text =
            std::fs::read_to_string(&path).with_context(|| format!("lendo {}", path.display()))?;
        let raw: Raw =
            toml::from_str(&text).with_context(|| format!("interpretando {}", path.display()))?;
        let d = raw.defaults.unwrap_or(Defaults {
            llama_dir: None,
            models_dir: None,
            results_dir: None,
            port: None,
            temperature: None,
            seed: None,
        });
        let rel = |p: Option<String>, default: &str| {
            let p = PathBuf::from(p.unwrap_or_else(|| default.into()));
            if p.is_absolute() { p } else { root().join(p) }
        };
        let profiles = raw
            .profiles
            .into_iter()
            .map(|(k, mut p)| {
                p.id = k.clone();
                (k, p)
            })
            .collect();
        Ok(Config {
            llama_dir: rel(d.llama_dir, "vendor/llama.cpp/current"),
            models_dir: rel(d.models_dir, "models"),
            results_dir: rel(d.results_dir, "results"),
            port: d.port.unwrap_or(8089),
            temperature: d.temperature.unwrap_or(0.0),
            seed: d.seed.unwrap_or(42),
            models: raw.models,
            embedders: raw.embedders,
            profiles,
        })
    }

    pub fn model(&self, id: &str) -> Result<&Model> {
        match self
            .models
            .iter()
            .chain(&self.embedders)
            .find(|m| m.id == id)
        {
            Some(m) => Ok(m),
            None => bail!("modelo desconhecido: {id} (veja `evals list`)"),
        }
    }

    pub fn profile(&self, id: &str) -> Result<&Profile> {
        match self.profiles.get(id) {
            Some(p) => Ok(p),
            None => bail!(
                "perfil desconhecido: {id} (opções: {})",
                self.profiles.keys().cloned().collect::<Vec<_>>().join(", ")
            ),
        }
    }

    pub fn path_of(&self, m: &Model) -> PathBuf {
        self.models_dir.join(&m.file)
    }

    pub fn bin(&self, name: &str) -> PathBuf {
        self.llama_dir.join(name)
    }
}

//! Sobe e derruba o `llama-server` (como o sidecar do Tauri fará) e lê o que ele reporta de memória.

use crate::config::{Config, Model, Profile};
use crate::hardware;
use anyhow::{Context, Result, bail};
use regex::Regex;
use serde::Serialize;
use std::collections::BTreeMap;
use std::fs::File;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};

pub fn llama_version(cfg: &Config) -> String {
    let tag = std::fs::read_to_string(cfg.llama_dir.join("TAG"))
        .map(|s| s.trim().to_string())
        .unwrap_or_default();
    match Command::new(cfg.bin("llama-server"))
        .arg("--version")
        .output()
    {
        Ok(o) => {
            let all = format!(
                "{}{}",
                String::from_utf8_lossy(&o.stdout),
                String::from_utf8_lossy(&o.stderr)
            );
            let line = all
                .lines()
                .find(|l| l.contains("version"))
                .unwrap_or("")
                .trim()
                .to_string();
            format!("{tag} {line}").trim().to_string()
        }
        Err(e) => format!("{tag} (erro: {e})"),
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct ServerInfo {
    pub argv: Vec<String>,
    pub env: BTreeMap<String, String>,
    pub load_secs: f64,
    pub vram_before_mib: Option<f64>,
    pub vram_loaded_mib: Option<f64>,
    /// Buffers alocados pelo llama.cpp, em MiB, por "dispositivo tipo" (vale para qualquer backend).
    pub buffers_mib: BTreeMap<String, f64>,
    /// Tabela "memory breakdown" que o llama.cpp imprime ao carregar.
    pub memory_breakdown: Vec<String>,
}

pub struct Server {
    pub url: String,
    child: Child,
    log_path: PathBuf,
    argv: Vec<String>,
    env: BTreeMap<String, String>,
    load_secs: f64,
    vram_before: Option<f64>,
    vram_loaded: Option<f64>,
}

impl Server {
    pub fn start(
        cfg: &Config,
        model: &Model,
        profile: &Profile,
        extra: &[String],
        log_dir: &Path,
        embedding: bool,
    ) -> Result<Self> {
        let path = cfg.path_of(model);
        if !path.exists() {
            bail!(
                "arquivo do modelo não encontrado: {}\n  rode: evals download {}",
                path.display(),
                model.id
            );
        }
        let bin = cfg.bin("llama-server");
        if !bin.exists() {
            bail!(
                "{} não existe; rode scripts/setup-llama-cpp.sh",
                bin.display()
            );
        }
        let port = cfg.port.to_string();
        let mut argv: Vec<String> = vec![
            "-m".into(),
            path.display().to_string(),
            "--host".into(),
            "127.0.0.1".into(),
            "--port".into(),
            port,
            "--no-webui".into(),
            "-np".into(),
            "1".into(),
            // Nível 4 registra os buffers e a tabela de memória por dispositivo.
            "-lv".into(),
            "4".into(),
        ];
        argv.extend(profile.args.iter().cloned());
        argv.extend(model.args.iter().cloned());
        argv.extend(extra.iter().cloned());
        if embedding && !argv.iter().any(|a| a == "--embedding") {
            argv.push("--embedding".into());
        }
        std::fs::create_dir_all(log_dir)?;
        let log_path = log_dir.join(format!("llama-server-{}.log", model.id));
        let log = File::create(&log_path)?;
        let vram_before = hardware::vram_used_mib();
        let t0 = Instant::now();
        let child = Command::new(&bin)
            .args(&argv)
            .envs(&profile.env)
            .stdout(log.try_clone()?)
            .stderr(log)
            .stdin(Stdio::null())
            .spawn()
            .with_context(|| format!("iniciando {}", bin.display()))?;
        let mut s = Server {
            url: format!("http://127.0.0.1:{}", cfg.port),
            child,
            log_path,
            argv,
            env: profile.env.clone(),
            load_secs: 0.0,
            vram_before,
            vram_loaded: None,
        };
        s.wait_ready(Duration::from_secs(900))?;
        s.load_secs = (t0.elapsed().as_secs_f64() * 100.0).round() / 100.0;
        s.vram_loaded = hardware::vram_used_mib();
        Ok(s)
    }

    fn wait_ready(&mut self, timeout: Duration) -> Result<()> {
        let agent = crate::client::agent(Duration::from_secs(2));
        let deadline = Instant::now() + timeout;
        while Instant::now() < deadline {
            if let Some(status) = self.child.try_wait()? {
                let log = std::fs::read_to_string(&self.log_path).unwrap_or_default();
                let tail: String = log
                    .chars()
                    .rev()
                    .take(3000)
                    .collect::<Vec<_>>()
                    .into_iter()
                    .rev()
                    .collect();
                bail!("llama-server saiu ({status}). Fim do log:\n{tail}");
            }
            if let Ok(r) = agent.get(format!("{}/health", self.url)).call()
                && r.status().is_success()
            {
                return Ok(());
            }
            std::thread::sleep(Duration::from_millis(500));
        }
        bail!(
            "llama-server não ficou pronto em {timeout:?} (log: {})",
            self.log_path.display()
        )
    }

    pub fn info(&self) -> ServerInfo {
        ServerInfo {
            argv: self.argv.clone(),
            env: self.env.clone(),
            load_secs: self.load_secs,
            vram_before_mib: self.vram_before,
            vram_loaded_mib: self.vram_loaded,
            buffers_mib: memory_report(&self.log_path),
            memory_breakdown: std::fs::read_to_string(&self.log_path)
                .unwrap_or_default()
                .lines()
                .filter(|l| l.contains("memory_breakdown_print"))
                .map(|l| l.split_once("print: ").map_or(l, |(_, r)| r).to_string())
                .collect(),
        }
    }
}

impl Drop for Server {
    fn drop(&mut self) {
        // SIGINT primeiro para o llama-server encerrar limpo; SIGKILL se não sair. No Windows não há
        // SIGINT para outro processo: `kill` (TerminateProcess) direto, sem esperar 20 s à toa.
        #[cfg(unix)]
        let _ = Command::new("kill")
            .args(["-INT", &self.child.id().to_string()])
            .status();
        #[cfg(windows)]
        let deadline = Instant::now();
        #[cfg(unix)]
        let deadline = Instant::now() + Duration::from_secs(20);
        while Instant::now() < deadline {
            if let Ok(Some(_)) = self.child.try_wait() {
                return;
            }
            std::thread::sleep(Duration::from_millis(200));
        }
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

/// "load_tensors: Vulkan0 model buffer size = 1234.56 MiB", "llama_kv_cache: Vulkan0 KV buffer size = …"
pub fn memory_report(log: &Path) -> BTreeMap<String, f64> {
    let re =
        Regex::new(r"(\S+)\s+(model|KV|compute|RS|output) buffer size =\s+([\d.]+) MiB").unwrap();
    let mut out = BTreeMap::new();
    let text = std::fs::read_to_string(log).unwrap_or_default();
    for c in re.captures_iter(&text) {
        let v: f64 = c[3].parse().unwrap_or(0.0);
        *out.entry(format!("{} {}", &c[1], &c[2])).or_insert(0.0) += v;
    }
    out.values_mut()
        .for_each(|v| *v = (*v * 10.0).round() / 10.0);
    out
}

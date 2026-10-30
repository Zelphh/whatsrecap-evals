//! Impressão digital do hardware e amostragem de VRAM.
//!
//! Cada resultado leva o hardware em que rodou. Velocidade e VRAM só valem na máquina alvo
//! (Ryzen 7 5700X + RX 7600); o relatório separa o que veio de outra máquina.

use serde::{Deserialize, Serialize};
#[cfg(not(windows))]
use std::fs;
#[cfg(not(windows))]
use std::path::PathBuf;
use std::process::Command;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle};
use std::time::Duration;

pub const TARGET_GPU: &str = "RX 7600";

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Fingerprint {
    pub host: String,
    pub os: String,
    pub cpu: String,
    pub ram_gb: f64,
    pub gpus: Vec<String>,
    pub is_target: bool,
}

/// Saída de um comando do PowerShell, sem espaços nas pontas. `None` se falhar ou vier vazia.
#[cfg(windows)]
fn powershell(cmd: &str) -> Option<String> {
    let out = Command::new("powershell")
        .args(["-NoProfile", "-NonInteractive", "-Command", cmd])
        .output()
        .ok()?;
    let s = String::from_utf8_lossy(&out.stdout).trim().to_string();
    (out.status.success() && !s.is_empty()).then_some(s)
}

#[cfg(windows)]
fn cpu() -> String {
    powershell("(Get-CimInstance Win32_Processor | Select-Object -First 1).Name")
        .unwrap_or_else(|| "?".into())
}

#[cfg(windows)]
fn ram_gb() -> f64 {
    powershell("(Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory")
        .and_then(|b| b.parse::<f64>().ok())
        .map(|b| (b / 1073741824.0 * 10.0).round() / 10.0)
        .unwrap_or(0.0)
}

#[cfg(windows)]
fn host() -> String {
    std::env::var("COMPUTERNAME").unwrap_or_default()
}

#[cfg(windows)]
fn os() -> String {
    powershell(r#"$o = Get-CimInstance Win32_OperatingSystem; "$($o.Caption) $($o.Version)""#)
        .unwrap_or_else(|| std::env::consts::OS.into())
}

#[cfg(not(windows))]
fn cpu() -> String {
    fs::read_to_string("/proc/cpuinfo")
        .ok()
        .and_then(|s| {
            s.lines()
                .find(|l| l.starts_with("model name"))
                .and_then(|l| l.split_once(':'))
                .map(|(_, v)| v.trim().to_string())
        })
        .unwrap_or_else(|| "?".into())
}

#[cfg(not(windows))]
fn ram_gb() -> f64 {
    fs::read_to_string("/proc/meminfo")
        .ok()
        .and_then(|s| {
            s.lines()
                .find(|l| l.starts_with("MemTotal:"))
                .and_then(|l| l.split_whitespace().nth(1))
                .and_then(|kb| kb.parse::<f64>().ok())
        })
        .map(|kb| (kb / 1024.0 / 1024.0 * 10.0).round() / 10.0)
        .unwrap_or(0.0)
}

fn gpus() -> Vec<String> {
    if let Ok(out) = Command::new("vulkaninfo").arg("--summary").output() {
        let s = String::from_utf8_lossy(&out.stdout);
        let names: Vec<String> = s
            .lines()
            .filter_map(|l| l.trim().strip_prefix("deviceName"))
            .filter_map(|l| l.split_once('=').map(|(_, v)| v.trim().to_string()))
            .filter(|n| !n.contains("llvmpipe"))
            .collect();
        if !names.is_empty() {
            return names;
        }
    }
    #[cfg(windows)]
    {
        powershell("(Get-CimInstance Win32_VideoController).Name")
            .map(|s| s.lines().map(|l| l.trim().to_string()).collect())
            .unwrap_or_default()
    }
    #[cfg(not(windows))]
    Command::new("lspci")
        .output()
        .map(|o| {
            String::from_utf8_lossy(&o.stdout)
                .lines()
                .filter(|l| l.contains("VGA") || l.contains("3D") || l.contains("Display"))
                .filter_map(|l| l.split_once(": ").map(|(_, v)| v.to_string()))
                .collect()
        })
        .unwrap_or_default()
}

#[cfg(not(windows))]
fn host() -> String {
    fs::read_to_string("/proc/sys/kernel/hostname")
        .map(|s| s.trim().to_string())
        .unwrap_or_default()
}

#[cfg(not(windows))]
fn os() -> String {
    fs::read_to_string("/proc/sys/kernel/osrelease")
        .map(|s| format!("Linux {}", s.trim()))
        .unwrap_or_else(|_| std::env::consts::OS.into())
}

pub fn fingerprint() -> Fingerprint {
    let gpus = gpus();
    Fingerprint {
        host: host(),
        os: os(),
        cpu: cpu(),
        ram_gb: ram_gb(),
        is_target: gpus.iter().any(|g| g.contains(TARGET_GPU)),
        gpus,
    }
}

/// VRAM dedicada em uso, somada entre os adaptadores, pelo contador "GPU Adapter Memory" do
/// Windows (vale para qualquer fabricante). Cada leitura leva ~1 s (o `typeperf` amostra uma vez).
#[cfg(windows)]
pub fn vram_used_mib() -> Option<f64> {
    let out = Command::new("typeperf")
        .args([r"\GPU Adapter Memory(*)\Dedicated Usage", "-sc", "1"])
        .output()
        .ok()?;
    let text = String::from_utf8_lossy(&out.stdout);
    // CSV: linha 1 = cabeçalho, linha 2 = "data","bytes adaptador 1","bytes adaptador 2",…
    let row = text.lines().filter(|l| l.starts_with('"')).nth(1)?;
    let bytes: f64 = row
        .split(',')
        .skip(1)
        .filter_map(|v| v.trim_matches('"').parse::<f64>().ok())
        .sum();
    Some((bytes / 1048576.0 * 10.0).round() / 10.0)
}

/// `mem_info_vram_used` das placas AMD (vendor 0x1002). Vazio em outras GPUs.
#[cfg(not(windows))]
fn amdgpu_vram_files() -> Vec<PathBuf> {
    let Ok(dir) = fs::read_dir("/sys/class/drm") else {
        return vec![];
    };
    dir.filter_map(Result::ok)
        .map(|e| e.path().join("device"))
        .filter(|d| {
            d.parent().and_then(|p| p.file_name()).is_some_and(|n| {
                n.to_string_lossy().starts_with("card") && !n.to_string_lossy().contains('-')
            })
        })
        .filter(|d| fs::read_to_string(d.join("vendor")).is_ok_and(|v| v.trim() == "0x1002"))
        .map(|d| d.join("mem_info_vram_used"))
        .filter(|f| f.exists())
        .collect()
}

#[cfg(not(windows))]
pub fn vram_used_mib() -> Option<f64> {
    let files = amdgpu_vram_files();
    if files.is_empty() {
        return None;
    }
    let bytes: u64 = files
        .iter()
        .filter_map(|f| fs::read_to_string(f).ok()?.trim().parse::<u64>().ok())
        .sum();
    Some((bytes as f64 / 1048576.0 * 10.0).round() / 10.0)
}

/// Amostra a VRAM usada (amdgpu no Linux, contador de GPU no Windows) em segundo plano e guarda o pico.
pub struct VramSampler {
    stop: Arc<AtomicBool>,
    peak: Arc<Mutex<Option<f64>>>,
    handle: Option<JoinHandle<()>>,
}

impl VramSampler {
    pub fn start() -> Self {
        let stop = Arc::new(AtomicBool::new(false));
        let peak = Arc::new(Mutex::new(None));
        let (s, p) = (stop.clone(), peak.clone());
        let handle = thread::spawn(move || {
            while !s.load(Ordering::Relaxed) {
                if let Some(v) = vram_used_mib() {
                    let mut g = p.lock().unwrap();
                    *g = Some(g.map_or(v, |x: f64| x.max(v)));
                }
                thread::sleep(Duration::from_millis(250));
            }
        });
        VramSampler {
            stop,
            peak,
            handle: Some(handle),
        }
    }

    pub fn finish(mut self) -> Option<f64> {
        self.stop.store(true, Ordering::Relaxed);
        if let Some(h) = self.handle.take() {
            h.join().ok();
        }
        *self.peak.lock().unwrap()
    }
}

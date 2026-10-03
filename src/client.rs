//! Cliente HTTP da API compatível com OpenAI do `llama-server`.

use anyhow::{Result, anyhow};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value, json};
use std::time::{Duration, Instant};
use ureq::Agent;

pub struct Client {
    pub url: String,
    pub temperature: f64,
    pub seed: u64,
    extra: Map<String, Value>,
    agent: Agent,
}

/// Resultado de uma chamada de chat. `json` é `None` se a saída não parseou.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ChatResult {
    pub content: Option<String>,
    pub json: Option<Value>,
    pub error: Option<String>,
    pub latency_ms: f64,
    /// `timings` do llama-server: prompt_n, prompt_per_second, predicted_n, predicted_per_second…
    pub timings: Value,
}

impl ChatResult {
    pub fn timing(&self, key: &str) -> Option<f64> {
        self.timings.get(key).and_then(Value::as_f64)
    }
}

pub fn agent(timeout: Duration) -> Agent {
    Agent::config_builder()
        .timeout_global(Some(timeout))
        .http_status_as_error(false)
        .build()
        .into()
}

impl Client {
    pub fn new(url: &str, temperature: f64, seed: u64, extra: Map<String, Value>) -> Self {
        Client {
            url: url.trim_end_matches('/').to_string(),
            temperature,
            seed,
            extra,
            agent: agent(Duration::from_secs(1800)),
        }
    }

    fn post(&self, path: &str, body: &Value) -> Result<Value> {
        let mut resp = self
            .agent
            .post(format!("{}{path}", self.url))
            .send_json(body)?;
        let status = resp.status();
        let text = resp
            .body_mut()
            .with_config()
            .limit(256 << 20)
            .read_to_string()?;
        if !status.is_success() {
            return Err(anyhow!(
                "HTTP {status}: {}",
                text.chars().take(2000).collect::<String>()
            ));
        }
        Ok(serde_json::from_str(&text)?)
    }

    pub fn props(&self) -> Value {
        self.agent
            .get(format!("{}/props", self.url))
            .call()
            .ok()
            .and_then(|mut r| r.body_mut().read_json().ok())
            .unwrap_or(Value::Null)
    }

    /// Uma chamada a /v1/chat/completions com a saída restrita pelo schema.
    pub fn chat(
        &self,
        messages: &[Value],
        schema: &Value,
        max_tokens: u32,
        seed: u64,
        cache_prompt: bool,
    ) -> ChatResult {
        let mut body = json!({
            "messages": messages,
            "temperature": self.temperature,
            "seed": seed,
            "max_tokens": max_tokens,
            "cache_prompt": cache_prompt,
            "response_format": {"type": "json_schema", "json_schema": {"name": "out", "schema": schema}},
        });
        for (k, v) in &self.extra {
            body[k] = v.clone();
        }
        let t0 = Instant::now();
        let r = self.post("/v1/chat/completions", &body);
        let latency_ms = (t0.elapsed().as_secs_f64() * 10000.0).round() / 10.0;
        let r = match r {
            Ok(r) => r,
            Err(e) => {
                return ChatResult {
                    error: Some(e.to_string()),
                    latency_ms,
                    ..Default::default()
                };
            }
        };
        let choice = &r["choices"][0];
        let content = choice["message"]["content"]
            .as_str()
            .unwrap_or_default()
            .to_string();
        let mut error = None;
        let parsed = match serde_json::from_str::<Value>(&content) {
            Ok(v) => Some(v),
            Err(e) => {
                error = Some(format!("JSON inválido: {e}"));
                None
            }
        };
        if choice["finish_reason"] == "length" {
            error = Some(error.map_or("cortado por max_tokens".into(), |e| {
                format!("{e}; cortado por max_tokens")
            }));
        }
        ChatResult {
            content: Some(content),
            json: parsed,
            error,
            latency_ms,
            timings: r["timings"].clone(),
        }
    }

    pub fn tokenize(&self, text: &str) -> Result<usize> {
        let r = self.post("/tokenize", &json!({"content": text}))?;
        Ok(r["tokens"].as_array().map_or(0, Vec::len))
    }

    pub fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        let r = self.post("/v1/embeddings", &json!({"input": texts}))?;
        let mut data: Vec<&Value> = r["data"]
            .as_array()
            .ok_or_else(|| anyhow!("resposta sem data"))?
            .iter()
            .collect();
        data.sort_by_key(|d| d["index"].as_u64());
        Ok(data
            .iter()
            .map(|d| {
                d["embedding"]
                    .as_array()
                    .map(|a| {
                        a.iter()
                            .filter_map(|x| x.as_f64())
                            .map(|x| x as f32)
                            .collect()
                    })
                    .unwrap_or_default()
            })
            .collect())
    }
}

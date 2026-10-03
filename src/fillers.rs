//! Janelas de ruído para o corpus de retrieval e para o chunk de throughput: conversa cotidiana
//! determinística, com parte do vocabulário das perguntas (viagem, festa, carro, aluguel…) para que
//! a busca não seja trivial.

use crate::tasks::{Rng, Window};
use chrono::{Duration, NaiveDate};

const DAILY: &[&str] = &[
    "bom dia",
    "bom dia amor",
    "chegou bem?",
    "cheguei",
    "tô saindo do trabalho",
    "já tô no ônibus",
    "o que vai querer jantar?",
    "pode ser pizza",
    "kkkkkk",
    "kkk verdade",
    "vc viu meu áudio?",
    "<áudio omitido>",
    "<figurinha omitida>",
    "<imagem omitida>",
    "tô morta de sono",
    "vou tomar banho",
    "boa noite",
    "boa noite ❤️",
    "saudade",
    "te amo",
    "também te amo",
    "que preguiça de segunda",
    "choveu muito aí?",
    "aqui tá um calor absurdo",
    "lembra de comprar pão",
    "comprei",
    "esqueci kkk",
    "a Pipoca destruiu o tapete",
    "olha esse vídeo",
    "kkkk que isso",
    "vamos ver série hoje?",
    "bora",
    "tô com fome",
    "pede um lanche",
    "minha mãe mandou beijo",
    "manda outro pra ela",
];

const TOPICAL: &[&str] = &[
    "vamos viajar no feriado?",
    "acho que não dá, tô sem grana",
    "a festa da firma foi chata demais",
    "o carro tá fazendo um barulho estranho de novo",
    "vou levar no mecânico semana que vem",
    "o condomínio aumentou de novo",
    "o aluguel vence dia 10 né?",
    "esse restaurante novo é caro",
    "o filme de ontem foi meio fraco",
    "vi um apê pra alugar perto do metrô",
    "a Carol passou na prova",
    "o Rafa chamou pra jogar bola",
    "o Diego vai fazer churrasco",
    "minha chefe tá insuportável",
    "pensei em fazer academia",
    "a passagem pra Salvador tá barata",
    "a vacina da Pipoca é mês que vem",
    "a senha do banco expirou",
    "o wifi caiu de novo",
    "achei um japonês bom perto do trabalho",
    "precisamos ver o presente da sua mãe",
    "o mecânico disse que é a bateria",
];

pub fn filler_windows(n: usize, authors: &[String], seed: u64, start_id: i64) -> Vec<Window> {
    let mut rng = Rng::new(seed);
    let day0 = NaiveDate::from_ymd_opt(2021, 3, 14).unwrap();
    let mut next_id = start_id;
    (0..n)
        .map(|w| {
            let date = day0 + Duration::days(rng.below(1800) as i64);
            let len = 4 + rng.below(6);
            let messages = (0..len)
                .map(|_| {
                    let pool = if rng.unit() < 0.25 { TOPICAL } else { DAILY };
                    let m = (
                        next_id,
                        authors[rng.below(2)].clone(),
                        pool[rng.below(pool.len())].to_string(),
                    );
                    next_id += 1;
                    m
                })
                .collect();
            Window {
                id: format!("fill{w:03}"),
                date: date.to_string(),
                messages,
            }
        })
        .collect()
}

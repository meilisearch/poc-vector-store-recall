use std::{collections::HashMap, time::Duration};
use std::{env, thread};

use rust_bert::pipelines::keywords_extraction::{Keyword, KeywordExtractionModel};
use rustyline::error::ReadlineError;
use rustyline::DefaultEditor;
use serde::Deserialize;
use serde_json::Value;

const TOP_N_KEYWORDS: usize = 3;
const THRESHOLD_KEYWORDS: f32 = 0.3;
const THRESHOLD_SEMANTIC_DISTANCE: f32 = 0.8;
const TOP_N_DOCUMENTS: usize = 4;

fn main() -> anyhow::Result<()> {
    let kem = KeywordExtractionModel::new(Default::default())?;

    let mut rl = DefaultEditor::new()?;
    if rl.load_history("history.txt").is_err() {
        println!("No previous history.");
    }

    loop {
        let readline = rl.readline(">> ");
        match readline {
            Ok(query) => {
                rl.add_history_entry(query.as_str())?;
                // Ask OpenAI the embedding of the query
                let embedding = compute_embeddings(&query)?;
                // Ask Meilisearch about the best results that corresponds
                let mut response = ask_vector_store(&embedding, TOP_N_DOCUMENTS)?;
                // Only keep the ones that are not too far
                response
                    .hits
                    .retain(|h| h.semantic_score.unwrap() > THRESHOLD_SEMANTIC_DISTANCE);
                eprintln!(
                    "We kept {} documents from the semantic search",
                    response.hits.len()
                );
                // Extarct the keywords from those documents and only keep the ones that are not too far
                let keywords: Vec<_> = response
                    .hits
                    .into_iter()
                    .map(|h| {
                        let keywords: Vec<_> = extract_best_keywords(&kem, &h)
                            .unwrap()
                            .into_iter()
                            .filter(|kw| kw.score > THRESHOLD_KEYWORDS)
                            .map(|kw| kw.text)
                            .take(TOP_N_KEYWORDS)
                            .collect();
                        keywords.join(" ")
                    })
                    .collect();
                eprintln!("We associated {query:?} to {keywords:?}");
                // Change the Meilisearch synonyms and associate those keywords to the current query
                let task_uid = set_synonyms_and_wait(&query, &keywords)?;
                eprintln!("We generated http://localhost:7700/tasks/{task_uid} with the synonyms.");
                // Now we can send the request to Meilisearch again, the synonyms are set.
                let response = ask_meilisearch(&query, 10)?;
                for (i, mut hit) in response.hits.into_iter().enumerate() {
                    hit.content.remove("_vectors");
                    println!("{i}. {hit:?}");
                }
            }
            Err(ReadlineError::Eof) => break,
            Err(err) => {
                eprintln!("Error: {:?}", err);
                break;
            }
        }
    }

    rl.save_history("history.txt")?;

    Ok(())
}

#[derive(Debug, Deserialize)]
struct OpenAiResponse {
    data: Vec<Embedding>,
}

#[derive(Debug, Deserialize)]
struct Embedding {
    embedding: Vec<f32>,
    // object: String,
    // index: usize,
}

fn compute_embeddings(query: &str) -> anyhow::Result<Vec<f32>> {
    let openai_api_key = env::var("OPENAI_API_KEY").expect("Missing OPENAI_API_KEY env variable");
    let mut response: OpenAiResponse = ureq::post("https://api.openai.com/v1/embeddings")
        .set("Content-Type", "application/json")
        .set("Authorization", &format!("Bearer {openai_api_key}"))
        .send_json(ureq::json!({
            "input": query,
            "model": "text-embedding-ada-002",
            "encoding_format": "float"
        }))?
        .into_json()?;

    Ok(response.data.pop().unwrap().embedding)
}

#[derive(Debug, Deserialize)]
struct MeilisearchResponse {
    hits: Vec<Hit>,
}

#[derive(Debug, Deserialize)]
struct Hit {
    #[serde(rename = "_semanticScore")]
    semantic_score: Option<f32>,
    #[serde(flatten)]
    content: HashMap<String, Value>,
}

fn ask_vector_store(vector: &[f32], limit: usize) -> anyhow::Result<MeilisearchResponse> {
    let response: MeilisearchResponse = ureq::post("http://localhost:7700/indexes/vec-test/search")
        .set("Content-Type", "application/json")
        .send_json(ureq::json!({
            "vector": vector,
            "limit": limit,
        }))
        .unwrap()
        .into_json()?;
    Ok(response)
}

fn ask_meilisearch(query: &str, limit: usize) -> anyhow::Result<MeilisearchResponse> {
    let response: MeilisearchResponse = ureq::post("http://localhost:7700/indexes/vec-test/search")
        .set("Content-Type", "application/json")
        .send_json(ureq::json!({
            "q": query,
            "limit": limit,
        }))
        .unwrap()
        .into_json()?;
    Ok(response)
}

fn extract_best_keywords(
    keyword_extraction_model: &KeywordExtractionModel<'_>,
    hit: &Hit,
) -> anyhow::Result<Vec<Keyword>> {
    let strings: Vec<_> = hit.content.values().flat_map(|v| v.as_str()).collect();
    let text = strings.join(". ");
    let mut output = keyword_extraction_model.predict(&[text])?;
    Ok(output.pop().unwrap())
}

#[derive(Debug, Deserialize)]
struct MeilisearchSettingsResponse {
    #[serde(rename = "taskUid")]
    task_uid: u64,
}

#[derive(Debug, Deserialize)]
struct MeilisearchTask {
    uid: u64,
    status: String,
}

/// Sets the synonyms of the query and return the task UID.
fn set_synonyms_and_wait(query: &str, keywords: &[String]) -> anyhow::Result<u64> {
    let mut body = HashMap::new();
    body.insert(query, keywords);

    let response: MeilisearchSettingsResponse =
        ureq::put("http://localhost:7700/indexes/vec-test/settings/synonyms")
            .set("Content-Type", "application/json")
            .send_json(body)?
            .into_json()?;

    let task_uid = response.task_uid;
    loop {
        thread::sleep(Duration::from_millis(100));
        let task: MeilisearchTask = ureq::get(&format!("http://localhost:7700/tasks/{task_uid}"))
            .call()?
            .into_json()?;
        if task.status == "succeeded" {
            break;
        }
    }

    Ok(response.task_uid)
}

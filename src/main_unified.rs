//! ZIGGURAT MIND - Unified Entry Point
//!
//! Unified architecture: Embedding Engine + Dialogue Memory + Mistral 7B
//! Memory flow: Query → Embed → Search → Context → Generate → Save

mod logos;
mod priests;
mod totems;
mod utils;

use anyhow::{Error as E, Result};
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::generation::{LogitsProcessor, Sampling};
use candle_transformers::models::mistral::{Config, Model as Mistral};
use clap::Parser;
use hf_hub::{api::sync::Api, Repo, RepoType};
use std::io::Write;
use std::path::Path;
use std::sync::Arc;
use tokenizers::Tokenizer;

use crate::priests::device::select_device;
use crate::priests::embeddings::{Embedder, EmbeddingEngine};
use crate::totems::episodic::DialogueManager;
use crate::utils::hub_load_safetensors;

const DEFAULT_SAMPLE_LEN: usize = 2048;

struct UnifiedPipeline {
    model: Mistral,
    tokenizer: Tokenizer,
    device: Device,
    logits_processor: LogitsProcessor,
    repeat_penalty: f32,
    repeat_last_n: usize,
}

impl UnifiedPipeline {
    fn new(
        model: Mistral,
        tokenizer: Tokenizer,
        device: Device,
        temperature: Option<f64>,
        top_p: Option<f64>,
        top_k: Option<usize>,
        repeat_penalty: f32,
        repeat_last_n: usize,
        seed: u64,
    ) -> Self {
        let temperature = temperature.unwrap_or(0.);
        let sampling = if temperature <= 0. {
            Sampling::ArgMax
        } else {
            match (top_k, top_p) {
                (None, None) => Sampling::All { temperature },
                (Some(k), None) => Sampling::TopK { k, temperature },
                (None, Some(p)) => Sampling::TopP { p, temperature },
                (Some(k), Some(p)) => Sampling::TopKThenTopP { k, p, temperature },
            }
        };
        let logits_processor = LogitsProcessor::from_sampling(seed, sampling);

        Self {
            model,
            tokenizer,
            device,
            logits_processor,
            repeat_penalty,
            repeat_last_n,
        }
    }

    fn run(&mut self, prompt: &str, sample_len: usize) -> Result<String> {
        let mut tokens = self
            .tokenizer
            .encode(prompt, true)
            .map_err(E::msg)?
            .get_ids()
            .to_vec();

        let mut generated_tokens = 0usize;
        let eos_token = match self.tokenizer.get_vocab(false).get("</s>") {
            Some(&t) => t,
            None => 2,
        };

        let start_gen = std::time::Instant::now();
        let mut output_tokens = Vec::new();

        for index in 0..sample_len {
            let context_size = if index > 0 { 1 } else { tokens.len() };
            let start_pos = tokens.len().saturating_sub(context_size);
            let ctxt = &tokens[start_pos..];
            let input = Tensor::new(ctxt, &self.device)?.unsqueeze(0)?;
            let logits = self
                .model
                .forward(&input, start_pos)?
                .squeeze(0)?
                .squeeze(0)?
                .to_dtype(DType::F32)?;

            let logits = if self.repeat_penalty == 1. {
                logits
            } else {
                let start_at = tokens.len().saturating_sub(self.repeat_last_n);
                candle_transformers::utils::apply_repeat_penalty(
                    &logits,
                    self.repeat_penalty,
                    &tokens[start_at..],
                )?
            };

            let next_token = self.logits_processor.sample(&logits)?;
            tokens.push(next_token);
            output_tokens.push(next_token);
            generated_tokens += 1;

            if next_token == eos_token {
                break;
            }
        }

        let dt = start_gen.elapsed();
        println!(
            "\n{generated_tokens} tokens generated ({:.2} token/s)",
            generated_tokens as f64 / dt.as_secs_f64(),
        );

        self.tokenizer.decode(&output_tokens, true).map_err(E::msg)
    }
}

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Run on CPU rather than on GPU.
    #[arg(long)]
    cpu: bool,

    /// Enable CUDA kernels.
    #[arg(long)]
    use_flash_attn: bool,

    /// Prompt to process
    #[arg(long)]
    prompt: Option<String>,

    /// Temperature for generation (0 = deterministic)
    #[arg(long, default_value_t = 0.7)]
    temperature: f64,

    /// Nucleus sampling probability cutoff.
    #[arg(long)]
    top_p: Option<f64>,

    /// Only sample among the top K samples.
    #[arg(long)]
    top_k: Option<usize>,

    /// The seed to use when generating random samples.
    #[arg(long, default_value_t = 299792458)]
    seed: u64,

    /// The length of the sample to generate (in tokens).
    #[arg(long, short = 'n', default_value_t = DEFAULT_SAMPLE_LEN)]
    sample_len: usize,

    /// Embedding model path
    #[arg(long, default_value = "models/embeddings")]
    embedding_path: String,

    /// Enable episodic memory
    #[arg(long)]
    enable_memory: bool,

    /// Number of similar dialogues to retrieve
    #[arg(long, default_value_t = 3)]
    memory_top_k: usize,

    /// Persona name for the session
    #[arg(long, default_value = "assistant")]
    persona: String,

    /// Model ID to use
    #[arg(long)]
    model_id: Option<String>,

    /// Model revision
    #[arg(long, default_value = "main")]
    revision: String,

    /// Interactive mode - keep running for multiple queries
    #[arg(long)]
    interactive: bool,
}

fn build_prompt_with_context(
    user_input: &str,
    memory_context: &str,
    enable_memory: bool,
) -> String {
    if !enable_memory || memory_context.is_empty() {
        return format!("[INST] {} [/INST]", user_input);
    }

    format!(
        "[INST] {} [/INST]\n\nRelevant context from memory:\n{}",
        user_input, memory_context
    )
}

fn process_query(
    prompt: &str,
    pipeline: &mut UnifiedPipeline,
    dialogue_manager: &mut Option<DialogueManager>,
    args: &Args,
) -> Result<()> {
    let memory_context = if let Some(ref mut dm) = *dialogue_manager {
        let similar = dm.find_similar_dialogues(prompt, args.memory_top_k)?;
        eprintln!("DEBUG: Found {} similar dialogues", similar.len());
        for (i, s) in similar.iter().enumerate() {
            eprintln!("DEBUG: Similar[{}] = {}", i, s);
        }
        if !similar.is_empty() {
            println!("🧠 Found {} relevant memory entries", similar.len());
            similar.join("\n\n")
        } else {
            String::new()
        }
    } else {
        String::new()
    };

    let enhanced_prompt = build_prompt_with_context(prompt, &memory_context, args.enable_memory);

    println!("\n📝 User: {}", prompt);
    println!("\n🤖 Assistant:");

    let response = pipeline.run(&enhanced_prompt, args.sample_len)?;

    println!("{}", response);

    if let Some(ref mut dm) = *dialogue_manager {
        dm.add_exchange(prompt.to_string(), response.clone())?;
        let stats = dm.stats();
        println!(
            "\n💾 Memory: {} turns in current session",
            stats.current_session_turns
        );
    }

    Ok(())
}

fn main() -> Result<()> {
    let args = Args::parse();

    println!("🏛️ ZIGGURAT MIND - Initializing...");

    let device = select_device(args.cpu)?;
    println!("📱 Device: {:?}", device);

    println!("🧠 Loading embedding engine from: {}", args.embedding_path);
    let embedder: Arc<dyn Embedder> = if Path::new(&args.embedding_path).exists() {
        Arc::new(EmbeddingEngine::new(&args.embedding_path, device.clone())?)
    } else {
        anyhow::bail!("Embedding model not found at: {}", args.embedding_path);
    };
    println!(
        "✅ Embedding engine loaded (dim: {})",
        embedder.embedding_dim()
    );

    let mut dialogue_manager = if args.enable_memory {
        Some(DialogueManager::new(embedder.clone(), args.persona.clone()))
    } else {
        None
    };

    println!("🤖 Loading Mistral 7B...");

    let model_id = args
        .model_id
        .clone()
        .unwrap_or_else(|| "mistralai/Mistral-7B-Instruct-v0.2".to_string());

    let (tokenizer, filenames, config_path): (
        Tokenizer,
        Vec<std::path::PathBuf>,
        std::path::PathBuf,
    ) = if Path::new(&model_id).exists() && Path::new(&model_id).join("tokenizer.json").exists() {
        let local_path = std::path::Path::new(&model_id).to_path_buf();
        let tokenizer = Tokenizer::from_file(local_path.join("tokenizer.json")).map_err(E::msg)?;

        // Читаем safetensors index для определения файлов
        let index_path = local_path.join("model.safetensors.index.json");
        let index_content = std::fs::read_to_string(&index_path)?;
        let index: serde_json::Value = serde_json::from_str(&index_content)?;

        // Извлекаем уникальные имена файлов из weight_map
        let mut unique_files = std::collections::HashSet::<String>::new();
        if let Some(weight_map) = index.get("weight_map").and_then(|v| v.as_object()) {
            for file in weight_map.values() {
                if let Some(file_str) = file.as_str() {
                    unique_files.insert(file_str.to_string());
                }
            }
        }

        // Сортируем для последовательной загрузки
        let mut filenames: Vec<_> = unique_files.into_iter().collect();
        filenames.sort();

        (
            tokenizer,
            filenames.into_iter().map(|f| local_path.join(f)).collect(),
            local_path.join("config.json"),
        )
    } else {
        let api = Api::new()?;
        let revision = args.revision.clone();
        let repo = api.repo(Repo::with_revision(
            model_id.clone(),
            RepoType::Model,
            revision,
        ));
        let tokenizer_filename = repo.get("tokenizer.json")?;
        let tokenizer = Tokenizer::from_file(tokenizer_filename).map_err(E::msg)?;
        let filenames = hub_load_safetensors(&repo, "model.safetensors-index.json")?;
        (tokenizer, filenames, repo.get("config.json")?)
    };

    let config: Config = serde_json::from_slice(&std::fs::read(config_path)?)?;

    let dtype = if device.is_cuda() {
        DType::BF16
    } else {
        DType::F32
    };
    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&filenames, dtype, &device)? };
    let model = Mistral::new(&config, vb)?;

    println!("✅ Mistral 7B loaded");

    let mut pipeline = UnifiedPipeline::new(
        model,
        tokenizer,
        device,
        Some(args.temperature),
        args.top_p,
        args.top_k,
        1.1,
        64,
        args.seed,
    );

    if args.interactive {
        println!("\n🗣️ Interactive mode - type 'quit' to exit");
        println!("========================================");

        if let Some(ref initial_prompt) = args.prompt {
            process_query(initial_prompt, &mut pipeline, &mut dialogue_manager, &args)?;
        }

        loop {
            print!("\n📝 You: ");
            std::io::stdout().flush()?;

            let mut input = String::new();
            std::io::stdin().read_line(&mut input)?;
            let input = input.trim();

            if input.is_empty() {
                continue;
            }
            if input.eq_ignore_ascii_case("quit") || input.eq_ignore_ascii_case("exit") {
                println!("👋 Goodbye!");
                break;
            }

            if let Err(e) = process_query(input, &mut pipeline, &mut dialogue_manager, &args) {
                eprintln!("Error: {}", e);
            }
        }
    } else {
        let Some(ref prompt) = args.prompt else {
            eprintln!("Error: --prompt is required (or use --interactive)");
            std::process::exit(1);
        };
        let args_ref = &args;
        process_query(prompt, &mut pipeline, &mut dialogue_manager, args_ref)?;
    }

    Ok(())
}

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
use std::cmp::min;
use std::io::Write;
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
    /// Очищает KV кэш между запросами
    pub fn clear_cache(&mut self) {
        self.model.clear_kv_cache();
    }

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

    fn run(&mut self, prompt: &str, sample_len: usize, seed: u64) -> Result<String> {
        let mut tokens = self
            .tokenizer
            .encode(prompt, true)
            .map_err(E::msg)?
            .get_ids()
            .to_vec();

        eprintln!("DEBUG: Input tokens count: {}", tokens.len());

        let mut generated_tokens = 0usize;
        let eos_token = match self.tokenizer.get_vocab(false).get("</s>") {
            Some(&t) => t,
            None => 2,
        };

        let temperature = 0.;
        let sampling = if temperature <= 0. {
            Sampling::ArgMax
        } else {
            Sampling::All { temperature }
        };
        let _logits_processor = LogitsProcessor::from_sampling(seed, sampling);

        let start_gen = std::time::Instant::now();
        let mut output_tokens = Vec::new();

        for index in 0..sample_len {
            let start_pos = if index == 0 {
                0
            } else {
                tokens.len().saturating_sub(1)
            };
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

    /// Disable memory context after first exchange (workaround for Candle compatibility)
    #[arg(long)]
    disable_memory_context: bool,

    /// Number of similar dialogues to retrieve
    #[arg(long, default_value_t = 5)]
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

const MAX_DIALOGUE_LENGTH: usize = 100;

fn truncate_text(text: &str, max_chars: usize) -> String {
    let char_count = text.chars().count();
    if char_count <= max_chars {
        return text.to_string();
    }

    let char_pos = text.char_indices().nth(max_chars);
    match char_pos {
        Some((byte_pos, _)) => {
            let truncated = &text[..byte_pos];
            if let Some(last_newline) = truncated.rfind('\n') {
                truncated[..last_newline].to_string()
            } else if let Some(last_space) = truncated.rfind(' ') {
                truncated[..last_space].to_string() + "..."
            } else {
                truncated.to_string() + "..."
            }
        }
        None => text.to_string(),
    }
}

fn build_prompt_with_context(
    user_input: &str,
    memory_context: &str,
    current_context: &str,
    enable_memory: bool,
) -> String {
    if !enable_memory || (memory_context.is_empty() && current_context.is_empty()) {
        return format!("[INST] {} [/INST]", user_input);
    }

    if memory_context.is_empty() {
        return format!(
            "[INST] Current conversation:\n{}\n\nUser: {} [/INST]",
            current_context, user_input
        );
    }

    if current_context.is_empty() {
        return format!(
            "[INST] {} [/INST]\n\nMEMORY (use this to answer about user's preferences):\n{}",
            user_input, memory_context
        );
    }

    format!(
        "[INST] Current conversation:\n{}\n\nUser: {} [/INST]\n\nMEMORY (use this to answer about user's preferences):\n{}",
        current_context, user_input, memory_context
    )
}

fn process_query(
    prompt: &str,
    pipeline: &mut UnifiedPipeline,
    dialogue_manager: &mut Option<DialogueManager>,
    persistence_manager: &std::sync::Arc<totems::episodic::persistence::PersistenceManager>,
    embedder: &Arc<dyn crate::priests::embeddings::Embedder>,
    args: &Args,
) -> Result<()> {
    let (similar_dialogues, current_context) = if let Some(ref mut dm) = *dialogue_manager {
        if args.disable_memory_context {
            (String::new(), String::new())
        } else {
            let similar = dm.find_similar_dialogues(prompt, args.memory_top_k)?;
            eprintln!("DEBUG: Found {} similar dialogues", similar.len());
            for (i, s) in similar.iter().enumerate() {
                eprintln!("DEBUG: Similar[{}] = {}", i, s);
            }

            let current_ctx = dm.get_current_context(5);
            eprintln!(
                "DEBUG: Current session context length: {}",
                current_ctx.len()
            );

            let similar_text = if !similar.is_empty() {
                println!("🧠 Found {} relevant memory entries", similar.len());
                let truncated: Vec<String> = similar
                    .iter()
                    .map(|s| truncate_text(s, MAX_DIALOGUE_LENGTH))
                    .collect();
                truncated.join("\n\n")
            } else {
                String::new()
            };

            (similar_text, current_ctx)
        }
    } else {
        (String::new(), String::new())
    };

    let enhanced_prompt = build_prompt_with_context(
        prompt,
        &similar_dialogues,
        &current_context,
        args.enable_memory,
    );

    eprintln!("DEBUG: Enhanced prompt length: {}", enhanced_prompt.len());
    eprintln!(
        "DEBUG: Enhanced prompt preview: {}",
        &enhanced_prompt[..min(200, enhanced_prompt.len())]
    );

    println!("\n📝 User: {}", prompt);
    println!("\n🤖 Assistant:");

    let response = pipeline.run(&enhanced_prompt, args.sample_len, args.seed)?;

    println!("{}", response);

    if let Some(ref mut dm) = *dialogue_manager {
        dm.add_exchange(prompt.to_string(), response.clone())?;
        let stats = dm.stats();
        println!(
            "\n💾 Memory: {} turns in current session",
            stats.current_session_turns
        );

        if let Err(e) = persistence_manager.save_with_embeddings(dm, embedder.embedding_dim()) {
            eprintln!("WARNING: Failed to save memory: {}", e);
        }
    }

    Ok(())
}

fn resolve_path(path: &str) -> std::path::PathBuf {
    let path = std::path::Path::new(path);
    if path.is_absolute() {
        return path.to_path_buf();
    }

    let exe_path = std::env::current_exe().unwrap_or(std::path::PathBuf::from("."));
    let mut current = exe_path.as_path();

    while let Some(parent) = current.parent() {
        if parent.join("Cargo.toml").exists() {
            return parent.join(path);
        }
        current = parent;
    }

    std::env::current_dir()
        .unwrap_or(std::path::PathBuf::from("."))
        .join(path)
}

fn main() -> Result<()> {
    let args = Args::parse();

    println!("🏛️ ZIGGURAT MIND - Initializing...");

    let device = select_device(args.cpu)?;
    println!("📱 Device: {:?}", device);

    let embedding_path = resolve_path(&args.embedding_path);
    println!(
        "🧠 Loading embedding engine from: {}",
        embedding_path.display()
    );

    if !embedding_path.exists() {
        anyhow::bail!(
            "Embedding model not found at: {}\n\
             Current directory: {:?}\n\
             Resolved from: {:?}",
            embedding_path.display(),
            std::env::current_dir().unwrap_or_default(),
            args.embedding_path
        );
    }

    let embedder: Arc<dyn Embedder> = Arc::new(EmbeddingEngine::new(
        embedding_path.to_str().unwrap_or(&args.embedding_path),
        device.clone(),
    )?);
    println!(
        "✅ Embedding engine loaded (dim: {})",
        embedder.embedding_dim()
    );

    // Инициализируем persistence manager
    let persistence_manager = std::sync::Arc::new(
        totems::episodic::persistence::PersistenceManager::new(Some(&resolve_path("")), true)?,
    );

    let mut dialogue_manager = if args.enable_memory {
        match persistence_manager.load_with_embeddings(embedder.clone(), args.persona.clone())? {
            Some((manager, sessions)) => {
                println!("💾 Loaded {} saved sessions", sessions.len());
                let stats = manager.stats();
                println!("📊 Vector store: {} entries", stats.total_turns);
                Some(manager)
            }
            None => Some(DialogueManager::new(embedder.clone(), args.persona.clone())),
        }
    } else {
        None
    };

    println!("🤖 Loading Mistral 7B...");

    let model_id = args
        .model_id
        .clone()
        .unwrap_or_else(|| "mistralai/Mistral-7B-Instruct-v0.2".to_string());

    let local_mistral_path = resolve_path("models/mistral-7b-instruct");
    let use_local_path = local_mistral_path.exists()
        && local_mistral_path.join("tokenizer.json").exists()
        && local_mistral_path
            .join("model.safetensors.index.json")
            .exists();

    let (tokenizer, filenames, config_path): (
        Tokenizer,
        Vec<std::path::PathBuf>,
        std::path::PathBuf,
    ) = if use_local_path {
        let local_path = local_mistral_path;
        eprintln!("DEBUG: Loading from local path: {:?}", local_path);

        let tokenizer = Tokenizer::from_file(local_path.join("tokenizer.json")).map_err(E::msg)?;

        let index_path = local_path.join("model.safetensors.index.json");
        let index_content = std::fs::read_to_string(&index_path)?;
        let index: serde_json::Value = serde_json::from_str(&index_content)?;

        let mut unique_files = std::collections::HashSet::<String>::new();
        if let Some(weight_map) = index.get("weight_map").and_then(|v| v.as_object()) {
            for file in weight_map.values() {
                if let Some(file_str) = file.as_str() {
                    unique_files.insert(file_str.to_string());
                }
            }
        }

        if unique_files.is_empty() {
            anyhow::bail!("No weight files found in safetensors index");
        }

        let mut filenames: Vec<_> = unique_files.into_iter().collect();
        filenames.sort();

        eprintln!(
            "DEBUG: Found {} weight files: {:?}",
            filenames.len(),
            filenames
        );

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

    // Validate config for Mistral 7B
    if config.hidden_size != 4096 {
        eprintln!(
            "WARNING: Expected hidden_size=4096 for Mistral 7B, got {}. This may cause issues.",
            config.hidden_size
        );
    }
    if config.num_attention_heads != 32 {
        eprintln!(
            "WARNING: Expected num_attention_heads=32 for Mistral 7B, got {}.",
            config.num_attention_heads
        );
    }
    if config.num_hidden_layers != 32 {
        eprintln!(
            "WARNING: Expected num_hidden_layers=32 for Mistral 7B, got {}.",
            config.num_hidden_layers
        );
    }

    eprintln!(
        "DEBUG: Config loaded - hidden_size: {}, num_heads: {}, num_layers: {}",
        config.hidden_size, config.num_attention_heads, config.num_hidden_layers
    );

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
            pipeline.clear_cache();
            process_query(
                initial_prompt,
                &mut pipeline,
                &mut dialogue_manager,
                &persistence_manager,
                &embedder,
                &args,
            )?;
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
                // Сохраняем память перед выходом
                if let Some(ref dm) = dialogue_manager {
                    if let Err(e) =
                        persistence_manager.save_with_embeddings(dm, embedder.embedding_dim())
                    {
                        eprintln!("WARNING: Failed to save memory on exit: {}", e);
                    } else {
                        println!("💾 Memory saved to disk");
                    }
                }
                println!("👋 Goodbye!");
                break;
            }

            pipeline.clear_cache();

            if let Err(e) = process_query(
                input,
                &mut pipeline,
                &mut dialogue_manager,
                &persistence_manager,
                &embedder,
                &args,
            ) {
                eprintln!("Error: {}", e);
            }
        }
    } else {
        let Some(ref prompt) = args.prompt else {
            eprintln!("Error: --prompt is required (or use --interactive)");
            std::process::exit(1);
        };
        pipeline.clear_cache();
        let args_ref = &args;
        process_query(
            prompt,
            &mut pipeline,
            &mut dialogue_manager,
            &persistence_manager,
            &embedder,
            args_ref,
        )?;

        // Сохраняем память после выполнения
        if let Some(ref dm) = dialogue_manager {
            if let Err(e) = persistence_manager.save_with_embeddings(dm, embedder.embedding_dim()) {
                eprintln!("WARNING: Failed to save memory: {}", e);
            } else {
                println!("💾 Memory saved to disk");
            }
        }
    }

    Ok(())
}

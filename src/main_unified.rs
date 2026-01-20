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
use std::sync::Arc;
use tokenizers::Tokenizer;

use crate::priests::device::select_device;
use crate::priests::embeddings::{Embedder, EmbeddingEngine};
use crate::totems::episodic::DialogueManager;
use crate::totems::semantic::{SemanticMemoryManager};
use crate::totems::semantic::concept::ConceptCategory;
use crate::utils::hub_load_safetensors;

const DEFAULT_SAMPLE_LEN: usize = 2048;

struct ConceptExtractorImpl {
    pipeline: std::sync::Arc<std::sync::Mutex<UnifiedPipeline>>,
}

impl ConceptExtractorImpl {
    fn new(pipeline: std::sync::Arc<std::sync::Mutex<UnifiedPipeline>>) -> Self {
        Self { pipeline }
    }
}

impl totems::semantic::ConceptExtractor for ConceptExtractorImpl {
    fn extract(
        &mut self,
        user_query: &str,
        assistant_response: &str,
        _session_id: &str,
    ) -> Result<totems::semantic::ExtractionResult> {
        let prompt = format!(
            r#"<s>[INST] You are a knowledge extraction assistant. Extract facts, preferences, rules, skills from the dialogue below.

Return ONLY a JSON array. Each object must have:
- "text": concise knowledge statement (max 100 chars)
- "category": one of facts, rules, preferences, skills, general
- "confidence": float between 0.0 and 1.0

If no knowledge found, return empty array [].

Dialogue:
User: {user_query}
Assistant: {assistant_response}

Output format: [{{"text":"...","category":"...","confidence":0.8}}]
NO markdown, NO explanations, NO text before or after. Only JSON.
[/INST]</s>"#,
            user_query = user_query,
            assistant_response = assistant_response
        );

        let response = {
            let mut pipeline = self.pipeline.lock().unwrap();
            pipeline.clear_cache();
            pipeline.run(&prompt, 200, 0)?
        };

        let cleaned = response
            .trim()
            .trim_start_matches("```json")
            .trim_start_matches("```")
            .trim_end_matches("```")
            .trim()
            .to_string();

        let parse_json = |s: &str| -> Result<Vec<serde_json::Value>, _> {
            serde_json::from_str(s)
        };

        let concepts: Vec<serde_json::Value> = match parse_json(&cleaned) {
            Ok(c) => c,
            Err(e) => {
                eprintln!("DEBUG: Direct JSON parse failed: {}", e);
                let without_brackets = cleaned.trim_start_matches('[').trim_end_matches(']');
                match parse_json(&format!("[{}]", without_brackets)) {
                    Ok(c) => c,
                    Err(e2) => {
                        eprintln!("DEBUG: Bracket-wrapped parse also failed: {} | Input: {}", e2, cleaned);
                        return Ok(Vec::new());
                    }
                }
            }
        };

        let mut results = Vec::new();
        for value in concepts {
            let text = match value.get("text").and_then(|v| v.as_str()) {
                Some(t) => t.to_string(),
                None => continue,
            };

            let category = value
                .get("category")
                .and_then(|v| v.as_str())
                .map(|s| s.to_string())
                .unwrap_or_else(|| "general".to_string());

            let confidence: f32 = value
                .get("confidence")
                .and_then(|v| v.as_f64())
                .unwrap_or(0.5) as f32;

            results.push((text, category, confidence));
        }

        Ok(results)
    }
}

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

    /// Enable semantic memory (facts, rules, preferences)
    #[arg(long)]
    enable_semantic: bool,

    /// Disable memory context after first exchange (workaround for Candle compatibility)
    #[arg(long)]
    disable_memory_context: bool,

    /// Number of similar dialogues to retrieve
    #[arg(long, default_value_t = 5)]
    memory_top_k: usize,

    /// Number of semantic concepts to retrieve
    #[arg(long, default_value_t = 10)]
    semantic_top_k: usize,

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

    /// Maximum number of sessions to keep in memory
    #[arg(long, default_value_t = 50)]
    max_sessions: usize,
}

const MAX_DIALOGUE_LENGTH: usize = 100;

fn get_memory_mb() -> u64 {
    #[cfg(target_os = "linux")]
    {
        if let Ok(status) = std::fs::read_to_string("/proc/self/status") {
            for line in status.lines() {
                if line.starts_with("VmRSS:") {
                    if let Some(val) = line.split_whitespace().nth(1) {
                        return val.parse::<u64>().unwrap_or(0) / 1024;
                    }
                }
            }
        }
    }
    0
}

fn get_gpu_memory_mb() -> Option<u64> {
    std::process::Command::new("nvidia-smi")
        .args(&["--query-gpu=memory.used", "--format=csv,noheader,nounits"])
        .output()
        .ok()
        .and_then(|output| {
            let stdout = String::from_utf8_lossy(&output.stdout);
            stdout.trim().parse::<u64>().ok()
        })
}

fn log_memory_usage(_label: &str) {
    let mem_mb = get_memory_mb();
    if mem_mb > 0 {
        eprintln!("DEBUG [{}]: RAM: {} MB", _label, mem_mb);
    }
    if let Some(gpu_mb) = get_gpu_memory_mb() {
        eprintln!("DEBUG [{}]: VRAM: {} MB", _label, gpu_mb);
    }
}

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
    episodic_context: &str,
    semantic_context: &str,
    current_context: &str,
    enable_memory: bool,
) -> String {
    if !enable_memory || (episodic_context.is_empty() && semantic_context.is_empty() && current_context.is_empty()) {
        return format!("[INST] {} [/INST]", user_input);
    }

    let mut context_parts = Vec::new();

    if !current_context.is_empty() {
        context_parts.push(format!("Current conversation:\n{}", current_context));
    }

    if !semantic_context.is_empty() {
        context_parts.push(format!("KNOWLEDGE:\n{}", semantic_context));
    }

    if !episodic_context.is_empty() {
        context_parts.push(format!("RELATED MEMORY:\n{}", episodic_context));
    }

    let combined_context = context_parts.join("\n\n");

    format!(
        "[INST] {} [/INST]\n\n{}",
        user_input, combined_context
    )
}

fn process_query(
    prompt: &str,
    pipeline_arc: &std::sync::Arc<std::sync::Mutex<UnifiedPipeline>>,
    dialogue_manager: &mut Option<DialogueManager>,
    semantic_manager: &mut Option<std::sync::Mutex<SemanticMemoryManager>>,
    persistence_manager: &std::sync::Arc<totems::episodic::persistence::PersistenceManager>,
    embedder: &Arc<dyn crate::priests::embeddings::Embedder>,
    args: &Args,
) -> Result<()> {
    log_memory_usage("process_query start");

    let (similar_dialogues, current_context) = if let Some(ref mut dm) = *dialogue_manager {
        if args.disable_memory_context {
            (String::new(), String::new())
        } else {
            let similar = dm.find_similar_dialogues(prompt, args.memory_top_k)?;
            eprintln!("DEBUG: Found {} similar dialogues", similar.len());

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

    let semantic_context = if args.enable_semantic {
        if let Some(ref sm) = *semantic_manager {
            let sm = sm.lock().unwrap();
            let results = sm.search_by_text(prompt, args.semantic_top_k);
            if !results.is_empty() {
                println!("📚 Found {} relevant concepts", results.len());
                let context: Vec<String> = results
                    .iter()
                    .map(|(sim, concept)| {
                        format!("[{} {:.2}] {}", concept.category, sim, truncate_text(&concept.text, 200))
                    })
                    .collect();
                context.join("\n")
            } else {
                String::new()
            }
        } else {
            String::new()
        }
    } else {
        String::new()
    };

    let enhanced_prompt = build_prompt_with_context(
        prompt,
        &similar_dialogues,
        &semantic_context,
        &current_context,
        args.enable_memory || args.enable_semantic,
    );

    eprintln!("DEBUG: Enhanced prompt length: {}", enhanced_prompt.len());

    println!("\n📝 User: {}", prompt);
    println!("\n🤖 Assistant:");

    let response = pipeline_arc.lock().unwrap().run(&enhanced_prompt, args.sample_len, args.seed)?;

    println!("{}", response);

    let session_id = dialogue_manager
        .as_ref()
        .map(|dm| dm.current_session().id.to_string())
        .unwrap_or_else(|| "unknown".to_string());

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

    if args.enable_semantic {
        if let Some(ref sm) = *semantic_manager {
            let mut sm = sm.lock().unwrap();
            if let Err(e) = sm.extract_from_dialogue(prompt, &response, &session_id) {
                eprintln!("DEBUG: Failed to extract concepts: {}", e);
            }
            eprintln!("DEBUG: Semantic memory now has {} concepts", sm.count());
        }
    }

    log_memory_usage("process_query end");
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

    let mut semantic_manager = if args.enable_semantic {
        let semantic_persistence = totems::semantic::persistence::SemanticPersistenceManager::new(
            Some(&resolve_path("memory_data")),
        )?;
        let sm = SemanticMemoryManager::new(embedder.clone(), semantic_persistence)?;
        let count = sm.count();
        if count > 0 {
            println!("📚 Loaded {} semantic concepts", count);
        }
        Some(std::sync::Mutex::new(sm))
    } else {
        None
    };

    println!("🤖 Loading Mistral 7B...");

    // Show device selection status
    if device.is_cuda() {
        println!("🚀 Device: GPU (CUDA) - using VRAM, not system RAM");
    } else {
        let mem_mb = get_memory_mb();
        println!("💻 Device: CPU - System RAM: {} MB", mem_mb);
        if mem_mb > 0 && mem_mb < 16000 {
            eprintln!("⚠️  WARNING: CPU mode requires ~16GB RAM. GPU recommended!");
        }
    }

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
        let local_path = local_mistral_path.clone();
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

        (tokenizer, filenames.into_iter().map(|f| local_path.join(f)).collect(), local_path.join("config.json"))
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

    // Check available memory before loading model
    let available_memory_mb = get_memory_mb();
    let is_cuda = device.is_cuda();

    if !is_cuda && available_memory_mb > 0 {
        let required_memory_mb = 18000; // ~18GB for full model + overhead
        if available_memory_mb < required_memory_mb {
            eprintln!("\n⚠️  WARNING: Low memory situation!");
            eprintln!("   Available: {} MB", available_memory_mb);
            eprintln!("   Required:  ~{} MB for Mistral 7B", required_memory_mb);
            eprintln!("\n   Options:");
            eprintln!("   1. Use GPU (CUDA) - recommended");
            eprintln!("   2. Close other applications to free RAM");
            eprintln!("   3. Use a smaller model (7B quantized)");
            eprintln!("\n   Continuing anyway, but may encounter OOM...\n");
        }
    }

    log_memory_usage("before_model_load");

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
        println!("🎯 Using GPU (BF16 precision)");
        DType::BF16
    } else {
        // CPU fallback: use quantized types to save memory
        let available_memory_mb = get_memory_mb();
        let mem_threshold = 16000; // 16GB threshold

        if available_memory_mb > mem_threshold {
            println!("💻 CPU mode: {} MB RAM available, using F32", available_memory_mb);
            DType::F32
        } else {
            // Low memory: warn user
            if available_memory_mb > 0 {
                eprintln!("⚠️  WARNING: Only {} MB RAM available!", available_memory_mb);
                eprintln!("    Mistral 7B requires ~16GB on CPU. Consider using GPU.");
            }
            println!("💻 CPU mode: F32 (full precision)");
            DType::F32
        }
    };
    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&filenames, dtype, &device)? };
    let model = Mistral::new(&config, vb)?;

    let pipeline_arc: std::sync::Arc<std::sync::Mutex<UnifiedPipeline>> =
        std::sync::Arc::new(std::sync::Mutex::new(UnifiedPipeline::new(
            model,
            tokenizer,
            device.clone(),
            Some(args.temperature),
            args.top_p,
            args.top_k,
            1.1,
            64,
            args.seed,
        )));

    log_memory_usage("after_model_load");

    if device.is_cuda() {
        println!("✅ Mistral 7B loaded on GPU (using VRAM)");
    } else {
        let mem_mb = get_memory_mb();
        println!("✅ Mistral 7B loaded on CPU (using {} MB RAM)", mem_mb);
        if mem_mb > 20000 {
            println!("💡 Tip: Use --features cuda for GPU inference (faster + less RAM)");
        }
    }

    if args.enable_semantic {
        let extractor = Arc::new(std::sync::Mutex::new(ConceptExtractorImpl::new(pipeline_arc.clone())));

        if let Some(ref mut sm) = semantic_manager {
            let mut sm = sm.lock().unwrap();
            sm.set_extractor(extractor);
            drop(sm);
        }
    }

    if args.interactive {
        println!("\n🗣️ Interactive mode - type 'quit' to exit");
        println!("   /semantic - Manage semantic memory");
        println!("   /mem - Show memory usage");
        println!("========================================");

        if let Some(ref initial_prompt) = args.prompt {
            pipeline_arc.lock().unwrap().clear_cache();
            process_query(
                initial_prompt,
                &pipeline_arc,
                &mut dialogue_manager,
                &mut semantic_manager,
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
                        println!("💾 Episodic memory saved to disk");
                    }
                }
                if let Some(ref sm) = semantic_manager {
                    let sm = sm.lock().unwrap();
                    let count = sm.count();
                    if count > 0 {
                        println!("📚 Semantic memory: {} concepts saved", count);
                    }
                }
                println!("👋 Goodbye!");
                break;
            }

            pipeline_arc.lock().unwrap().clear_cache();

            if input.starts_with("/semantic") || input.starts_with("/s") {
                if !args.enable_semantic {
                    println!("Semantic memory is disabled. Use --enable-semantic to enable.");
                    continue;
                }
                if let Some(ref sm) = semantic_manager {
                    handle_semantic_command(input, sm);
                } else {
                    println!("Semantic memory not initialized.");
                }
                continue;
            }

            if input == "/mem" || input == "/memory" {
                let mem_mb = get_memory_mb();
                if mem_mb > 0 {
                    println!("💻 RAM: {} MB", mem_mb);
                }
                if let Some(gpu_mb) = get_gpu_memory_mb() {
                    println!("🚀 VRAM: {} MB", gpu_mb);
                }
                continue;
            }

            if let Err(e) = process_query(
                input,
                &pipeline_arc,
                &mut dialogue_manager,
                &mut semantic_manager,
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
        pipeline_arc.lock().unwrap().clear_cache();
        let args_ref = &args;
        process_query(
            prompt,
            &pipeline_arc,
            &mut dialogue_manager,
            &mut semantic_manager,
            &persistence_manager,
            &embedder,
            args_ref,
        )?;

        // Сохраняем память после выполнения
        if let Some(ref dm) = dialogue_manager {
            if let Err(e) = persistence_manager.save_with_embeddings(dm, embedder.embedding_dim()) {
                eprintln!("WARNING: Failed to save memory: {}", e);
            } else {
                println!("💾 Episodic memory saved to disk");
            }
        }
        if let Some(ref sm) = semantic_manager {
            let sm = sm.lock().unwrap();
            let count = sm.count();
            if count > 0 {
                println!("📚 Semantic memory: {} concepts saved", count);
            }
        }
    }

    Ok(())
}

fn handle_semantic_command(input: &str, sm: &std::sync::Mutex<totems::semantic::SemanticMemoryManager>) {
    let parts: Vec<&str> = input.split_whitespace().collect();
    let command = parts.get(1).map(|s| *s).unwrap_or("");

    match command {
        "help" | "h" => {
            println!(r#"
 📚 Semantic Memory Commands:
   /semantic help           - Show this help
   /semantic list [n]       - List concepts (default: 10)
   /semantic list <category> [n] - List concepts by category (facts|rules|preferences|skills|general)
   /semantic stats          - Show statistics
   /semantic search <query> - Search concepts
   /semantic add "<text>" <category> [confidence] - Add new concept
   /semantic vote <id> <up|down> - Vote to adjust concept confidence
   /semantic delete <id>    - Delete concept by ID
   /semantic merge          - Merge duplicate concepts
   /semantic get <id>       - Show concept by ID
   Short form: /s instead of /semantic
"#);
        }

        "list" | "l" => {
            let sm = sm.lock().unwrap();
            if parts.len() >= 3 {
                let cat_arg = parts[2].to_lowercase();
                let category = match cat_arg.as_str() {
                    "facts" | "rules" | "preferences" | "pref" | "skills" | "general" => {
                        let category = match cat_arg.as_str() {
                            "facts" => ConceptCategory::Facts,
                            "rules" => ConceptCategory::Rules,
                            "preferences" | "pref" => ConceptCategory::Preferences,
                            "skills" => ConceptCategory::Skills,
                            "general" => ConceptCategory::General,
                            _ => unreachable!()
                        };
                        let limit = parts.get(3).and_then(|s| s.parse::<usize>().ok()).unwrap_or(20);
                        let concepts = sm.list_by_category(&category, limit);
                        let total_in_cat = sm.get_concepts_by_category(&category).len();
                        if concepts.is_empty() {
                            println!("No concepts found in category '{}'.", category);
                        } else {
                            println!("📚 Concepts in {} (showing {} of {}):", category, concepts.len(), total_in_cat);
                            for (i, c) in concepts.iter().enumerate() {
                                println!("{}. [{}] {} (conf: {:.2})", i + 1, c.category, truncate_text(&c.text, 60), c.confidence);
                            }
                        }
                        return;
                    }
                    _ => {
                        let limit = parts.get(2).and_then(|s| s.parse::<usize>().ok()).unwrap_or(10);
                        let concepts = sm.list_concepts(limit, 0);
                        if concepts.is_empty() {
                            println!("No concepts found.");
                        } else {
                            println!("📚 Concepts (showing {} of {}):", concepts.len(), sm.count());
                            for (i, c) in concepts.iter().enumerate() {
                                println!("{}. [{}] {} (conf: {:.2})", i + 1, c.category, truncate_text(&c.text, 60), c.confidence);
                            }
                        }
                    }
                };
            } else {
                let limit = parts.get(2).and_then(|s| s.parse::<usize>().ok()).unwrap_or(10);
                let concepts = sm.list_concepts(limit, 0);
                if concepts.is_empty() {
                    println!("No concepts found.");
                } else {
                    println!("📚 Concepts (showing {} of {}):", concepts.len(), sm.count());
                    for (i, c) in concepts.iter().enumerate() {
                        println!("{}. [{}] {} (conf: {:.2})", i + 1, c.category, truncate_text(&c.text, 60), c.confidence);
                    }
                }
            }
        }

        "stats" | "st" => {
            let sm = sm.lock().unwrap();
            println!("{}", sm.stats_pretty());
        }

        "search" | "s" => {
            if parts.len() < 3 {
                println!("Usage: /semantic search <query>");
                return;
            }
            let query = parts[2..].join(" ");
            let sm = sm.lock().unwrap();
            println!("{}", sm.search_pretty(&query, 10));
        }

        "add" | "a" => {
            if parts.len() < 4 {
                println!(r#"Usage: /semantic add "<text>" <category> [confidence]
Categories: facts, rules, preferences, skills"#);
                return;
            }
            let mut text = parts[2].to_string();
            let mut idx = 3;
            if text.starts_with('"') {
                text = text.trim_start_matches('"').to_string();
                while idx < parts.len() && !parts[idx].ends_with('"') {
                    text.push(' ');
                    text.push_str(parts[idx]);
                    idx += 1;
                }
                if idx < parts.len() {
                    text.push(' ');
                    text.push_str(parts[idx].trim_end_matches('"'));
                    idx += 1;
                }
            }
            if idx >= parts.len() {
                println!(r#"Usage: /semantic add "<text>" <category> [confidence]"#);
                return;
            }
            let category = parts[idx];
            idx += 1;
            let confidence = parts.get(idx).and_then(|s| s.parse::<f32>().ok()).unwrap_or(0.5);

            let cat = match category {
                "facts" => ConceptCategory::Facts,
                "rules" => ConceptCategory::Rules,
                "preferences" | "pref" => ConceptCategory::Preferences,
                "skills" => ConceptCategory::Skills,
                _ => {
                    println!("Unknown category: {}. Use: facts, rules, preferences, skills", category);
                    return;
                }
            };

            let mut sm = sm.lock().unwrap();
            match sm.add_concept(text.to_string(), cat, "manual".to_string(), Some(confidence)) {
                Ok(c) => println!("✅ Added concept: {} ({})", c.id, c.category),
                Err(e) => println!("❌ Error: {}", e),
            }
        }

        "delete" | "del" | "remove" | "rm" => {
            if parts.len() < 3 {
                println!("Usage: /semantic delete <id>");
                return;
            }
            let id = parts[2];
            let mut sm = sm.lock().unwrap();
            if let Ok(uuid) = uuid::Uuid::parse_str(id) {
                if sm.remove_concept(uuid) {
                    println!("✅ Deleted concept: {}", id);
                } else {
                    println!("❌ Concept not found: {}", id);
                }
            } else {
                println!("Invalid UUID: {}", id);
            }
        }

        "merge" => {
            let mut sm = sm.lock().unwrap();
            match sm.merge_similar(0.8) {
                Ok(count) => println!("✅ Merged {} duplicate concepts", count),
                Err(e) => println!("❌ Error: {}", e),
            }
        }

        "get" => {
            if parts.len() < 3 {
                println!("Usage: /semantic get <id>");
                return;
            }
            let sm = sm.lock().unwrap();
            if let Some(c) = sm.get_concept_by_id(parts[2]) {
                println!("{}", sm.format_concept(c));
            } else {
                println!("Concept not found: {}", parts[2]);
            }
        }

        "vote" | "v" => {
            if parts.len() < 4 {
                println!(r#"Usage: /semantic vote <id> <up|down>
  /semantic vote <id> up   - Increase confidence by 0.1
  /semantic vote <id> down - Decrease confidence by 0.1"#);
                return;
            }
            let id = parts[2];
            let direction = parts[3].to_lowercase();
            let delta = match direction.as_str() {
                "up" | "u" | "+" => 0.1,
                "down" | "d" | "-" => -0.1,
                _ => {
                    println!("Invalid vote direction: {}. Use 'up' or 'down'", parts[3]);
                    return;
                }
            };
            let mut sm = sm.lock().unwrap();
            if let Ok(uuid) = uuid::Uuid::parse_str(id) {
                if let Err(e) = sm.update_concept_confidence(uuid, delta) {
                    println!("❌ Error: {}", e);
                } else {
                    if let Some(c) = sm.get_concept(uuid) {
                        println!("✅ Voted {} on concept: confidence = {:.2}", if delta > 0.0 { "UP" } else { "DOWN" }, c.confidence);
                    }
                }
            } else {
                println!("Invalid UUID: {}", id);
            }
        }

        "" => {
            let sm = sm.lock().unwrap();
            println!("📚 Semantic memory: {} concepts", sm.count());
        }

        _ => {
            println!("Unknown command: {}. Use /semantic help", command);
        }
    }
}

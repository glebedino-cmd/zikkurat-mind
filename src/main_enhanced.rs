mod logos;
mod priests;
mod totems;
mod utils;

use crate::logos::tokenizer::TokenOutputStream;
use crate::priests::embeddings::EmbeddingEngine;
use crate::totems::{PersistenceFormat, PersistenceManager, UnifiedMemoryManager};
use crate::utils::hub_load_safetensors;
use tracing_chrome::ChromeLayerBuilder;
use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::util::SubscriberInitExt;

#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

#[cfg(feature = "accelerate")]
extern crate accelerate_src;

use anyhow::{Error as E, Result};
use clap::Parser;
use serde::{Deserialize, Serialize};
use std::path::Path;
use std::sync::Arc;

use candle_transformers::models::mistral::{Config, Model as Mistral};
use candle_transformers::models::quantized_mistral::Model as QMistral;

#[derive(Serialize, Deserialize, Debug, Default)]
struct MistralConfig {
    #[serde(default)]
    temperature: Option<f64>,

    #[serde(default)]
    top_p: Option<f64>,

    #[serde(default)]
    top_k: Option<usize>,

    #[serde(default)]
    repeat_penalty: Option<f32>,

    #[serde(default)]
    repeat_last_n: Option<usize>,
}

use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::generation::{LogitsProcessor, Sampling};
use hf_hub::{api::sync::Api, Repo, RepoType};
use tokenizers::Tokenizer;

enum Model {
    Mistral(Mistral),
    Quantized(QMistral),
}

struct TextGeneration {
    model: Model,
    device: Device,
    tokenizer: TokenOutputStream,
    logits_processor: LogitsProcessor,
    repeat_penalty: f32,
    repeat_last_n: usize,
}

impl TextGeneration {
    #[allow(clippy::too_many_arguments)]
    fn new(
        model: Model,
        tokenizer: Tokenizer,
        seed: u64,
        temp: Option<f64>,
        top_p: Option<f64>,
        top_k: Option<usize>,
        repeat_penalty: f32,
        repeat_last_n: usize,
        device: &Device,
    ) -> Self {
        let logits_processor = {
            let temperature = temp.unwrap_or(0.);
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
            LogitsProcessor::from_sampling(seed, sampling)
        };

        Self {
            model,
            tokenizer: TokenOutputStream::new(tokenizer),
            logits_processor,
            repeat_penalty,
            repeat_last_n,
            device: device.clone(),
        }
    }

    fn run(&mut self, prompt: &str, sample_len: usize) -> Result<String> {
        use std::io::Write;
        self.tokenizer.clear();
        let mut tokens = self
            .tokenizer
            .tokenizer()
            .encode(prompt, true)
            .map_err(E::msg)?
            .get_ids()
            .to_vec();
        for &t in tokens.iter() {
            if let Some(t) = self.tokenizer.next_token(t)? {
                print!("{t}")
            }
        }
        std::io::stdout().flush()?;

        let mut generated_tokens = 0usize;
        let eos_token = match self.tokenizer.get_token("</s>") {
            Some(token) => token,
            None => anyhow::bail!("cannot find </s> token"),
        };
        let start_gen = std::time::Instant::now();
        let mut output_tokens = Vec::new();

        for index in 0..sample_len {
            let context_size = if index > 0 { 1 } else { tokens.len() };
            let start_pos = tokens.len().saturating_sub(context_size);
            let ctxt = &tokens[start_pos..];
            let input = Tensor::new(ctxt, &self.device)?.unsqueeze(0)?;
            let logits = match &mut self.model {
                Model::Mistral(m) => m.forward(&input, start_pos)?,
                Model::Quantized(m) => m.forward(&input, start_pos)?,
            };
            let logits = logits.squeeze(0)?.squeeze(0)?.to_dtype(DType::F32)?;
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
            if let Some(t) = self.tokenizer.next_token(next_token)? {
                print!("{t}");
                std::io::stdout().flush()?;
            }
        }

        let dt = start_gen.elapsed();
        if let Some(rest) = self.tokenizer.decode_rest().map_err(E::msg)? {
            print!("{rest}");
        }
        std::io::stdout().flush()?;
        println!(
            "\n{generated_tokens} tokens generated ({:.2} token/s)",
            generated_tokens as f64 / dt.as_secs_f64(),
        );

        // Декодируем сгенерированный текст
        let generated_text = self
            .tokenizer
            .tokenizer()
            .decode(&output_tokens, true)
            .map_err(E::msg)?;

        Ok(generated_text)
    }
}

#[derive(Clone, Debug, Copy, PartialEq, Eq, clap::ValueEnum)]
enum Which {
    #[value(name = "7b-v0.1")]
    Mistral7bV01,
    #[value(name = "7b-v0.2")]
    Mistral7bV02,
    #[value(name = "7b-instruct-v0.1")]
    Mistral7bInstructV01,
    #[value(name = "7b-instruct-v0.2")]
    Mistral7bInstructV02,
    #[value(name = "7b-maths-v0.1")]
    Mathstral7bV01,
    #[value(name = "nemo-2407")]
    MistralNemo2407,
    #[value(name = "nemo-instruct-2407")]
    MistralNemoInstruct2407,
}

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Run on CPU rather than on GPU.
    #[arg(long)]
    cpu: bool,

    /// Enable tracing (generates a trace-timestamp.json file).
    #[arg(long)]
    tracing: bool,

    #[arg(long)]
    use_flash_attn: bool,

    #[arg(long)]
    prompt: Option<String>,

    /// Show current configuration and exit
    #[arg(long)]
    show_config: bool,

    /// Save current parameters to config file
    #[arg(long)]
    save_config: bool,

    /// The temperature used to generate samples.
    #[arg(long)]
    temperature: Option<f64>,

    /// Nucleus sampling probability cutoff.
    #[arg(long)]
    top_p: Option<f64>,

    /// Only sample among the top K samples.
    #[arg(long)]
    top_k: Option<usize>,

    /// The seed to use when generating random samples.
    #[arg(long, default_value_t = 299792458)]
    seed: u64,

    /// The length of sample to generate (in tokens).
    #[arg(long, short = 'n', default_value_t = 10000)]
    sample_len: usize,

    /// The model size to use.
    #[arg(long, default_value = "7b-v0.1")]
    which: Which,

    #[arg(long)]
    model_id: Option<String>,

    #[arg(long, default_value = "main")]
    revision: String,

    #[arg(long)]
    tokenizer_file: Option<String>,

    #[arg(long)]
    config_file: Option<String>,

    #[arg(long)]
    weight_files: Option<String>,

    #[arg(long)]
    quantized: bool,

    /// Penalty to be applied for repeating tokens, 1. means no penalty.
    #[arg(long, default_value_t = 1.1)]
    repeat_penalty: f32,

    /// The context size to consider for repeat penalty.
    #[arg(long, default_value_t = 64)]
    repeat_last_n: usize,

    /// Use the slower dmmv cuda kernel.
    #[arg(long)]
    force_dmmv: bool,

    /// Enable unified memory system (experimental)
    #[arg(long)]
    enable_memory: bool,

    /// Embedding model path for memory system
    #[arg(long, default_value = "models/embeddings")]
    embedding_model: String,

    /// Number of similar episodes to retrieve (0 = disable)
    #[arg(long, default_value_t = 3)]
    memory_episodes_count: usize,

    /// Number of concepts to retrieve (0 = disable)
    #[arg(long, default_value_t = 2)]
    memory_concepts_count: usize,

    /// Persona name for session
    #[arg(long, default_value = "assistant")]
    persona: String,

    /// Memory data directory
    #[arg(long, default_value = "data")]
    memory_data_dir: String,

    /// Persistence format (json, binary, hybrid)
    #[arg(long, default_value = "hybrid")]
    persistence_format: String,

    /// Auto-save interval (operations)
    #[arg(long, default_value_t = 10)]
    auto_save_interval: usize,

    /// Export memory and exit
    #[arg(long)]
    export_memory: bool,

    /// Import memory from file
    #[arg(long)]
    import_memory: Option<String>,

    /// Show memory statistics
    #[arg(long)]
    memory_stats: bool,

    /// Cleanup old memories (days)
    #[arg(long, default_value = 30)]
    cleanup_days: Option<i64>,
}

fn load_config(config_path: Option<&str>) -> Result<MistralConfig> {
    if let Some(path) = config_path {
        let content = std::fs::read_to_string(path)?;
        serde_json::from_str(&content)
            .map_err(|e| E::msg(format!("Failed to parse config file {}: {}", path, e)))
    } else if Path::new(".mistral-config.json").exists() {
        let content = std::fs::read_to_string(".mistral-config.json")?;
        serde_json::from_str(&content)
            .map_err(|e| E::msg(format!("Failed to parse .mistral-config.json: {}", e)))
    } else {
        Ok(MistralConfig::default())
    }
}

fn save_config(args: &Args, config: &MistralConfig) -> Result<()> {
    let config_path = if let Some(path) = &args.config_file {
        path.clone()
    } else {
        ".mistral-config.json".to_string()
    };

    let content = serde_json::to_string_pretty(config)?;
    std::fs::write(&config_path, content)?;
    println!("Configuration saved to: {}", config_path);
    Ok(())
}

fn show_config(effective_config: &MistralConfig) -> Result<()> {
    println!("Effective Configuration:");
    println!("  Temperature: {:?}", effective_config.temperature);
    println!("  Top P: {:?}", effective_config.top_p);
    println!("  Top K: {:?}", effective_config.top_k);
    println!("  Repeat Penalty: {:?}", effective_config.repeat_penalty);
    println!("  Repeat Last N: {:?}", effective_config.repeat_last_n);
    Ok(())
}

fn merge_config_with_args(config: MistralConfig, args: &Args) -> MistralConfig {
    MistralConfig {
        temperature: args.temperature.or(config.temperature),
        top_p: args.top_p.or(config.top_p),
        top_k: args.top_k.or(config.top_k),
        repeat_penalty: if args.repeat_penalty != 1.1 {
            Some(args.repeat_penalty)
        } else {
            config.repeat_penalty
        },
        repeat_last_n: if args.repeat_last_n != 64 {
            Some(args.repeat_last_n)
        } else {
            config.repeat_last_n
        },
    }
}

/// Создает расширенный промпт с контекстом памяти
fn create_enhanced_prompt(
    user_input: &str,
    memory_context: Option<&crate::totems::MemoryContext>,
) -> String {
    let mut prompt_parts = Vec::new();

    // Добавляем контекст памяти если доступен
    if let Some(context) = memory_context {
        if !context.relevant_concepts.is_empty() || !context.relevant_episodes.is_empty() {
            let formatted_context = crate::totems::UnifiedMemoryManager::format_context_for_prompt(
                &crate::totems::UnifiedMemoryManager::new(
                    Arc::new(EmbeddingEngine::new("dummy", Device::Cpu)),
                    "dummy".to_string(),
                ),
                context,
            );
            prompt_parts.push(formatted_context);
            prompt_parts.push(String::new());
        }
    }

    // Добавляем текущий ввод
    prompt_parts.push(format!("=== User Input ===\n{}", user_input));
    prompt_parts.push("=== Assistant Response ===".to_string());

    prompt_parts.join("\n\n")
}

fn parse_persistence_format(format_str: &str) -> Result<PersistenceFormat> {
    match format_str.to_lowercase().as_str() {
        "json" => Ok(PersistenceFormat::Json),
        "binary" => Ok(PersistenceFormat::Binary),
        "hybrid" => Ok(PersistenceFormat::Hybrid),
        _ => Err(anyhow::anyhow!(
            "Invalid persistence format: {}. Use: json, binary, hybrid",
            format_str
        )),
    }
}

fn main() -> Result<()> {
    let args = Args::parse();

    // Handle special commands
    if args.show_config {
        let config = load_config(args.config_file.as_deref())?;
        let effective_config = merge_config_with_args(config, &args);
        return show_config(&effective_config);
    }

    if args.save_config {
        let config = load_config(args.config_file.as_deref())?;
        let effective_config = merge_config_with_args(config, &args);
        return save_config(&args, &effective_config);
    }

    if args.memory_stats {
        // Показываем статистику памяти и выходим
        let persistence = PersistenceManager::new(&args.memory_data_dir);
        let stats = persistence.get_persistence_stats()?;
        println!("{}", stats.format());
        return Ok(());
    }

    if let Some(ref import_path) = args.import_memory {
        // Импортируем память из файла
        let import_data = std::fs::read_to_string(import_path)?;
        let embedder = Arc::new(EmbeddingEngine::new(
            &args.embedding_model,
            crate::priests::device::select_device(args.cpu)?,
        )?);
        let mut memory = UnifiedMemoryManager::new(embedder, args.persona.clone());
        memory.import_memory(&import_data)?;
        println!("✅ Memory imported successfully");
        return Ok(());
    }

    // Check if prompt is provided for main functionality
    let user_input = if let Some(ref p) = args.prompt {
        p.clone()
    } else {
        eprintln!("Error: --prompt <PROMPT> is required when not using special flags");
        std::process::exit(1);
    };

    #[cfg(feature = "mkl")]
    extern crate intel_mkl_src;

    #[cfg(feature = "accelerate")]
    extern crate accelerate_src;

    #[cfg(feature = "cuda")]
    candle_core::quantized::cuda::set_force_dmmv(args.force_dmmv);

    let _guard = if args.tracing {
        let (chrome_layer, guard) = ChromeLayerBuilder::new().build();
        tracing_subscriber::registry().with(chrome_layer).init();
        Some(guard)
    } else {
        None
    };

    // Load and merge configuration
    let config = load_config(args.config_file.as_deref())?;
    let effective_config = merge_config_with_args(config, &args);

    println!(
        "avx: {}, neon: {}, simd128: {}, f16c: {}",
        candle_core::utils::with_avx(),
        candle_core::utils::with_neon(),
        candle_core::utils::with_simd128(),
        candle_core::utils::with_f16c()
    );
    println!(
        "temp: {:.2} repeat-penalty: {:.2} repeat-last-n: {}",
        effective_config.temperature.unwrap_or(0.0),
        effective_config.repeat_penalty.unwrap_or(1.1),
        effective_config.repeat_last_n.unwrap_or(64)
    );

    // Инициализация памяти (если включена)
    let mut memory = if args.enable_memory {
        println!("🧠 Initializing unified memory system...");

        // Инициализируем персистентность
        let persistence_format = parse_persistence_format(&args.persistence_format)?;
        let mut persistence = PersistenceManager::with_config(
            &args.memory_data_dir,
            persistence_format,
            args.auto_save_interval,
        );

        if let Err(e) = persistence.initialize() {
            println!(
                "⚠️  Warning: Failed to initialize persistence: {}. Memory will be in-memory only.",
                e
            );
        }

        // Проверяем наличие модели эмбеддингов
        let device = crate::priests::device::select_device(args.cpu)?;
        match EmbeddingEngine::new(&args.embedding_model, device) {
            Ok(embedder) => {
                let memory_manager =
                    UnifiedMemoryManager::new(Arc::new(embedder), args.persona.clone());

                // Загружаем существующую память
                match persistence.load_vector_store(embedder.embedding_dim()) {
                    Ok(vector_store) => {
                        // TODO: Загрузить векторное хранилище в память
                        println!("📥 Loaded existing memory from disk");
                    }
                    Err(_) => {
                        println!("🆕 Starting with fresh memory");
                    }
                }

                println!("✅ Memory system initialized");
                Some(memory_manager)
            }
            Err(e) => {
                println!(
                    "⚠️  Failed to initialize memory: {}. Memory will be disabled.",
                    e
                );
                None
            }
        }
    } else {
        None
    };

    // Выполняем очистку если нужно
    if let (Some(ref mut memory_manager), Some(cleanup_days)) =
        (&mut memory, Some(args.cleanup_days))
    {
        let removed = memory_manager.cleanup_old_memories(cleanup_days)?;
        println!("🧹 Cleaned up {} old memory entries", removed);

        if args.export_memory {
            let export_data = memory_manager.export_memory()?;
            println!("📤 Memory export ready");
            return Ok(());
        }
    }

    // Загрузка Mistral модели
    let start = std::time::Instant::now();
    let api = Api::new()?;
    let model_id = match args.model_id {
        Some(model_id) => model_id,
        None => {
            if args.quantized {
                if args.which != Which::Mistral7bV01 {
                    anyhow::bail!("only 7b-v0.1 is available as a quantized model for now")
                }
                "lmz/candle-mistral".to_string()
            } else {
                let name = match args.which {
                    Which::Mistral7bV01 => "mistralai/Mistral-7B-v0.1",
                    Which::Mistral7bV02 => "mistralai/Mistral-7B-v0.2",
                    Which::Mistral7bInstructV01 => "mistralai/Mistral-7B-Instruct-v0.1",
                    Which::Mistral7bInstructV02 => "mistralai/Mistral-7B-Instruct-v0.2",
                    Which::Mathstral7bV01 => "mistralai/mathstral-7B-v0.1",
                    Which::MistralNemo2407 => "mistralai/Mistral-Nemo-Base-2407",
                    Which::MistralNemoInstruct2407 => "mistralai/Mistral-Nemo-Instruct-2407",
                };
                name.to_string()
            }
        }
    };
    let repo = api.repo(Repo::with_revision(
        model_id,
        RepoType::Model,
        args.revision,
    ));
    let tokenizer_filename = match args.tokenizer_file {
        Some(file) => std::path::PathBuf::from(file),
        None => repo.get("tokenizer.json")?,
    };
    let filenames = match args.weight_files {
        Some(files) => files
            .split(',')
            .map(std::path::PathBuf::from)
            .collect::<Vec<_>>(),
        None => {
            if args.quantized {
                vec![repo.get("model-q4k.gguf")?]
            } else {
                hub_load_safetensors(&repo, "model.safetensors.index.json")?
            }
        }
    };
    println!("retrieved the files in {:?}", start.elapsed());
    let tokenizer = Tokenizer::from_file(tokenizer_filename).map_err(E::msg)?;

    let start = std::time::Instant::now();
    let config = match args.config_file {
        Some(config_file) => serde_json::from_slice(&std::fs::read(config_file)?)?,
        None => {
            if args.quantized {
                Config::config_7b_v0_1(args.use_flash_attn)
            } else {
                let config_file = repo.get("config.json")?;
                serde_json::from_slice(&std::fs::read(config_file)?)?
            }
        }
    };
    let device = crate::priests::device::select_device(args.cpu)?;
    let (model, device) = if args.quantized {
        let filename = &filenames[0];
        let vb =
            candle_transformers::quantized_var_builder::VarBuilder::from_gguf(filename, &device)?;
        let model = QMistral::new(&config, vb)?;
        (Model::Quantized(model), device)
    } else {
        let dtype = if device.is_cuda() {
            DType::BF16
        } else {
            DType::F32
        };
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&filenames, dtype, &device)? };
        let model = Mistral::new(&config, vb)?;
        (Model::Mistral(model), device)
    };

    println!("loaded the model in {:?}", start.elapsed());

    let mut pipeline = TextGeneration::new(
        model,
        tokenizer,
        args.seed,
        effective_config.temperature,
        effective_config.top_p,
        effective_config.top_k,
        effective_config.repeat_penalty.unwrap_or(1.1),
        effective_config.repeat_last_n.unwrap_or(64),
        &device,
    );

    // === Memory Integration ===
    let mut memory_context = None;
    let mut search_stats = None;

    // Получаем контекст из памяти если она доступна
    if let (Some(ref mut memory_manager), true) = (&mut memory, args.enable_memory) {
        // Выполняем полный поиск по памяти
        match memory_manager.recall(
            &user_input,
            args.memory_episodes_count,
            args.memory_concepts_count,
        ) {
            Ok(context) => {
                memory_context = Some(context.clone());
                search_stats = Some(context.search_stats.clone());

                // Показываем статистику поиска
                println!("\n🔍 Memory Search Results:");
                println!(
                    "   📝 Episodes found: {} ({}ms)",
                    context.search_stats.episodes_found,
                    context.search_stats.episode_search_time_ms
                );
                println!(
                    "   🧠 Concepts found: {} ({}ms)",
                    context.search_stats.concepts_found,
                    context.search_stats.concept_search_time_ms
                );
                println!(
                    "   ⏱️  Total search time: {}ms",
                    context.search_stats.episode_search_time_ms
                        + context.search_stats.concept_search_time_ms
                );
            }
            Err(e) => {
                println!("⚠️  Memory search failed: {}", e);
            }
        }
    }

    // Создаем улучшенный промпт
    let enhanced_prompt = create_enhanced_prompt(&user_input, memory_context.as_ref());

    // Показываем контекст памяти (для отладки)
    if args.enable_memory {
        if let Some(ref context) = memory_context {
            println!("\n=== 🧠 Memory Context ===");
            let mut dummy_memory = UnifiedMemoryManager::new(
                Arc::new(EmbeddingEngine::new("dummy", Device::Cpu).unwrap()),
                "dummy".to_string(),
            );
            let formatted = dummy_memory.format_context_for_prompt(&context);
            println!("{}", formatted);
            println!("=======================\n");
        }
    }

    // Генерируем ответ
    println!("🤖 Assistant:");
    let response = pipeline.run(&enhanced_prompt, args.sample_len)?;

    // Сохраняем диалог в память
    if let (Some(ref mut memory_manager), true) = (&mut memory, args.enable_memory) {
        match memory_manager.add_exchange(user_input.clone(), response.clone()) {
            Ok(()) => {
                println!("💾 Dialogue saved to memory");

                // Проверяем необходимость автосохранения
                let mut persistence = PersistenceManager::with_config(
                    &args.memory_data_dir,
                    parse_persistence_format(&args.persistence_format)?,
                    args.auto_save_interval,
                );

                if persistence.should_auto_save() {
                    println!("🔄 Auto-saving memory...");
                    // TODO: Реализовать автосохранение
                }

                // Показываем статистику памяти
                let stats = memory_manager.get_comprehensive_stats();
                println!("{}", stats.format());
            }
            Err(e) => {
                println!("⚠️  Failed to save dialogue to memory: {}", e);
            }
        }
    }

    Ok(())
}

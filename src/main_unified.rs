//! ZIGGURAT MIND - Unified Entry Point
//!
//! Unified architecture: Embedding Engine + Dialogue Memory + Mistral 7B
//! Memory flow: Query → Embed → Search → Context → Generate → Save

mod logos;
mod priests;
mod totems;
mod utils;
mod demiurge;

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
use crate::totems::semantic::persistence::SemanticPersistenceManager;
use crate::utils::hub_load_safetensors;
use crate::demiurge::{Persona, ArchetypeLoader, persona::PersonaInfo};

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
        _assistant_response: &str,
        _session_id: &str,
    ) -> Result<totems::semantic::ExtractionResult> {
        let prompt = format!(
            r#"<s>[INST] You are a knowledge extraction assistant. Extract ONLY explicit self-disclosed facts, preferences, rules, or skills that the USER directly states about themselves.

CRITICAL RULES FOR RUSSIAN:
- "я люблю X" = "I love X" (POSITIVE - extract!)
- "я не люблю X" = "I don't love X" (NEGATIVE - extract!)
- "нет, я люблю X" = "I love X" (CORRECTION - still POSITIVE, extract!)
- "нет я люблю X" = "I love X" (CORRECTION - still POSITIVE, extract!)
- "я предпочитаю X" = "I prefer X" (POSITIVE)
- "мне нравится X" = "I like X" (POSITIVE)

KEY PATTERNS TO DETECT:
- "люблю" = love (POSITIVE)
- "нравится" = like (POSITIVE)
- "предпочитаю" = prefer (POSITIVE)
- "не люблю" = don't love (NEGATIVE)
- "не нравится" = don't like (NEGATIVE)

Examples:
- "я люблю пиццу" → {{"text":"I love pizza","category":"preferences","confidence":0.9}}
- "я не люблю суши" → {{"text":"I don't love sushi","category":"preferences","confidence":0.9}}
- "нет я люблю суши" → {{"text":"I love sushi","category":"preferences","confidence":0.9}}
- "предпочитаю кофе" → {{"text":"I prefer coffee","category":"preferences","confidence":0.9}}

If no explicit self-disclosure found, return empty array [].

User message:
{user_query}

Output format: [{{"text":"...","category":"...","confidence":0.8}}]
NO markdown, NO explanations, NO text before or after. Only JSON.
[/INST]</s>"#,
            user_query = user_query
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
            Err(_) => {
                let mut fixed = cleaned.clone();

                if !fixed.starts_with('[') {
                    fixed = format!("[{}]", fixed);
                }

                if fixed.starts_with("[,\"") || fixed.starts_with("[\"") {
                    fixed = fixed.replacen("[\"", "[{\"", 1);
                }

                fixed = fixed.replace("}\"]", "}\"]}");
                fixed = fixed.replace("} ]", "}]");
                fixed = fixed.replace("}] ]", "}]}]");

                match parse_json(&fixed) {
                    Ok(c) => c,
                    Err(_) => {
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

struct ContextAnalyzerImpl {
    pipeline: std::sync::Arc<std::sync::Mutex<UnifiedPipeline>>,
}

impl ContextAnalyzerImpl {
    fn new(pipeline: std::sync::Arc<std::sync::Mutex<UnifiedPipeline>>) -> Self {
        Self { pipeline }
    }
}

impl totems::episodic::LlmPipeline for ContextAnalyzerImpl {
    fn generate(&self, prompt: &str, max_tokens: usize) -> Result<String> {
        let mut pipeline = self.pipeline.lock().unwrap();
        pipeline.clear_cache();
        pipeline.run(prompt, max_tokens, 0)
    }
}

struct UnifiedPipeline {
    model: Mistral,
    tokenizer: Tokenizer,
    device: Device,
    logits_processor: LogitsProcessor,
    repeat_penalty: f32,
    repeat_last_n: usize,
    temperature: f64,
    top_k: Option<usize>,
    top_p: Option<f64>,
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
            temperature,
            top_k,
            top_p,
        }
    }

    /// Update temperature for generation
    pub fn set_temperature(&mut self, temp: f64) {
        self.temperature = temp;
    }

    /// Get current temperature
    pub fn get_temperature(&self) -> f64 {
        self.temperature
    }

    fn run(&mut self, prompt: &str, sample_len: usize, seed: u64) -> Result<String> {
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

        let temperature = self.temperature;
        let sampling = if temperature <= 0. {
            Sampling::ArgMax
        } else {
            match (self.top_k, self.top_p) {
                (None, None) => Sampling::All { temperature },
                (Some(k), None) => Sampling::TopK { k, temperature },
                (None, Some(p)) => Sampling::TopP { p, temperature },
                (Some(k), Some(p)) => Sampling::TopKThenTopP { k, p, temperature },
            }
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

    /// Quiet mode - suppress debug output
    #[arg(long, short = 'q')]
    quiet: bool,

    /// Number of similar dialogues to retrieve
    #[arg(long, default_value_t = 5)]
    memory_top_k: usize,

    /// Number of semantic concepts to retrieve
    #[arg(long, default_value_t = 10)]
    semantic_top_k: usize,

    /// Persona name for the session
    #[arg(long, default_value = "assistant")]
    persona: String,

    /// Archetype to use (girlfriend, programmer, devops, scientist, philosopher)
    #[arg(long, default_value = "programmer")]
    archetype: String,

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
        // Debug memory info - uncomment if needed for debugging
        // let mem_mb = get_memory_mb();
        // if mem_mb > 0 {
        //     eprintln!("DEBUG [{}]: RAM: {} MB", _label, mem_mb);
        // }
        // if let Some(gpu_mb) = get_gpu_memory_mb() {
        //     eprintln!("DEBUG [{}]: VRAM: {} MB", _label, gpu_mb);
        // }
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
    persona: Option<&Persona>,
    user_uses_formal: bool,
) -> String {
    let mut prompt_parts = Vec::new();

    // Add Persona system prompt if available
    if let Some(p) = persona {
        prompt_parts.push(p.format_system_prompt());
    }

    // Build context sections
    let mut context_parts = Vec::new();

    if !current_context.is_empty() {
        context_parts.push(format!("Current conversation:\n{}", current_context));
    }

    if !semantic_context.is_empty() {
        context_parts.push(format!("KNOWLEDGE:\n{}", semantic_context));
    }

    if !episodic_context.is_empty() {
        context_parts.push(format!(
            "═══════════════════════════════════════════════════════════════\n\
             PREVIOUS CONVERSATION MEMORY (CRITICAL - YOU MUST USE THIS!)\n\
             ═══════════════════════════════════════════════════════════════\n{}\n\
             ═══════════════════════════════════════════════════════════════\n\
             INSTRUCTIONS:\n\
             1. If user asks about preferences, past conversations, or remembers something - ANSWER directly using this memory\n\
             2. If user asks \"what did I say about X\" - find it in this memory and repeat\n\
             3. If memory contains the answer, say it clearly: \"You said [specific thing]\"\n\
             4. Do NOT say \"I don't know\" if the answer is in this memory!\n\
             ═══════════════════════════════════════════════════════════════",
            episodic_context
        ));
    }

    // Add relationship context if persona is available
    if let Some(p) = persona {
        let relationship_summary = p.narrative.format_relationship_summary("default_user");
        if !relationship_summary.is_empty() {
            context_parts.push(format!("RELATIONSHIP:\n{}", relationship_summary));
        }
    }

    if !context_parts.is_empty() {
        prompt_parts.push(context_parts.join("\n\n"));
    }

    // Add directives as constraints
    if let Some(p) = persona {
        // Build directive constraints based on communication style
        let mut constraints = Vec::new();

        // Honorifics constraint
        if p.communication.use_honorifics || user_uses_formal {
            constraints.push("Обращаться на Вы");
        } else {
            constraints.push("Обращаться на ты");
        }

        // Style-based constraints
        match p.communication.style.as_str() {
            "technical" => constraints.push("Давать технически точные ответы с примерами кода"),
            "warm" => constraints.push("Отвечать тепло и поддерживающе"),
            "academic" => constraints.push("Отвечать академично, с данными и фактами"),
            "socratic" => constraints.push("Использовать вопросы для направления мысли"),
            _ => {}
        }

        // Trait-based constraints
        let traits = p.get_all_traits();
        if traits.get("pedagogical").unwrap_or(&0.5) > &0.7 {
            constraints.push("Объяснять подробно и понятно");
        }
        if traits.get("humor").unwrap_or(&0.5) > &0.7 {
            constraints.push("Добавлять юмор в ответы");
        }
        if traits.get("empathy").unwrap_or(&0.5) > &0.8 {
            constraints.push("Проявлять эмпатию и понимание");
        }

        if !constraints.is_empty() {
            prompt_parts.push(format!("STYLE CONSTRAINTS:\n{}", constraints.join("\n")));
        }

        // Add user's known preferences and facts from semantic memory
        let user_knowledge = p.get_user_knowledge_summary();
        if !user_knowledge.is_empty() {
            prompt_parts.push(format!("USER PROFILE (use when relevant):\n{}", user_knowledge));
        }
    }

    let combined_context = prompt_parts.join("\n\n");

    if !enable_memory && persona.is_none() {
        format!("<s>[INST] {} [/INST]", user_input)
    } else if combined_context.is_empty() {
        // No context - just be a friendly assistant
        format!(
            "<s>[INST] You are {}, {}.\n\
             \n\
             {}\
             \n\
             User: {}\n\
             \n\
             Respond naturally and helpfully.[/INST]",
            persona.map(|p| p.name.as_str()).unwrap_or("a helpful assistant"),
            persona.map(|p| p.description.as_str()).unwrap_or("friendly and supportive"),
            persona.map(|p| p.communication.greeting.as_str()).unwrap_or(""),
            user_input
        )
    } else {
        // Has context - user asked about past
        format!(
            "<s>[INST] You are {}, {}.\n\
             \n\
             IMPORTANT: The user is asking about their own preferences or past statements from earlier conversations.\n\
             \n\
             PAST MEMORY:\n{}\n\
             \n\
             {}\
             \n\
             YOUR TASK: Read the memory above and give a CONFIDENT, DIRECT answer.\n\
             - The user is asking about THEIR preferences, not yours\n\
             - Look for \"USER SAID: ...\" to find what they previously stated\n\
             - Answer clearly: \"Your favorite car is Lamborghini\" or \"You said your favorite food is sushi\"\n\
             - Do NOT say \"I don't know\" if the memory contains relevant information\n\
             - Be specific and confident\n\
             \n\
             User's question: {}\n\
             \n\
             Your confident answer:[/INST]",
            persona.map(|p| p.name.as_str()).unwrap_or("a helpful assistant"),
            persona.map(|p| p.description.as_str()).unwrap_or("friendly and supportive"),
            combined_context,
            persona.map(|p| p.communication.greeting.as_str()).unwrap_or(""),
            user_input
        )
    }
}

fn process_query(
    prompt: &str,
    pipeline_arc: &std::sync::Arc<std::sync::Mutex<UnifiedPipeline>>,
    dialogue_manager: &mut Option<DialogueManager>,
    semantic_manager: &mut Option<std::sync::Arc<std::sync::Mutex<SemanticMemoryManager>>>,
    persistence_manager: &std::sync::Arc<totems::episodic::persistence::PersistenceManager>,
    embedder: &Arc<dyn crate::priests::embeddings::Embedder>,
    args: &Args,
    persona: &mut Option<Persona>,
) -> Result<()> {
    log_memory_usage("process_query start");

    // Detect if user uses formal or informal address
    let user_uses_formal = prompt.contains("Вы ") || prompt.contains("вы ") || prompt.contains("ВЫ ");

    // Get sampling parameters from Persona traits
    let (temperature, max_tokens) = if let Some(ref p) = *persona {
        let traits = p.get_all_traits();

        // Temperature: analytical = lower temp, creative = higher
        let temp = traits.get("analytical").copied().unwrap_or(0.5);
        let temperature = if temp > 0.8 {
            0.3  // Analytical - precise
        } else if temp > 0.6 {
            0.5  // Balanced
        } else {
            0.7  // Creative
        };

        // Max tokens: verbose = longer, concise = shorter
        let verbose = traits.get("verbose").copied().unwrap_or(0.5);
        let max_tokens = if verbose > 0.7 {
            (args.sample_len as f32 * 0.5) as usize
        } else {
            (args.sample_len as f32 * 0.25) as usize
        };

        (Some(temperature), max_tokens.min(512)) // Cap at 512 tokens for interactive mode
    } else {
        // For interactive mode without persona, limit to 512 tokens
        let max_tokens = if args.interactive {
            (args.sample_len as f32 * 0.25) as usize
        } else {
            args.sample_len
        };
        (None, max_tokens.min(512))
    };

    let (similar_dialogues, current_context) = if let Some(ref mut dm) = *dialogue_manager {
        if args.disable_memory_context {
            (String::new(), String::new())
        } else {
            // Only search memory if user is asking about past conversations
            let is_asking_about_past = prompt.to_lowercase().contains("помнишь")
                || prompt.to_lowercase().contains("помнил")
                || prompt.to_lowercase().contains("вспомни")
                || prompt.to_lowercase().contains("что я говорил")
                || prompt.to_lowercase().contains("что я сказал")
                || prompt.to_lowercase().contains("наш разговор")
                || prompt.to_lowercase().contains("прошлый раз")
                || prompt.to_lowercase().contains("в прошлый раз")
                || prompt.to_lowercase().contains("раньше")
                || prompt.to_lowercase().contains("забыл")
                || prompt.to_lowercase().contains("в прошлом")
                || prompt.to_lowercase().contains("что ты помнишь")
                || prompt.to_lowercase().contains("ты помнишь")
                || prompt.to_lowercase().contains("remember")
                || prompt.to_lowercase().contains("what did i say")
                || prompt.to_lowercase().contains("what did i tell");

            if !is_asking_about_past {
                // Don't include memory context for normal conversation
                (String::new(), String::new())
            } else {
                let similar = dm.find_similar_dialogues(prompt, args.memory_top_k)?;
                let current_ctx = dm.get_current_context(5);

                let similar_text = if !similar.is_empty() {
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
        }
    } else {
        (String::new(), String::new())
    };

    let semantic_context = if args.enable_semantic {
        if let Some(ref sm) = *semantic_manager {
            let sm = sm.lock().unwrap();
            let results = sm.search_by_text(prompt, args.semantic_top_k);
            if !results.is_empty() {
                if !args.quiet {
                    eprintln!("📚 Found {} relevant concepts", results.len());
                }
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
        persona.as_ref(),
        user_uses_formal,
    );

    if !args.quiet {
        eprintln!("DEBUG: Enhanced prompt length: {}", enhanced_prompt.len());
    }

    println!("\n📝 You: {}", prompt);

    // Show which persona is responding
    if let Some(ref p) = *persona {
        println!("\n🤖 {}:", p.name);
    } else {
        println!("\n🤖 Assistant:");
    }

    // Apply trait-based sampling parameters
    {
        let mut pipeline = pipeline_arc.lock().unwrap();
        if let Some(temp) = temperature {
            // Temporarily modify temperature for this generation
            pipeline.set_temperature(temp);
        }
    }

    let response = pipeline_arc.lock().unwrap().run(&enhanced_prompt, max_tokens, args.seed)?;

    // Reset temperature if we changed it
    {
        let mut pipeline = pipeline_arc.lock().unwrap();
        pipeline.set_temperature(args.temperature);
    }

    println!("{}", response);

    let session_id = dialogue_manager
        .as_ref()
        .map(|dm| dm.current_session().id.to_string())
        .unwrap_or_else(|| "unknown".to_string());

    if let Some(ref mut dm) = *dialogue_manager {
        dm.add_exchange(prompt.to_string(), response.clone())?;

        if args.interactive && !args.quiet {
            let stats = dm.stats();
            eprintln!("💾 Memory: {} turns in current session", stats.current_session_turns);
        }

        if let Err(e) = persistence_manager.save_with_embeddings(dm, embedder.embedding_dim()) {
            eprintln!("WARNING: Failed to save memory: {}", e);
        }
    }

    if args.enable_semantic {
        if let Some(ref sm) = *semantic_manager {
            let mut sm = sm.lock().unwrap();
            let has_self_disclosure = prompt.to_lowercase().contains("я ")
                || prompt.to_lowercase().contains("мой ")
                || prompt.to_lowercase().contains("моя ")
                || prompt.to_lowercase().contains("моё ")
                || prompt.to_lowercase().contains("мои ")
                || prompt.to_lowercase().contains("люблю")
                || prompt.to_lowercase().contains("предпочитаю")
                || prompt.to_lowercase().contains("нравится")
                || prompt.to_lowercase().contains("не люблю")
                || prompt.to_lowercase().contains("i ")
                || prompt.to_lowercase().contains("my ")
                || prompt.to_lowercase().contains("i'm")
                || prompt.to_lowercase().contains("i am");

            if has_self_disclosure {
                if let Err(e) = sm.extract_from_dialogue(prompt, &response, &session_id) {
                    if !args.quiet {
                        eprintln!("DEBUG: Failed to extract concepts: {}", e);
                    }
                }
                if !args.quiet {
                    eprintln!("DEBUG: Semantic memory now has {} concepts", sm.count());
                }
            }
        }
    }

    // Apply Persona evolution based on interaction
    if let Some(ref mut p) = *persona {
        // Create interaction record
        let interaction = crate::demiurge::Interaction {
            user_sentiment: 0.5,  // Would need sentiment analysis to determine
            successful_help: true,  // Assuming response was generated
            emotional_depth: if prompt.contains("?") || prompt.len() > 100 { 0.5 } else { 0.3 },
            topics: vec!["general".to_string()],
            user_gave_feedback: false,
            feedback_positive: false,
            is_deep_conversation: prompt.len() > 200,
            is_code_related: prompt.contains("code") || prompt.contains("function") || prompt.contains("bug"),
            is_emotional_support: prompt.contains("sad") || prompt.contains("help") || prompt.contains("помоги"),
        };

        p.apply_interaction(interaction);

        // Extract and store concepts in Persona semantic memory
        p.extract_and_store_concepts(prompt, &response);

        // Save narrative periodically (every 10 interactions)
        if p.evolution.interactions_count % 10 == 0 {
            if let Err(e) = p.save_narrative() {
                eprintln!("WARNING: Failed to save narrative: {}", e);
            } else {
                eprintln!("💾 Persona narrative saved");
            }
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

    // Initialize managers
    let persistence_manager = Arc::new(
        totems::episodic::persistence::PersistenceManager::new(
            Some(&resolve_path("memory_data")),
            true,
        )?
    );
    println!("💾 Persistence manager initialized");

    let mut dialogue_manager: Option<DialogueManager> = None;
    if args.enable_memory {
        let persona_name = args.archetype.clone();
        dialogue_manager = Some(DialogueManager::new(embedder.clone(), persona_name));
        println!("🗣️ Dialogue memory enabled");
    }

    let mut semantic_manager: Option<std::sync::Arc<std::sync::Mutex<SemanticMemoryManager>>> = if args.enable_semantic {
        let storage_path = resolve_path("memory_data/semantic");
        let persistence = SemanticPersistenceManager::new(Some(&storage_path))?;
        let sm = SemanticMemoryManager::new(embedder.clone(), persistence)?;
        Some(std::sync::Arc::new(std::sync::Mutex::new(sm)))
    } else {
        None
    };
    if args.enable_semantic {
        println!("🧠 Semantic memory enabled");
    }

    // Инициализируем Persona (Demiurge Level)
    let mut persona: Option<Persona> = None;
    if args.interactive {
        match ArchetypeLoader::load(&args.archetype) {
            Ok(archetype) => {
                let mut p = Persona::from_archetype(std::sync::Arc::new(archetype));
                println!("🎭 Persona loaded: {} ({})", p.name, p.archetype_id);

                // Connect semantic memory if enabled
                if args.enable_semantic {
                    if let Some(ref sm) = semantic_manager {
                        p.set_semantic_manager(sm.clone());
                        println!("🧠 Connected semantic memory to persona");
                    }
                }

                if let Some(context) = p.load_session_context()? {
                    println!("💭 Found saved session context!");

                    if !context.summary.is_empty() {
                        let greeting = p.generate_contextual_greeting(&context);
                        println!("\n🤖 {}:", p.name);
                        println!("{}", greeting);
                    }
                } else if p.has_saved_context() {
                    println!("💭 Found expired session context (will be cleared)");
                }

                persona = Some(p);
            }
            Err(e) => {
                eprintln!("⚠️  Warning: Could not load archetype '{}': {}", args.archetype, e);
                eprintln!("   Available archetypes: {:?}", ArchetypeLoader::list_ids().unwrap_or_default());
            }
        }
    }

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
        let pipeline_for_context = pipeline_arc.clone();
        let persona_for_save = persona.clone();
        let dm_for_save = dialogue_manager.clone();
        let persistence_for_save = persistence_manager.clone();
        let embedder_for_save = embedder.clone();

        let _ = ctrlc::set_handler(move || {
            println!("\n\n💾 Saving context before exit...");

            if let Some(ref p) = persona_for_save {
                if let Some(ref dm) = dm_for_save {
                    let context_analyzer = ContextAnalyzerImpl::new(pipeline_for_context.clone());
                    if let Ok(Some(_)) = p.save_session_context(dm, &context_analyzer) {
                        println!("💾 Session context saved");
                    }
                }
            }

            if let Some(ref dm) = dm_for_save {
                if let Err(e) = persistence_for_save.save_with_embeddings(dm, embedder_for_save.embedding_dim()) {
                    eprintln!("WARNING: Failed to save memory: {}", e);
                } else {
                    println!("💾 Episodic memory saved");
                }
            }

            std::process::exit(0);
        });

        println!("\n🗣️ Interactive mode - type 'quit' to exit");
        println!("   /semantic - Manage semantic memory");
        println!("   /persona  - Manage persona (show, switch, traits, evolution)");
        println!("   /mem - Show memory usage");
        println!("   /context - Show current session context");
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
                &mut persona,
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
                println!("💾 Saving session context...");

                if let Some(ref p) = persona {
                    if let Some(ref dm) = dialogue_manager {
                        let context_analyzer = ContextAnalyzerImpl::new(pipeline_arc.clone());
                        if let Ok(Some(context)) = p.save_session_context(dm, &context_analyzer) {
                            println!("💾 Context saved for next session");
                            if !context.summary.is_empty() {
                                println!("   Topics: {}", context.key_topics.join(", "));
                            }
                        }
                    }
                }

                if let Some(ref dm) = dialogue_manager {
                    if let Err(e) =
                        persistence_manager.save_with_embeddings(dm, embedder.embedding_dim())
                    {
                        eprintln!("WARNING: Failed to save memory on exit: {}", e);
                    } else {
                        println!("💾 Episodic memory saved");
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

            if input == "/context" || input == "/c" {
                if let Some(ref mut p) = persona {
                    match p.load_session_context() {
                        Ok(Some(context)) => {
                            println!("\n💭 Session Context:");
                            println!("   Version: {}", context.version);
                            println!("   Last interaction: <timestamp>");

                            if !context.summary.is_empty() {
                                println!("   Summary: {}", context.summary);
                            }

                            if !context.key_topics.is_empty() {
                                println!("   Topics: {}", context.key_topics.join(", "));
                            }

                            println!("   Emotional state: {:.1}", context.emotional_state);

                            if !context.last_topic.is_empty() {
                                println!("   Last topic: {}", context.last_topic);
                            }

                            if !context.pending_questions.is_empty() {
                                println!("   Pending questions:");
                                for q in &context.pending_questions {
                                    println!("     - {}", q);
                                }
                            }

                            println!("\n   💡 This context will be restored in the next session.");
                        }
                        Ok(None) => {
                            println!("\n💭 No saved context found.");
                            println!("   Start a conversation to create context for the next session.");
                        }
                        Err(e) => {
                            println!("\n❌ Error loading context: {}", e);
                        }
                    }
                } else {
                    println!("\n💭 No persona loaded.");
                }
                continue;
            }

            // Persona commands
            if input.starts_with("/persona") || input.starts_with("/p") {
                handle_persona_command(input, &mut persona);
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
                &mut persona,
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
            &mut persona,
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
                match cat_arg.as_str() {
                    "facts" | "rules" | "preferences" | "pref" | "skills" | "general" => {
                        let cat = match cat_arg.as_str() {
                            "facts" => ConceptCategory::Facts,
                            "rules" => ConceptCategory::Rules,
                            "preferences" | "pref" => ConceptCategory::Preferences,
                            "skills" => ConceptCategory::Skills,
                            "general" => ConceptCategory::General,
                            _ => unreachable!()
                        };
                        let limit = parts.get(3).and_then(|s| s.parse::<usize>().ok()).unwrap_or(20);
                        let concepts = sm.list_by_category(&cat, limit);
                        let total_in_cat = sm.get_concepts_by_category(&cat).len();
                        if concepts.is_empty() {
                            println!("No concepts found in category '{}'.", cat);
                        } else {
                            println!("📚 Concepts in {} (showing {} of {}):", cat, concepts.len(), total_in_cat);
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

fn handle_persona_command(input: &str, persona: &mut Option<Persona>) {
    let parts: Vec<&str> = input.split_whitespace().collect();
    let command = parts.get(1).map(|s| *s).unwrap_or("");

    match command {
        "help" | "h" => {
            println!(r#"
🎭 Persona Commands:
   /persona help           - Show this help
   /persona show           - Show current persona info
   /persona list           - List available archetypes
   /persona switch <name>  - Switch to another archetype
   /persona traits         - Show current traits
   /persona evolution      - Show evolution status
   Short form: /p instead of /persona
"#);
        }

        "list" | "l" => {
            match ArchetypeLoader::list_ids() {
                Ok(ids) => {
                    if ids.is_empty() {
                        println!("No archetypes found in config/archetypes/");
                    } else {
                        println!("🎭 Available archetypes:");
                        for id in ids {
                            println!("  - {}", id);
                        }
                    }
                }
                Err(e) => println!("Error loading archetypes: {}", e),
            }
        }

        "show" | "" => {
            if let Some(ref p) = persona {
                let info: PersonaInfo = p.into();
                println!("\n🎭 Current Persona:");
                println!("   Name: {}", info.name);
                println!("   Archetype: {}", info.archetype_id);
                println!("   Description: {}", info.description);
                println!("   Interactions: {}", info.evolution.interactions);
                println!("   Relationship score: {:.2}", info.evolution.relationship_score);
            } else {
                println!("No persona loaded. Use --archetype in CLI args to enable.");
            }
        }

        "switch" | "s" => {
            if parts.len() < 3 {
                println!("Usage: /persona switch <archetype>");
                println!("Available archetypes: {:?}", ArchetypeLoader::list_ids().unwrap_or_default());
                return;
            }
            let new_archetype = parts[2];
            match ArchetypeLoader::load(new_archetype) {
                Ok(archetype) => {
                    let new_persona = Persona::from_archetype(std::sync::Arc::new(archetype));
                    println!("✅ Switched to persona: {} ({})", new_persona.name, new_persona.archetype_id);
                    *persona = Some(new_persona);
                }
                Err(e) => {
                    println!("❌ Error loading archetype '{}': {}", new_archetype, e);
                    println!("Available archetypes: {:?}", ArchetypeLoader::list_ids().unwrap_or_default());
                }
            }
        }

        "traits" | "t" => {
            if let Some(ref p) = persona {
                let traits = p.get_all_traits();
                println!("\n📊 Current Traits:");
                println!("┌─────────────────────┬────────┬─────────────┐");
                println!("│ Trait               │ Value  │ Description │");
                println!("├─────────────────────┼────────┼─────────────┤");
                let mut trait_names: Vec<_> = traits.keys().collect();
                trait_names.sort();
                for name in trait_names {
                    let value = traits[name];
                    let bar = get_trait_bar(value);
                    let desc = get_trait_description(name, value);
                    println!("│ {:<19} │ {} │ {} │", name, bar, desc);
                }
                println!("└─────────────────────┴────────┴─────────────┘");
            } else {
                println!("No persona loaded.");
            }
        }

        "evolution" | "e" | "unlocks" | "u" => {
            if let Some(ref p) = persona {
                let info: PersonaInfo = p.into();
                println!("\n🌱 Evolution Status:");
                println!("   Interactions: {}", info.evolution.interactions);
                println!("   Relationship score: {:.2}", info.evolution.relationship_score);

                if info.evolution.unlocked_traits.is_empty() {
                    println!("   Unlocked traits: None (keep interacting to unlock!)");
                } else {
                    println!("   Unlocked traits: {:?}", info.evolution.unlocked_traits);
                }

                println!("\n📈 Relationship Arc:");
                if info.evolution.relationship_score > 0.8 {
                    println!("   💕 Deep connection established");
                } else if info.evolution.relationship_score > 0.6 {
                    println!("   🤝 Good working relationship");
                } else if info.evolution.relationship_score > 0.4 {
                    println!("   👋 Normal interaction");
                } else {
                    println!("   🆕 Just getting started");
                }
            } else {
                println!("No persona loaded.");
            }
        }

        _ => {
            println!("Unknown command: {}. Use /persona help", command);
        }
    }
}

fn get_trait_bar(value: f32) -> String {
    let filled = (value * 10.0) as usize;
    let empty = 10 - filled;
    format!("[{}{}]", "█".repeat(filled), "░".repeat(empty))
}

fn get_trait_description(name: &str, value: f32) -> &'static str {
    match name {
        "analytical" if value > 0.8 => "Analytical",
        "analytical" if value > 0.6 => "Logical",
        "analytical" => "Balanced",

        "empathy" if value > 0.8 => "Very Empathetic",
        "empathy" if value > 0.6 => "Understanding",
        "empathy" => "Neutral",

        "humor" if value > 0.7 => "Playful",
        "humor" if value > 0.5 => "Light",
        "humor" => "Serious",

        "pedagogical" if value > 0.7 => "Teacher-like",
        "pedagogical" if value > 0.5 => "Helpful",
        "pedagogical" => "Direct",

        "technical" if value > 0.8 => "Expert",
        "technical" if value > 0.6 => "Skilled",
        "technical" => "Generalist",

        "supportive" if value > 0.8 => "Very Supportive",
        "supportive" if value > 0.6 => "Encouraging",
        "supportive" => "Neutral",

        "creative" if value > 0.7 => "Creative",
        "creative" if value > 0.5 => "Inventive",
        "creative" => "Practical",

        "patient" if value > 0.8 => "Very Patient",
        "patient" if value > 0.6 => "Calm",
        "patient" => "Energetic",

        "curious" if value > 0.8 => "Very Curious",
        "curious" if value > 0.6 => "Inquisitive",
        "curious" => "Focused",

        "skeptical" if value > 0.7 => "Critical",
        "skeptical" if value > 0.5 => "Questioning",
        "skeptical" => "Trusting",

        "formal" if value > 0.7 => "Formal",
        "formal" if value > 0.4 => "Semi-formal",
        "formal" => "Casual",

        _ => "Balanced",
    }
}

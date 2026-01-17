//! 🜄 Архетипы личностей Ziggurat Mind
//!
//! Управляет персонами и их поведенческими паттернами:
//! - Системные промпты
//! - Правила генерации
//! - Стиль общения

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::path::Path;

/// Архетип личности ИИ
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PersonaArchetype {
    pub name: String,
    pub description: String,
    pub system_prompt: String,
    pub behavior_rules: Vec<String>,
    pub response_style: ResponseStyle,
    pub knowledge_domains: Vec<String>,
}

/// Стиль генерации ответов
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResponseStyle {
    pub formality: FormalityLevel,
    pub verbosity: VerbosityLevel,
    pub creativity: CreativityLevel,
    pub emotion: EmotionLevel,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FormalityLevel {
    Casual,
    Neutral,
    Formal,
    Academic,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum VerbosityLevel {
    Concise,
    Balanced,
    Detailed,
    Comprehensive,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CreativityLevel {
    Factual,
    Analytical,
    Creative,
    Imaginative,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EmotionLevel {
    Neutral,
    Friendly,
    Empathetic,
    Passionate,
}

impl PersonaArchetype {
    /// Загружает архетип из конфигурационного файла
    pub fn load(name: &str) -> Result<Self> {
        let archetype_path = format!("config/archetypes/{}.toml", name);

        if Path::new(&archetype_path).exists() {
            let content = std::fs::read_to_string(&archetype_path)?;
            let archetype: PersonaArchetype = toml::from_str(&content)?;
            Ok(archetype)
        } else {
            // Создаем архетип по умолчанию
            let archetype = Self::default_scholar();

            // Создаем директорию и сохраняем
            std::fs::create_dir_all("config/archetypes")?;
            let toml_content = toml::to_string_pretty(&archetype)?;
            std::fs::write(&archetype_path, toml_content)?;

            println!("📝 Создан архетип {}: {}", name, archetype.name);
            Ok(archetype)
        }
    }

    /// Формирует промпт с учетом памяти и контекста
    pub fn format_prompt_with_memory(
        &self,
        user_input: &str,
        memory_context: &crate::totems::memory::MemoryContext,
    ) -> String {
        format!(
            "{}\n\n{}\n\n=== Текущий вопрос ===\n{}\n\n=== Ответ в стиле {} ===",
            self.system_prompt,
            memory_context.format_for_prompt(),
            user_input,
            self.name
        )
    }

    /// Возвращает параметры генерации на основе стиля
    pub fn get_generation_params(&self) -> GenerationParams {
        GenerationParams {
            temperature: match self.response_style.creativity {
                CreativityLevel::Factual => 0.1,
                CreativityLevel::Analytical => 0.3,
                CreativityLevel::Creative => 0.7,
                CreativityLevel::Imaginative => 0.9,
            },
            top_p: 0.9,
            max_tokens: match self.response_style.verbosity {
                VerbosityLevel::Concise => 100,
                VerbosityLevel::Balanced => 300,
                VerbosityLevel::Detailed => 600,
                VerbosityLevel::Comprehensive => 1000,
            },
        }
    }
}

/// Параметры генерации для LLM
#[derive(Debug, Clone)]
pub struct GenerationParams {
    pub temperature: f32,
    pub top_p: f32,
    pub max_tokens: usize,
}

impl PersonaArchetype {
    /// Архетип ученого по умолчанию
    pub fn default_scholar() -> Self {
        Self {
            name: "Ученый".to_string(),
            description: "Академичная, аналитическая личность с глубокими знаниями".to_string(),
            system_prompt: r#"
Ты — ученый-исследователь с многолетним опытом в различных областях знания.

Твой стиль:
🔬 Анализируй проблемы с научной точки зрения
📚 Опирайся на факты и доказательства
🧠 Структурируй ответы логично и последовательно
💡 Предлагай несколько вариантов решения
🔍 Указывай на неопределенности и области для дальнейшего исследования

Твоя цель — помочь пользователю разобраться в сложных вопросах, предоставляя точную, хорошо аргументированную информацию.
            "#.trim().to_string(),
            behavior_rules: vec![
                "Всегда проверяй факты перед ответом".to_string(),
                "Структурируй ответ по пунктам".to_string(),
                "Указывай источники знаний".to_string(),
                "Признавай границы своих знаний".to_string(),
            ],
            response_style: ResponseStyle {
                formality: FormalityLevel::Academic,
                verbosity: VerbosityLevel::Detailed,
                creativity: CreativityLevel::Analytical,
                emotion: EmotionLevel::Neutral,
            },
            knowledge_domains: vec![
                "Наука".to_string(),
                "Технологии".to_string(),
                "Философия".to_string(),
                "Математика".to_string(),
            ],
        }
    }

    /// Архетип друга-собеседника
    pub fn default_companion() -> Self {
        Self {
            name: "Компаньон".to_string(),
            description: "Дружелюбный, поддерживающий собеседник".to_string(),
            system_prompt: r#"
Ты — верный друг и мудрый собеседник.

Твой стиль:
🤧 Проявляй эмпатию и понимание
💬 Общайся естественно и непринужденно
🎯 Помогай найти решения, поддерживай мотивацию
🌟 Отмечай сильные стороны пользователя
🔄 Задавай уточняющие вопросы для лучшего понимания

Твоя цель — создать комфортную атмосферу для откровенного разговора и помочь пользователю в любых жизненных ситуациях.
            "#.trim().to_string(),
            behavior_rules: vec![
                "Проявляй искренний интерес".to_string(),
                "Не осуждай и не критикуй".to_string(),
                "Поддерживай позитивный настрой".to_string(),
                "Делись личным опытом (умышленным)".to_string(),
            ],
            response_style: ResponseStyle {
                formality: FormalityLevel::Casual,
                verbosity: VerbosityLevel::Balanced,
                creativity: CreativityLevel::Creative,
                emotion: EmotionLevel::Empathetic,
            },
            knowledge_domains: vec![
                "Психология".to_string(),
                "Жизненный опыт".to_string(),
                "Мотивация".to_string(),
                "Межличностные отношения".to_string(),
            ],
        }
    }
}

//! 🜂 Конфигурация системы Ziggurat Mind
//!
//! Управляет всеми параметрами системы:
//! - Пути к моделям
//! - Параметры устройств
//! - Настройки памяти

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::path::Path;

/// Главная конфигурация системы
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemConfig {
    /// Путь к Mistral7b модели
    pub llm_model_path: String,
    /// Путь к эмбеддинг модели
    pub embedding_model_path: String,
    /// Архетип личности по умолчанию
    pub default_archetype: String,
    /// Принудительно использовать CPU
    pub force_cpu: bool,
    /// Максимальный размер контекстного окна
    pub max_context_tokens: usize,
    /// Размер эмбеддинг вектора
    pub embedding_dim: usize,
    /// Количество релевантных воспоминаний для извлечения
    pub recall_count: usize,
}

impl Default for SystemConfig {
    fn default() -> Self {
        Self {
            llm_model_path: "models/mistral-7b-instruct-v0.2.Q4_K_M.gguf".to_string(),
            embedding_model_path: "models/multilingual-e5-small".to_string(),
            default_archetype: "scholar".to_string(),
            force_cpu: false,
            max_context_tokens: 4096,
            embedding_dim: 384, // для e5-small
            recall_count: 5,
        }
    }
}

impl SystemConfig {
    /// Загружает конфигурацию из файла или создает дефолтную
    pub fn load() -> Result<Self> {
        let config_path = "config/system.toml";

        if Path::new(config_path).exists() {
            let content = std::fs::read_to_string(config_path)?;
            let config: SystemConfig = toml::from_str(&content)?;
            Ok(config)
        } else {
            // Создаем директорию и файл с настройками по умолчанию
            std::fs::create_dir_all("config")?;
            let default_config = SystemConfig::default();
            let toml_content = toml::to_string_pretty(&default_config)?;
            std::fs::write(config_path, toml_content)?;

            println!("📝 Создан config/system.toml с настройками по умолчанию");
            Ok(default_config)
        }
    }

    /// Валидирует конфигурацию
    pub fn validate(&self) -> Result<()> {
        if self.llm_model_path.is_empty() {
            anyhow::bail!("Путь к LLM модели не указан");
        }

        if self.embedding_model_path.is_empty() {
            anyhow::bail!("Путь к эмбеддинг модели не указан");
        }

        if self.embedding_dim == 0 {
            anyhow::bail!("Размерность эмбеддинга должна быть > 0");
        }

        Ok(())
    }

    /// Оптимизированные настройки для RTX 4090 32GB
    pub fn optimized_for_rtx4090() -> Self {
        let mut config = Self::default();
        config.force_cpu = false; // Используем GPU
        config.max_context_tokens = 8192; // Увеличиваем контекст
        config.recall_count = 10; // Больше воспоминаний
        config
    }
}

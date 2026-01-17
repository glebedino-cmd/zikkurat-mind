//! 🜂 Уровень 0: Инициация
//! 
//! Модуль инициации отвечает за:
//! - Загрузку конфигурации системы
//! - Инициацию личности (архетипов)
//! - Оркестрацию запуска всех уровней
//! - Управление параметрами модели

pub mod archetypes;
pub mod config;

use crate::priests::device::Device;
use crate::logos::inference::InferenceEngine;
use crate::totems::memory::MemorySystem;
use anyhow::Result;

/// Главный оркестратор инициации системы Ziggurat Mind
pub struct InitiationManager {
    config: config::SystemConfig,
    archetype: archetypes::PersonaArchetype,
}

impl InitiationManager {
    /// Создает новый менеджер инициации
    pub fn new() -> Result<Self> {
        let config = config::SystemConfig::load()?;
        let archetype = archetypes::PersonaArchetype::load(&config.default_archetype)?;
        
        Ok(Self { config, archetype })
    }
    
    /// Инициализирует полную систему Ziggurat Mind
    pub fn init_system(&self) -> Result<SystemComponents> {
        println!("🏛️ Инициализация Ziggurat Mind...");
        println!("📋 Архетип: {}", self.archetype.name);
        
        // Выбор устройства (GPU/CPU)
        let device = Device::select(!self.config.force_cpu)?;
        println!("⚡ Устройство: {}", device.info());
        
        // Инициализация эмбеддинг движка (память)
        let embedder = std::sync::Arc::new(
            crate::priests::embeddings::EmbeddingEngine::new(
                &self.config.embedding_model_path,
                device.clone()
            )?
        );
        println!("🧠 Эмбеддинг модель: {}", self.config.embedding_model_path);
        
        // Инициализация системы памяти
        let memory = MemorySystem::new(embedder.clone())?;
        println!("💾 Система памяти инициализирована");
        
        // Инициализация Mistral7b (Логос)
        let inference = InferenceEngine::new(
            &self.config.llm_model_path,
            device
        )?;
        println!("🤖 LLM движок готов");
        
        Ok(SystemComponents {
            inference,
            memory,
            archetype: self.archetype.clone(),
            config: self.config.clone(),
        })
    }
}

/// Компоненты полной системы после инициации
pub struct SystemComponents {
    pub inference: InferenceEngine,     // Логос - генерация
    pub memory: MemorySystem,           // Тотемы - память
    pub archetype: archetypes::PersonaArchetype, // Демиург - личность
    pub config: config::SystemConfig,   // Конфигурация
}
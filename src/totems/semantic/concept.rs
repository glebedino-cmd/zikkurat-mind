//! 📚 Семантическая память - Концепты, знания и факты
//!
//! Хранит структурированные знания: факты, правила, предпочтения и навыки
//! Извлекается автоматически из диалогов или добавляется вручную

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use uuid::Uuid;

/// Категории концептов в семантической памяти
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ConceptCategory {
    /// Факты о пользователе или мире
    Facts,
    /// Правила и инструкции
    Rules,
    /// Предпочтения и вкусы
    Preferences,
    /// Навыки и способности
    Skills,
    /// Общие знания
    General,
}

impl std::fmt::Display for ConceptCategory {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ConceptCategory::Facts => write!(f, "facts"),
            ConceptCategory::Rules => write!(f, "rules"),
            ConceptCategory::Preferences => write!(f, "preferences"),
            ConceptCategory::Skills => write!(f, "skills"),
            ConceptCategory::General => write!(f, "general"),
        }
    }
}

impl std::str::FromStr for ConceptCategory {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "facts" => Ok(ConceptCategory::Facts),
            "rules" => Ok(ConceptCategory::Rules),
            "preferences" => Ok(ConceptCategory::Preferences),
            "skills" => Ok(ConceptCategory::Skills),
            "general" => Ok(ConceptCategory::General),
            _ => Err(format!("Unknown category: {}", s)),
        }
    }
}

impl Eq for ConceptCategory {}

impl PartialEq for ConceptCategory {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (ConceptCategory::Facts, ConceptCategory::Facts) => true,
            (ConceptCategory::Rules, ConceptCategory::Rules) => true,
            (ConceptCategory::Preferences, ConceptCategory::Preferences) => true,
            (ConceptCategory::Skills, ConceptCategory::Skills) => true,
            (ConceptCategory::General, ConceptCategory::General) => true,
            _ => false,
        }
    }
}

impl Hash for ConceptCategory {
    fn hash<H: Hasher>(&self, state: &mut H) {
        match self {
            ConceptCategory::Facts => 0u8.hash(state),
            ConceptCategory::Rules => 1u8.hash(state),
            ConceptCategory::Preferences => 2u8.hash(state),
            ConceptCategory::Skills => 3u8.hash(state),
            ConceptCategory::General => 4u8.hash(state),
        }
    }
}

/// Единица семантической памяти - концепт
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Concept {
    /// Уникальный идентификатор
    pub id: Uuid,
    /// Текст концепта
    pub text: String,
    /// Категория
    pub category: ConceptCategory,
    /// Уверенность в концепте (0.0 - 1.0)
    pub confidence: f32,
    /// Источник: session_id или "manual"
    pub source: String,
    /// Векторное представление
    #[serde(skip)]
    pub embedding: Vec<f32>,
    /// Метаданные
    pub metadata: HashMap<String, String>,
    /// Время создания
    pub created_at: DateTime<Utc>,
    /// Время последнего обновления
    pub updated_at: DateTime<Utc>,
    /// Количество использований
    pub usage_count: u32,
}

impl Concept {
    /// Создает новый концепт
    pub fn new(text: String, category: ConceptCategory, source: String) -> Self {
        let now = Utc::now();
        Self {
            id: Uuid::new_v4(),
            text,
            category,
            confidence: 0.5,
            source,
            embedding: Vec::new(),
            metadata: HashMap::new(),
            created_at: now,
            updated_at: now,
            usage_count: 0,
        }
    }

    /// Создает с кастомной уверенностью
    pub fn with_confidence(mut self, confidence: f32) -> Self {
        self.confidence = confidence.clamp(0.0, 1.0);
        self
    }

    /// Добавляет метаданные
    pub fn with_metadata(mut self, key: String, value: String) -> Self {
        self.metadata.insert(key, value);
        self
    }

    /// Проверяет валидность концепта
    pub fn is_valid(&self) -> bool {
        !self.text.trim().is_empty()
    }

    /// Обновляет счетчик использования
    pub fn increment_usage(&mut self) {
        self.usage_count += 1;
        self.updated_at = Utc::now();
    }

    /// Обновляет уверенность
    pub fn update_confidence(&mut self, delta: f32) {
        self.confidence = (self.confidence + delta).clamp(0.0, 1.0);
        self.updated_at = Utc::now();
    }
}

impl Default for ConceptCategory {
    fn default() -> Self {
        ConceptCategory::General
    }
}

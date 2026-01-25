//! 📚 Семантическая память - Концепты, знания и факты
//!
//! Хранит структурированные знания: факты, правила, предпочтения и навыки
//! Извлекается автоматически из диалогов или добавляется вручную

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet, VecDeque};
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
    /// Цели и мечты
    Goals,
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
            ConceptCategory::Goals => write!(f, "goals"),
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
            "goals" => Ok(ConceptCategory::Goals),
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
            (ConceptCategory::Goals, ConceptCategory::Goals) => true,
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
            ConceptCategory::Goals => 5u8.hash(state),
            ConceptCategory::General => 4u8.hash(state),
        }
    }
}

/// Конфигурация временного затухания для категорий концептов
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecayConfig {
    /// Период затухания в днях
    pub period_days: u32,
    /// Коэффициент затухания за период (0.0 - 1.0)
    pub decay_rate: f32,
    /// Минимальная уверенность (ниже - концепт удаляется)
    pub min_confidence: f32,
}

impl Default for DecayConfig {
    fn default() -> Self {
        Self {
            period_days: 30,
            decay_rate: 0.9,
            min_confidence: 0.1,
        }
    }
}

impl ConceptCategory {
    /// Получить конфигурацию затухания для категории
    pub fn get_decay_config(&self) -> DecayConfig {
        match self {
            ConceptCategory::Facts => DecayConfig {
                period_days: 30,
                decay_rate: 0.95, // медленное затухание
                min_confidence: 0.05,
            },
            ConceptCategory::Rules => DecayConfig {
                period_days: 60,
                decay_rate: 0.98, // очень медленное затухание
                min_confidence: 0.05,
            },
            ConceptCategory::Preferences => DecayConfig {
                period_days: 20,
                decay_rate: 0.90, // среднее затухание
                min_confidence: 0.1,
            },
            ConceptCategory::Skills => DecayConfig {
                period_days: 90,
                decay_rate: 0.98, // очень медленное затухание
                min_confidence: 0.05,
            },
            ConceptCategory::Goals => DecayConfig {
                period_days: 15,
                decay_rate: 0.85, // быстрое затухание
                min_confidence: 0.1,
            },
            ConceptCategory::General => DecayConfig {
                period_days: 25,
                decay_rate: 0.92, // умеренное затухание
                min_confidence: 0.05,
            },
        }
    }
}

/// RDF-подобный Triple (Subject-Predicate-Object)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Triple {
    /// Subject concept ID
    pub subject: Uuid,
    /// Predicate (relationship type)
    pub predicate: String,
    /// Object concept ID
    pub object: Uuid,
    /// Confidence in this relationship (0.0 - 1.0)
    pub confidence: f32,
    /// When this relationship was established
    pub created_at: DateTime<Utc>,
    /// When this relationship was last verified/updated
    pub updated_at: DateTime<Utc>,
    /// Metadata about the relationship
    pub metadata: HashMap<String, String>,
}

impl Triple {
    /// Create a new triple
    pub fn new(subject: Uuid, predicate: String, object: Uuid) -> Self {
        let now = Utc::now();
        Self {
            subject,
            predicate,
            object,
            confidence: 0.5,
            created_at: now,
            updated_at: now,
            metadata: HashMap::new(),
        }
    }

    /// Create with custom confidence
    pub fn with_confidence(mut self, confidence: f32) -> Self {
        self.confidence = confidence.clamp(0.0, 1.0);
        self
    }

    /// Add metadata
    pub fn with_metadata(mut self, key: String, value: String) -> Self {
        self.metadata.insert(key, value);
        self
    }

    /// Get effective confidence with temporal decay
    pub fn get_effective_confidence(&self) -> f32 {
        let days_old = (Utc::now() - self.updated_at).num_days() as f32;
        let decay_factor = (-days_old / 90.0).exp(); // 90-day half-life
        self.confidence * decay_factor
    }
}

/// Knowledge Graph - хранит связи между концептами
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KnowledgeGraph {
    /// All triples in the graph
    pub triples: HashMap<Uuid, Triple>,
    /// Index by subject for quick lookup
    subject_index: HashMap<Uuid, Vec<Uuid>>,
    /// Index by object for quick reverse lookup
    object_index: HashMap<Uuid, Vec<Uuid>>,
    /// Index by predicate for type-based queries
    predicate_index: HashMap<String, Vec<Uuid>>,
}

impl KnowledgeGraph {
    /// Create new knowledge graph
    pub fn new() -> Self {
        Self {
            triples: HashMap::new(),
            subject_index: HashMap::new(),
            object_index: HashMap::new(),
            predicate_index: HashMap::new(),
        }
    }

    /// Add a triple to the graph
    pub fn add_triple(&mut self, triple: Triple) -> Uuid {
        let triple_id = triple.subject; // Use subject as ID for now
        let uuid = triple_id;

        // Index by subject
        self.subject_index
            .entry(triple.subject)
            .or_insert_with(Vec::new)
            .push(uuid);

        // Index by object
        self.object_index
            .entry(triple.object)
            .or_insert_with(Vec::new)
            .push(uuid);

        // Index by predicate
        self.predicate_index
            .entry(triple.predicate.clone())
            .or_insert_with(Vec::new)
            .push(uuid);

        self.triples.insert(uuid, triple);
        uuid
    }

    /// Find triples by subject
    pub fn find_by_subject(&self, subject_id: &Uuid) -> Vec<&Triple> {
        if let Some(triple_ids) = self.subject_index.get(subject_id) {
            triple_ids
                .iter()
                .filter_map(|id| self.triples.get(id))
                .collect()
        } else {
            Vec::new()
        }
    }

    /// Find triples by object (reverse lookup)
    pub fn find_by_object(&self, object_id: &Uuid) -> Vec<&Triple> {
        if let Some(triple_ids) = self.object_index.get(object_id) {
            triple_ids
                .iter()
                .filter_map(|id| self.triples.get(id))
                .collect()
        } else {
            Vec::new()
        }
    }

    /// Find triples by predicate
    pub fn find_by_predicate(&self, predicate: &str) -> Vec<&Triple> {
        if let Some(triple_ids) = self.predicate_index.get(predicate) {
            triple_ids
                .iter()
                .filter_map(|id| self.triples.get(id))
                .collect()
        } else {
            Vec::new()
        }
    }

    /// Find all related concepts (both directions)
    pub fn find_related_concepts(&self, concept_id: &Uuid) -> Vec<(Uuid, &str, f32)> {
        let mut related = Vec::new();

        // Outgoing relationships (as subject)
        for triple in self.find_by_subject(concept_id) {
            related.push((
                triple.object,
                triple.predicate.as_str(),
                triple.get_effective_confidence(),
            ));
        }

        // Incoming relationships (as object)
        for triple in self.find_by_object(concept_id) {
            related.push((
                triple.subject,
                triple.predicate.as_str(),
                triple.get_effective_confidence(),
            ));
        }

        related
    }

    /// Find paths between two concepts (simple breadth-first search)
    pub fn find_paths(&self, from: &Uuid, to: &Uuid, max_depth: usize) -> Vec<Vec<Uuid>> {
        let mut paths = Vec::new();
        let mut visited = HashSet::new();
        let mut queue = VecDeque::new();

        queue.push_back((vec![*from], 0));
        visited.insert(*from);

        while let Some((current_path, depth)) = queue.pop_front() {
            if depth >= max_depth {
                continue;
            }

            let current = current_path.last().unwrap();
            if current == to {
                paths.push(current_path.clone());
                continue;
            }

            // Find related concepts
            let related = self.find_related_concepts(current);
            for (next_id, _, _) in related {
                if !visited.contains(&next_id) {
                    visited.insert(next_id);
                    let mut new_path = current_path.clone();
                    new_path.push(next_id);
                    queue.push_back((new_path, depth + 1));
                }
            }
        }

        paths
    }

    /// Get graph statistics
    pub fn get_stats(&self) -> GraphStats {
        GraphStats {
            total_triples: self.triples.len(),
            total_predicates: self.predicate_index.len(),
            avg_degree: if !self.subject_index.is_empty() {
                self.triples.len() as f32 / self.subject_index.len() as f32
            } else {
                0.0
            },
        }
    }
}

/// Category statistics for decay
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CategoryDecayStats {
    pub total: usize,
    pub low_confidence: usize,
    pub avg_confidence: f32,
}

impl Default for CategoryDecayStats {
    fn default() -> Self {
        Self {
            total: 0,
            low_confidence: 0,
            avg_confidence: 0.0,
        }
    }
}

/// Decay statistics for concepts
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecayStats {
    pub total_concepts: usize,
    pub decayed_concepts: usize,
    pub low_confidence_concepts: usize,
    pub category_stats: HashMap<ConceptCategory, CategoryDecayStats>,
}

/// Graph statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphStats {
    pub total_triples: usize,
    pub total_predicates: usize,
    pub avg_degree: f32,
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
    /// Связанные концепты (IDs) для быстрого доступа
    #[serde(skip)]
    pub related_concepts: Vec<Uuid>,
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
            related_concepts: Vec::new(),
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

    /// Применить временное затухание к уверенности концепта
    pub fn apply_temporal_decay(&mut self) -> bool {
        let config = self.category.get_decay_config();
        let now = Utc::now();
        let days_since_update = (now - self.updated_at).num_days() as u32;

        if days_since_update < config.period_days {
            return true; // еще рано для затухания
        }

        let periods_passed = days_since_update / config.period_days;
        let decay_factor = config.decay_rate.powi(periods_passed as i32);

        self.confidence *= decay_factor;
        self.updated_at = now; // обновляем время последней обработки

        if self.confidence < config.min_confidence {
            return false; // концепт нужно удалить
        }

        true // концепт остается актуальным
    }

    /// Получить актуальную уверенность с учетом затухания (без изменения)
    pub fn get_effective_confidence(&self) -> f32 {
        let config = self.category.get_decay_config();
        let now = Utc::now();
        let days_since_update = (now - self.updated_at).num_days() as u32;

        if days_since_update < config.period_days {
            return self.confidence;
        }

        let periods_passed = days_since_update / config.period_days;
        let decay_factor = config.decay_rate.powi(periods_passed as i32);

        (self.confidence * decay_factor).max(config.min_confidence)
    }
}

impl Default for ConceptCategory {
    fn default() -> Self {
        ConceptCategory::General
    }
}

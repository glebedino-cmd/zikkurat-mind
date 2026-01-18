//! 📚 Семантическая память - База знаний и концептов
//!
//! Управляет извлеченными знаниями, концептами и убеждениями
//! Автоматически выявляет и структурирует знания из диалогов

use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use uuid::Uuid;

use crate::priests::embeddings::EmbeddingEngine;
use crate::totems::retrieval::{MemoryEntry, MemoryType, VectorStore};

/// Концепт или знание в семантической памяти
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Concept {
    /// Уникальный идентификатор концепта
    pub id: Uuid,
    /// Название концепта
    pub name: String,
    /// Определение или описание
    pub definition: String,
    /// Категория знаний
    pub category: String,
    /// Связанные концепты
    pub related_concepts: Vec<String>,
    /// Источник знания
    pub source: KnowledgeSource,
    /// Уверенность в знании (0.0 - 1.0)
    pub confidence: f32,
    /// Количество упоминаний в диалогах
    pub mention_count: usize,
    /// Последнее упоминание
    pub last_mentioned: chrono::DateTime<chrono::Utc>,
    /// Дополнительные метаданные
    pub metadata: HashMap<String, String>,
}

/// Источник знания
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum KnowledgeSource {
    /// Извлечено из диалога
    Dialogue { session_id: Uuid, turn: usize },
    /// Предустановленное знание
    Predefined,
    /// Обновлено пользователем
    UserCorrection,
    /// Выведено системой
    Inferred,
}

/// Менеджер семантической памяти
pub struct SemanticMemory {
    /// Векторное хранилище для быстрого поиска концептов
    vector_store: VectorStore,
    /// Эмбеддинг движок
    embedder: Arc<EmbeddingEngine>,
    /// Индекс концептов по имени для быстрого доступа
    concept_index: HashMap<String, Uuid>,
    /// Хранилище концептов
    concepts: HashMap<Uuid, Concept>,
    /// Категории для организации знаний
    categories: HashMap<String, Vec<Uuid>>,
    /// Статистика извлечения
    extraction_stats: ExtractionStats,
}

/// Статистика извлечения знаний
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExtractionStats {
    /// Всего обработанных диалогов
    pub total_dialogues_processed: usize,
    /// Всего извлеченных концептов
    pub total_concepts_extracted: usize,
    /// Успешных извлечений
    pub successful_extractions: usize,
    /// Последнее извлечение
    pub last_extraction: chrono::DateTime<chrono::Utc>,
}

impl Default for ExtractionStats {
    fn default() -> Self {
        Self {
            total_dialogues_processed: 0,
            total_concepts_extracted: 0,
            successful_extractions: 0,
            last_extraction: chrono::Utc::now(),
        }
    }
}

impl SemanticMemory {
    /// Создает новую семантическую память
    pub fn new(embedder: Arc<EmbeddingEngine>) -> Self {
        let dimension = embedder.embedding_dim();
        Self {
            vector_store: VectorStore::new(dimension),
            embedder,
            concept_index: HashMap::new(),
            concepts: HashMap::new(),
            categories: HashMap::new(),
            extraction_stats: ExtractionStats::default(),
        }
    }

    /// Добавляет новый концепт в память
    pub fn add_concept(&mut self, concept: Concept) -> Result<()> {
        let concept_id = concept.id;

        // Векторизуем текст концепта
        let concept_text = format!(
            "{}: {} (Category: {})",
            concept.name, concept.definition, concept.category
        );
        let embedding = self.embedder.embed(&concept_text)?;

        // Создаем запись в векторной памяти
        let memory_entry = MemoryEntry::new(
            concept_text.clone(),
            embedding,
            MemoryType::Semantic {
                category: concept.category.clone(),
            },
        )
        .with_metadata("concept_id".to_string(), concept_id.to_string())
        .with_metadata("concept_name".to_string(), concept.name.clone())
        .with_metadata("category".to_string(), concept.category.clone())
        .with_metadata("confidence".to_string(), concept.confidence.to_string());

        // Сохраняем в векторное хранилище
        self.vector_store.add(memory_entry)?;

        // Обновляем индексы
        self.concept_index.insert(concept.name.clone(), concept_id);
        let concept_clone = concept.clone();
        self.concepts.insert(concept_id, concept_clone);

        // Добавляем в категорию
        self.categories
            .entry(concept.category.clone())
            .or_insert_with(Vec::new)
            .push(concept_id);

        // Обновляем статистику
        self.extraction_stats.total_concepts_extracted += 1;

        Ok(())
    }

    /// Ищет концепты по запросу
    pub fn query_concepts(&mut self, query: &str, top_k: usize) -> Result<Vec<ConceptResult>> {
        // Векторизуем запрос
        let query_embedding = self.embedder.embed(query)?;

        // Ищем в векторной памяти
        let memory_type = MemoryType::Semantic {
            category: String::new(),
        };
        let results = self
            .vector_store
            .search_by_type(&query_embedding, &memory_type, top_k);

        // Конвертируем в результаты концептов
        let mut concept_results = Vec::new();
        for (similarity, entry) in results {
            if let Some(concept_id_str) = entry.metadata.get("concept_id") {
                if let Ok(concept_id) = Uuid::parse_str(concept_id_str) {
                    if let Some(concept) = self.concepts.get(&concept_id) {
                        concept_results.push(ConceptResult {
                            concept: concept.clone(),
                            similarity,
                            metadata: entry.metadata.clone(),
                        });
                    }
                }
            }
        }

        Ok(concept_results)
    }

    /// Автоматически извлекает концепты из диалога
    pub fn extract_concepts_from_dialogue(
        &mut self,
        dialogue: &str,
        session_id: Uuid,
        turn: usize,
    ) -> Result<usize> {
        self.extraction_stats.total_dialogues_processed += 1;

        let mut extracted_count = 0;

        // 1. Извлекаем определения (простая эвристика)
        for definition in self.extract_definitions(dialogue) {
            match self.create_concept_from_definition(definition, session_id, turn) {
                Ok(concept) => {
                    self.add_concept(concept)?;
                    extracted_count += 1;
                }
                Err(e) => {
                    eprintln!("Failed to create concept: {}", e);
                }
            }
        }

        // 2. Извлекаем факты и утверждения
        for fact in self.extract_facts(dialogue) {
            match self.create_concept_from_fact(fact, session_id, turn) {
                Ok(concept) => {
                    self.add_concept(concept)?;
                    extracted_count += 1;
                }
                Err(e) => {
                    eprintln!("Failed to create concept from fact: {}", e);
                }
            }
        }

        // 3. Извлекаем связи между концептами
        self.extract_concept_relations(dialogue, session_id, turn)?;

        if extracted_count > 0 {
            self.extraction_stats.successful_extractions += 1;
            self.extraction_stats.last_extraction = chrono::Utc::now();
        }

        Ok(extracted_count)
    }

    /// Извлекает определения из текста (X - это Y)
    fn extract_definitions(&self, text: &str) -> Vec<(String, String)> {
        let mut definitions = Vec::new();

        // Паттерны для определений
        let patterns = [
            " - это ",
            " является ",
            " — это ",
            " = ",
            " означает ",
            " это ",
        ];

        for line in text.lines() {
            for pattern in &patterns {
                if line.contains(pattern) {
                    if let Some((concept, definition)) = self.parse_definition(line, pattern) {
                        definitions.push((concept, definition));
                    }
                }
            }
        }

        definitions
    }

    /// Парсит определение из строки
    fn parse_definition(&self, line: &str, pattern: &str) -> Option<(String, String)> {
        let parts: Vec<&str> = line.split(pattern).collect();
        if parts.len() == 2 {
            Some((parts[0].trim().to_string(), parts[1].trim().to_string()))
        } else {
            None
        }
    }

    /// Извлекает факты из текста
    fn extract_facts(&self, text: &str) -> Vec<String> {
        let mut facts = Vec::new();

        // Простые паттерны для фактов
        let _fact_patterns = [
            r#"(\w+) содержит (\w+)"#,
            r#"(\w+) включает (\w+)"#,
            r#"(\w+) состоит из (\w+)"#,
            r#"(\w+) имеет (\w+)"#,
            r#"(\w+) находится в (\w+)"#,
        ];

        // Упрощенная реализация - ищем информативные предложения
        for line in text.lines() {
            // Пропускаем короткие строки
            if line.len() < 10 {
                continue;
            }

            // Ищем ключевые слова указывающие на факты
            let fact_indicators = ["известно что", "согласно", "как правило", "важно отметить"];
            for indicator in &fact_indicators {
                if line.to_lowercase().contains(indicator) {
                    facts.push(line.trim().to_string());
                    break;
                }
            }
        }

        facts
    }

    /// Создает концепт из определения
    fn create_concept_from_definition(
        &self,
        (name, definition): (String, String),
        session_id: Uuid,
        turn: usize,
    ) -> Result<Concept> {
        Ok(Concept {
            id: Uuid::new_v4(),
            name: name.clone(),
            definition,
            category: self.categorize_concept(&name),
            related_concepts: Vec::new(),
            source: KnowledgeSource::Dialogue { session_id, turn },
            confidence: 0.8, // Высокая уверенность для прямых определений
            mention_count: 1,
            last_mentioned: chrono::Utc::now(),
            metadata: HashMap::new(),
        })
    }

    /// Создает концепт из факта
    fn create_concept_from_fact(
        &self,
        fact: String,
        session_id: Uuid,
        turn: usize,
    ) -> Result<Concept> {
        Ok(Concept {
            id: Uuid::new_v4(),
            name: format!("Факт: {}", &fact[..std::cmp::min(50, fact.len())]),
            definition: fact,
            category: "факты".to_string(),
            related_concepts: Vec::new(),
            source: KnowledgeSource::Dialogue { session_id, turn },
            confidence: 0.6, // Средняя уверенность для фактов
            mention_count: 1,
            last_mentioned: chrono::Utc::now(),
            metadata: HashMap::new(),
        })
    }

    /// Категоризирует концепт на основе названия
    fn categorize_concept(&self, concept_name: &str) -> String {
        let name_lower = concept_name.to_lowercase();

        // Научные категории
        if name_lower.contains("квант") || name_lower.contains("физик") {
            return "физика".to_string();
        }
        if name_lower.contains("математик") || name_lower.contains("число") {
            return "математика".to_string();
        }
        if name_lower.contains("биолог") || name_lower.contains("клетк") {
            return "биология".to_string();
        }
        if name_lower.contains("хим") {
            return "химия".to_string();
        }

        // Технологические категории
        if name_lower.contains("программ") || name_lower.contains("код") {
            return "программирование".to_string();
        }
        if name_lower.contains("нейросет") || name_lower.contains("ai") {
            return "искусственный интеллект".to_string();
        }

        // Общие категории
        if name_lower.contains("человек") || name_lower.contains("личност") {
            return "психология".to_string();
        }
        if name_lower.contains("истор") {
            return "история".to_string();
        }

        "общие".to_string()
    }

    /// Извлекает связи между концептами
    fn extract_concept_relations(
        &mut self,
        _text: &str,
        _session_id: Uuid,
        _turn: usize,
    ) -> Result<()> {
        // TODO: Реализовать извлечение связей в будущем
        Ok(())
    }

    /// Получает концепт по имени
    pub fn get_concept_by_name(&self, name: &str) -> Option<&Concept> {
        self.concept_index
            .get(name)
            .and_then(|id| self.concepts.get(id))
    }

    /// Получает все концепты в категории
    pub fn get_concepts_by_category(&self, category: &str) -> Vec<&Concept> {
        self.categories
            .get(category)
            .map(|ids| ids.iter().filter_map(|id| self.concepts.get(id)).collect())
            .unwrap_or_default()
    }

    /// Обновляет концепт
    pub fn update_concept(&mut self, concept_id: Uuid, update: ConceptUpdate) -> Result<bool> {
        if let Some(concept) = self.concepts.get_mut(&concept_id) {
            if let Some(definition) = update.definition {
                concept.definition = definition;
                concept.last_mentioned = chrono::Utc::now();
            }
            if let Some(confidence) = update.confidence {
                concept.confidence = confidence;
            }
            if let Some(category) = update.category {
                // Удаляем из старой категории
                self.categories
                    .get_mut(&concept.category)
                    .map(|ids| ids.retain(|&id| id != concept_id));

                // Добавляем в новую
                concept.category = category.clone();
                self.categories
                    .entry(category)
                    .or_insert_with(Vec::new)
                    .push(concept_id);
            }
            concept.mention_count += 1;
            Ok(true)
        } else {
            Ok(false)
        }
    }

    /// Возвращает статистику извлечения
    pub fn get_extraction_stats(&self) -> &ExtractionStats {
        &self.extraction_stats
    }

    /// Возвращает всю статистику семантической памяти
    pub fn get_stats(&self) -> SemanticMemoryStats {
        SemanticMemoryStats {
            total_concepts: self.concepts.len(),
            total_categories: self.categories.len(),
            extraction_stats: self.extraction_stats.clone(),
            vector_store_stats: self.vector_store.stats(),
        }
    }

    /// Очищает старые концепты
    pub fn cleanup_old_concepts(&mut self, before: chrono::DateTime<chrono::Utc>) -> usize {
        let mut removed = 0;
        let mut to_remove = Vec::new();

        for (id, concept) in &self.concepts {
            if concept.last_mentioned < before {
                to_remove.push(*id);
            }
        }

        for id in to_remove {
            if let Some(concept) = self.concepts.remove(&id) {
                // Удаляем из индекса
                self.concept_index.remove(&concept.name);

                // Удаляем из категории
                if let Some(category_concepts) = self.categories.get_mut(&concept.category) {
                    category_concepts.retain(|&cat_id| cat_id != id);
                }

                removed += 1;
            }
        }

        removed
    }
}

/// Результат поиска концептов
#[derive(Debug, Clone)]
pub struct ConceptResult {
    pub concept: Concept,
    pub similarity: f32,
    pub metadata: HashMap<String, String>,
}

/// Обновление концепта
#[derive(Debug, Clone)]
pub struct ConceptUpdate {
    pub definition: Option<String>,
    pub confidence: Option<f32>,
    pub category: Option<String>,
}

/// Статистика семантической памяти
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SemanticMemoryStats {
    pub total_concepts: usize,
    pub total_categories: usize,
    pub extraction_stats: ExtractionStats,
    pub vector_store_stats: crate::totems::retrieval::VectorStoreStats,
}

impl SemanticMemoryStats {
    /// Форматирует статистику для вывода
    pub fn format(&self) -> String {
        format!(
            "📚 Semantic Memory Stats:\n   Concepts: {} in {} categories\n   Extraction Rate: {:.1}%\n   Last Extraction: {}\n   Vector Store: {} entries",
            self.total_concepts,
            self.total_categories,
            if self.extraction_stats.total_dialogues_processed > 0 {
                (self.extraction_stats.successful_extractions as f32 / self.extraction_stats.total_dialogues_processed as f32) * 100.0
            } else {
                0.0
            },
            self.extraction_stats.last_extraction.format("%Y-%m-%d %H:%M"),
            self.vector_store_stats.total_entries
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::priests::embeddings::{EmbeddingConfig, EmbeddingEngine};
    use candle_core::Device;

    #[test]
    fn test_concept_extraction() {
        let embedder = EmbeddingEngine::new("dummy_path", Device::Cpu);
        let mut semantic = SemanticMemory::new(Arc::new(embedder));

        let dialogue = "Квантовая запутанность - это явление в квантовой механике.";

        let result = semantic.extract_concepts_from_dialogue(dialogue, Uuid::new_v4(), 0);

        // В реальном тесте здесь будет проверка извлечения
        assert!(result.is_ok());
    }

    #[test]
    fn test_concept_categorization() {
        let embedder = EmbeddingEngine::new("dummy_path", Device::Cpu);
        let semantic = SemanticMemory::new(Arc::new(embedder));

        assert_eq!(semantic.categorize_concept("квантовая механика"), "физика");
        assert_eq!(
            semantic.categorize_concept("нейронные сети"),
            "искусственный интеллект"
        );
        assert_eq!(
            semantic.categorize_concept("программирование"),
            "программирование"
        );
    }
}

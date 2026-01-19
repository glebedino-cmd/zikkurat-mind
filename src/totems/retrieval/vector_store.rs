//! 🜃 Уровень 2: Тотемы Памяти - Векторный поиск
//!
//! In-memory векторная база данных для поиска семантически схожих записей
//! Оптимизирована для cosine similarity и быстрого извлечения

#![allow(dead_code)]

use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;

/// Тип памяти для классификации записей
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum MemoryType {
    /// Эпизодическая память (диалоги, события)
    Episodic { session_id: Uuid, turn: usize },
    /// Семантическая память (знания, концепты)
    Semantic { category: String },
    /// Кратковременная память (текущий контекст)
    ShortTerm,
}

/// Запись в векторной базе данных
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryEntry {
    /// Уникальный идентификатор записи
    pub id: Uuid,
    /// Исходный текст
    pub text: String,
    /// Векторное представление
    pub embedding: Vec<f32>,
    /// Метаданные для дополнительной информации
    pub metadata: HashMap<String, String>,
    /// Временная метка создания
    pub timestamp: chrono::DateTime<chrono::Utc>,
    /// Тип памяти
    pub memory_type: MemoryType,
}

impl MemoryEntry {
    /// Создает новую запись
    pub fn new(text: String, embedding: Vec<f32>, memory_type: MemoryType) -> Self {
        Self {
            id: Uuid::new_v4(),
            text,
            embedding,
            metadata: HashMap::new(),
            timestamp: chrono::Utc::now(),
            memory_type,
        }
    }

    /// Добавляет метаданные
    pub fn with_metadata(mut self, key: String, value: String) -> Self {
        self.metadata.insert(key, value);
        self
    }
}

/// In-memory векторное хранилище с поиском по косинусному сходству
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VectorStore {
    /// Векторные записи
    entries: Vec<MemoryEntry>,
    /// Размерность векторов
    dimension: usize,
    /// Общее количество запросов к хранилищу
    #[serde(skip)]
    query_count: u64,
}

impl VectorStore {
    /// Создает новое хранилище
    pub fn new(dimension: usize) -> Self {
        Self {
            entries: Vec::new(),
            dimension,
            query_count: 0,
        }
    }

    /// Добавляет запись в хранилище
    pub fn add(&mut self, entry: MemoryEntry) -> Result<()> {
        // Проверяем размерность вектора
        if entry.embedding.len() != self.dimension {
            return Err(anyhow!(
                "Embedding dimension mismatch: expected {}, got {}",
                self.dimension,
                entry.embedding.len()
            ));
        }

        self.entries.push(entry);
        Ok(())
    }

    /// Добавляет несколько записей (batch operation)
    pub fn add_batch(&mut self, entries: Vec<MemoryEntry>) -> Result<()> {
        for entry in entries {
            self.add(entry)?;
        }
        Ok(())
    }

    /// Ищет наиболее похожие записи по косинусному сходству
    pub fn search(&mut self, query_embedding: &[f32], top_k: usize) -> Vec<(f32, &MemoryEntry)> {
        self.query_count += 1;

        if query_embedding.len() != self.dimension {
            return Vec::new();
        }

        let mut similarities: Vec<(f32, &MemoryEntry)> = self
            .entries
            .iter()
            .map(|entry| {
                let similarity = cosine_similarity(query_embedding, &entry.embedding);
                (similarity, entry)
            })
            .collect();

        // Сортируем по убыванию сходства
        similarities.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());

        // Возвращаем top_k результатов
        similarities.truncate(top_k);
        similarities
    }

    /// Ищет записи по типу памяти
    pub fn search_by_type(
        &mut self,
        query_embedding: &[f32],
        memory_type: &MemoryType,
        top_k: usize,
    ) -> Vec<(f32, &MemoryEntry)> {
        self.query_count += 1;

        if query_embedding.len() != self.dimension {
            return Vec::new();
        }

        eprintln!(
            "DEBUG search_by_type: entries.len() = {}, dimension = {}",
            self.entries.len(),
            self.dimension
        );

        // Фильтруем по типу памяти
        let filtered_entries: Vec<&MemoryEntry> = self
            .entries
            .iter()
            .filter(|entry| match (&entry.memory_type, memory_type) {
                (MemoryType::Episodic { .. }, MemoryType::Episodic { .. }) => true,
                (MemoryType::Semantic { .. }, MemoryType::Semantic { .. }) => true,
                (MemoryType::ShortTerm, MemoryType::ShortTerm) => true,
                _ => false,
            })
            .collect();

        eprintln!(
            "DEBUG search_by_type: filtered_entries.len() = {}",
            filtered_entries.len()
        );

        let mut similarities: Vec<(f32, &MemoryEntry)> = filtered_entries
            .iter()
            .map(|entry| {
                let similarity = cosine_similarity(query_embedding, &entry.embedding);
                eprintln!(
                    "DEBUG search_by_type: similarity = {:.4}, text = {}",
                    similarity, entry.text
                );
                (similarity, *entry)
            })
            .collect();

        similarities.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
        similarities.truncate(top_k);
        similarities
    }

    /// Возвращает все записи указанного типа
    pub fn get_by_type(&self, memory_type: &MemoryType) -> Vec<&MemoryEntry> {
        self.entries
            .iter()
            .filter(|entry| match (&entry.memory_type, memory_type) {
                (MemoryType::Episodic { .. }, MemoryType::Episodic { .. }) => true,
                (MemoryType::Semantic { .. }, MemoryType::Semantic { .. }) => true,
                (MemoryType::ShortTerm, MemoryType::ShortTerm) => true,
                _ => false,
            })
            .collect()
    }

    /// Удаляет записи старше указанного времени
    pub fn cleanup_old(&mut self, before: chrono::DateTime<chrono::Utc>) -> usize {
        let initial_len = self.entries.len();
        self.entries.retain(|entry| entry.timestamp > before);
        initial_len - self.entries.len()
    }

    /// Удаляет записи по типу
    pub fn clear_by_type(&mut self, memory_type: &MemoryType) -> usize {
        let initial_len = self.entries.len();
        self.entries
            .retain(|entry| !match (&entry.memory_type, memory_type) {
                (MemoryType::Episodic { .. }, MemoryType::Episodic { .. }) => true,
                (MemoryType::Semantic { .. }, MemoryType::Semantic { .. }) => true,
                (MemoryType::ShortTerm, MemoryType::ShortTerm) => true,
                _ => false,
            });
        initial_len - self.entries.len()
    }

    /// Статистика хранилища
    pub fn stats(&self) -> VectorStoreStats {
        let mut episodic_count = 0;
        let mut semantic_count = 0;
        let mut short_term_count = 0;

        for entry in &self.entries {
            match entry.memory_type {
                MemoryType::Episodic { .. } => episodic_count += 1,
                MemoryType::Semantic { .. } => semantic_count += 1,
                MemoryType::ShortTerm => short_term_count += 1,
            }
        }

        VectorStoreStats {
            total_entries: self.entries.len(),
            episodic_count,
            semantic_count,
            short_term_count,
            dimension: self.dimension,
            query_count: self.query_count,
        }
    }

    /// Размер хранилища в байтах (приблизительно)
    pub fn size_bytes(&self) -> usize {
        let base_size = std::mem::size_of::<VectorStore>();
        let entries_size = self
            .entries
            .iter()
            .map(|e| {
                std::mem::size_of::<MemoryEntry>()
                    + e.text.len()
                    + e.embedding.len() * std::mem::size_of::<f32>()
                    + e.metadata.len() * (std::mem::size_of::<String>() + 32) // примерно
            })
            .sum::<usize>();

        base_size + entries_size
    }

    /// Очищает все записи
    pub fn clear(&mut self) {
        self.entries.clear();
        self.query_count = 0;
    }

    /// Возвращает количество записей
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Проверяет пустое ли хранилище
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Возвращает размерность векторов
    pub fn dimension(&self) -> usize {
        self.dimension
    }
}

/// Статистика векторного хранилища
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VectorStoreStats {
    pub total_entries: usize,
    pub episodic_count: usize,
    pub semantic_count: usize,
    pub short_term_count: usize,
    pub dimension: usize,
    pub query_count: u64,
}

impl VectorStoreStats {
    /// Форматирует статистику для вывода
    pub fn format(&self) -> String {
        format!(
            "📊 VectorStore Stats:\n   Entries: {} total ({} episodic, {} semantic, {} short-term)\n   Dimension: {}D\n   Queries: {}",
            self.total_entries,
            self.episodic_count,
            self.semantic_count,
            self.short_term_count,
            self.dimension,
            self.query_count
        )
    }
}

/// Вычисляет косинусное сходство между двумя векторами
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() {
        return 0.0;
    }

    let dot_product: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

    if norm_a == 0.0 || norm_b == 0.0 {
        return 0.0;
    }

    dot_product / (norm_a * norm_b)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cosine_similarity() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![0.0, 1.0, 0.0];
        let c = vec![1.0, 0.0, 0.0];
        let d = vec![2.0, 0.0, 0.0];

        assert_eq!(cosine_similarity(&a, &b), 0.0);
        assert_eq!(cosine_similarity(&a, &c), 1.0);
        assert_eq!(cosine_similarity(&a, &d), 1.0);
    }

    #[test]
    fn test_vector_store_basic() {
        let mut store = VectorStore::new(3);

        let entry1 = MemoryEntry::new(
            "hello".to_string(),
            vec![1.0, 0.0, 0.0],
            MemoryType::ShortTerm,
        );

        let entry2 = MemoryEntry::new(
            "world".to_string(),
            vec![0.0, 1.0, 0.0],
            MemoryType::ShortTerm,
        );

        store.add(entry1).unwrap();
        store.add(entry2).unwrap();

        assert_eq!(store.len(), 2);
        assert_eq!(store.dimension(), 3);
    }

    #[test]
    fn test_search() {
        let mut store = VectorStore::new(3);

        store
            .add(MemoryEntry::new(
                "hello".to_string(),
                vec![1.0, 0.0, 0.0],
                MemoryType::ShortTerm,
            ))
            .unwrap();

        store
            .add(MemoryEntry::new(
                "world".to_string(),
                vec![0.0, 1.0, 0.0],
                MemoryType::ShortTerm,
            ))
            .unwrap();

        let results = store.search(&vec![1.0, 0.0, 0.0], 2);
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].0, 1.0); // Первое совпадение
        assert_eq!(results[1].0, 0.0); // Второе совпадение
    }

    #[test]
    fn test_memory_type_filtering() {
        let mut store = VectorStore::new(3);

        store
            .add(MemoryEntry::new(
                "dialogue".to_string(),
                vec![1.0, 0.0, 0.0],
                MemoryType::Episodic {
                    session_id: Uuid::new_v4(),
                    turn: 1,
                },
            ))
            .unwrap();

        store
            .add(MemoryEntry::new(
                "knowledge".to_string(),
                vec![0.0, 1.0, 0.0],
                MemoryType::Semantic {
                    category: "science".to_string(),
                },
            ))
            .unwrap();

        let episodic_entries = store.get_by_type(&MemoryType::Episodic {
            session_id: Uuid::nil(),
            turn: 0,
        });
        assert_eq!(episodic_entries.len(), 1);

        let semantic_entries = store.get_by_type(&MemoryType::Semantic {
            category: String::new(),
        });
        assert_eq!(semantic_entries.len(), 1);
    }
}

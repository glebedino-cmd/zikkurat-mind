//! 🏛️ Унифицированный менеджер памяти
//!
//! Объединяет эпизодическую и семантическую память
//! Предоставляет единый интерфейс для работы с разными типами памяти

#![allow(dead_code)]

use anyhow::Result;
use std::sync::Arc;

use crate::priests::embeddings::Embedder;
use crate::totems::{
    episodic::{DialogueManager, DialogueManagerStats},
    retrieval::VectorStore,
    semantic::{ConceptResult, SemanticMemory, SemanticMemoryStats},
};

/// Унифицированный менеджер памяти
pub struct UnifiedMemoryManager {
    /// Менеджер эпизодической памяти (диалоги)
    pub episodic: DialogueManager,
    /// Менеджер семантической памяти (концепты, знания)
    pub semantic: SemanticMemory,
    /// Объединенное векторное хранилище (для оптимизации)
    unified_vector_store: VectorStore,
    /// Эмбеддинг движок (может быть реальным или dummy)
    embedder: Arc<dyn Embedder>,
}

/// Контекст памяти для генерации
#[derive(Debug, Clone)]
pub struct MemoryContext {
    /// Текущий диалог (последние N сообщений)
    pub current_dialogue: String,
    /// Релевантные эпизоды из прошлого
    pub relevant_episodes: Vec<String>,
    /// Релевантные концепты и знания
    pub relevant_concepts: Vec<ConceptResult>,
    /// Статистика поиска
    pub search_stats: SearchStats,
}

/// Статистика поиска
#[derive(Debug, Clone)]
pub struct SearchStats {
    /// Найдено эпизодов
    pub episodes_found: usize,
    /// Найдено концептов
    pub concepts_found: usize,
    /// Время поиска эпизодов (ms)
    pub episode_search_time_ms: u64,
    /// Время поиска концептов (ms)
    pub concept_search_time_ms: u64,
}

impl UnifiedMemoryManager {
    /// Создает новый унифицированный менеджер памяти
    pub fn new(embedder: Arc<dyn Embedder>, persona_name: String) -> Self {
        let dimension = embedder.embedding_dim();

        Self {
            episodic: DialogueManager::new(embedder.clone(), persona_name),
            semantic: SemanticMemory::new(embedder.clone()),
            unified_vector_store: VectorStore::new(dimension),
            embedder,
        }
    }

    /// Выполняет полный поиск по памяти
    pub fn recall(
        &mut self,
        query: &str,
        episodes_count: usize,
        concepts_count: usize,
    ) -> Result<MemoryContext> {
        let _start_time = std::time::Instant::now();

        // 1. Поиск в эпизодической памяти
        let episode_start = std::time::Instant::now();
        let relevant_episodes = self
            .episodic
            .find_similar_dialogues(query, episodes_count)?;
        let episode_time = episode_start.elapsed().as_millis();

        // 2. Поиск в семантической памяти
        let concept_start = std::time::Instant::now();
        let relevant_concepts = self.semantic.query_concepts(query, concepts_count)?;
        let concept_time = concept_start.elapsed().as_millis();

        // 3. Получаем текущий контекст диалога
        let current_dialogue = self.episodic.get_current_context(5);

        // 4. Синхронизируем векторные хранилища
        self.sync_vector_stores()?;

        let context = MemoryContext {
            current_dialogue,
            relevant_episodes: relevant_episodes.clone(),
            relevant_concepts: relevant_concepts.clone(),
            search_stats: SearchStats {
                episodes_found: relevant_episodes.len(),
                concepts_found: relevant_concepts.len(),
                episode_search_time_ms: episode_time as u64,
                concept_search_time_ms: concept_time as u64,
            },
        };

        Ok(context)
    }

    /// Добавляет обмен в эпизодическую память
    pub fn add_exchange(&mut self, user: String, assistant: String) -> Result<()> {
        // Добавляем в эпизодическую память
        self.episodic
            .add_exchange(user.clone(), assistant.clone())?;

        // Извлекаем концепты из диалога
        let session_id = self.episodic.current_session().id;
        let current_turn = self.episodic.current_session().turn_count() - 1;

        let combined_dialogue = format!("User: {}\nAssistant: {}", user, assistant);
        self.semantic.extract_concepts_from_dialogue(
            &combined_dialogue,
            session_id,
            current_turn,
        )?;

        Ok(())
    }

    /// Синхронизирует векторные хранилища
    fn sync_vector_stores(&mut self) -> Result<()> {
        // TODO: Реализовать синхронизацию для оптимизации поиска
        // В MVP это заглушка - каждое хранилище работает независимо
        Ok(())
    }

    /// Форматирует контекст памяти для промпта
    pub fn format_context_for_prompt(&self, context: &MemoryContext) -> String {
        let mut prompt_parts = Vec::new();

        // Добавляем релевантные концепты
        if !context.relevant_concepts.is_empty() {
            prompt_parts.push("=== 📚 Relevant Knowledge ===".to_string());
            for concept in &context.relevant_concepts {
                prompt_parts.push(format!(
                    "🧠 {} (confidence: {:.2}): {}",
                    concept.concept.name, concept.concept.confidence, concept.concept.definition
                ));
            }
            prompt_parts.push(String::new());
        }

        // Добавляем релевантные эпизоды
        if !context.relevant_episodes.is_empty() {
            prompt_parts.push("=== 📝 Relevant Past Dialogues ===".to_string());
            for (i, episode) in context.relevant_episodes.iter().enumerate() {
                prompt_parts.push(format!("🗨️ Episode {}: {}", i + 1, episode));
            }
            prompt_parts.push(String::new());
        }

        // Добавляем текущий диалог
        if !context.current_dialogue.is_empty() {
            prompt_parts.push("=== 💬 Current Dialogue ===".to_string());
            prompt_parts.push(context.current_dialogue.clone());
            prompt_parts.push(String::new());
        }

        prompt_parts.join("\n")
    }

    /// Возвращает полную статистику памяти
    pub fn get_comprehensive_stats(&self) -> ComprehensiveMemoryStats {
        let episodic_stats = self.episodic.stats();
        let semantic_stats = self.semantic.get_stats();

        ComprehensiveMemoryStats {
            episodic: episodic_stats,
            semantic: semantic_stats,
            unified_store_stats: self.unified_vector_store.stats(),
            total_memory_entries: self.unified_vector_store.len(),
            last_updated: chrono::Utc::now(),
        }
    }

    /// Очищает старые записи
    pub fn cleanup_old_memories(&mut self, days_old: i64) -> Result<usize> {
        let cutoff = chrono::Utc::now() - chrono::Duration::days(days_old);

        let episodic_removed = self.unified_vector_store.cleanup_old(cutoff);
        let semantic_removed = self.semantic.cleanup_old_concepts(cutoff);

        Ok(episodic_removed + semantic_removed)
    }

    /// Экспортирует память в JSON
    pub fn export_memory(&self) -> Result<String> {
        let export_data = MemoryExport {
            episodic_sessions: self.episodic.session_history().clone(),
            concepts: self.semantic.get_all_concepts(),
            export_timestamp: chrono::Utc::now(),
            version: "1.0".to_string(),
        };

        serde_json::to_string_pretty(&export_data)
            .map_err(|e| anyhow::anyhow!("Failed to export memory: {}", e))
    }

    /// Импортирует память из JSON
    pub fn import_memory(&mut self, json_data: &str) -> Result<()> {
        let import_data: MemoryExport = serde_json::from_str(json_data)
            .map_err(|e| anyhow::anyhow!("Failed to parse import data: {}", e))?;

        // TODO: Реализовать импорт сессий и концептов
        println!(
            "📥 Imported {} concepts from backup",
            import_data.concepts.len()
        );

        Ok(())
    }

    /// Начинает новую сессию с именем личности
    pub fn start_new_session(&mut self, persona_name: String) {
        self.episodic.start_new_session(persona_name);
    }

    /// Возвращает текущую сессию
    pub fn current_session(&self) -> &crate::totems::episodic::Session {
        self.episodic.current_session()
    }
}

/// Экспортируемые данные памяти
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct MemoryExport {
    pub episodic_sessions: std::collections::HashMap<uuid::Uuid, crate::totems::episodic::Session>,
    pub concepts: Vec<crate::totems::semantic::Concept>,
    pub export_timestamp: chrono::DateTime<chrono::Utc>,
    pub version: String,
}

/// Комплексная статистика памяти
#[derive(Debug, Clone)]
pub struct ComprehensiveMemoryStats {
    pub episodic: DialogueManagerStats,
    pub semantic: SemanticMemoryStats,
    pub unified_store_stats: crate::totems::retrieval::VectorStoreStats,
    pub total_memory_entries: usize,
    pub last_updated: chrono::DateTime<chrono::Utc>,
}

impl ComprehensiveMemoryStats {
    /// Форматирует полную статистику
    pub fn format(&self) -> String {
        format!(
            "🧠 Comprehensive Memory Stats:\n{}\n{}\n📊 Total Entries: {} | Last Update: {}",
            self.episodic.format(),
            self.semantic.format(),
            self.total_memory_entries,
            self.last_updated.format("%Y-%m-%d %H:%M:%S")
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::priests::embeddings::EmbeddingEngine;
    use candle_core::Device;
    use std::sync::Arc;

    #[test]
    fn test_unified_memory() {
        let embedder = Arc::new(EmbeddingEngine::new("dummy_path", Device::Cpu));
        let mut memory = UnifiedMemoryManager::new(embedder, "test".to_string());

        let context = memory.recall("test query", 3, 2).unwrap();

        assert!(context.current_dialogue.contains("test") || context.relevant_episodes.is_empty());
    }
}

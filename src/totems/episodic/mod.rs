//! 📜 Эпизодическая память - История диалогов и событий
//!
//! Управляет диалоговыми сессиями с автоматической векторизацией
//! и поиском похожих разговоров из прошлого

#![allow(dead_code)]

use anyhow::Result;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use uuid::Uuid;

use crate::priests::embeddings::Embedder;
use crate::totems::retrieval::{MemoryEntry, MemoryType, VectorStore};

/// Обмен в диалоге (пользователь - ассистент)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Turn {
    /// Вопрос пользователя
    pub user: String,
    /// Ответ ассистента
    pub assistant: String,
    /// Временная метка
    pub timestamp: DateTime<Utc>,
    /// Дополнительные метаданные
    pub metadata: HashMap<String, String>,
}

impl Turn {
    /// Создает новый обмен
    pub fn new(user: String, assistant: String) -> Self {
        Self {
            user,
            assistant,
            timestamp: Utc::now(),
            metadata: HashMap::new(),
        }
    }

    /// Объединенный текст для векторизации
    pub fn combined_text(&self) -> String {
        format!("User: {}\nAssistant: {}", self.user, self.assistant)
    }

    /// Добавляет метаданные
    pub fn with_metadata(mut self, key: String, value: String) -> Self {
        self.metadata.insert(key, value);
        self
    }
}

/// Диалоговая сессия
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Session {
    /// Уникальный идентификатор сессии
    pub id: Uuid,
    /// Имя личности (архетипа)
    pub persona_name: String,
    /// Список обменов в диалоге
    pub turns: Vec<Turn>,
    /// Время создания сессии
    pub created_at: DateTime<Utc>,
    /// Время последнего обновления
    pub updated_at: DateTime<Utc>,
    /// Метаданные сессии
    pub metadata: HashMap<String, String>,
}

impl Session {
    /// Создает новую сессию
    pub fn new(persona_name: String) -> Self {
        let now = Utc::now();
        Self {
            id: Uuid::new_v4(),
            persona_name,
            turns: Vec::new(),
            created_at: now,
            updated_at: now,
            metadata: HashMap::new(),
        }
    }

    /// Добавляет обмен в сессию
    pub fn add_turn(&mut self, turn: Turn) {
        self.turns.push(turn);
        self.updated_at = Utc::now();
    }

    /// Возвращает количество обменов
    pub fn turn_count(&self) -> usize {
        self.turns.len()
    }

    /// Возвращает текст последнего обмена
    pub fn last_turn(&self) -> Option<&Turn> {
        self.turns.last()
    }

    /// Получает последние N обменов
    pub fn last_turns(&self, n: usize) -> &[Turn] {
        let start = if self.turns.len() > n {
            self.turns.len() - n
        } else {
            0
        };
        &self.turns[start..]
    }

    /// Формирует контекст из последних обменов
    pub fn format_context(&self, max_turns: usize) -> String {
        let recent_turns = self.last_turns(max_turns);
        let mut context = String::new();

        for turn in recent_turns {
            context.push_str(&format!(
                "User: {}\nAssistant: {}\n\n",
                turn.user, turn.assistant
            ));
        }

        context.trim_end().to_string()
    }
}

/// Менеджер эпизодической памяти
pub struct DialogueManager {
    /// Текущая сессия
    current_session: Session,
    /// Векторное хранилище для быстрого поиска
    vector_store: VectorStore,
    /// Эмбеддинг движок
    embedder: Arc<dyn Embedder>,
    /// История всех сессий
    session_history: HashMap<Uuid, Session>,
    /// Максимальное количество хранимых сессий
    max_sessions: usize,
}

impl DialogueManager {
    /// Создает новый менеджер диалогов
    pub fn new(embedder: Arc<dyn Embedder>, persona_name: String) -> Self {
        let dimension = embedder.embedding_dim();
        Self {
            current_session: Session::new(persona_name),
            vector_store: VectorStore::new(dimension),
            embedder,
            session_history: HashMap::new(),
            max_sessions: 100, // Ограничиваем количество сессий
        }
    }

    /// Создает с кастомными параметрами
    pub fn with_config(
        embedder: Arc<dyn Embedder>,
        persona_name: String,
        max_sessions: usize,
    ) -> Self {
        let dimension = embedder.embedding_dim();
        Self {
            current_session: Session::new(persona_name),
            vector_store: VectorStore::new(dimension),
            embedder,
            session_history: HashMap::new(),
            max_sessions,
        }
    }

    /// Добавляет обмен в текущую сессию и векторизует его
    pub fn add_exchange(&mut self, user: String, assistant: String) -> Result<()> {
        let turn = Turn::new(user.clone(), assistant.clone());
        let turn_id = self.current_session.turn_count();

        // Сохраняем обмен в сессии
        self.current_session.add_turn(turn.clone());

        // Векторизуем объединенный текст
        let context_text = turn.combined_text();
        let embedding = self.embedder.embed(&context_text)?;
        eprintln!("DEBUG add_exchange: embedding.len() = {}", embedding.len());

        // Создаем запись в векторной памяти
        let memory_entry = MemoryEntry::new(
            context_text.clone(),
            embedding,
            MemoryType::Episodic {
                session_id: self.current_session.id,
                turn: turn_id,
            },
        )
        .with_metadata(
            "session_id".to_string(),
            self.current_session.id.to_string(),
        )
        .with_metadata("turn".to_string(), turn_id.to_string())
        .with_metadata(
            "persona".to_string(),
            self.current_session.persona_name.clone(),
        );

        // Добавляем в векторное хранилище
        self.vector_store.add(memory_entry)?;
        eprintln!(
            "DEBUG add_exchange: vector_store.len() = {}",
            self.vector_store.len()
        );

        Ok(())
    }

    /// Ищет похожие диалоги по запросу
    pub fn find_similar_dialogues(&mut self, query: &str, top_k: usize) -> Result<Vec<String>> {
        // Векторизуем запрос
        let query_embedding = self.embedder.embed(query)?;

        // Ищем похожие эпизодические записи
        let memory_type = MemoryType::Episodic {
            session_id: Uuid::nil(),
            turn: 0, // Используем нулевой turn для поиска всех эпизодов
        };

        let results = self
            .vector_store
            .search_by_type(&query_embedding, &memory_type, top_k);

        // Формируем текстовые результаты
        let mut dialogues = Vec::new();
        for (similarity, entry) in results {
            let formatted = format!(
                "[Similarity: {:.3}] Session: {} - {}",
                similarity,
                entry
                    .metadata
                    .get("session_id")
                    .unwrap_or(&"unknown".to_string()),
                entry.text
            );
            dialogues.push(formatted);
        }

        Ok(dialogues)
    }

    /// Ищет диалоги с конкретной сессии
    pub fn find_session_dialogues(&self, session_id: &Uuid, top_k: usize) -> Vec<String> {
        let memory_type = MemoryType::Episodic {
            session_id: *session_id,
            turn: 0,
        };

        let entries = self.vector_store.get_by_type(&memory_type);
        let mut dialogues = Vec::new();

        for entry in entries.iter().take(top_k) {
            dialogues.push(format!(
                "Turn {}: {}",
                entry.metadata.get("turn").unwrap_or(&"?".to_string()),
                entry.text
            ));
        }

        dialogues
    }

    /// Получает контекст текущей сессии
    pub fn get_current_context(&self, max_turns: usize) -> String {
        self.current_session.format_context(max_turns)
    }

    /// Начинает новую сессию
    pub fn start_new_session(&mut self, persona_name: String) -> Uuid {
        // Сохраняем текущую сессию в историю
        let old_session_id = self.current_session.id;
        self.session_history
            .insert(old_session_id, self.current_session.clone());

        // Очищаем старую сессию из векторной памяти (опционально)
        let cutoff = Utc::now() - chrono::Duration::days(7); // Удаляем сессии старше недели
        self.vector_store.cleanup_old(cutoff);

        // Ограничиваем количество сессий
        if self.session_history.len() > self.max_sessions {
            let oldest_sessions = self
                .session_history
                .iter()
                .min_by_key(|(_, s)| s.created_at)
                .map(|(id, _)| *id);

            if let Some(oldest_id) = oldest_sessions {
                self.session_history.remove(&oldest_id);
                // Также очищаем связанные записи из векторной памяти
                let memory_type = MemoryType::Episodic {
                    session_id: oldest_id,
                    turn: 0,
                };
                self.vector_store.clear_by_type(&memory_type);
            }
        }

        // Создаем новую сессию
        self.current_session = Session::new(persona_name);
        self.current_session.id
    }

    /// Возвращает текущую сессию
    pub fn current_session(&self) -> &Session {
        &self.current_session
    }

    /// Возвращает историю сессий
    pub fn session_history(&self) -> &HashMap<Uuid, Session> {
        &self.session_history
    }

    /// Возвращает статистику
    pub fn stats(&self) -> DialogueManagerStats {
        let store_stats = self.vector_store.stats();

        DialogueManagerStats {
            current_session_id: self.current_session.id,
            current_session_turns: self.current_session.turn_count(),
            total_sessions: self.session_history.len() + 1, // +1 for current
            total_turns: store_stats.episodic_count,
            last_activity: self.current_session.updated_at,
        }
    }

    /// Загружает сессию из истории
    pub fn load_session(&mut self, session_id: Uuid) -> Result<bool> {
        if let Some(session) = self.session_history.get(&session_id).cloned() {
            // Сохраняем текущую сессию
            let current_id = self.current_session.id;
            self.session_history
                .insert(current_id, self.current_session.clone());

            // Загружаем запрошенную сессию
            self.current_session = session;
            Ok(true)
        } else {
            Ok(false)
        }
    }

    /// Удаляет сессию из истории и векторной памяти
    pub fn delete_session(&mut self, session_id: Uuid) -> bool {
        let existed = self.session_history.remove(&session_id).is_some();

        if existed {
            // Очищаем записи из векторной памяти
            let memory_type = MemoryType::Episodic {
                session_id,
                turn: 0,
            };
            self.vector_store.clear_by_type(&memory_type);
        }

        existed
    }
}

/// Статистика менеджера диалогов
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DialogueManagerStats {
    pub current_session_id: Uuid,
    pub current_session_turns: usize,
    pub total_sessions: usize,
    pub total_turns: usize,
    pub last_activity: DateTime<Utc>,
}

impl DialogueManagerStats {
    /// Форматирует статистику для вывода
    pub fn format(&self) -> String {
        format!(
            "💬 Dialogue Manager Stats:\n   Current Session: {} ({} turns)\n   Total Sessions: {}\n   Total Turns: {}\n   Last Activity: {}",
            self.current_session_id,
            self.current_session_turns,
            self.total_sessions,
            self.total_turns,
            self.last_activity.format("%Y-%m-%d %H:%M:%S")
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::priests::embeddings::{EmbeddingConfig, EmbeddingEngine};
    use candle_core::Device;

    #[tokio::test]
    async fn test_dialogue_manager() -> Result<()> {
        // Создаем тестовый эмбеддинг движок (заглушка)
        // В реальном коде здесь будет настоящая модель
        let embedder = Arc::new(create_test_embedder()?);
        let mut manager = DialogueManager::new(embedder.clone(), "test_persona".to_string());

        // Добавляем обмен
        manager
            .add_exchange(
                "Hello, how are you?".to_string(),
                "I'm doing well, thank you!".to_string(),
            )
            .await?;

        // Проверяем статистику
        let stats = manager.stats();
        assert_eq!(stats.current_session_turns, 1);

        // Проверяем контекст
        let context = manager.get_current_context(5);
        assert!(context.contains("Hello, how are you?"));
        assert!(context.contains("I'm doing well, thank you!"));

        Ok(())
    }

    fn create_test_embedder() -> Result<EmbeddingEngine> {
        // В реальных тестах здесь будет настоящая модель
        // Пока возвращаем ошибку для демонстрации
        Err(anyhow!("Test embedder not implemented"))
    }
}

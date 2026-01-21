//! 📜 Эпизодическая память - История диалогов и событий
//!
//! Управляет диалоговыми сессиями с автоматической векторизацией
//! и поиском похожих разговоров из прошлого

#![allow(dead_code)]

pub mod persistence;

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
    pub fn format_context(&self, max_turns: usize, max_chars: usize) -> String {
        let recent_turns = self.last_turns(max_turns);
        let mut context = String::new();

        for turn in recent_turns {
            let user_char_count = turn.user.chars().count();
            let user = if user_char_count > max_chars / 4 {
                let byte_pos = turn
                    .user
                    .char_indices()
                    .nth(max_chars / 4)
                    .unwrap_or((turn.user.len(), ' '))
                    .0;
                &turn.user[..byte_pos]
            } else {
                &turn.user
            };

            let assistant_char_count = turn.assistant.chars().count();
            let assistant = if assistant_char_count > max_chars / 4 {
                let byte_pos = turn
                    .assistant
                    .char_indices()
                    .nth(max_chars / 4)
                    .unwrap_or((turn.assistant.len(), ' '))
                    .0;
                &turn.assistant[..byte_pos]
            } else {
                &turn.assistant
            };
            context.push_str(&format!("User: {}\nAssistant: {}\n\n", user, assistant));

            if context.chars().count() > max_chars {
                let context_byte_pos = context
                    .char_indices()
                    .nth(max_chars)
                    .unwrap_or((context.len(), ' '))
                    .0;
                return context[..context_byte_pos].to_string();
            }
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

impl Clone for DialogueManager {
    fn clone(&self) -> Self {
        Self {
            current_session: self.current_session.clone(),
            vector_store: self.vector_store.clone(),
            embedder: self.embedder.clone(),
            session_history: self.session_history.clone(),
            max_sessions: self.max_sessions,
        }
    }
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

        self.current_session.add_turn(turn.clone());

        let query_for_embedding = format!("User query: {}", user);
        let embedding = self.embedder.embed(&query_for_embedding)?;

        let memory_entry = MemoryEntry::new(
            user.clone(),
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
        )
        .with_metadata("user_query".to_string(), user)
        .with_metadata("assistant_response".to_string(), assistant);

        self.vector_store.add(memory_entry)?;

        self.cleanup_if_needed();

        Ok(())
    }

    /// Очищает старые сессии если превышен лимит
    fn cleanup_if_needed(&mut self) {
        let total = self.session_history.len() + 1; // +1 для текущей сессии
        if total > self.max_sessions {
            let to_remove = total - self.max_sessions;
            let mut session_ids: Vec<Uuid> = self.session_history.keys().copied().collect();
            session_ids.sort_by_key(|id| {
                self.session_history.get(id)
                    .map(|s| s.updated_at)
                    .unwrap_or_else(Utc::now)
            });

            for id in session_ids.into_iter().take(to_remove) {
                self.session_history.remove(&id);
                self.vector_store.clear_by_type(&MemoryType::Episodic {
                    session_id: id,
                    turn: 0,
                });
            }
        }
    }

    /// Ищет похожие диалоги по запросу
    pub fn find_similar_dialogues(&mut self, query: &str, top_k: usize) -> Result<Vec<String>> {
        let query_embedding = self.embedder.embed(query)?;

        let memory_type = MemoryType::Episodic {
            session_id: Uuid::nil(),
            turn: 0,
        };

        let results: Vec<(f32, crate::totems::retrieval::MemoryEntry)> = self
            .vector_store
            .search_by_type(&query_embedding, &memory_type, top_k * 3)
            .into_iter()
            .map(|(s, e)| (s, e.clone()))
            .collect();

        let keyword_matches: Vec<(f32, crate::totems::retrieval::MemoryEntry)> = self
            .keyword_search(query, top_k)
            .into_iter()
            .map(|(s, e)| (s + 0.1, e.clone()))
            .collect();

        let mut all_entries: Vec<(f32, crate::totems::retrieval::MemoryEntry)> = results
            .into_iter()
            .chain(keyword_matches.into_iter())
            .collect();

        all_entries.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        all_entries.truncate(top_k);

        let mut dialogues = Vec::new();
        let mut seen = std::collections::HashSet::new();

        for (similarity, entry) in all_entries {
            let key = format!(
                "{}-{}",
                entry.metadata.get("session_id").unwrap_or(&"".to_string()),
                entry.metadata.get("turn").unwrap_or(&"".to_string())
            );

            if seen.contains(&key) {
                continue;
            }
            seen.insert(key);

            // Only include high-similarity memories (above 0.3)
            if similarity < 0.3 {
                continue;
            }

            let user_query = entry
                .metadata
                .get("user_query")
                .cloned()
                .unwrap_or_else(|| entry.text.clone());

            // Skip test/placeholder entries
            if user_query.contains("# Test") || user_query.contains("TEST") || user_query.is_empty() {
                continue;
            }

            let assistant_response = entry
                .metadata
                .get("assistant_response")
                .cloned()
                .unwrap_or_default();

            let context = format!("FROM PAST: User said \"{}\"", user_query);

            let truncated = if context.chars().count() > 200 {
                if let Some((byte_pos, _)) = context.char_indices().nth(200) {
                    let trunc = &context[..byte_pos];
                    if let Some(newline_pos) = trunc.rfind('"') {
                        &context[..=newline_pos]
                    } else if let Some(space_pos) = trunc.rfind(' ') {
                        &context[..space_pos]
                    } else {
                        trunc
                    }
                } else {
                    &context
                }
                .to_string()
                    + "\"..."
            } else {
                context
            };

            let score_pct = (similarity * 100.0) as u32;
            let formatted = format!("[Relevance: {}%] {}", score_pct, truncated);
            dialogues.push(formatted);
        }

        Ok(dialogues)
    }

    fn keyword_search(
        &self,
        query: &str,
        top_k: usize,
    ) -> Vec<(f32, crate::totems::retrieval::MemoryEntry)> {
        let keywords: Vec<&str> = query.split_whitespace().filter(|w| w.len() > 3).collect();

        if keywords.is_empty() {
            return Vec::new();
        }

        let mut matches: Vec<(f32, crate::totems::retrieval::MemoryEntry)> = Vec::new();

        for entry in self.vector_store.entries() {
            let user_text = entry
                .metadata
                .get("user_query")
                .unwrap_or(&entry.text)
                .to_lowercase();

            let assistant_text = entry
                .metadata
                .get("assistant_response")
                .unwrap_or(&String::new())
                .to_lowercase();

            let full_text = format!("{} {}", user_text, assistant_text);

            let keyword_count = keywords
                .iter()
                .filter(|k| full_text.contains(&*k.to_lowercase()))
                .count();
            if keyword_count > 0 {
                let score = (keyword_count as f32 / keywords.len() as f32).min(1.0);
                matches.push((score, entry.clone()));
            }
        }

        matches.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        matches.truncate(top_k);
        matches
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
        self.current_session.format_context(max_turns, 512)
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

    pub fn get_turns_for_context(&self, max_turns: usize) -> Vec<Turn> {
        self.current_session.last_turns(max_turns).to_vec()
    }

    pub fn analyze_for_context(
        &self,
        pipeline: &dyn LlmPipeline,
        max_turns: usize,
    ) -> Result<SessionAnalysis> {
        let turns = self.get_turns_for_context(max_turns);

        let analyzer = ContextAnalyzer::new(pipeline);

        let summary = analyzer.summarize_session(&turns)?;
        let key_topics = analyzer.extract_topics(&turns)?;
        let emotional_state = analyzer.analyze_emotions(&turns)?;
        let last_topic = analyzer.extract_last_topic(&turns)?;

        Ok(SessionAnalysis {
            summary,
            key_topics,
            emotional_state,
            last_topic,
            turn_count: turns.len(),
        })
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

pub trait LlmPipeline: Send + Sync {
    fn generate(&self, prompt: &str, max_tokens: usize) -> Result<String>;
}

struct ContextAnalyzer<'a> {
    pipeline: &'a dyn LlmPipeline,
}

impl<'a> ContextAnalyzer<'a> {
    fn new(pipeline: &'a dyn LlmPipeline) -> Self {
        Self { pipeline }
    }

    fn summarize_session(&self, turns: &[Turn]) -> Result<String> {
        if turns.is_empty() {
            return Ok(String::new());
        }

        let dialogue_text = turns
            .iter()
            .enumerate()
            .map(|(i, t)| format!("Turn {}:\nUser: {}\nAssistant: {}\n", i + 1, t.user, t.assistant))
            .collect::<Vec<_>>()
            .join("\n");

        let prompt = format!(
            r#"<s>[INST] Ты — ассистент по анализу диалогов. Кратко опиши, о чём был разговор (2-3 предложения на русском).

Диалог:
{dialogue_text}

Краткое содержание:[/INST]"#,
            dialogue_text = dialogue_text
        );

        let response = self.pipeline.generate(&prompt, 300)?;
        Ok(response.trim().to_string())
    }

    fn extract_topics(&self, turns: &[Turn]) -> Result<Vec<String>> {
        if turns.is_empty() {
            return Ok(Vec::new());
        }

        let dialogue_text = turns
            .iter()
            .map(|t| format!("User: {}\nAssistant: {}", t.user, t.assistant))
            .collect::<Vec<_>>()
            .join("\n---\n");

        let prompt = format!(
            r#"<s>[INST] Извлеки ключевые темы из диалога. Верни только JSON массив строк, например: ["тема1", "тема2", "тема3"].
Не более 5 тем. Темы должны быть короткими (1-2 слова), на русском языке.

Диалог:
{dialogue_text}

Темы:[/INST]"#,
            dialogue_text = dialogue_text
        );

        let response = self.pipeline.generate(&prompt, 200)?;
        self.parse_topics(&response)
    }

    fn parse_topics(&self, response: &str) -> Result<Vec<String>> {
        let cleaned = response
            .trim()
            .trim_start_matches("```json")
            .trim_start_matches("```")
            .trim_end_matches("```")
            .trim()
            .to_string();

        let topics: Result<Vec<String>, _> = serde_json::from_str(&cleaned);

        match topics {
            Ok(t) => Ok(t),
            Err(_) => {
                let without_brackets = cleaned.trim_start_matches('[').trim_end_matches(']');
                let items: Result<Vec<String>, _> = serde_json::from_str(&format!("[{}]", without_brackets));
                items.map_err(|e| anyhow::anyhow!("Failed to parse topics: {}", e))
            }
        }
    }

    fn analyze_emotions(&self, turns: &[Turn]) -> Result<f32> {
        if turns.is_empty() {
            return Ok(0.5);
        }

        let dialogue_text = turns
            .iter()
            .map(|t| format!("User: {}\nAssistant: {}", t.user, t.assistant))
            .collect::<Vec<_>>()
            .join("\n---\n");

        let prompt = format!(
            r#"<s>[INST] Определи эмоциональное состояние пользователя по диалогу.
 Верни только число от 0.0 (негативное/грустное) до 1.0 (позитивное/радостное).

Диалог:
{dialogue_text}

Число:[/INST]"#,
            dialogue_text = dialogue_text
        );

        let response = self.pipeline.generate(&prompt, 50)?;
        let cleaned = response.trim();

        cleaned
            .parse::<f32>()
            .map(|v| v.clamp(0.0, 1.0))
            .map_err(|_| anyhow::anyhow!("Failed to parse emotional state"))
    }

    fn extract_last_topic(&self, turns: &[Turn]) -> Result<String> {
        if let Some(last_turn) = turns.last() {
            let prompt = format!(
                r#"<s>[INST] Определи, о чём был последний вопрос пользователя (1-2 слова на русском).
Вопрос: {question}

Тема:[/INST]"#,
                question = last_turn.user
            );

            let response = self.pipeline.generate(&prompt, 50)?;
            return Ok(response.trim().to_string());
        }
        Ok(String::new())
    }
}

pub struct SessionAnalysis {
    pub summary: String,
    pub key_topics: Vec<String>,
    pub emotional_state: f32,
    pub last_topic: String,
    pub turn_count: usize,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::priests::embeddings::{EmbeddingConfig, EmbeddingEngine};
    use candle_core::Device;

    #[tokio::test]
    async fn test_dialogue_manager() -> Result<()> {
        let embedder = Arc::new(create_test_embedder()?);
        let mut manager = DialogueManager::new(embedder.clone(), "test_persona".to_string());

        manager
            .add_exchange(
                "Hello, how are you?".to_string(),
                "I'm doing well, thank you!".to_string(),
            )
            .await?;

        let stats = manager.stats();
        assert_eq!(stats.current_session_turns, 1);

        let context = manager.get_current_context(5);
        assert!(context.contains("Hello, how are you?"));
        assert!(context.contains("I'm doing well, thank you!"));

        Ok(())
    }

    fn create_test_embedder() -> Result<EmbeddingEngine> {
        Err(anyhow!("Test embedder not implemented"))
    }
}

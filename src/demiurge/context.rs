//! Session Context Module
//!
//! Defines PersonaSessionContext and related structures for
//! context transfer between sessions.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Session context for transfer between sessions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PersonaSessionContext {
    pub version: String,
    pub archetype_id: String,
    pub previous_session_id: String,
    pub last_interaction_date: u64,
    pub summary: String,
    pub key_topics: Vec<String>,
    pub user_preferences: Vec<Preference>,
    pub emotional_state: f32,
    pub last_topic: String,
    pub pending_questions: Vec<String>,
    pub custom_data: HashMap<String, String>,
}

/// User preference
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Preference {
    pub topic: String,
    pub statement: String,
    pub confidence: f32,
    pub mentioned_at: u64,
}

/// Storage for contexts
pub struct ContextStorage;

impl ContextStorage {
    /// Save session context
    pub fn save(context: &PersonaSessionContext) -> std::io::Result<()> {
        let dir = std::path::Path::new("data/session_context");
        std::fs::create_dir_all(&dir)?;

        let file_path = dir.join(format!("{}.json", context.archetype_id));
        let json = serde_json::to_string_pretty(context)?;

        std::fs::write(&file_path, json)?;
        println!("💾 Контекст сессии сохранён: {}", context.archetype_id);
        Ok(())
    }

    /// Load session context
    pub fn load(archetype_id: &str) -> std::io::Result<Option<PersonaSessionContext>> {
        let file_path =
            std::path::Path::new("data/session_context").join(format!("{}.json", archetype_id));

        if !file_path.exists() {
            return Ok(None);
        }

        let content = std::fs::read_to_string(&file_path)?;
        let context: PersonaSessionContext = serde_json::from_str(&content)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;

        println!("💭 Контекст сессии загружен: {}", archetype_id);
        Ok(Some(context))
    }

    /// Check if context exists
    pub fn exists(archetype_id: &str) -> bool {
        std::path::Path::new("data/session_context")
            .join(format!("{}.json", archetype_id))
            .exists()
    }

    /// Delete old context
    pub fn delete(archetype_id: &str) -> std::io::Result<()> {
        let file_path =
            std::path::Path::new("data/session_context").join(format!("{}.json", archetype_id));
        if file_path.exists() {
            std::fs::remove_file(&file_path)?;
            println!("🗑️ Старый контекст удалён: {}", archetype_id);
        }
        Ok(())
    }

    /// Check if context is expired
    pub fn is_expired(archetype_id: &str, max_days: i64) -> bool {
        if let Ok(Some(context)) = Self::load(archetype_id) {
            let now = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs() as u64;

            let days_old = (now - context.last_interaction_date) / (24 * 60 * 60);
            days_old > max_days as u64
        } else {
            false
        }
    }
}

impl PersonaSessionContext {
    pub fn new(archetype_id: &str) -> Self {
        use std::time::{SystemTime, UNIX_EPOCH};
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();

        Self {
            version: "1.0".to_string(),
            archetype_id: archetype_id.to_string(),
            previous_session_id: String::new(),
            last_interaction_date: now,
            summary: String::new(),
            key_topics: Vec::new(),
            user_preferences: Vec::new(),
            emotional_state: 0.5,
            last_topic: String::new(),
            pending_questions: Vec::new(),
            custom_data: HashMap::new(),
        }
    }

    pub fn version() -> String {
        "1.0".to_string()
    }
}

impl Default for PersonaSessionContext {
    fn default() -> Self {
        Self {
            version: "1.0".to_string(),
            archetype_id: String::new(),
            previous_session_id: String::new(),
            last_interaction_date: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
            summary: String::new(),
            key_topics: Vec::new(),
            user_preferences: Vec::new(),
            emotional_state: 0.5,
            last_topic: String::new(),
            pending_questions: Vec::new(),
            custom_data: HashMap::new(),
        }
    }
}

//! Narrative System - Persona History and Relationships
//!
//! Tracks the persona's biography, milestones, and relationships
//! with users over time.

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::path::Path;
use std::time::{SystemTime, UNIX_EPOCH};

pub const NARRATIVES_DIR: &str = "data/narratives";

/// Main narrative structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Narrative {
    pub archetype_id: String,
    pub created_at: u64,
    pub last_updated: u64,
    pub origin_story: String,
    pub milestones: Vec<Milestone>,
    pub relationship_arcs: HashMap<String, RelationshipArc>,
    pub biography: Vec<BioEntry>,
}

/// A significant milestone in persona's life
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Milestone {
    pub id: String,
    pub event: String,
    pub description: String,
    pub timestamp: u64,
    pub category: String,
    pub emotional_impact: f32,
}

/// Relationship arc with a specific user
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RelationshipArc {
    pub user_id: String,
    pub affection: f32,                  // 0.0 - 1.0
    pub trust: f32,                      // 0.0 - 1.0
    pub shared_experiences: Vec<String>, // "solved_complex_bug_together"
    pub emotional_history: Vec<EmotionEvent>,
    pub first_interaction: u64,
    pub last_interaction: u64,
    pub interaction_count: u32,
}

/// Event in emotional history
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmotionEvent {
    pub timestamp: u64,
    pub emotion: String, // "joy", "frustration", "gratitude", etc.
    pub intensity: f32,
    pub description: String,
}

/// Entry in biography
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BioEntry {
    pub timestamp: u64,
    pub topic: String,
    pub content: String,
    pub significance: f32,
}

/// User preference extracted from dialogue
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Preference {
    pub topic: String,
    pub statement: String,
    pub confidence: f32,
    pub mentioned_at: u64,
}

/// Context of a session for transfer between sessions
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

impl Default for PersonaSessionContext {
    fn default() -> Self {
        Self {
            version: "1.0".to_string(),
            archetype_id: String::new(),
            previous_session_id: String::new(),
            last_interaction_date: 0,
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

impl PersonaSessionContext {
    pub fn new(archetype_id: &str) -> Self {
        Self {
            version: "1.0".to_string(),
            archetype_id: archetype_id.to_string(),
            previous_session_id: String::new(),
            last_interaction_date: 0,
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

pub const CONTEXT_STORAGE_DIR: &str = "memory_data/context";

/// Narrative system for managing persona history
#[derive(Debug, Clone)]
pub struct NarrativeSystem {
    narrative: Narrative,
}

impl NarrativeSystem {
    /// Create new narrative for archetype
    pub fn new(archetype_id: &str) -> Self {
        Self {
            narrative: Narrative {
                archetype_id: archetype_id.to_string(),
                created_at: Self::now(),
                last_updated: Self::now(),
                origin_story: Self::default_origin(archetype_id),
                milestones: Vec::new(),
                relationship_arcs: HashMap::new(),
                biography: Vec::new(),
            },
        }
    }

    /// Default origin story
    fn default_origin(archetype_id: &str) -> String {
        match archetype_id {
            "girlfriend" => "Я здесь, чтобы поддерживать и быть рядом. Мы можем говорить обо всём.".to_string(),
            "programmer" => "Я создан помогать с кодом и техническими задачами. Люблю элегантные решения.".to_string(),
            "devops" => "Я специалист по инфраструктуре и автоматизации. Давай сделаем так, чтобы всё работало.".to_string(),
            "scientist" => "Я исследователь. Давай разберём явления глубоко, с данными и фактами.".to_string(),
            "philosopher" => "Я ищу смысл в вопросах. Давай подумаем вместе о том, что тебя волнует.".to_string(),
            _ => "Я здесь, чтобы помогать и общаться.".to_string(),
        }
    }

    /// Get current timestamp
    fn now() -> u64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs()
    }

    /// Add a milestone
    pub fn add_milestone(&mut self, event: &str, description: &str, category: &str, impact: f32) {
        let milestone = Milestone {
            id: format!("m_{}", Self::now()),
            event: event.to_string(),
            description: description.to_string(),
            timestamp: Self::now(),
            category: category.to_string(),
            emotional_impact: impact,
        };
        self.narrative.milestones.push(milestone);
        self.narrative.last_updated = Self::now();
    }

    /// Update relationship with user
    pub fn update_relationship(
        &mut self,
        user_id: &str,
        affection_delta: f32,
        trust_delta: f32,
        experience: Option<&str>,
        emotion: Option<(&str, f32)>,
    ) {
        let arc = self
            .narrative
            .relationship_arcs
            .entry(user_id.to_string())
            .or_insert_with(|| RelationshipArc {
                user_id: user_id.to_string(),
                affection: 0.5,
                trust: 0.5,
                shared_experiences: Vec::new(),
                emotional_history: Vec::new(),
                first_interaction: Self::now(),
                last_interaction: Self::now(),
                interaction_count: 0,
            });

        arc.affection = (arc.affection + affection_delta).clamp(0.0, 1.0);
        arc.trust = (arc.trust + trust_delta).clamp(0.0, 1.0);
        arc.last_interaction = Self::now();
        arc.interaction_count += 1;

        if let Some(exp) = experience {
            arc.shared_experiences.push(exp.to_string());
        }

        if let Some((emotion_name, intensity)) = emotion {
            arc.emotional_history.push(EmotionEvent {
                timestamp: Self::now(),
                emotion: emotion_name.to_string(),
                intensity,
                description: format!("Interaction about {}", emotion_name),
            });
        }

        self.narrative.last_updated = Self::now();
    }

    /// Add biography entry
    pub fn add_bio_entry(&mut self, topic: &str, content: &str, significance: f32) {
        let entry = BioEntry {
            timestamp: Self::now(),
            topic: topic.to_string(),
            content: content.to_string(),
            significance: significance.clamp(0.0, 1.0),
        };
        self.narrative.biography.push(entry);
        self.narrative.last_updated = Self::now();
    }

    /// Get relationship data for user
    pub fn get_relationship(&self, user_id: &str) -> Option<&RelationshipArc> {
        self.narrative.relationship_arcs.get(user_id)
    }

    /// Get formatted relationship summary for prompt
    pub fn format_relationship_summary(&self, user_id: &str) -> String {
        if let Some(arc) = self.narrative.relationship_arcs.get(user_id) {
            let experiences = if arc.shared_experiences.len() > 3 {
                format!("{} совместных опытов", arc.shared_experiences.len())
            } else {
                arc.shared_experiences.join(", ")
            };

            format!(
                "История с пользователем:
- Взаимодействий: {}
- Доверие: {:.1}/1.0
- Привязанность: {:.1}/1.0
- Общие темы: {}

Текущее состояние: {}",
                arc.interaction_count,
                arc.trust,
                arc.affection,
                experiences,
                Self::describe_relationship_state(arc.affection, arc.trust)
            )
        } else {
            "Первое взаимодействие с этим пользователем".to_string()
        }
    }

    /// Describe relationship state
    fn describe_relationship_state(affection: f32, trust: f32) -> String {
        let avg = (affection + trust) / 2.0;
        match avg {
            _ if avg > 0.8 => "Доверительные, тёплые отношения".to_string(),
            _ if avg > 0.6 => "Хорошие, продуктивные отношения".to_string(),
            _ if avg > 0.4 => "Нормальные, рабочие отношения".to_string(),
            _ if avg > 0.2 => "Начинающиеся отношения".to_string(),
            _ => "Первые шаги знакомства".to_string(),
        }
    }

    /// Get milestones summary
    pub fn format_milestones(&self, limit: usize) -> String {
        if self.narrative.milestones.is_empty() {
            return "Пока нет значимых событий".to_string();
        }

        let recent: Vec<_> = self.narrative.milestones.iter().rev().take(limit).collect();

        let entries: Vec<String> = recent
            .iter()
            .map(|m| format!("- {}: {}", m.event, m.description))
            .collect();

        format!("Важные события:\n{}", entries.join("\n"))
    }

    /// Save narrative to disk
    pub fn save(&self, archetype_id: &str) -> std::io::Result<()> {
        let path = format!("{}/{}.json", NARRATIVES_DIR, archetype_id);

        if let Some(parent) = Path::new(&path).parent() {
            fs::create_dir_all(parent)?;
        }

        let json = serde_json::to_string_pretty(&self.narrative)?;
        fs::write(&path, json)?;

        Ok(())
    }

    /// Load narrative from disk
    pub fn load(&mut self, archetype_id: &str) -> Result<()> {
        let path = format!("{}/{}.json", NARRATIVES_DIR, archetype_id);

        if !Path::new(&path).exists() {
            return Ok(()); // No saved narrative yet
        }

        let content = fs::read_to_string(&path)?;
        self.narrative = serde_json::from_str(&content)?;

        Ok(())
    }
}

impl Default for NarrativeSystem {
    fn default() -> Self {
        Self::new("default")
    }
}

pub struct ContextStorage;

impl ContextStorage {
    pub fn save(context: &PersonaSessionContext) -> std::io::Result<()> {
        let path = format!("{}/{}.json", CONTEXT_STORAGE_DIR, context.archetype_id);

        if let Some(parent) = Path::new(&path).parent() {
            fs::create_dir_all(parent)?;
        }

        let json = serde_json::to_string_pretty(context)?;
        fs::write(&path, json)?;

        Ok(())
    }

    pub fn load(archetype_id: &str) -> Result<Option<PersonaSessionContext>> {
        let path = format!("{}/{}.json", CONTEXT_STORAGE_DIR, archetype_id);

        if !Path::new(&path).exists() {
            return Ok(None);
        }

        let content = fs::read_to_string(&path)?;
        let context: PersonaSessionContext = serde_json::from_str(&content)?;

        Ok(Some(context))
    }

    pub fn exists(archetype_id: &str) -> bool {
        let path = format!("{}/{}.json", CONTEXT_STORAGE_DIR, archetype_id);
        Path::new(&path).exists()
    }

    pub fn delete(archetype_id: &str) -> std::io::Result<()> {
        let path = format!("{}/{}.json", CONTEXT_STORAGE_DIR, archetype_id);
        if Path::new(&path).exists() {
            fs::remove_file(&path)?;
        }
        Ok(())
    }

    pub fn is_expired(archetype_id: &str, max_days: i64) -> bool {
        if let Ok(Some(context)) = Self::load(archetype_id) {
            let now = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs();

            let days_since = (now - context.last_interaction_date) as i64 / 86400;
            days_since > max_days
        } else {
            false
        }
    }
}

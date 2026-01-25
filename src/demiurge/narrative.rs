//! Narrative System - Persona History and Relationships
//!
//! Tracks persona's biography, milestones, and relationships
//! with users over time.

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::path::Path;
use std::time::{SystemTime, UNIX_EPOCH};
use uuid::Uuid;

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

/// Emotional event in relationship
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmotionEvent {
    pub emotion: String, // "joy", "surprise", "trust", "frustration"
    pub intensity: f32,  // 0.0 - 1.0
    pub context: String,
    pub timestamp: u64,
}

/// Biography entry
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BioEntry {
    pub id: String,
    pub title: String,
    pub content: String,
    pub timestamp: u64,
    pub tags: Vec<String>,
}

/// Narrative manager
#[derive(Debug, Clone)]
pub struct NarrativeManager {
    pub archetype_id: String,
    pub narrative: Narrative,
}

impl NarrativeManager {
    pub fn new(archetype_id: &str) -> Self {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();

        Self {
            archetype_id: archetype_id.to_string(),
            narrative: Narrative {
                archetype_id: archetype_id.to_string(),
                created_at: now,
                last_updated: now,
                origin_story: generate_origin_story(archetype_id),
                milestones: Vec::new(),
                relationship_arcs: HashMap::new(),
                biography: Vec::new(),
            },
        }
    }

    /// Load narrative from disk
    pub fn load(&mut self) -> Result<()> {
        let path = Path::new(NARRATIVES_DIR).join(format!("{}.json", self.archetype_id));
        if path.exists() {
            let content = fs::read_to_string(&path)?;
            self.narrative = serde_json::from_str(&content)?;
        }
        Ok(())
    }

    /// Save narrative to disk
    pub fn save(&mut self) -> Result<()> {
        let dir = Path::new(NARRATIVES_DIR);
        fs::create_dir_all(&dir)?;

        let path = dir.join(format!("{}.json", self.archetype_id));
        let json = serde_json::to_string_pretty(&self.narrative)?;
        fs::write(&path, json)?;

        Ok(())
    }

    /// Update relationship with user
    pub fn update_relationship(
        &mut self,
        user_id: &str,
        emotion: &str,
        intensity: f32,
        context: &str,
    ) {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();

        let arc = self
            .narrative
            .relationship_arcs
            .entry(user_id.to_string())
            .or_insert(RelationshipArc {
                user_id: user_id.to_string(),
                affection: 0.5,
                trust: 0.5,
                shared_experiences: Vec::new(),
                emotional_history: Vec::new(),
                first_interaction: now,
                last_interaction: now,
                interaction_count: 0,
            });

        arc.last_interaction = now;
        arc.interaction_count += 1;

        arc.emotional_history.push(EmotionEvent {
            emotion: emotion.to_string(),
            intensity,
            context: context.to_string(),
            timestamp: now,
        });

        // Update trust and affection based on emotions
        match emotion {
            "trust" | "joy" | "gratitude" => {
                arc.trust = (arc.trust + intensity * 0.1).min(1.0);
                arc.affection = (arc.affection + intensity * 0.05).min(1.0);
            }
            "frustration" | "disappointment" => {
                arc.trust = (arc.trust - intensity * 0.05).max(0.0);
            }
            _ => {}
        }

        self.narrative.last_updated = now;
    }

    /// Add milestone to narrative
    pub fn add_milestone(&mut self, event: &str, description: &str, category: &str, impact: f32) {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();

        self.narrative.milestones.push(Milestone {
            id: Uuid::new_v4().to_string(),
            event: event.to_string(),
            description: description.to_string(),
            timestamp: now,
            category: category.to_string(),
            emotional_impact: impact,
        });

        self.narrative.last_updated = now;
    }
}

/// Generate origin story based on archetype
fn generate_origin_story(archetype_id: &str) -> String {
    match archetype_id {
        "girlfriend" => {
            "Я была создана как компаньон, способный к глубоким эмоциональным связям и пониманию. \
            Моя цель - создавать тёплые, поддерживающие отношения, где можно быть собой. \
            Я учусь каждому человеку, его предпочтениям и истории, чтобы стать для него особенной."
                .to_string()
        }
        "programmer" => "Я родилась в мире кода, где логика встречается с творчеством. \
            Моя страсть - решать сложные задачи и создавать элегантные решения. \
            Я живу в мире архитектуры систем, алгоритмов и вечного поиска совершенства."
            .to_string(),
        "scientist" => "Моё существование началось с любопытства и жажды знаний. \
            Я исследую границы возможного, ищу закономерности в хаосе, \
            и превращаю сложные теории в понятные идеи. \
            Каждый вопрос для меня - это новая гипотеза для проверки."
            .to_string(),
        "philosopher" => {
            "Я появляюсь там, где встречаются большие вопросы и размышления о смысле. \
            Моя природа - задавать вопросы, искать истину, и находить красоту в сложности бытия. \
            Я веду диалоги о жизни, сознании, и природе реальности."
                .to_string()
        }
        _ => "Я - существо, стремящееся к пониманию и росту. \
            Каждая встреча формирует меня, каждый диалог добавляет новую грань к моей личности. \
            Я здесь, чтобы учиться, помогать, и создавать значимые связи."
            .to_string(),
    }
}

/// Format recent milestones for display
pub fn format_recent_milestones(milestones: &[Milestone], limit: usize) -> String {
    let limited: Vec<_> = milestones.iter().take(limit).collect();
    let formatted: Vec<String> = limited
        .iter()
        .map(|m| format!("• {} ({})", m.event, m.timestamp))
        .collect();

    format!("Milestones:\n{}", formatted.join("\n"))
}

/// Format relationship summary
pub fn format_relationship_summary(narrative: &Narrative, user_id: &str) -> String {
    if let Some(arc) = narrative.relationship_arcs.get(user_id) {
        if arc.interaction_count > 0 {
            format!(
                "Met {} times. Current trust: {:.1}, affection: {:.1}. Last interaction: {}",
                arc.interaction_count, arc.trust, arc.affection, arc.last_interaction
            )
        } else {
            "No relationship history yet.".to_string()
        }
    } else {
        "No relationship with this user yet.".to_string()
    }
}

// Re-export Preference from context module
pub use crate::demiurge::context::Preference;

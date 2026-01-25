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
    pub timestamp: u64,
    pub emotion: String,
    pub intensity: f32,
    pub trigger: String,
}

/// Biography entry
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BioEntry {
    pub topic: String,
    pub content: String,
    pub timestamp: u64,
    pub significance: f32,
}

/// Main narrative system
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NarrativeSystem {
    pub archetype_id: String,
    pub narrative: Narrative,
}

// Helper functions for narrative system
pub fn format_milestones(narrative: &Narrative, limit: usize) -> String {
    let mut milestones: Vec<_> = narrative.milestones.iter().collect();
    milestones.sort_by(|a, b| b.timestamp.cmp(&a.timestamp));
    
    if milestones.is_empty() {
        return "No milestones yet.".to_string();
    }
    
    let limited: Vec<_> = milestones.into_iter().take(limit).collect();
    let formatted: Vec<String> = limited
        .iter()
        .map(|m| format!("• {} ({})", m.event, m.timestamp))
        .collect();
    
    format!("Milestones:\n{}", formatted.join("\n"))
}

pub fn format_relationship_summary(narrative: &Narrative, user_id: &str) -> String {
    if let Some(arc) = narrative.relationship_arcs.get(user_id) {
        if arc.interaction_count > 0 {
            format!(
                "Met {} times. Current trust: {:.1}, affection: {:.1}. Last interaction: {}",
                arc.interaction_count,
                arc.trust,
                arc.affection,
                arc.last_interaction
            )
        } else {
            "No relationship history yet.".to_string()
        }
        } else {
            "No relationship with this user yet.".to_string()
    
    }
        }
    }

        let content = fs::read_to_string(&path)?;
        self.narrative = serde_json::from_str(&content)?;
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
}

// Re-export Preference from context module
pub use crate::demiurge::context::Preference;

    let limited: Vec<_> = milestones.into_iter().take(limit).collect();
    let formatted: Vec<String> = limited
        .iter()
        .map(|m| format!("• {} ({})", m.event, m.timestamp))
        .collect();

    format!("Milestones:\n{}", formatted.join("\n"))
}

/// Format relationship summary
pub fn format_relationship_summary(&self, user_id: &str) -> String {
    if let Some(arc) = self.narrative.relationship_arcs.get(user_id) {
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

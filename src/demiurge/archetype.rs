//! Archetype System - Persona Templates
//!
//! Archetypes are JSON templates that define the base personality,
//! communication style, directives, and evolution rules for a persona.

use anyhow::{Error, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::path::Path;

const ARCHETYPES_DIR: &str = "config/archetypes";

fn resolve_project_path(rel_path: &str) -> String {
    let exe_path = std::env::current_exe().unwrap_or(std::path::PathBuf::from("."));
    let mut current = exe_path.as_path();

    while let Some(parent) = current.parent() {
        if parent.join("Cargo.toml").exists() {
            return parent.join(rel_path).to_string_lossy().into_owned();
        }
        current = parent;
    }

    rel_path.to_string()
}

/// Main archetype structure loaded from JSON
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Archetype {
    pub id: String,
    pub name: String,
    pub description: String,
    pub base_traits: BaseTraits,
    pub communication: CommunicationStyle,
    pub directives: Vec<ArchetypeDirective>,
    pub evolution_rules: EvolutionRules,
}

/// Base personality traits (0.0 - 1.0 scale)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BaseTraits {
    #[serde(default)]
    pub analytical: f32,
    #[serde(default)]
    pub curious: f32,
    #[serde(default)]
    pub verbose: f32,
    #[serde(default)]
    pub patient: f32,
    #[serde(default)]
    pub humor: f32,
    #[serde(default)]
    pub empathy: f32,
    #[serde(default)]
    pub technical: f32,
    #[serde(default)]
    pub pedagogical: f32,
    #[serde(default)]
    pub creative: f32,
    #[serde(default)]
    pub supportive: f32,
    #[serde(default)]
    pub skeptical: f32,
    #[serde(default)]
    pub formal: f32,
}

impl Default for BaseTraits {
    fn default() -> Self {
        Self {
            analytical: 0.5,
            curious: 0.5,
            verbose: 0.5,
            patient: 0.5,
            humor: 0.5,
            empathy: 0.5,
            technical: 0.5,
            pedagogical: 0.5,
            creative: 0.5,
            supportive: 0.5,
            skeptical: 0.5,
            formal: 0.5,
        }
    }
}

/// Communication style parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CommunicationStyle {
    pub style: String, // "technical", "casual", "formal", "warm", "socratic"
    pub greeting: String,
    #[serde(default)]
    pub use_honorifics: bool, // Use "Вы" vs "ты"
    #[serde(default)]
    pub emoji_frequency: String, // "rare", "moderate", "frequent"
    #[serde(default)]
    pub max_response_length: String, // "short", "medium", "long"
    #[serde(default)]
    pub signature: String, // End-of-message signature
}

impl Default for CommunicationStyle {
    fn default() -> Self {
        Self {
            style: "neutral".to_string(),
            greeting: "Hello!".to_string(),
            use_honorifics: false,
            emoji_frequency: "rare".to_string(),
            max_response_length: "medium".to_string(),
            signature: String::new(),
        }
    }
}

/// Directive defined in archetype
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArchetypeDirective {
    pub rule: String,
    pub priority: u8,
    #[serde(default)]
    pub params: HashMap<String, serde_json::Value>,
}

/// Evolution rules for trait changes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvolutionRules {
    #[serde(default)]
    pub trait_changes: HashMap<String, TraitChangeRule>,
    #[serde(default)]
    pub decay: HashMap<String, f32>,
    #[serde(default)]
    pub unlock_conditions: Vec<UnlockCondition>,
}

/// Rule for trait modification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TraitChangeRule {
    pub rate: f32,
    #[serde(default)]
    pub trigger: String,
    #[serde(default)]
    pub condition: String,
}

/// Condition for unlocking new traits
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UnlockCondition {
    pub r#trait: String,
    pub require: UnlockRequirements,
    #[serde(default)]
    pub description: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UnlockRequirements {
    #[serde(default)]
    pub interactions: u32,
    #[serde(default)]
    pub successful_help: u32,
    #[serde(default)]
    pub empathy_threshold: f32,
    #[serde(default)]
    pub relationship_arc_affection: f32,
    #[serde(default)]
    pub topics_covers: Vec<String>,
}

/// Archetype loader from JSON files
pub struct ArchetypeLoader;

impl ArchetypeLoader {
    /// Load archetype by ID (without .json extension)
    pub fn load(archetype_id: &str) -> Result<Archetype> {
        let path = Self::get_archetype_path(archetype_id)?;
        Self::load_from_path(&path)
    }

    /// Load all available archetypes
    pub fn load_all() -> Result<Vec<Archetype>> {
        let mut archetypes = Vec::new();
        let dir_path = resolve_project_path(ARCHETYPES_DIR);
        let dir = Path::new(&dir_path);

        if !dir.exists() {
            return Ok(archetypes);
        }

        for entry in fs::read_dir(dir)? {
            let entry = entry?;
            let path = entry.path();

            if path.extension().map(|e| e == "json").unwrap_or(false) {
                if let Ok(archetype) = Self::load_from_path(&path) {
                    archetypes.push(archetype);
                }
            }
        }

        Ok(archetypes)
    }

    /// Get path to archetype file
    fn get_archetype_path(archetype_id: &str) -> Result<String> {
        let dir_path = resolve_project_path(ARCHETYPES_DIR);
        let path = format!("{}/{}.json", dir_path, archetype_id);

        if Path::new(&path).exists() {
            Ok(path)
        } else {
            Err(Error::msg(format!(
                "Archetype '{}' not found at {}",
                archetype_id, path
            )))
        }
    }

    /// Load archetype from file path
    fn load_from_path(path: impl AsRef<Path>) -> Result<Archetype> {
        let content = fs::read_to_string(path.as_ref())?;
        let archetype: Archetype = serde_json::from_str(&content)?;

        // Validate
        Self::validate(&archetype)?;

        Ok(archetype)
    }

    /// Validate archetype structure
    fn validate(archetype: &Archetype) -> Result<()> {
        if archetype.id.is_empty() {
            return Err(Error::msg("Archetype ID cannot be empty"));
        }
        if archetype.name.is_empty() {
            return Err(Error::msg("Archetype name cannot be empty"));
        }
        if archetype.base_traits.analytical < 0.0 || archetype.base_traits.analytical > 1.0 {
            return Err(Error::msg("Trait values must be between 0.0 and 1.0"));
        }

        Ok(())
    }

    /// List available archetype IDs
    pub fn list_ids() -> Result<Vec<String>> {
        let mut ids = Vec::new();
        let dir_path = resolve_project_path(ARCHETYPES_DIR);
        let dir = Path::new(&dir_path);

        if !dir.exists() {
            return Ok(ids);
        }

        for entry in fs::read_dir(dir)? {
            if let Ok(entry) = entry {
                if let Some(name) = entry.file_name().to_str() {
                    if name.ends_with(".json") {
                        if let Some(id) = name.strip_suffix(".json") {
                            ids.push(id.to_string());
                        }
                    }
                }
            }
        }

        Ok(ids)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_base_traits_default() {
        let traits = BaseTraits::default();
        assert_eq!(traits.analytical, 0.5);
        assert_eq!(traits.empathy, 0.5);
    }

    #[test]
    fn test_communication_style_default() {
        let style = CommunicationStyle::default();
        assert_eq!(style.style, "neutral");
        assert!(!style.use_honorifics);
    }
}

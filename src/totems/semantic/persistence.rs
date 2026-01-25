//! 💾 Persistence Layer для семантической памяти
//!
//! Сохраняет и загружает концепты в/из JSON файла

use anyhow::{Context, Result};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::path::PathBuf;
use uuid::Uuid;

use super::concept::Concept;
use super::concept::ConceptCategory;

const SEMANTIC_MEMORY_FILE: &str = "semantic_memory.json";

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SemanticStorage {
    pub version: String,
    pub created_at: DateTime<Utc>,
    pub last_saved_at: DateTime<Utc>,
    pub total_concepts: usize,
    pub concepts: Vec<SerializedConcept>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SerializedConcept {
    pub id: String,
    pub text: String,
    pub category: String,
    pub confidence: f32,
    pub source: String,
    pub metadata: serde_json::Value,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
    pub usage_count: u32,
}

pub struct SemanticPersistenceManager {
    storage_path: PathBuf,
}

impl SemanticPersistenceManager {
    pub fn new(base_path: Option<&PathBuf>) -> Result<Self> {
        let storage_path = base_path
            .clone()
            .unwrap_or(&PathBuf::from("memory_data"))
            .join(SEMANTIC_MEMORY_FILE);

        if let Some(parent) = storage_path.parent() {
            if !parent.exists() {
                fs::create_dir_all(parent)
                    .with_context(|| format!("Failed to create directory: {:?}", parent))?;
            }
        }

        Ok(Self { storage_path })
    }

    pub fn save(&self, concepts: &[Concept]) -> Result<()> {
        let serialized_concepts: Vec<SerializedConcept> =
            concepts.iter().map(|c| self.serialize_concept(c)).collect();

        let storage = SemanticStorage {
            version: "1.0".to_string(),
            created_at: Utc::now(),
            last_saved_at: Utc::now(),
            total_concepts: concepts.len(),
            concepts: serialized_concepts,
        };

        let content = serde_json::to_string_pretty(&storage)
            .context("Failed to serialize semantic memory")?;

        fs::write(&self.storage_path, content).with_context(|| {
            format!("Failed to write semantic memory to {:?}", self.storage_path)
        })?;

        eprintln!(
            "DEBUG: Saved {} semantic concepts to {:?}",
            concepts.len(),
            self.storage_path
        );

        Ok(())
    }

    pub fn load(&self) -> Result<Option<Vec<Concept>>> {
        if !self.storage_path.exists() {
            eprintln!(
                "DEBUG: No semantic memory file found at {:?}",
                self.storage_path
            );
            return Ok(None);
        }

        let content = fs::read_to_string(&self.storage_path).with_context(|| {
            format!(
                "Failed to read semantic memory from {:?}",
                self.storage_path
            )
        })?;

        let storage: SemanticStorage =
            serde_json::from_str(&content).context("Failed to deserialize semantic memory")?;

        eprintln!(
            "DEBUG: Loaded {} semantic concepts from {:?}",
            storage.total_concepts, self.storage_path
        );

        let concepts: Vec<Concept> = storage
            .concepts
            .into_iter()
            .filter_map(|c| self.deserialize_concept(c).ok())
            .collect();

        Ok(Some(concepts))
    }

    pub fn storage_path(&self) -> &PathBuf {
        &self.storage_path
    }

    fn serialize_concept(&self, concept: &Concept) -> SerializedConcept {
        let category = concept.category.to_string();
        let metadata = serde_json::to_value(&concept.metadata).unwrap_or(serde_json::Value::Null);

        SerializedConcept {
            id: concept.id.to_string(),
            text: concept.text.clone(),
            category,
            confidence: concept.confidence,
            source: concept.source.clone(),
            metadata,
            created_at: concept.created_at,
            updated_at: concept.updated_at,
            usage_count: concept.usage_count,
        }
    }

    fn deserialize_concept(&self, serialized: SerializedConcept) -> Result<Concept> {
        let id: Uuid = Uuid::parse_str(&serialized.id)
            .with_context(|| format!("Invalid concept UUID: {}", serialized.id))?;

        let category: ConceptCategory = serialized
            .category
            .parse()
            .unwrap_or(ConceptCategory::General);

        let metadata: HashMap<String, String> = match serialized.metadata {
            serde_json::Value::Object(map) => map
                .into_iter()
                .filter_map(|(k, v)| match v {
                    serde_json::Value::String(s) => Some((k, s)),
                    _ => None,
                })
                .collect(),
            _ => HashMap::new(),
        };

        Ok(Concept {
            id,
            text: serialized.text,
            category,
            confidence: serialized.confidence,
            source: serialized.source,
            embedding: Vec::new(),
            metadata,
            created_at: serialized.created_at,
            updated_at: serialized.updated_at,
            usage_count: serialized.usage_count,
            related_concepts: Vec::new(),
        })
    }
}

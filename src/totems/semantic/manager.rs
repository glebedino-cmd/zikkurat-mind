//! 🧠 Менеджер семантической памяти
//!
//! Управляет концептами: добавление, поиск, объединение
//! Извлечение концептов выполняется отдельно через SemanticExtractor

use anyhow::Result;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use super::concept::{Concept, ConceptCategory};
use super::persistence::SemanticPersistenceManager;
use crate::priests::embeddings::Embedder;
use crate::totems::retrieval::vector_store::cosine_similarity;

pub type ExtractionResult = Vec<(String, String, f32)>; // (text, category, confidence)

pub trait ConceptExtractor: Send + Sync {
    fn extract(
        &mut self,
        user_query: &str,
        assistant_response: &str,
        session_id: &str,
    ) -> Result<ExtractionResult>;
}

pub struct SemanticMemoryManager {
    concepts: HashMap<uuid::Uuid, Concept>,
    embedder: Arc<dyn Embedder>,
    persistence: SemanticPersistenceManager,
    category_index: HashMap<ConceptCategory, Vec<uuid::Uuid>>,
    extractor: Option<Arc<std::sync::Mutex<dyn ConceptExtractor>>>,
}

impl SemanticMemoryManager {
    pub fn new(
        embedder: Arc<dyn Embedder>,
        persistence: SemanticPersistenceManager,
    ) -> Result<Self> {
        let mut manager = Self {
            concepts: HashMap::new(),
            embedder,
            persistence,
            category_index: HashMap::new(),
            extractor: None,
        };

        if let Some(loaded) = manager.persistence.load()? {
            let loaded_concepts = loaded;
            let count = loaded_concepts.len();
            for mut concept in loaded_concepts.into_iter() {
                manager.index_concept(&concept.id, &concept.category);
                concept.embedding = manager.embedder.embed(&concept.text)?;
                manager.concepts.insert(concept.id, concept);
            }
            eprintln!("DEBUG: Loaded {} concepts from storage", count);
        }

        Ok(manager)
    }

    pub fn with_extractor(
        embedder: Arc<dyn Embedder>,
        persistence: SemanticPersistenceManager,
        extractor: Arc<std::sync::Mutex<dyn ConceptExtractor>>,
    ) -> Result<Self> {
        let mut manager = Self::new(embedder, persistence)?;
        manager.extractor = Some(extractor);
        Ok(manager)
    }

    pub fn set_extractor(&mut self, extractor: Arc<std::sync::Mutex<dyn ConceptExtractor>>) {
        self.extractor = Some(extractor);
    }

    pub fn with_concepts(
        embedder: Arc<dyn Embedder>,
        persistence: SemanticPersistenceManager,
        concepts: Vec<Concept>,
    ) -> Result<Self> {
        let mut manager = Self {
            concepts: HashMap::new(),
            embedder,
            persistence,
            category_index: HashMap::new(),
            extractor: None,
        };

        for mut concept in concepts {
            concept.embedding = manager.embedder.embed(&concept.text)?;
            manager.concepts.insert(concept.id, concept.clone());
            manager.index_concept(&concept.id, &concept.category);
        }

        Ok(manager)
    }

    fn index_concept(&mut self, id: &uuid::Uuid, category: &ConceptCategory) {
        self.category_index
            .entry(category.clone())
            .or_insert_with(Vec::new)
            .push(*id);
    }

    pub fn add_concept(
        &mut self,
        text: String,
        category: ConceptCategory,
        source: String,
        confidence: Option<f32>,
    ) -> Result<Concept> {
        let embedding = self.embedder.embed(&text)?;

        let mut concept = Concept::new(text, category, source);
        if let Some(c) = confidence {
            concept = concept.with_confidence(c);
        }
        concept.embedding = embedding;

        self.index_concept(&concept.id, &concept.category);
        self.concepts.insert(concept.id, concept.clone());

        self.save()?;
        eprintln!(
            "DEBUG: Added concept '{}' (category: {})",
            concept.id, concept.category
        );

        Ok(concept)
    }

    pub fn search(
        &self,
        query: &str,
        top_k: usize,
        category: Option<ConceptCategory>,
    ) -> Vec<(f32, &Concept)> {
        let query_embedding = match self.embedder.embed(query) {
            Ok(e) => e,
            Err(e) => {
                eprintln!("DEBUG: Failed to embed query: {}", e);
                return Vec::new();
            }
        };

        let candidates: Vec<&Concept> = match category {
            Some(cat) => {
                if let Some(ids) = self.category_index.get(&cat) {
                    ids.iter().filter_map(|id| self.concepts.get(id)).collect()
                } else {
                    Vec::new()
                }
            }
            None => self.concepts.values().collect(),
        };

        let mut similarities: Vec<(f32, &Concept)> = candidates
            .iter()
            .map(|concept| {
                let similarity = cosine_similarity(&query_embedding, &concept.embedding);
                (similarity, *concept)
            })
            .collect();

        similarities.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        similarities.truncate(top_k);

        eprintln!(
            "DEBUG: Search found {} concepts (top_k={})",
            similarities.len(),
            top_k
        );

        similarities
    }

    pub fn search_by_text(&self, query: &str, top_k: usize) -> Vec<(f32, &Concept)> {
        self.search(query, top_k, None)
    }

    pub fn search_by_category(
        &self,
        query: &str,
        category: ConceptCategory,
        top_k: usize,
    ) -> Vec<(f32, &Concept)> {
        self.search(query, top_k, Some(category))
    }

    pub fn extract_from_dialogue(
        &mut self,
        user_query: &str,
        assistant_response: &str,
        session_id: &str,
    ) -> Result<Vec<Concept>> {
        let extractor = match &self.extractor {
            Some(e) => e,
            None => {
                eprintln!("DEBUG: No extractor configured for semantic memory");
                return Ok(Vec::new());
            }
        };

        let raw_results = {
            let mut extractor = extractor.lock().unwrap();
            extractor.extract(user_query, assistant_response, session_id)?
        };
        self.parse_extraction(raw_results, session_id)
    }

    fn parse_extraction(
        &mut self,
        results: ExtractionResult,
        session_id: &str,
    ) -> Result<Vec<Concept>> {
        let mut extracted = Vec::new();

        for (text, category_str, confidence) in results {
            if text.trim().is_empty() {
                continue;
            }

            let category: ConceptCategory =
                category_str.parse().unwrap_or(ConceptCategory::General);

            if let Ok(concept) = self.add_concept(
                text.trim().to_string(),
                category,
                session_id.to_string(),
                Some(confidence),
            ) {
                extracted.push(concept);
            }
        }

        eprintln!(
            "DEBUG: Extracted {} concepts from dialogue",
            extracted.len()
        );
        Ok(extracted)
    }

    pub fn merge_similar(&mut self, _threshold: f32) -> Result<usize> {
        let mut merged = 0;
        let ids: Vec<_> = self.concepts.keys().copied().collect();
        let mut processed = std::collections::HashSet::new();

        for id1 in &ids {
            if processed.contains(id1) {
                continue;
            }

            let concept1 = match self.concepts.get(id1) {
                Some(c) => c.clone(),
                None => continue,
            };

            let mut to_merge: Vec<uuid::Uuid> = Vec::new();

            for id2 in &ids {
                if id1 == id2 || processed.contains(id2) {
                    continue;
                }

                let concept2 = match self.concepts.get(id2) {
                    Some(c) => c.clone(),
                    None => continue,
                };

                if concept1.text.to_lowercase() == concept2.text.to_lowercase() {
                    to_merge.push(*id2);
                    processed.insert(*id2);

                    if concept2.confidence > concept1.confidence {
                        if let Some(c1) = self.concepts.get_mut(id1) {
                            c1.confidence = concept2.confidence;
                            c1.usage_count += concept2.usage_count;
                        }
                    }
                }
            }

            if !to_merge.is_empty() {
                for id in &to_merge {
                    self.concepts.remove(id);
                    if let Some(cat) = self.category_index.get_mut(&concept1.category) {
                        cat.retain(|i| i != id);
                    }
                }
                merged += to_merge.len();
            }

            processed.insert(*id1);
        }

        if merged > 0 {
            self.save()?;
            eprintln!("DEBUG: Merged {} duplicate concepts", merged);
        }

        Ok(merged)
    }

    pub fn update_concept_confidence(&mut self, concept_id: uuid::Uuid, delta: f32) -> Result<()> {
        let new_confidence = {
            if let Some(concept) = self.concepts.get_mut(&concept_id) {
                concept.update_confidence(delta);
                concept.confidence
            } else {
                return Ok(());
            }
        };
        self.save()?;
        eprintln!(
            "DEBUG: Updated concept {} confidence to {:.2}",
            concept_id, new_confidence
        );
        Ok(())
    }

    pub fn increment_usage(&mut self, concept_id: uuid::Uuid) {
        if let Some(concept) = self.concepts.get_mut(&concept_id) {
            concept.increment_usage();
        }
    }

    pub fn get_concept(&self, id: uuid::Uuid) -> Option<&Concept> {
        self.concepts.get(&id)
    }

    pub fn get_concepts_by_category(&self, category: &ConceptCategory) -> Vec<&Concept> {
        if let Some(ids) = self.category_index.get(category) {
            ids.iter().filter_map(|id| self.concepts.get(id)).collect()
        } else {
            Vec::new()
        }
    }

    pub fn get_all_concepts(&self) -> Vec<&Concept> {
        self.concepts.values().collect()
    }

    pub fn count(&self) -> usize {
        self.concepts.len()
    }

    pub fn count_by_category(&self) -> HashMap<String, usize> {
        let mut counts = HashMap::new();
        for concept in self.concepts.values() {
            let cat = concept.category.to_string();
            *counts.entry(cat).or_insert(0) += 1;
        }
        counts
    }

    pub fn remove_concept(&mut self, id: uuid::Uuid) -> bool {
        if let Some(concept) = self.concepts.remove(&id) {
            if let Some(ids) = self.category_index.get_mut(&concept.category) {
                ids.retain(|i| i != &id);
            }
            self.save().ok();
            true
        } else {
            false
        }
    }

    pub fn clear(&mut self) {
        self.concepts.clear();
        self.category_index.clear();
        self.save().ok();
    }

    fn save(&self) -> Result<()> {
        let concepts: Vec<Concept> = self.concepts.values().cloned().collect();
        self.persistence.save(&concepts)
    }

    pub fn list_concepts(&self, limit: usize, offset: usize) -> Vec<&Concept> {
        let mut all: Vec<&Concept> = self.concepts.values().collect();
        all.sort_by(|a, b| b.updated_at.cmp(&a.updated_at));
        all.into_iter().skip(offset).take(limit).collect()
    }

    pub fn list_by_category(&self, category: &ConceptCategory, limit: usize) -> Vec<&Concept> {
        if let Some(ids) = self.category_index.get(category) {
            let mut result: Vec<&Concept> =
                ids.iter().filter_map(|id| self.concepts.get(id)).collect();
            result.sort_by(|a, b| b.updated_at.cmp(&a.updated_at));
            result.into_iter().take(limit).collect()
        } else {
            Vec::new()
        }
    }

    pub fn search_pretty(&self, query: &str, top_k: usize) -> String {
        let results = self.search_by_text(query, top_k);
        if results.is_empty() {
            return "No concepts found.".to_string();
        }
        let mut output = format!("Found {} concepts:\n", results.len());
        for (i, (sim, concept)) in results.iter().enumerate() {
            output += &format!(
                "{}. [{} {:.2}] {} - {}\n",
                i + 1,
                concept.category,
                sim,
                truncate_text(&concept.text, 80),
                concept.id
            );
        }
        output
    }

    pub fn stats_pretty(&self) -> String {
        let by_cat = self.count_by_category();
        let total = self.count();
        let avg_confidence: f32 = if total > 0 {
            self.concepts.values().map(|c| c.confidence).sum::<f32>() / total as f32
        } else {
            0.0
        };
        let total_usage: u32 = self.concepts.values().map(|c| c.usage_count).sum();

        format!(
            "📚 Semantic Memory Stats:\n   Total concepts: {}\n   Average confidence: {:.2}\n   Total usage count: {}\n   By category:\n{}",
            total,
            avg_confidence,
            total_usage,
            by_cat.iter()
                .map(|(cat, count)| format!("      - {}: {}", cat, count))
                .collect::<Vec<_>>()
                .join("\n")
        )
    }

    pub fn format_concept(&self, concept: &Concept) -> String {
        format!(
            "[{}] {} (conf: {:.2}, used: {}, updated: {})\n   Source: {}",
            concept.id,
            concept.text,
            concept.confidence,
            concept.usage_count,
            concept.updated_at.format("%Y-%m-%d %H:%M"),
            concept.source
        )
    }

    pub fn get_concept_by_id(&self, id: &str) -> Option<&Concept> {
        if let Ok(uuid) = uuid::Uuid::parse_str(id) {
            self.concepts.get(&uuid)
        } else {
            None
        }
    }

    pub fn find_similar_text(&self, text: &str, threshold: f32) -> Vec<&Concept> {
        let target = text.to_lowercase();
        self.concepts
            .values()
            .filter(|c| {
                let c_text = c.text.to_lowercase();
                let similarity = text_similarity(&target, &c_text);
                similarity > threshold
            })
            .collect()
    }
}

fn truncate_text(text: &str, max_chars: usize) -> String {
    let char_count = text.chars().count();
    if char_count <= max_chars {
        return text.to_string();
    }
    let byte_pos = text
        .char_indices()
        .nth(max_chars)
        .map(|(p, _)| p)
        .unwrap_or(text.len());
    format!("{}...", &text[..byte_pos])
}

fn text_similarity(a: &str, b: &str) -> f32 {
    let a_words: HashSet<&str> = a.split_whitespace().collect();
    let b_words: HashSet<&str> = b.split_whitespace().collect();
    if a_words.is_empty() || b_words.is_empty() {
        return 0.0;
    }
    let intersection = a_words.intersection(&b_words).count();
    let union = a_words.union(&b_words).count();
    intersection as f32 / union as f32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_concept_creation() {
        let concept = Concept::new(
            "User likes cats".to_string(),
            ConceptCategory::Preferences,
            "test".to_string(),
        );
        assert!(concept.is_valid());
        assert_eq!(concept.confidence, 0.5);
    }

    #[test]
    fn test_concept_with_confidence() {
        let concept = Concept::new(
            "Test fact".to_string(),
            ConceptCategory::Facts,
            "test".to_string(),
        )
        .with_confidence(0.9);
        assert_eq!(concept.confidence, 0.9);
    }

    #[test]
    fn test_category_display() {
        assert_eq!(ConceptCategory::Facts.to_string(), "facts");
        assert_eq!(ConceptCategory::Preferences.to_string(), "preferences");
    }
}

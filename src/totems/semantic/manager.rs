//! 🧠 Менеджер семантической памяти
//!
//! Управляет концептами: добавление, поиск, объединение
//! Извлечение концептов выполняется отдельно через SemanticExtractor

use anyhow::Result;
use regex::Regex;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use super::concept::{
    CategoryDecayStats, Concept, ConceptCategory, DecayStats, GraphStats, KnowledgeGraph, Triple,
};
use super::persistence::SemanticPersistenceManager;
use crate::priests::embeddings::Embedder;
use crate::totems::retrieval::vector_store::cosine_similarity;

fn remove_negation(text: &str) -> String {
    let mut result = text.to_string();
    let negations = [
        "don't ",
        "don't ",
        "doesn't ",
        "didn't ",
        "not ",
        "n't ",
        "не ",
        "нельзя ",
        "не люблю ",
        "не нравится ",
    ];
    for neg in &negations {
        result = result.replace(neg, "");
    }
    result.trim().to_string()
}

fn is_contradiction(text1: &str, text2: &str) -> bool {
    let t1 = text1.to_lowercase();
    let t2 = text2.to_lowercase();

    let t1_neg = t1.contains("n't")
        || t1.contains("not ")
        || t1.contains("не ")
        || t1.contains("нельзя")
        || t1.contains("не люблю")
        || t1.contains("не нравится");
    let t2_neg = t2.contains("n't")
        || t2.contains("not ")
        || t2.contains("не ")
        || t2.contains("нельзя")
        || t2.contains("не люблю")
        || t2.contains("не нравится");

    if t1_neg != t2_neg {
        let base1 = remove_negation(&t1);
        let base2 = remove_negation(&t2);

        let check_words = [
            "love",
            "loves",
            "loved",
            "люблю",
            "любит",
            "любил",
            "like",
            "likes",
            "liked",
            "нравится",
            "нравилось",
            "понравилось",
            "prefer",
            "prefers",
            "preferred",
            "предпочитаю",
            "предпочитает",
            "предпочитал",
            "hate",
            "hates",
            "hated",
            "ненавижу",
            "ненавидит",
            "ненавидел",
            "enjoy",
            "enjoys",
            "enjoyed",
        ];

        let has_match1 = check_words.iter().any(|w| base1.contains(*w));
        let has_match2 = check_words.iter().any(|w| base2.contains(*w));
        if has_match1 && has_match2 {
            return true;
        }
    }

    false
}

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
    knowledge_graph: KnowledgeGraph,
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
            knowledge_graph: KnowledgeGraph::new(),
        };

        if let Some(loaded) = manager.persistence.load()? {
            let _count = loaded.len();
            for mut concept in loaded.into_iter() {
                manager.index_concept(&concept.id, &concept.category);
                concept.embedding = manager.embedder.embed(&concept.text)?;
                manager.concepts.insert(concept.id, concept);
            }
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
            knowledge_graph: KnowledgeGraph::new(),
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
        let cleaned_text = text
            .trim()
            .replace("  ", " ")
            .replace(" .", ".")
            .replace(" ,", ",");

        let embedding = self.embedder.embed(&cleaned_text)?;

        let normalized_text = cleaned_text.to_lowercase();

        // Check for contradictions
        for (_, existing) in &self.concepts {
            if is_contradiction(&normalized_text, &existing.text.to_lowercase()) {
                // Keep higher confidence
                let new_conf = confidence.unwrap_or(0.5);
                if new_conf > existing.confidence {
                    continue; // This replaces the existing one
                } else {
                    return Ok(existing.clone()); // Keep existing, return it
                }
            }
        }

        // Check for duplicates using similarity
        for (_id, existing) in &self.concepts {
            let similarity = cosine_similarity(&embedding, &existing.embedding);
            if similarity > 0.95 {
                // Merge concepts - keep higher confidence
                let mut merged = existing.clone();
                if let Some(new_conf) = confidence {
                    if new_conf > existing.confidence {
                        merged.confidence = new_conf;
                        merged.updated_at = chrono::Utc::now();
                    }
                }
                return Ok(merged);
            }
        }

        // Create new concept
        let mut concept = Concept::new(cleaned_text, category.clone(), source);
        if let Some(conf) = confidence {
            concept = concept.with_confidence(conf);
        }
        concept.embedding = embedding.clone();
        self.index_concept(&concept.id, &category);
        self.concepts.insert(concept.id, concept.clone());
        Ok(concept)
    }

    pub fn search(
        &self,
        query: &str,
        top_k: usize,
        category: Option<ConceptCategory>,
    ) -> Vec<(f32, &Concept)> {
        let query_embedding = match self.embedder.embed(query) {
            Ok(embedding) => embedding,
            Err(_) => return Vec::new(),
        };

        let mut candidates = self
            .concepts
            .values()
            .filter(|c| {
                if let Some(cat) = &category {
                    c.category == *cat
                } else {
                    true
                }
            })
            .collect::<Vec<_>>();

        candidates.sort_by(|a, b| {
            let sim_a = cosine_similarity(&query_embedding, &a.embedding);
            let sim_b = cosine_similarity(&query_embedding, &b.embedding);
            sim_b
                .partial_cmp(&sim_a)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        candidates
            .into_iter()
            .take(top_k)
            .map(|c| {
                let sim = cosine_similarity(&query_embedding, &c.embedding);
                (sim, c)
            })
            .collect()
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

    pub fn get_concepts_by_category(&self, category: &ConceptCategory) -> Vec<&Concept> {
        if let Some(ids) = self.category_index.get(category) {
            ids.iter().filter_map(|id| self.concepts.get(id)).collect()
        } else {
            Vec::new()
        }
    }

    pub fn count(&self) -> usize {
        self.concepts.len()
    }

    /// Get concept by ID
    pub fn get_concept(&self, id: &uuid::Uuid) -> Option<&Concept> {
        self.concepts.get(id)
    }

    pub fn extract_from_dialogue(
        &mut self,
        user_query: &str,
        assistant_response: &str,
        session_id: &str,
    ) -> Result<usize> {
        let raw_results = if let Some(extractor) = &self.extractor {
            let mut extractor = extractor.lock().unwrap();
            extractor.extract(user_query, assistant_response, session_id)?
        } else {
            Vec::new()
        };

        let parsed =
            self.parse_extraction(raw_results, session_id, user_query, assistant_response)?;
        Ok(parsed.len())
    }

    fn parse_extraction(
        &mut self,
        results: ExtractionResult,
        session_id: &str,
        user_query: &str,
        assistant_response: &str,
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
                category.clone(),
                session_id.to_string(),
                Some(confidence),
            ) {
                extracted.push(concept);
            }
        }

        // Extract relations from the dialogue
        let dialogue_text = format!("{} {}", user_query, assistant_response);
        self.extract_relations_from_text(&dialogue_text, session_id)?;

        Ok(extracted)
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

    /// Применить временное затухание ко всем концептам
    pub fn apply_temporal_decay(&mut self) -> Result<usize> {
        let mut concepts_to_remove = Vec::new();
        let mut updated_count = 0;

        for (id, concept) in &mut self.concepts {
            if !concept.apply_temporal_decay() {
                concepts_to_remove.push(*id);
            } else {
                updated_count += 1;
            }
        }

        // Удаляем концепты с низкой уверенностью
        for id in concepts_to_remove {
            if let Some(concept) = self.concepts.remove(&id) {
                // Удаляем из категорийного индекса
                if let Some(index) = self.category_index.get_mut(&concept.category) {
                    index.retain(|&x| x != id);
                }
            }
        }

        // Сохраняем изменения
        if !self.concepts.is_empty() {
            let concepts: Vec<Concept> = self.concepts.values().cloned().collect();
            self.persistence.save(&concepts)?;
        }

        Ok(updated_count)
    }

    /// Получить концепты с учетом временного затухания (без фактического применения)
    pub fn get_concepts_with_decay(&self, top_k: usize) -> Vec<(f32, &Concept)> {
        let mut concepts_with_decay: Vec<(f32, &Concept)> = self
            .concepts
            .values()
            .map(|concept| {
                let effective_confidence = concept.get_effective_confidence();
                (effective_confidence, concept)
            })
            .filter(|(confidence, _)| *confidence > 0.01) // фильтруем очень низкую уверенность
            .collect();

        // Сортируем по эффективной уверенности
        concepts_with_decay.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());

        concepts_with_decay.truncate(top_k);
        concepts_with_decay
    }

    /// Статистика по затуханию концептов
    pub fn get_decay_stats(&self) -> DecayStats {
        let mut total_concepts = 0;
        let mut decayed_concepts = 0;
        let mut low_confidence_concepts = 0;
        let mut category_stats: HashMap<ConceptCategory, CategoryDecayStats> = HashMap::new();

        for concept in self.concepts.values() {
            total_concepts += 1;
            let effective_confidence = concept.get_effective_confidence();

            if effective_confidence < concept.confidence * 0.9 {
                decayed_concepts += 1;
            }

            if effective_confidence < 0.3 {
                low_confidence_concepts += 1;
            }

            let cat_stats = category_stats
                .entry(concept.category.clone())
                .or_insert_with(CategoryDecayStats::default);
            cat_stats.total += 1;
            cat_stats.avg_confidence += effective_confidence;

            if effective_confidence < 0.3 {
                cat_stats.low_confidence += 1;
            }
        }

        // Вычисляем средние значения
        for cat_stats in category_stats.values_mut() {
            if cat_stats.total > 0 {
                cat_stats.avg_confidence /= cat_stats.total as f32;
            }
        }

        DecayStats {
            total_concepts,
            decayed_concepts,
            low_confidence_concepts,
            category_stats,
        }
    }

    // ============ Knowledge Graph Methods ============

    /// Добавить связь (triple) между концептами
    pub fn add_relation(
        &mut self,
        subject_id: &uuid::Uuid,
        predicate: &str,
        object_id: &uuid::Uuid,
        confidence: Option<f32>,
    ) -> Result<uuid::Uuid> {
        // Проверяем существование концептов
        if !self.concepts.contains_key(subject_id) {
            anyhow::bail!("Subject concept not found: {}", subject_id);
        }
        if !self.concepts.contains_key(object_id) {
            anyhow::bail!("Object concept not found: {}", object_id);
        }

        let mut triple = Triple::new(*subject_id, predicate.to_string(), *object_id);
        if let Some(conf) = confidence {
            triple = triple.with_confidence(conf);
        }

        let triple_id = self.knowledge_graph.add_triple(triple);

        // Обновляем related_concepts в концептах
        if let Some(subject_concept) = self.concepts.get_mut(subject_id) {
            if !subject_concept.related_concepts.contains(object_id) {
                subject_concept.related_concepts.push(*object_id);
            }
        }
        if let Some(object_concept) = self.concepts.get_mut(object_id) {
            if !object_concept.related_concepts.contains(subject_id) {
                object_concept.related_concepts.push(*subject_id);
            }
        }

        Ok(triple_id)
    }

    /// Найти связи от концепта
    pub fn find_outgoing_relations(&self, concept_id: &uuid::Uuid) -> Vec<&Triple> {
        self.knowledge_graph.find_by_subject(concept_id)
    }

    /// Найти связи к концепту
    pub fn find_incoming_relations(&self, concept_id: &uuid::Uuid) -> Vec<&Triple> {
        self.knowledge_graph.find_by_object(concept_id)
    }

    /// Найти все связанные концепты
    pub fn find_related_concepts(&self, concept_id: &uuid::Uuid) -> Vec<(uuid::Uuid, &str, f32)> {
        self.knowledge_graph.find_related_concepts(concept_id)
    }

    /// Автоматическое извлечение отношений из текста
    pub fn extract_relations_from_text(
        &mut self,
        text: &str,
        source_session: &str,
    ) -> Result<usize> {
        let mut relations_added = 0;

        // Простые паттерны для извлечения отношений
        let patterns = vec![
            // X is a Y -> (X, is_a, Y)
            (r#"(.+)\s+is\s+a\s+([a-z]+)"#, "is_a"),
            // X likes Y -> (X, likes, Y)
            (r#"(.+)\s+likes\s+(.+)"#, "likes"),
            // X wants Y -> (X, wants, Y)
            (r#"(.+)\s+wants\s+(.+)"#, "wants"),
            // X has Y -> (X, has, Y)
            (r#"(.+)\s+has\s+(.+)"#, "has"),
            // Russian patterns
            (r#"(.+)\s+—\s+это\s+(.+)"#, "is_a"),
            (r#"(.+)\s+любит\s+(.+)"#, "likes"),
            (r#"(.+)\s+хочет\s+(.+)"#, "wants"),
        ];

        for (pattern, predicate) in patterns {
            if let Ok(re) = Regex::new(pattern) {
                for caps in re.captures_iter(text) {
                    if let (Some(subject_match), Some(object_match)) = (caps.get(1), caps.get(2)) {
                        let subject_text = subject_match.as_str().trim().to_lowercase();
                        let object_text = object_match.as_str().trim().to_lowercase();

                        // Находим или создаем концепты
                        let subject_id =
                            self.find_or_create_concept(&subject_text, source_session)?;
                        let object_id =
                            self.find_or_create_concept(&object_text, source_session)?;

                        // Добавляем связь
                        if let Ok(_) =
                            self.add_relation(&subject_id, predicate, &object_id, Some(0.7))
                        {
                            relations_added += 1;
                        }
                    }
                }
            }
        }

        Ok(relations_added)
    }

    /// Найти или создать концепт
    fn find_or_create_concept(&mut self, text: &str, source: &str) -> Result<uuid::Uuid> {
        // Ищем существующий концепт
        for (id, concept) in &self.concepts {
            if concept.text.to_lowercase() == text && concept.source == source {
                return Ok(*id);
            }
        }

        // Создаем новый концепт
        let concept = Concept::new(
            text.to_string(),
            ConceptCategory::General,
            source.to_string(),
        );
        let concept_id = concept.id;
        self.add_concept_internal(concept)?;
        Ok(concept_id)
    }

    /// Внутренний метод добавления концепта без сохранения
    fn add_concept_internal(&mut self, concept: Concept) -> Result<()> {
        let id = concept.id;
        self.index_concept(&id, &concept.category);
        let mut concept_with_embedding = concept;
        concept_with_embedding.embedding = self.embedder.embed(&concept_with_embedding.text)?;
        self.concepts.insert(id, concept_with_embedding);
        Ok(())
    }

    /// Получить статистику графа
    pub fn get_graph_stats(&self) -> GraphStats {
        self.knowledge_graph.get_stats()
    }

    /// Сохранить все данные (концепты и граф)
    pub fn save(&self) -> Result<()> {
        // Save concepts
        let concepts: Vec<&Concept> = self.concepts.values().collect();
        let owned: Vec<Concept> = concepts.into_iter().cloned().collect();
        self.persistence.save(&owned)?;
        // Save knowledge graph
        self.save_graph()?;
        Ok(())
    }

    /// Сохранить граф
    pub fn save_graph(&self) -> Result<()> {
        use std::fs;
        // Сохраняем граф в отдельный файл
        let graph_path = std::path::Path::new("memory_data/semantic/knowledge_graph.json");
        fs::create_dir_all(graph_path.parent().unwrap())?;
        let json = serde_json::to_string_pretty(&self.knowledge_graph)?;
        fs::write(graph_path, json)?;
        Ok(())
    }

    /// Загрузить граф
    pub fn load_graph(&mut self) -> Result<()> {
        use std::fs;
        let graph_path = std::path::Path::new("memory_data/semantic/knowledge_graph.json");
        if graph_path.exists() {
            let json = fs::read_to_string(graph_path)?;
            self.knowledge_graph = serde_json::from_str(&json)?;
        }
        Ok(())
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

    if a_words.is_empty() && b_words.is_empty() {
        return 1.0;
    }
    if a_words.is_empty() || b_words.is_empty() {
        return 0.0;
    }

    let intersection: HashSet<_> = a_words.intersection(&b_words).collect();
    let union: HashSet<_> = a_words.union(&b_words).collect();

    if union.is_empty() {
        return 0.0;
    }

    intersection.len() as f32 / union.len() as f32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_concept_creation() {
        let concept = Concept::new(
            "User likes coffee".to_string(),
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

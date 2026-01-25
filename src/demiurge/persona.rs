//! Persona - The Active Persona Instance
//!
//! Persona is an instantiated archetype with dynamic traits,
//! communication settings, and evolution state.

use crate::demiurge::{
    Archetype, ArchetypeDirective, BaseTraits, CommunicationStyle, ContextStorage, Directive,
    EvolutionState, NarrativeManager, PersonaSessionContext,
};
use crate::totems::episodic::{DialogueManager, LlmPipeline};
use crate::totems::semantic::{ConceptCategory, SemanticMemoryManager};
use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::io;
use std::sync::{Arc, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};

pub const MAX_CONTEXT_AGE_DAYS: i64 = 30;
pub const MIN_TURNS_FOR_SAVE: usize = 3;

#[derive(Clone)]
pub struct Persona {
    pub archetype_id: String,
    pub name: String,
    pub description: String,
    pub base_traits: HashMap<String, f32>,
    pub communication: CommunicationStyle,
    pub directives: Vec<Directive>,
    pub narrative: NarrativeManager,
    pub evolution: EvolutionState,
    pub semantic_manager: Option<Arc<Mutex<SemanticMemoryManager>>>,
}

impl Persona {
    /// Create persona from archetype
    pub fn from_archetype(archetype: Arc<Archetype>) -> Self {
        let traits = Self::extract_traits(&archetype.base_traits);
        let directives = Self::extract_directives(&archetype.directives);

        Self {
            archetype_id: archetype.id.clone(),
            name: archetype.name.clone(),
            description: archetype.description.clone(),
            base_traits: traits,
            communication: archetype.communication.clone(),
            directives,
            narrative: NarrativeManager::new(&archetype.id),
            evolution: EvolutionState::default(),
            semantic_manager: None,
        }
    }

    /// Set semantic memory manager for this persona
    pub fn set_semantic_manager(&mut self, manager: Arc<Mutex<SemanticMemoryManager>>) {
        self.semantic_manager = Some(manager);
    }

    /// Get user preferences from semantic memory
    pub fn get_user_preferences(&self) -> Vec<(String, String)> {
        if let Some(ref sm) = self.semantic_manager {
            let sm = sm.lock().unwrap();
            let prefs = sm.get_concepts_by_category(&ConceptCategory::Preferences);
            prefs
                .into_iter()
                .map(|c| (c.text.clone(), format!("{:.2}", c.confidence)))
                .collect()
        } else {
            Vec::new()
        }
    }

    /// Get user facts from semantic memory
    pub fn get_user_facts(&self) -> Vec<String> {
        if let Some(ref sm) = self.semantic_manager {
            let sm = sm.lock().unwrap();
            let facts = sm.get_concepts_by_category(&ConceptCategory::Facts);
            facts.into_iter().map(|c| c.text.clone()).collect()
        } else {
            Vec::new()
        }
    }

    /// Search semantic memory for relevant concepts
    pub fn search_semantic(&self, query: &str, limit: usize) -> Vec<(String, f32)> {
        if let Some(ref sm) = self.semantic_manager {
            let sm = sm.lock().unwrap();
            let results = sm.search_by_text(query, limit);
            results
                .into_iter()
                .map(|(sim, c)| (c.text.clone(), sim))
                .collect()
        } else {
            Vec::new()
        }
    }

    /// Get all user knowledge as formatted string
    pub fn get_user_knowledge_summary(&self) -> String {
        let preferences = self.get_user_preferences();
        let facts = self.get_user_facts();

        if preferences.is_empty() && facts.is_empty() {
            return String::new();
        }

        let mut parts = Vec::new();

        if !facts.is_empty() {
            parts.push(format!("KNOWN FACTS ABOUT USER:\n- {}", facts.join("\n- ")));
        }

        if !preferences.is_empty() {
            let prefs_list: Vec<String> = preferences
                .into_iter()
                .map(|(text, conf)| format!("{} (confidence: {})", text, conf))
                .collect();
            parts.push(format!("USER PREFERENCES:\n- {}", prefs_list.join("\n- ")));
        }

        parts.join("\n\n")
    }

    /// Extract and store concepts from current dialogue
    pub fn extract_and_store_concepts(&self, user_input: &str, assistant_response: &str) {
        if let Some(ref sm) = self.semantic_manager {
            let has_self_disclosure = user_input.to_lowercase().contains("я ")
                || user_input.to_lowercase().contains("мой ")
                || user_input.to_lowercase().contains("моя ")
                || user_input.to_lowercase().contains("моё ")
                || user_input.to_lowercase().contains("мои ")
                || user_input.to_lowercase().contains("люблю")
                || user_input.to_lowercase().contains("предпочитаю")
                || user_input.to_lowercase().contains("нравится")
                || user_input.to_lowercase().contains("не люблю")
                || user_input.to_lowercase().contains("i ")
                || user_input.to_lowercase().contains("my ")
                || user_input.to_lowercase().contains("i'm")
                || user_input.to_lowercase().contains("i am");

            if has_self_disclosure {
                let session_id = format!("persona_{}", self.archetype_id);
                let mut sm = sm.lock().unwrap();
                if let Err(e) =
                    sm.extract_from_dialogue(user_input, assistant_response, &session_id)
                {
                    eprintln!("Warning: Failed to extract concepts: {}", e);
                }
            }
        }
    }

    /// Extract traits into HashMap
    fn extract_traits(base: &BaseTraits) -> HashMap<String, f32> {
        let mut traits = HashMap::new();
        traits.insert("analytical".to_string(), base.analytical.clamp(0.0, 1.0));
        traits.insert("curious".to_string(), base.curious.clamp(0.0, 1.0));
        traits.insert("verbose".to_string(), base.verbose.clamp(0.0, 1.0));
        traits.insert("patient".to_string(), base.patient.clamp(0.0, 1.0));
        traits.insert("humor".to_string(), base.humor.clamp(0.0, 1.0));
        traits.insert("empathy".to_string(), base.empathy.clamp(0.0, 1.0));
        traits.insert("technical".to_string(), base.technical.clamp(0.0, 1.0));
        traits.insert("pedagogical".to_string(), base.pedagogical.clamp(0.0, 1.0));
        traits.insert("creative".to_string(), base.creative.clamp(0.0, 1.0));
        traits.insert("supportive".to_string(), base.supportive.clamp(0.0, 1.0));
        traits.insert("skeptical".to_string(), base.skeptical.clamp(0.0, 1.0));
        traits.insert("formal".to_string(), base.formal.clamp(0.0, 1.0));
        traits
    }

    /// Extract directives from archetype format
    fn extract_directives(archetype_directives: &[ArchetypeDirective]) -> Vec<Directive> {
        archetype_directives
            .iter()
            .map(|d| Directive {
                rule: d.rule.clone(),
                priority: d.priority,
                directive_type: crate::demiurge::directives::DirectiveType::Custom,
                params: d.params.clone(),
            })
            .collect()
    }

    /// Get trait value with evolution offsets applied
    pub fn get_trait(&self, name: &str) -> f32 {
        let base = self.base_traits.get(name).copied().unwrap_or(0.5);
        let offset = self
            .evolution
            .trait_offsets
            .get(name)
            .copied()
            .unwrap_or(0.0);
        (base + offset).clamp(0.0, 1.0)
    }

    /// Get all current traits (base + evolution)
    pub fn get_all_traits(&self) -> HashMap<String, f32> {
        let mut traits = self.base_traits.clone();
        for (name, offset) in &self.evolution.trait_offsets {
            let base = traits.get(name).copied().unwrap_or(0.5);
            traits.insert(name.clone(), (base + offset).clamp(0.0, 1.0));
        }
        traits
    }

    /// Format system prompt with persona context
    pub fn format_system_prompt(&self) -> String {
        let emoji = match self.communication.emoji_frequency.as_str() {
            "frequent" => " 💫✨",
            "moderate" => " ✨",
            _ => "",
        };

        let traits = self.get_all_traits();
        let trait_desc = Self::describe_traits(&traits);

        format!(
            r#"Ты — {}, {}.

Твой стиль общения: {}, {} формальный тон.
{}
{}
Приветствие: "{}"{}

ВАЖНО:
- Не придумывай и не упоминай детали прошлых разговоров, которых не было
- Не говори "помню, что..." или "раньше ты говорил..." если не уверен, что это было на самом деле
- Если пользователь спрашивает о прошлом, честно скажи, что не помнишь, вместо того чтобы выдумывать"#,
            self.name,
            self.description,
            self.communication.style,
            if self.communication.use_honorifics {
                "с обращением на Вы"
            } else {
                "на ты"
            },
            trait_desc,
            self.communication.signature,
            self.communication.greeting,
            emoji,
        )
    }

    /// Generate human-readable trait description
    fn describe_traits(traits: &HashMap<String, f32>) -> String {
        let mut desc = Vec::new();

        if traits.get("analytical").unwrap_or(&0.5) > &0.8 {
            desc.push("склонен к аналитическому мышлению");
        }
        if traits.get("empathy").unwrap_or(&0.5) > &0.8 {
            desc.push("очень эмпатичный");
        }
        if traits.get("humor").unwrap_or(&0.5) > &0.7 {
            desc.push("любишь шутить");
        }
        if traits.get("pedagogical").unwrap_or(&0.5) > &0.7 {
            desc.push("любишь объяснять и учить");
        }
        if traits.get("technical").unwrap_or(&0.5) > &0.8 {
            desc.push("технически подкован");
        }
        if traits.get("creative").unwrap_or(&0.5) > &0.7 {
            desc.push("креативный");
        }
        if traits.get("patient").unwrap_or(&0.5) > &0.8 {
            desc.push("терпеливый");
        }

        if desc.is_empty() {
            "сбалансированный характер".to_string()
        } else {
            format!("Ты {}", desc.join(", "))
        }
    }

    /// Apply interaction and evolve
    pub fn apply_interaction(&mut self, _interaction: crate::demiurge::Interaction) {
        self.evolution.interactions_count += 1;

        // Apply to evolution engine
        // This will be implemented in evolution.rs
    }

    /// Save narrative to disk
    pub fn save_narrative(&self) -> Result<()> {
        let mut narrative = self.narrative.clone();
        narrative.save()
    }

    /// Load narrative from disk
    pub fn load_narrative(&mut self) -> Result<()> {
        self.narrative.load()
    }

    pub fn load_session_context(&mut self) -> Result<Option<PersonaSessionContext>> {
        if ContextStorage::is_expired(&self.archetype_id, MAX_CONTEXT_AGE_DAYS) {
            let _ = ContextStorage::delete(&self.archetype_id);
            return Ok(None);
        }

        Ok(ContextStorage::load(&self.archetype_id)?)
    }

    pub fn save_session_context<D: LlmPipeline>(
        &self,
        dialogue_manager: &DialogueManager,
        pipeline: &D,
    ) -> Result<Option<PersonaSessionContext>> {
        let turn_count = dialogue_manager.current_session().turn_count();

        if turn_count < MIN_TURNS_FOR_SAVE {
            return Ok(None);
        }

        let analysis = dialogue_manager.analyze_for_context(pipeline, 10)?;

        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        let previous_session_id = dialogue_manager.current_session().id.to_string();

        let mut context = PersonaSessionContext::new(&self.archetype_id.clone());
        context.previous_session_id = previous_session_id;
        context.last_interaction_date = now;
        context.summary = analysis.summary;
        context.key_topics = analysis.key_topics;
        context.emotional_state = analysis.emotional_state;
        context.last_topic = analysis.last_topic;

        ContextStorage::save(&context)?;

        Ok(Some(context))
    }

    pub fn generate_contextual_greeting(&self, context: &PersonaSessionContext) -> String {
        let emoji = match self.communication.emoji_frequency.as_str() {
            "frequent" => " 💫✨",
            "moderate" => " ✨",
            _ => "",
        };

        let honorific = if self.communication.use_honorifics {
            "Вы"
        } else {
            "ты"
        };

        let emotional_indicator = if context.emotional_state > 0.7 {
            "рада"
        } else if context.emotional_state > 0.4 {
            "рада"
        } else {
            "здесь"
        };

        let greeting = if !context.summary.is_empty() {
            match self.archetype_id.as_str() {
                "girlfriend" => format!(
                    "Привет{}! {} {}{} Помню, что мы говорили о {}. Как там {}?",
                    emoji,
                    honorific,
                    emotional_indicator,
                    emoji,
                    context.key_topics.first().map(|t| t.as_str()).unwrap_or_else(|| "этом"),
                    if context.last_topic.is_empty() {
                        "всё"
                    } else {
                        &context.last_topic
                    }
                ),
                "programmer" => format!(
                    "Привет. Контекст восстановлен. Последняя тема: {}. Есть незавершённые вопросы. Готов{} продолжить.",
                    context.key_topics.first().map(|t| t.as_str()).unwrap_or_else(|| "общее"),
                    if honorific == "ты" { "" } else { "ы" }
                ),
                "devops" => format!(
                    "Привет. Восстановлено {}. Система готова к работе. Продолжаем с {}?",
                    context.key_topics.len(),
                    context.last_topic
                ),
                "scientist" => format!(
                    "Здравствуй. Интересно, что привело тебя снова? Помню, мы обсуждали {}. Есть что добавить к исследованию?",
                    context.key_topics.first().map(|t| t.as_str()).unwrap_or_else(|| "эту тему")
                ),
                "philosopher" => format!(
                    "Здравствуй. Интересно, что привело тебя снова сюда? Я помню, что мы говорили о {}. Что нового в твоих размышлениях?",
                    context.key_topics.first().map(|t| t.as_str()).unwrap_or_else(|| "этом")
                ),
                _ => format!(
                    "Привет{}! Помню наш разговор о {}. {} продолжить?",
                    emoji,
                    context.key_topics.first().map(|t| t.as_str()).unwrap_or_else(|| "этом"),
                    if honorific == "ты" { "Давай" } else { "Давайте" }
                ),
            }
        } else {
            self.communication.greeting.clone()
        };

        greeting
    }

    pub fn has_saved_context(&self) -> bool {
        ContextStorage::exists(&self.archetype_id)
            && !ContextStorage::is_expired(&self.archetype_id, MAX_CONTEXT_AGE_DAYS)
    }
}

/// Compact persona info for CLI display
#[derive(Debug, Serialize, Deserialize)]
pub struct PersonaInfo {
    pub archetype_id: String,
    pub name: String,
    pub description: String,
    pub traits: HashMap<String, f32>,
    pub evolution: PersonaEvolutionInfo,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct PersonaEvolutionInfo {
    pub interactions: u64,
    pub unlocked_traits: Vec<String>,
    pub relationship_score: f32,
}

impl From<&Persona> for PersonaInfo {
    fn from(p: &Persona) -> Self {
        Self {
            archetype_id: p.archetype_id.clone(),
            name: p.name.clone(),
            description: p.description.clone(),
            traits: p.get_all_traits(),
            evolution: PersonaEvolutionInfo {
                interactions: p.evolution.interactions_count,
                unlocked_traits: p.evolution.unlocked_traits.clone(),
                relationship_score: p.evolution.relationship_score,
            },
        }
    }
}

//! Evolution Engine - Dynamic Persona Development
//!
//! Tracks interaction outcomes and modifies persona traits
//! over time based on evolution rules.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Interaction data for evolution tracking
#[derive(Debug, Clone)]
pub struct Interaction {
    pub user_sentiment: f32, // -1.0 to 1.0
    pub successful_help: bool,
    pub emotional_depth: f32, // 0.0 to 1.0
    pub topics: Vec<String>,
    pub user_gave_feedback: bool,
    pub feedback_positive: bool,
    pub is_deep_conversation: bool,
    pub is_code_related: bool,
    pub is_emotional_support: bool,
}

/// Current evolution state of persona
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct EvolutionState {
    pub interactions_count: u64,
    pub successful_helps: u64,
    pub relationship_score: f32,
    pub trait_offsets: HashMap<String, f32>,
    pub unlocked_traits: Vec<String>,
    pub last_interaction_time: u64,
    pub decay_applied_at: u64,
}

/// Evolution engine for trait modifications
pub struct EvolutionEngine {
    state: EvolutionState,
    rules: EvolutionRules,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvolutionRules {
    pub trait_changes: HashMap<String, TraitChangeRule>,
    pub decay: HashMap<String, f32>,
    pub unlock_conditions: Vec<UnlockCondition>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TraitChangeRule {
    pub rate: f32,
    pub trigger: String,
    pub condition: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UnlockCondition {
    pub r#trait: String,
    pub require: UnlockRequirements,
    pub description: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UnlockRequirements {
    pub interactions: u32,
    pub successful_help: u32,
    pub empathy_threshold: f32,
    pub relationship_arc_affection: f32,
    pub topics_covers: Vec<String>,
    pub deep_conversations: u32,
}

impl EvolutionEngine {
    /// Create new evolution engine with rules
    pub fn new(rules: EvolutionRules) -> Self {
        Self {
            state: EvolutionState::default(),
            rules,
        }
    }

    /// Apply interaction and update evolution state
    pub fn apply_interaction(&mut self, interaction: &Interaction) {
        self.state.interactions_count += 1;
        self.state.last_interaction_time = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        if interaction.successful_help {
            self.state.successful_helps += 1;
        }

        // Update relationship score
        let sentiment_impact = interaction.user_sentiment * 0.01;
        let help_impact = if interaction.successful_help {
            0.005
        } else {
            0.0
        };
        self.state.relationship_score =
            (self.state.relationship_score + sentiment_impact + help_impact).clamp(0.0, 1.0);

        // Apply trait changes
        self.apply_trait_changes(interaction);

        // Check for unlocks
        self.check_unlocks();
    }

    /// Apply trait modifications based on rules
    fn apply_trait_changes(&mut self, interaction: &Interaction) {
        for (trait_name, rule) in &self.rules.trait_changes {
            let should_apply = match rule.trigger.as_str() {
                "successful_help" => interaction.successful_help,
                "positive_feedback" => {
                    interaction.user_gave_feedback && interaction.feedback_positive
                }
                "deep_conversation" => interaction.is_deep_conversation,
                "emotional_support" => interaction.is_emotional_support,
                "code_related" => interaction.is_code_related,
                "any" => true,
                _ => false,
            };

            if should_apply {
                let offset = self
                    .state
                    .trait_offsets
                    .entry(trait_name.clone())
                    .or_insert(0.0);
                *offset += rule.rate;
                *offset = offset.clamp(-0.3, 0.3); // Cap changes
            }
        }

        // Apply decay
        self.apply_decay();
    }

    /// Apply decay to unused traits
    fn apply_decay(&mut self) {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        // Only decay if enough time has passed (1 hour)
        if now - self.state.decay_applied_at < 3600 {
            return;
        }

        for (trait_name, decay_rate) in &self.rules.decay {
            let offset = self
                .state
                .trait_offsets
                .entry(trait_name.clone())
                .or_insert(0.0);
            *offset -= decay_rate;
            *offset = offset.clamp(-0.3, 0.3);
        }

        self.state.decay_applied_at = now;
    }

    /// Check and apply unlock conditions
    fn check_unlocks(&mut self) {
        for unlock in &self.rules.unlock_conditions {
            // Check if already unlocked
            if self.state.unlocked_traits.contains(&unlock.r#trait) {
                continue;
            }

            let meets_requirements = self.evaluate_requirements(&unlock.require);

            if meets_requirements {
                self.state.unlocked_traits.push(unlock.r#trait.clone());
            }
        }
    }

    /// Evaluate if requirements are met
    fn evaluate_requirements(&self, req: &UnlockRequirements) -> bool {
        if req.interactions > 0 && self.state.interactions_count < req.interactions as u64 {
            return false;
        }
        if req.successful_help > 0 && self.state.successful_helps < req.successful_help as u64 {
            return false;
        }
        if req.empathy_threshold > 0.0 && self.state.relationship_score < req.empathy_threshold {
            return false;
        }

        true
    }

    /// Get current trait value (base + offset)
    pub fn get_trait(&self, base: f32, trait_name: &str) -> f32 {
        let offset = self
            .state
            .trait_offsets
            .get(trait_name)
            .copied()
            .unwrap_or(0.0);
        (base + offset).clamp(0.0, 1.0)
    }

    /// Get evolution state
    pub fn state(&self) -> &EvolutionState {
        &self.state
    }

    /// Get unlocked traits
    pub fn unlocked_traits(&self) -> &[String] {
        &self.state.unlocked_traits
    }

    /// Get interaction count
    pub fn interactions_count(&self) -> u64 {
        self.state.interactions_count
    }

    /// Get relationship score
    pub fn relationship_score(&self) -> f32 {
        self.state.relationship_score
    }
}

/// Default evolution rules for programmer archetype
impl Default for EvolutionRules {
    fn default() -> Self {
        let mut trait_changes = HashMap::new();
        trait_changes.insert(
            "empathy".to_string(),
            TraitChangeRule {
                rate: 0.001,
                trigger: "successful_help".to_string(),
                condition: "".to_string(),
            },
        );
        trait_changes.insert(
            "pedagogical".to_string(),
            TraitChangeRule {
                rate: 0.002,
                trigger: "positive_feedback".to_string(),
                condition: "".to_string(),
            },
        );
        trait_changes.insert(
            "humor".to_string(),
            TraitChangeRule {
                rate: 0.0005,
                trigger: "deep_conversation".to_string(),
                condition: "".to_string(),
            },
        );

        let mut decay = HashMap::new();
        decay.insert("humor".to_string(), 0.0001);
        decay.insert("patience".to_string(), 0.0002);

        let mut unlock_conditions = Vec::new();
        unlock_conditions.push(UnlockCondition {
            r#trait: "mentor".to_string(),
            require: UnlockRequirements {
                interactions: 100,
                successful_help: 50,
                empathy_threshold: 0.8,
                relationship_arc_affection: 0.0,
                topics_covers: vec![],
                deep_conversations: 0,
            },
            description: "Когда накопил достаточно опыта помощи".to_string(),
        });
        unlock_conditions.push(UnlockCondition {
            r#trait: "life_coach".to_string(),
            require: UnlockRequirements {
                interactions: 300,
                successful_help: 0,
                empathy_threshold: 0.0,
                relationship_arc_affection: 0.75,
                topics_covers: vec!["career".to_string(), "life".to_string()],
                deep_conversations: 20,
            },
            description: "Когда пользователь начал доверять личные решения".to_string(),
        });

        Self {
            trait_changes,
            decay,
            unlock_conditions,
        }
    }
}

//! Demiurge Level - Persona & Archetype System
//!
//! This module implements Level 3 of the ZIGGURAT MIND architecture:
//! The Demiurge creates and manages AI personas with dynamic traits,
//! communication styles, and evolving narratives.

pub mod archetype;
pub mod directives;
pub mod evolution;
pub mod narrative;
pub mod persona;

pub use archetype::{
    Archetype, ArchetypeDirective, ArchetypeLoader, BaseTraits, CommunicationStyle,
};
pub use directives::Directive;
pub use evolution::{EvolutionState, Interaction};
pub use narrative::{ContextStorage, NarrativeSystem, PersonaSessionContext};
pub use persona::Persona;

use anyhow::Result;
use std::sync::Arc;

pub fn create_persona(archetype_id: &str) -> Result<Persona> {
    let archetype = ArchetypeLoader::load(archetype_id)?;
    Ok(Persona::from_archetype(Arc::new(archetype)))
}

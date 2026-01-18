pub mod episodic;
pub mod memory;
pub mod persistence;
pub mod retrieval;
pub mod semantic;

// Re-export для удобного импорта
pub use episodic::{DialogueManager, DialogueManagerStats, Session, Turn};
pub use memory::{
    ComprehensiveMemoryStats, MemoryContext, MemoryExport, SearchStats, UnifiedMemoryManager,
};
pub use persistence::{PersistenceFormat, PersistenceManager, PersistenceStats};
pub use semantic::{
    Concept, ConceptResult, ConceptUpdate, ExtractionStats, KnowledgeSource, SemanticMemory,
    SemanticMemoryStats,
};

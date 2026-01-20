//! 📚 Семантическая память
//!
//! Модуль для хранения и управления концептами: факты, правила, предпочтения, навыки
//!
//! # Пример использования
//!
//! ```rust
//! use totems::semantic::{SemanticMemoryManager, ConceptCategory};
//!
//! // Создание менеджера
//! let manager = SemanticMemoryManager::new(embedder, persistence)?;
//!
//! // Добавление концепта
//! manager.add_concept(
//!     "Пользователь предпочитает тёмную тему".to_string(),
//!     ConceptCategory::Preferences,
//!     "session-123".to_string(),
//!     Some(0.9),
//! )?;
//!
//! // Поиск
//! let results = manager.search_by_text("тема", 5);
//! ```

pub mod concept;
pub mod manager;
pub mod persistence;

pub use manager::{ConceptExtractor, ExtractionResult, SemanticMemoryManager};

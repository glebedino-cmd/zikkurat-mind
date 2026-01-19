//! 🌙 Dummy Embedding Engine - Fallback для случаев без моделей
//!
//! Используется когда модели эмбеддингов недоступны
//! Возвращает фиктивные векторы для тестирования системы памяти

use crate::priests::embeddings::Embedder;
use anyhow::{anyhow, Result};
use candle_core::Device;

/// Фиктивный эмбеддинг движок для тестирования без реальных моделей
pub struct DummyEmbeddingEngine {
    device: Device,
    embedding_dim: usize,
}

impl DummyEmbeddingEngine {
    pub fn new(device: Device, embedding_dim: usize) -> Self {
        println!("⚠️  Using DUMMY embedding engine (no models available)");
        Self {
            device,
            embedding_dim,
        }
    }

    pub fn embedding_dim(&self) -> usize {
        self.embedding_dim
    }

    /// Создает фиктивный эмбеддинг из текста
    fn embed(&self, text: &str) -> Result<Vec<f32>> {
        // Создаем детерминированный вектор на основе хеша текста
        let hash = self.hash_text(text);
        let mut embedding = Vec::with_capacity(self.embedding_dim);

        // Генерируем псевдо-случайный вектор на основе хеша
        for i in 0..self.embedding_dim {
            let value = ((hash >> (i % 32)) & 0xFF) as f32 / 255.0;
            embedding.push(value);
        }

        Ok(embedding)
    }

    fn hash_text(&self, text: &str) -> u64 {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();
        text.hash(&mut hasher);
        hasher.finish()
    }
}

impl Embedder for DummyEmbeddingEngine {
    fn embed(&self, text: &str) -> Result<Vec<f32>> {
        self.embed(text)
    }

    fn embedding_dim(&self) -> usize {
        self.embedding_dim()
    }
}

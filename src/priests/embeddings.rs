//! 🜂 Уровень 1: Жрецы Железа - Эмбеддинг движок
//!
//! Высокопроизводительный движок векторизации на базе intfloat/multilingual-e5-small
//! Оптимизирован для RTX 4090 32GB с батчингом и кэшированием

#![allow(dead_code)]

use anyhow::{anyhow, Result};
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::bert::{BertModel, Config};
use parking_lot::RwLock;
use std::collections::HashMap;
use std::sync::Arc;
use tokenizers::Tokenizer;

/// Trait для эмбеддингов, поддерживает разные реализации
pub trait Embedder: Send + Sync {
    fn embed(&self, text: &str) -> Result<Vec<f32>>;
    fn embedding_dim(&self) -> usize;
}

/// Конфигурация эмбеддинг движка
#[derive(Debug, Clone)]
pub struct EmbeddingConfig {
    /// Размерность эмбеддинга (384 для e5-small)
    pub embedding_dim: usize,
    /// Максимальная длина последовательности
    pub max_length: usize,
    /// Размер батча для оптимизации GPU
    pub batch_size: usize,
    /// Размер кэша для хранения результатов
    pub cache_size: usize,
    /// Нормализовать ли векторы (cosine similarity)
    pub normalize: bool,
}

impl Default for EmbeddingConfig {
    fn default() -> Self {
        Self {
            embedding_dim: 384, // e5-small
            max_length: 512,
            batch_size: 32,   // Оптимизировано для RTX 4090
            cache_size: 1000, // Кэш для часто используемых текстов
            normalize: true,  // Для cosine similarity
        }
    }
}

/// Высокопроизводительный эмбеддинг движок
pub struct EmbeddingEngine {
    /// BERT модель для векторизации
    model: BertModel,
    /// Токенайзер для предобработки текста
    tokenizer: Tokenizer,
    /// Устройство вычислений (GPU/CPU)
    device: Device,
    /// Конфигурация движка
    config: EmbeddingConfig,
    /// Кэш для хранения вычисленных эмбеддингов
    cache: Arc<RwLock<HashMap<String, Vec<f32>>>>,
    /// Статистика использования
    stats: Arc<RwLock<EmbeddingStats>>,
}

/// Статистика эмбеддинг движка
#[derive(Debug, Default, Clone)]
pub struct EmbeddingStats {
    pub total_requests: usize,
    pub cache_hits: usize,
    pub cache_misses: usize,
    pub total_tokens_processed: usize,
    pub avg_batch_size: f32,
}

impl EmbeddingEngine {
    /// Создает новый эмбеддинг движок
    pub fn new(model_path: &str, device: Device) -> Result<Self> {
        let config = EmbeddingConfig::default();
        Self::with_config(model_path, device, config)
    }

    /// Создает движок с кастомной конфигурацией
    pub fn with_config(model_path: &str, device: Device, config: EmbeddingConfig) -> Result<Self> {
        println!("🧠 Загрузка эмбеддинг модели: {}", model_path);

        // Загрузка конфигурации модели
        let config_path = std::path::Path::new(model_path).join("config.json");
        let config_content = std::fs::read_to_string(config_path)?;
        let model_config: Config = serde_json::from_str(&config_content)?;

        // Загрузка весов модели
        let weights_path = std::path::Path::new(model_path).join("model.safetensors");
        let vb =
            unsafe { VarBuilder::from_mmaped_safetensors(&[&weights_path], DType::F32, &device)? };
        let model = BertModel::load(vb, &model_config)?;

        // Загрузка токенайзера
        let tokenizer_path = std::path::Path::new(model_path).join("tokenizer.json");
        let tokenizer = Tokenizer::from_file(tokenizer_path)
            .map_err(|e| anyhow!("Failed to load tokenizer: {}", e))?;

        println!(
            "✅ Эмбеддинг движок загружен (dim: {})",
            config.embedding_dim
        );

        Ok(Self {
            model,
            tokenizer,
            device,
            config,
            cache: Arc::new(RwLock::new(HashMap::new())),
            stats: Arc::new(RwLock::new(EmbeddingStats::default())),
        })
    }

    /// Векторизует один текст
    pub fn embed(&self, text: &str) -> Result<Vec<f32>> {
        // Проверяем кэш
        if let Some(embedding) = self.get_from_cache(text) {
            self.update_stats(true, 0);
            return Ok(embedding);
        }

        // Вычисляем эмбеддинг
        let embedding = self.compute_embedding(text)?;

        // Сохраняем в кэш
        self.add_to_cache(text.to_string(), embedding.clone());
        self.update_stats(false, 1);

        Ok(embedding)
    }

    /// Векторизует батч текстов с оптимизацией GPU
    pub fn embed_batch(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(vec![]);
        }

        // Проверяем кэш для каждого текста
        let mut results: Vec<Vec<f32>> = Vec::with_capacity(texts.len());
        let mut uncached_texts: Vec<(usize, String)> = Vec::new();

        for (i, text) in texts.iter().enumerate() {
            if let Some(embedding) = self.get_from_cache(text) {
                results.push(embedding);
                self.update_stats(true, 0);
            } else {
                results.push(Vec::new()); // Заполнитель
                uncached_texts.push((i, text.clone()));
            }
        }

        // Вычисляем эмбеддинги для незакэшированных текстов батчами
        if !uncached_texts.is_empty() {
            let batch_embeddings = self.compute_batch_embeddings(
                &uncached_texts
                    .iter()
                    .map(|(_, t)| t.as_str())
                    .collect::<Vec<_>>(),
            )?;

            // Обновляем результаты и кэш
            for ((idx, text), embedding) in uncached_texts.iter().zip(batch_embeddings.iter()) {
                results[*idx] = embedding.clone();
                self.add_to_cache(text.clone(), embedding.clone());
            }

            self.update_stats(false, uncached_texts.len());
        }

        Ok(results)
    }

    /// Вычисляет эмбеддинг для одного текста
    fn compute_embedding(&self, text: &str) -> Result<Vec<f32>> {
        // Предобработка текста для e5-small
        let processed_text = format!("query: {}", text);

        // Токенизация
        let tokens = self
            .tokenizer
            .encode(processed_text.as_str(), true)
            .map_err(|e| anyhow!("Tokenization failed: {}", e))?;

        // Подготовка тензоров (2D: batch_size=1, seq_len)
        let token_ids = Tensor::new(tokens.get_ids(), &self.device)?.unsqueeze(0)?;
        let attention_mask =
            Tensor::new(tokens.get_attention_mask(), &self.device)?.unsqueeze(0)?;

        // Forward pass
        let output = self.model.forward(&token_ids, &attention_mask, None)?;

        // Mean pooling для получения эмбеддинга
        let pooled = output.mean(1)?.squeeze(0)?;
        let embedding = if self.config.normalize {
            self.l2_normalize(&pooled.to_vec1()?)?
        } else {
            pooled.to_vec1()?
        };

        Ok(embedding)
    }

    /// Вычисляет эмбеддинги для батча текстов
    fn compute_batch_embeddings(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        let mut embeddings = Vec::with_capacity(texts.len());

        // Обрабатываем батчами для оптимизации GPU
        for chunk in texts.chunks(self.config.batch_size) {
            // Предобработка для e5-small
            let processed_texts: Vec<String> =
                chunk.iter().map(|t| format!("query: {}", t)).collect();

            // Токенизация батча
            let mut all_token_ids: Vec<u32> = Vec::new();
            let mut all_attention_masks: Vec<u32> = Vec::new();

            for text in &processed_texts {
                let tokens = self
                    .tokenizer
                    .encode(text.as_str(), true)
                    .map_err(|e| anyhow!("Tokenization failed: {}", e))?;

                all_token_ids.extend(tokens.get_ids());
                all_attention_masks.extend(tokens.get_attention_mask());
            }

            // Создаем тензоры батча
            let batch_size = chunk.len();
            let seq_len = all_token_ids.len() / batch_size;

            let token_ids = Tensor::from_vec(all_token_ids, (batch_size, seq_len), &self.device)?;
            let attention_mask =
                Tensor::from_vec(all_attention_masks, (batch_size, seq_len), &self.device)?;

            // Forward pass
            let output = self.model.forward(&token_ids, &attention_mask, None)?;

            // Mean pooling для каждого элемента батча
            for i in 0..batch_size {
                let pooled = output.get(i)?.mean(0)?;
                let embedding = if self.config.normalize {
                    self.l2_normalize(&pooled.to_vec1()?)?
                } else {
                    pooled.to_vec1()?
                };
                embeddings.push(embedding);
            }
        }

        Ok(embeddings)
    }

    /// L2 нормализация вектора
    fn l2_normalize(&self, vec: &[f32]) -> Result<Vec<f32>> {
        let norm: f32 = vec.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm == 0.0 {
            return Ok(vec.to_vec());
        }
        Ok(vec.iter().map(|x| x / norm).collect())
    }

    /// Проверяет кэш и возвращает эмбеддинг если найден
    fn get_from_cache(&self, text: &str) -> Option<Vec<f32>> {
        let cache = self.cache.read();
        cache.get(text).cloned()
    }

    /// Добавляет эмбеддинг в кэш с LRU eviction
    fn add_to_cache(&self, text: String, embedding: Vec<f32>) {
        let mut cache = self.cache.write();

        // Если кэш переполнен, удаляем самые старые записи
        if cache.len() >= self.config.cache_size {
            // Простая LRU стратегия - удаляем первые 20% записей
            let remove_count = self.config.cache_size / 5;
            let keys_to_remove: Vec<String> = cache.keys().take(remove_count).cloned().collect();

            for key in keys_to_remove {
                cache.remove(&key);
            }
        }

        cache.insert(text, embedding);
    }

    /// Обновляет статистику
    fn update_stats(&self, cache_hit: bool, tokens_processed: usize) {
        let mut stats = self.stats.write();
        stats.total_requests += 1;

        if cache_hit {
            stats.cache_hits += 1;
        } else {
            stats.cache_misses += 1;
            stats.total_tokens_processed += tokens_processed;
        }
    }

    /// Вычисляет косинусное сходство между двумя векторами
    pub fn cosine_similarity(&self, a: &[f32], b: &[f32]) -> Result<f32> {
        if a.len() != b.len() {
            return Err(anyhow!(
                "Vector dimensions don't match: {} vs {}",
                a.len(),
                b.len()
            ));
        }

        let dot_product: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
        let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

        if norm_a == 0.0 || norm_b == 0.0 {
            return Ok(0.0);
        }

        Ok(dot_product / (norm_a * norm_b))
    }

    /// Возвращает статистику использования
    pub fn get_stats(&self) -> EmbeddingStats {
        self.stats.read().clone()
    }

    /// Очищает кэш
    pub fn clear_cache(&self) {
        self.cache.write().clear();
    }

    /// Возвращает размер кэша
    pub fn cache_size(&self) -> usize {
        self.cache.read().len()
    }

    /// Возвращает размерность эмбеддинга
    pub fn embedding_dim(&self) -> usize {
        self.config.embedding_dim
    }
}

impl Embedder for EmbeddingEngine {
    fn embed(&self, text: &str) -> Result<Vec<f32>> {
        self.embed(text)
    }

    fn embedding_dim(&self) -> usize {
        self.embedding_dim()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_embedding_config_default() {
        let config = EmbeddingConfig::default();
        assert_eq!(config.embedding_dim, 384);
        assert_eq!(config.max_length, 512);
        assert_eq!(config.batch_size, 32);
        assert!(config.normalize);
    }

    #[test]
    fn test_cosine_similarity() {
        let engine = EmbeddingEngine::new("dummy_path", Device::Cpu);

        let a = vec![1.0, 0.0, 0.0];
        let b = vec![0.0, 1.0, 0.0];
        let c = vec![1.0, 0.0, 0.0];

        assert_eq!(engine.cosine_similarity(&a, &b).unwrap(), 0.0);
        assert_eq!(engine.cosine_similarity(&a, &c).unwrap(), 1.0);
    }
}

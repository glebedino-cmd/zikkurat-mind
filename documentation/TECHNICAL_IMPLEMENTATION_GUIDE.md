# 🛠️ Technical Implementation Guide

## 📋 Quick Reference

### 🎯 RTX 4090 Optimized Parameters
```rust
pub const OPTIMAL_BATCH_SIZE: usize = 32;      // Embeddings
pub const GPU_CACHE_SIZE: usize = 1000;         // Vector cache
pub const RAM_CACHE_SIZE: usize = 10000;        // RAM storage
pub const MAX_VRAM_PERCENT: f32 = 0.75;       // 18GB of 24GB
pub const CHUNK_SIZE: usize = 1000;             // File chunks
```

### ⚡ Performance Targets
```rust
Embedding Speed:    > 500 texts/sec
Vector Search:      > 5,000 vectors/ms  
Generation Speed:   > 40 tokens/sec
Memory Capacity:    > 50,000 entries
VRAM Utilization:   ~75% (18GB)
Cache Hit Rate:     > 80%
```

---

## 🔧 Implementation Commands

### Build Commands
```bash
# Release build with CUDA
cargo build --release --features cuda

# Development build with debugging
cargo build --features cuda

# Run tests
cargo test --release --features cuda

# Run benchmarks
cargo bench --features cuda

# Check for memory leaks
cargo clippy --features cuda
```

### Run Commands
```bash
# Standard interaction
./target/release/ziggurat-mind.exe \
  --prompt "Расскажи о квантовой запутанности" \
  --temperature 0.7 \
  --sample-len 200

# With memory limits
./target/release/ziggurat-mind.exe \
  --prompt "Помнишь наш разговор?" \
  --max-memory-entries 100000 \
  --vram-limit 0.75 \
  --memory-dir "./data/memory"

# Memory management
./target/release/ziggurat-mind.exe --memory-stats
./target/release/ziggurat-mind.exe --clear-memory
```

---

## 🏗️ File Implementation Order

### Phase 1: Core Infrastructure
1. **`src/priests/embeddings.rs`** - Complete embedding engine
2. **`src/totems/retrieval/vector_store.rs`** - Vector storage system
3. **`src/totems/memory.rs`** - Memory manager

### Phase 2: Memory Systems  
4. **`src/totems/episodic/dialogue.rs`** - Dialogue management
5. **`src/totems/persistence.rs`** - Storage layer
6. **`src/totems/semantic/knowledge.rs`** - Knowledge base

### Phase 3: Integration
7. **`src/totems/mod.rs`** - Unified interface
8. **`src/main.rs`** - Main loop integration
9. **`tests/`** - Test suite

---

## 📊 Data Structures

### Memory Entry
```rust
#[derive(Serialize, Deserialize)]
pub struct MemoryEntry {
    pub id: Uuid,
    pub text: String,
    pub embedding: Vec<f32>,          // 384 dims
    pub metadata: HashMap<String, String>,
    pub timestamp: DateTime<Utc>,
    pub memory_type: MemoryType,
    pub on_disk: bool,               // Lazy loading
}
```

### Memory Context
```rust
pub struct MemoryContext {
    pub recent_dialogue: String,
    pub relevant_episodes: Vec<String>,
    pub relevant_knowledge: Vec<String>,
    pub confidence_score: f32,
    pub retrieval_time_ms: u64,
}
```

### Performance Monitor
```rust
pub struct PerformanceMonitor {
    pub vram_usage: Arc<RwLock<f32>>,
    pub ram_usage: Arc<RwLock<usize>>,
    pub search_latency: Arc<RwLock<f64>>,
    pub cache_hit_rate: Arc<RwLock<f32>>,
    pub total_operations: Arc<RwLock<usize>>,
}
```

---

## 🔍 Debugging & Monitoring

### Memory Pressure Detection
```rust
pub fn check_memory_pressure(&self) -> MemoryPressure {
    let vram_usage = self.performance_monitor.vram_usage.read();
    match *vram_usage {
        x if x < 0.60 => MemoryPressure::Low,
        x if x < 0.75 => MemoryPressure::Medium,
        x if x < 0.85 => MemoryPressure::High,
        _ => MemoryPressure::Critical,
    }
}
```

### Performance Logging
```rust
// Add to main loop
if memory.entry_count() % 10 == 0 {
    let stats = memory.get_performance_stats();
    println!("📊 Memory: {} entries, {:.1}% VRAM, {:.1}% cache hit", 
            stats.total_entries, stats.vram_usage, stats.cache_hit_rate);
}
```

### Debug Commands
```bash
# Show detailed memory stats
./target/release/ziggurat-mind.exe --memory-stats --verbose

# Stress test memory
./target/release/ziggurat-mind.exe --stress-test --entries 50000

# Validate storage integrity
./target/release/zikkurat-mind.exe --validate-storage

# Profile VRAM usage
nvprof --print-gpu-trace ./target/release/zikkurat-mind.exe
```

---

## 🚀 Performance Optimization Tips

### GPU Utilization
```rust
// Batch processing for embeddings
let batch_size = std::cmp::min(texts.len(), 32); // RTX 4090 optimal
let chunks: Vec<_> = texts.chunks(batch_size).collect();

// GPU-accelerated cosine similarity
let query_tensor = Tensor::new(query_embedding, &device)?;
let vectors_tensor = Tensor::from_vec(embeddings, (n, dim), &device)?;
let similarities = cosine_similarity_batch(&query_tensor, &vectors_tensor)?;
```

### Memory Management
```rust
// LRU eviction for GPU cache
if self.gpu_cache.len() > GPU_CACHE_SIZE {
    let to_remove = self.gpu_cache.len() - GPU_CACHE_SIZE + 100;
    self.gpu_cache.remove_lru(to_remove);
}

// Lazy loading for disk storage
pub fn get_embedding(&self, id: &Uuid) -> Result<Option<Vec<f32>>> {
    // Check RAM first
    if let Some(embedding) = self.ram_cache.get(id) {
        return Ok(Some(embedding.clone()));
    }
    
    // Check GPU cache second
    if let Some(tensor) = self.gpu_cache.get(id) {
        let embedding = tensor.to_vec1()?;
        self.ram_cache.insert(*id, embedding.clone());
        return Ok(Some(embedding));
    }
    
    // Load from disk last
    self.load_from_disk(id)
}
```

### Compression Strategy
```rust
// LZ4 compression for storage
pub fn save_compressed(&self, entries: &[MemoryEntry]) -> Result<()> {
    let serialized = bincode::serialize(entries)?;
    let compressed = lz4::block::compress(&serialized, None, true)?;
    
    let mut file = File::create(&self.chunk_path)?;
    file.write_all(&compressed)?;
    Ok(())
}
```

---

## 🧪 Testing Strategies

### Unit Tests
```rust
#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_embedding_dimensions() -> Result<()> {
        let engine = EmbeddingEngine::new("model_path", Device::Cpu)?;
        let embedding = engine.embed("test text")?;
        assert_eq!(embedding.len(), 384); // e5-small dimensions
        Ok(())
    }
    
    #[test]
    fn test_cosine_similarity() -> Result<()> {
        let engine = EmbeddingEngine::new("model_path", Device::Cpu)?;
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![0.0, 1.0, 0.0];
        let similarity = engine.cosine_similarity(&a, &b)?;
        assert!((similarity - 0.0).abs() < 1e-6);
        Ok(())
    }
}
```

### Integration Tests
```rust
#[test]
fn test_memory_recall_accuracy() -> Result<()> {
    let mut memory = MemorySystem::new()?;
    
    // Add test dialogues
    memory.add_exchange("What is quantum entanglement?", 
                      "Quantum entanglement is a phenomenon...")?;
    
    // Test recall
    let context = memory.recall("Tell me about entanglement")?;
    assert!(!context.relevant_episodes.is_empty());
    assert!(context.confidence_score > 0.5);
    Ok(())
}
```

### Stress Tests
```rust
#[test]
fn test_memory_stress_50k_entries() -> Result<()> {
    let mut memory = MemorySystem::new()?;
    
    // Add 50,000 entries
    for i in 0..50000 {
        memory.add_exchange(
            format!("Test query {}", i),
            format!("Test response {}", i)
        )?;
    }
    
    // Verify performance
    let stats = memory.get_performance_stats();
    assert!(stats.total_entries == 50000);
    assert!(stats.vram_usage < 0.80); // Should stay under 80%
    
    Ok(())
}
```

---

## 📈 Benchmarking

### Performance Benchmarks
```rust
// benches/memory_bench.rs
use criterion::{black_box, criterion_group, criterion_main, Criterion};

fn bench_embedding_batch(c: &mut Criterion) {
    let engine = EmbeddingEngine::new("model_path", Device::Cpu).unwrap();
    let texts: Vec<String> = (0..32).map(|i| format!("Test text {}", i)).collect();
    
    c.bench_function("embed_batch_32", |b| {
        b.iter(|| engine.embed_batch(black_box(&texts)))
    });
}

fn bench_vector_search(c: &mut Criterion) {
    let store = create_test_vector_store(50000);
    let query = create_test_embedding();
    
    c.bench_function("search_50k_vectors", |b| {
        b.iter(|| store.search(black_box(&query), 10))
    });
}

criterion_group!(benches, bench_embedding_batch, bench_vector_search);
criterion_main!(benches);
```

### Memory Profiling
```bash
# VRAM usage monitoring
watch -n 1 'nvidia-smi --query-gpu=memory.used,memory.total --format=csv'

# CPU and RAM monitoring  
htop

# Detailed profiling with Valgrind
valgrind --tool=massif ./target/release/ziggurat-mind.exe

# GPU profiling
nvprof --print-gpu-trace ./target/release/zikkurat-mind.exe
```

---

## 🔧 Troubleshooting

### Common Issues

#### VRAM Overflow
```
Error: CUDA out of memory
```
**Solution**: Reduce batch size or enable conservative mode
```bash
./target/release/ziggurat-mind.exe --vram-limit 0.5 --conservative-mode
```

#### Slow Performance
```
Warning: High latency detected
```
**Solution**: Check GPU utilization and increase cache size
```bash
./target/release/zikkurat-mind.exe --cache-size 2000 --batch-size 16
```

#### Memory Corruption
```
Error: Failed to load corrupted memory file
```
**Solution**: Clear corrupted memory and rebuild index
```bash
./target/release/zikkurat-mind.exe --clear-memory --rebuild-index
```

### Recovery Commands
```bash
# Backup current memory
cp -r data/memory data/memory_backup_$(date +%Y%m%d)

# Clear corrupted data
./target/release/zikkurat-mind.exe --clear-memory

# Validate storage integrity
./target/release/zikkurat-mind.exe --validate-storage --repair

# Restore from backup if needed
cp -r data/memory_backup_20260117 data/memory
```

---

## 📚 API Reference

### Core Methods
```rust
// Embedding engine
impl EmbeddingEngine {
    pub fn embed(&self, text: &str) -> Result<Vec<f32>>;
    pub fn embed_batch(&self, texts: &[String]) -> Result<Vec<Vec<f32>>>;
    pub fn cosine_similarity(&self, a: &[f32], b: &[f32]) -> Result<f32>;
}

// Vector store
impl VectorStore {
    pub fn add(&mut self, entry: MemoryEntry) -> Result<()>;
    pub fn search(&self, query: &[f32], top_k: usize) -> Vec<(f32, &MemoryEntry)>;
    pub fn save(&self, path: &Path) -> Result<()>;
    pub fn load(&mut self, path: &Path) -> Result<()>;
}

// Memory system
impl MemorySystem {
    pub fn recall(&self, query: &str) -> Result<MemoryContext>;
    pub fn add_exchange(&mut self, user: String, assistant: String) -> Result<()>;
    pub fn optimize(&mut self) -> Result<()>;
    pub fn get_stats(&self) -> PerformanceStats;
}
```

### Configuration Options
```rust
pub struct MemoryConfig {
    pub max_entries: usize,              // 50000
    pub vram_limit: f32,                // 0.75
    pub batch_size: usize,               // 32
    pub cache_size: usize,               // 1000
    pub compression: CompressionType,      // Lz4
    pub persistence_path: PathBuf,        // ./data/memory
}
```

---

*Last Updated: 2026-01-17*  
*Hardware: RTX 4090 24GB*  
*Optimization: Conservative 75% VRAM usage*
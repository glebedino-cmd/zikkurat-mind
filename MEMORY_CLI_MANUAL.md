# Ziggurat Mind Memory System - CLI Usage Manual

## Quick Start

```bash
# Basic memory-enabled query
ziggurat-enhanced.exe --prompt "Привет, как дела?" --enable-memory --sample-len 50
```

---

## Basic Commands

### Enable Memory System
```bash
ziggurat-enhanced.exe --prompt "Your question" --enable-memory --sample-len 100
```

### Second Query (Context Test)
```bash
ziggurat-enhanced.exe --prompt "О чем мы говорили?" --enable-memory --sample-len 50
```

---

## Persona System

### Using Different Personas
```bash
# Researcher persona
ziggurat-enhanced.exe --prompt "Я изучаю квантовую физику" --enable-memory --persona researcher --sample-len 50

# Query with researcher context
ziggurat-enhanced.exe --prompt "Расскажи что ты знаешь о квантовой физике" --enable-memory --persona researcher --sample-len 100

# Companion persona
ziggurat-enhanced.exe --prompt "Давай поговорим о технологиях" --enable-memory --persona companion --temperature 0.7 --sample-len 80
```

### Default Persona
```bash
ziggurat-enhanced.exe --prompt "Привет!" --enable-memory --persona assistant --sample-len 50
```

---

## Knowledge Extraction

### Extract Concepts from Dialogue
```bash
ziggurat-enhanced.exe --prompt "Машинное обучение - это подраздел искусственного интеллекта. Нейронные сети состоят из слоев нейронов. Глубокое обучение использует многослойные сети." --enable-memory --memory-concepts-count 10 --sample-len 30
```

### Query Extracted Concepts
```bash
ziggurat-enhanced.exe --prompt "Что такое машинное обучение?" --enable-memory --memory-concepts-count 5 --memory-episodes-count 0 --sample-len 50
```

### Search All Knowledge
```bash
ziggurat-enhanced.exe --prompt "Найди информацию о машинном обучении и нейронных сетях" --enable-memory --memory-episodes-count 3 --memory-concepts-count 5 --sample-len 100
```

---

## Memory Parameters

### Episode Retrieval (Past Dialogues)
```bash
# Retrieve 3 similar past dialogues
ziggurat-enhanced.exe --prompt "Вспомни наш разговор" --enable-memory --memory-episodes-count 3 --sample-len 80

# Disable episode retrieval
ziggurat-enhanced.exe --prompt "Новый вопрос" --enable-memory --memory-episodes-count 0 --sample-len 50
```

### Concept Retrieval (Knowledge Base)
```bash
# Retrieve 5 concepts from knowledge base
ziggurat-enhanced.exe --prompt "Расскажи о физике" --enable-memory --memory-concepts-count 5 --sample-len 80

# Retrieve 10 concepts
ziggurat-enhanced.exe --prompt "Проанализируй тему" --enable-memory --memory-concepts-count 10 --sample-len 100
```

### Combined Retrieval
```bash
ziggurat-enhanced.exe --prompt "Используй всю доступную информацию" --enable-memory --memory-episodes-count 5 --memory-concepts-count 5 --sample-len 150
```

---

## Memory Statistics

### View Memory Stats
```bash
ziggurat-enhanced.exe --memory-stats
```

### Cleanup and Stats
```bash
# Remove memories older than 7 days
ziggurat-enhanced.exe --cleanup-days 7 --memory-stats

# Remove ALL old memories (0 days)
ziggurat-enhanced.exe --cleanup-days 0 --memory-stats
```

---

## Data Persistence

### Export Memory to File
```bash
ziggurat-enhanced.exe --export-memory my_backup.json
```

### Import Memory from File
```bash
ziggurat-enhanced.exe --import-memory my_backup.json --enable-memory
```

### Export After Session
```bash
ziggurat-enhanced.exe --prompt "Заверши сессию" --enable-memory --export-memory session_backup.json --sample-len 30
```

---

## Configuration Options

### Custom Memory Data Directory
```bash
ziggurat-enhanced.exe --prompt "Test" --enable-memory --memory-data-dir ./my_memory_data --sample-len 50
```

### Persistence Format
```bash
# JSON format (human-readable)
ziggurat-enhanced.exe --prompt "Test" --enable-memory --persistence-format json --sample-len 50

# Binary format (faster, compact)
ziggurat-enhanced.exe --prompt "Test" --enable-memory --persistence-format binary --sample-len 50

# Hybrid format (default)
ziggurat-enhanced.exe --prompt "Test" --enable-memory --persistence-format hybrid --sample-len 50
```

### Auto-Save Interval
```bash
# Auto-save every 5 operations
ziggurat-enhanced.exe --prompt "Test" --enable-memory --auto-save-interval 5 --sample-len 50
```

---

## Embedding Models

### Using Real Embedding Model
```bash
# Download model to models/embeddings/
# Model should contain: config.json, model.safetensors, tokenizer.json

ziggurat-enhanced.exe --prompt "Test memory with real embeddings" --enable-memory --embedding-model models/embeddings --sample-len 50
```

### Fallback to Dummy Embeddings
If embedding models are not found, the system automatically uses dummy embeddings:
```bash
ziggurat-enhanced.exe --prompt "Test with dummy embeddings" --enable-memory --sample-len 50
```

Output when using dummy embeddings:
```
⚠️  Embedding models not found, using dummy embeddings
   Memory system will work with limited functionality
   For full functionality, download models to: models/embeddings
```

---

## GPU/CPU Selection

### Force CPU
```bash
ziggurat-enhanced.exe --prompt "Test" --enable-memory --cpu --sample-len 50
```

### Auto-Select GPU (Default)
```bash
ziggurat-enhanced.exe --prompt "Test" --enable-memory --sample-len 50
```

---

## Temperature and Sampling

### Low Temperature (Deterministic)
```bash
ziggurat-enhanced.exe --prompt "Calculate: 5 * 7" --enable-memory --temperature 0.0 --sample-len 30
```

### High Temperature (Creative)
```bash
ziggurat-enhanced.exe --prompt "Напиши историю" --enable-memory --temperature 0.8 --sample-len 100
```

---

## Complete Examples

### Research Session
```bash
# Session 1: Provide knowledge
ziggurat-enhanced.exe --prompt "Квантовая запутанность - это явление, при котором квантовые состояния двух частиц становятся связанными. Эйнштейн называл это 'жутким действием на расстоянии'." --enable-memory --persona researcher --memory-concepts-count 5 --sample-len 30

# Session 2: Query knowledge
ziggurat-enhanced.exe --prompt "Расскажи о квантовой запутанности" --enable-memory --persona researcher --memory-episodes-count 2 --memory-concepts-count 3 --sample-len 100

# Session 3: Check memory
ziggurat-enhanced.exe --prompt "О чем мы говорили?" --enable-memory --persona researcher --memory-episodes-count 3 --sample-len 80
```

### Export and Restore
```bash
# Export
ziggurat-enhanced.exe --export-memory research_backup.json

# Restore in new session
ziggurat-enhanced.exe --import-memory research_backup.json --enable-memory --persona researcher
```

---

## Troubleshooting

### Memory Disabled?
```bash
# Check if embedding model path exists
dir models\embeddings

# Use dummy embeddings (auto-fallback)
ziggurat-enhanced.exe --prompt "Test" --enable-memory --sample-len 50
```

### Check Memory Stats
```bash
ziggurat-enhanced.exe --memory-stats
```

### View Debug Information
```bash
ziggurat-enhanced.exe --prompt "Test" --enable-memory --tracing --sample-len 50
# Creates trace-timestamp.json for debugging
```

---

## All CLI Arguments Reference

| Argument | Description | Default |
|----------|-------------|---------|
| `--enable-memory` | Enable unified memory system | false |
| `--memory-episodes-count` | Number of episodes to retrieve | 3 |
| `--memory-concepts-count` | Number of concepts to retrieve | 2 |
| `--persona` | Persona name for session | "assistant" |
| `--embedding-model` | Path to embedding model | "models/embeddings" |
| `--memory-data-dir` | Memory data directory | "data" |
| `--persistence-format` | Format: json/binary/hybrid | "hybrid" |
| `--auto-save-interval` | Operations between auto-saves | 10 |
| `--memory-stats` | Show memory statistics and exit | - |
| `--export-memory` | Export memory and exit | - |
| `--import-memory <file>` | Import memory from file | - |
| `--cleanup-days <n>` | Cleanup memories older than n days | 30 |
| `--cpu` | Run on CPU instead of GPU | false |
| `--temperature` | Sampling temperature | 0.0 |
| `--top-p` | Nucleus sampling probability | - |
| `--top-k` | Top-K sampling | - |
| `--repeat-penalty` | Token repetition penalty | 1.1 |
| `--sample-len` | Length of sample to generate | 10000 |
| `--tracing` | Enable tracing (debug) | false |

# 🏛️ ZIGGURAT MIND - MVP Implementation Plan

## 🎯 Философия реализации

> "Память - это не кэш, а фундамент сознания. Личность рождается из воспоминаний."

Текущий `mistral-pure` - это **Уровень 4 (Логос)** в зачаточном состоянии. Мы обернём его слоями памяти и сознания, создав **истинную долговременную память**.

---

## 🧠 Архитектура памяти: Двойственная природа

### Философия разделения памяти

```
┌─────────────────────────────────────────────────────────┐
│          ТОТЕМЫ ПАМЯТИ (totems/)                        │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  📜 ЭПИЗОДИЧЕСКАЯ ПАМЯТЬ (episodic/)                    │
│  ├─ "Что происходило?"                                  │
│  ├─ Диалоги, события, контекст                         │
│  ├─ Временная привязка (когда?)                        │
│  └─ Эмоциональная окраска (как?)                       │
│                                                          │
│  📚 СЕМАНТИЧЕСКАЯ ПАМЯТЬ (semantic/)                    │
│  ├─ "Что я знаю?"                                       │
│  ├─ Факты, концепции, определения                      │
│  ├─ Абстрактные знания                                 │
│  └─ Вневременная мудрость                              │
│                                                          │
│  🔍 ВЕКТОРНЫЙ ПОИСК (retrieval/)                        │
│  ├─ Семантическое сходство                             │
│  ├─ Контекстуальная релевантность                     │
│  └─ Ранжирование по важности                           │
└─────────────────────────────────────────────────────────┘
```

### Пример: Как это работает в жизни

**Пользователь спрашивает:** "Расскажи о квантовой запутанности"

**Система извлекает:**
- **Эпизодическая память:** "3 месяца назад мы обсуждали эксперимент Алена Аспе"
- **Семантическая память:** "Квантовая запутанность - корреляция состояний частиц..."
- **Формирует ответ:** Объединяет знание + контекст предыдущего разговора

---

## 📐 Архитектурная трансформация

### Целевая структура MVP (расширенная)
```
ziggurat/
├── src/
│   ├── main.rs                    ← Оркестратор сознания
│   │
│   ├── initiation/                ← 🜂 Уровень 0: Инициация
│   │   ├── mod.rs
│   │   ├── archetypes.rs          ← Загрузка личностей
│   │   └── config.rs              ← Системные конфиги
│   │
│   ├── priests/                   ← 🜁 Уровень 1: Жрецы Железа
│   │   ├── mod.rs
│   │   ├── device.rs              ← Абстракция GPU/CPU
│   │   ├── resources.rs           ← Управление памятью
│   │   └── embeddings.rs          ← Генерация векторов (новое!)
│   │
│   ├── totems/                    ← 🜃 Уровень 2: Тотемы Памяти (КЛЮЧЕВОЙ МОДУЛЬ)
│   │   ├── mod.rs
│   │   │
│   │   ├── episodic/              ← ЭПИЗОДИЧЕСКАЯ ПАМЯТЬ
│   │   │   ├── mod.rs
│   │   │   ├── session.rs         ← Текущая сессия
│   │   │   ├── dialogue.rs        ← История диалогов
│   │   │   └── events.rs          ← Значимые события
│   │   │
│   │   ├── semantic/              ← СЕМАНТИЧЕСКАЯ ПАМЯТЬ
│   │   │   ├── mod.rs
│   │   │   ├── knowledge.rs       ← База знаний
│   │   │   ├── concepts.rs        ← Концепты и определения
│   │   │   └── beliefs.rs         ← Убеждения и принципы
│   │   │
│   │   ├── retrieval/             ← ВЕКТОРНЫЙ ПОИСК
│   │   │   ├── mod.rs
│   │   │   ├── vector_store.rs    ← In-memory векторная БД
│   │   │   ├── embedder.rs        ← Интерфейс для эмбеддингов
│   │   │   └── ranker.rs          ← Ранжирование результатов
│   │   │
│   │   ├── context_window.rs      ← Скользящее окно (кратковременная память)
│   │   └── persistence.rs         ← Сохранение на диск
│   │
│   ├── demiurge/                  ← 🜄 Уровень 3: Демиург Личности
│   │   ├── mod.rs
│   │   ├── persona.rs             ← Ядро личности
│   │   ├── narrative.rs           ← Эволюционирующий нарратив
│   │   └── directives.rs          ← Ограничения/правила
│   │
│   ├── logos/                     ← 🜂 Уровень 4: Логос
│   │   ├── mod.rs
│   │   ├── inference.rs           ← Обёртка Candle
│   │   ├── tokenizer.rs           ← TokenOutputStream
│   │   └── sampling.rs            ← Параметры генерации
│   │
│   └── spirit/                    ← 🜃 Уровень 5: Дух (будущее)
│       └── mod.rs                 ← Заглушка для автономии
│
├── config/
│   ├── archetypes/
│   │   ├── scholar.toml           ← Пример: учёный
│   │   └── companion.toml         ← Пример: компаньон
│   └── system.toml                ← Общие настройки
│
├── data/                          ← ДОЛГОВРЕМЕННАЯ ПАМЯТЬ (игнорируется git)
│   ├── episodic/
│   │   ├── sessions/              ← JSON файлы сессий
│   │   └── embeddings.bin         ← Векторы диалогов
│   └── semantic/
│       ├── knowledge.json         ← Структурированные знания
│       └── embeddings.bin         ← Векторы знаний
│
└── Cargo.toml
```

---

## 🚀 Пошаговая реализация

### **ФАЗА 0: Добавление эмбеддинг-модели** (2-3 часа)

#### Выбор модели для векторизации

Используем **компактную модель эмбеддингов** (не Mistral!):
- **Вариант 1:** `sentence-transformers/all-MiniLM-L6-v2` (80MB, 384 dims)
- **Вариант 2:** `BAAI/bge-small-en-v1.5` (130MB, 384 dims, лучше качество)
- **Вариант 3 (для русского):** `intfloat/multilingual-e5-small` (118MB, 384 dims)

```rust
// src/priests/embeddings.rs
use candle_core::{Tensor, Device};
use candle_nn::VarBuilder;
use candle_transformers::models::bert::{BertModel, Config};

pub struct EmbeddingEngine {
    model: BertModel,
    tokenizer: Tokenizer,
    device: Device,
}

impl EmbeddingEngine {
    pub fn new(model_path: &str, device: Device) -> Result<Self> {
        // Загрузка легковесной BERT-подобной модели
        let config = Config::default();
        let vb = VarBuilder::from_safetensors(...);
        let model = BertModel::load(vb, &config)?;
        
        Ok(Self { model, tokenizer, device })
    }
    
    pub fn embed(&self, text: &str) -> Result<Vec<f32>> {
        let tokens = self.tokenizer.encode(text, true)?;
        let input = Tensor::new(tokens.get_ids(), &self.device)?;
        let output = self.model.forward(&input)?;
        
        // Mean pooling
        let embedding = output.mean(1)?;
        Ok(embedding.to_vec1()?)
    }
    
    pub fn embed_batch(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        texts.iter().map(|t| self.embed(t)).collect()
    }
}
```

---

### **ФАЗА 1: Векторное хранилище (in-memory)** (3-4 часа)

#### Простая in-memory векторная БД

```rust
// src/totems/retrieval/vector_store.rs
use std::collections::HashMap;

#[derive(Clone, Serialize, Deserialize)]
pub struct MemoryEntry {
    pub id: Uuid,
    pub text: String,
    pub embedding: Vec<f32>,
    pub metadata: HashMap<String, String>,
    pub timestamp: DateTime<Utc>,
    pub memory_type: MemoryType,
}

#[derive(Clone, Serialize, Deserialize)]
pub enum MemoryType {
    Episodic { session_id: Uuid, turn: usize },
    Semantic { category: String },
}

pub struct VectorStore {
    entries: Vec<MemoryEntry>,
    dim: usize,
}

impl VectorStore {
    pub fn new(dim: usize) -> Self {
        Self {
            entries: Vec::new(),
            dim,
        }
    }
    
    pub fn add(&mut self, entry: MemoryEntry) {
        assert_eq!(entry.embedding.len(), self.dim);
        self.entries.push(entry);
    }
    
    pub fn search(&self, query_embedding: &[f32], top_k: usize) -> Vec<(f32, &MemoryEntry)> {
        let mut scores: Vec<_> = self.entries
            .iter()
            .map(|entry| {
                let score = cosine_similarity(query_embedding, &entry.embedding);
                (score, entry)
            })
            .collect();
        
        scores.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
        scores.truncate(top_k);
        scores
    }
    
    pub fn search_by_type(&self, query_embedding: &[f32], memory_type: MemoryType, top_k: usize) 
        -> Vec<(f32, &MemoryEntry)> 
    {
        let filtered: Vec<_> = self.entries
            .iter()
            .filter(|e| std::mem::discriminant(&e.memory_type) == std::mem::discriminant(&memory_type))
            .collect();
        
        let mut scores: Vec<_> = filtered
            .iter()
            .map(|entry| {
                let score = cosine_similarity(query_embedding, &entry.embedding);
                (score, *entry)
            })
            .collect();
        
        scores.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
        scores.truncate(top_k);
        scores
    }
}

fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    dot / (norm_a * norm_b)
}
```

#### Персистентность (сохранение на диск)

```rust
// src/totems/persistence.rs
use bincode;
use std::fs::File;
use std::io::{BufReader, BufWriter};

pub fn save_vector_store(store: &VectorStore, path: &Path) -> Result<()> {
    let file = File::create(path)?;
    let writer = BufWriter::new(file);
    bincode::serialize_into(writer, store)?;
    Ok(())
}

pub fn load_vector_store(path: &Path) -> Result<VectorStore> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);
    let store = bincode::deserialize_from(reader)?;
    Ok(store)
}
```

---

### **ФАЗА 2: Эпизодическая память** (4-5 часов)

#### Структура диалога с автоматической векторизацией

```rust
// src/totems/episodic/dialogue.rs

pub struct DialogueManager {
    current_session: Session,
    vector_store: VectorStore,
    embedder: Arc<EmbeddingEngine>,
}

impl DialogueManager {
    pub fn add_exchange(&mut self, user: String, assistant: String) -> Result<()> {
        let turn = self.current_session.turns.len();
        
        // Сохраняем сырой диалог
        self.current_session.add_turn(user.clone(), assistant.clone());
        
        // Векторизуем контекст (user + assistant как один блок)
        let context = format!("User: {}\nAssistant: {}", user, assistant);
        let embedding = self.embedder.embed(&context)?;
        
        let entry = MemoryEntry {
            id: Uuid::new_v4(),
            text: context,
            embedding,
            metadata: HashMap::from([
                ("session_id".into(), self.current_session.id.to_string()),
                ("turn".into(), turn.to_string()),
            ]),
            timestamp: Utc::now(),
            memory_type: MemoryType::Episodic { 
                session_id: self.current_session.id, 
                turn 
            },
        };
        
        self.vector_store.add(entry);
        Ok(())
    }
    
    pub fn recall_similar_dialogues(&self, query: &str, top_k: usize) -> Result<Vec<String>> {
        let query_embedding = self.embedder.embed(query)?;
        let results = self.vector_store.search_by_type(
            &query_embedding,
            MemoryType::Episodic { session_id: Uuid::nil(), turn: 0 },
            top_k
        );
        
        Ok(results.iter().map(|(score, entry)| {
            format!("[Score: {:.2}] {}", score, entry.text)
        }).collect())
    }
}

#[derive(Serialize, Deserialize)]
pub struct Session {
    pub id: Uuid,
    pub persona_name: String,
    pub turns: Vec<Turn>,
    pub created_at: DateTime<Utc>,
}

#[derive(Serialize, Deserialize)]
pub struct Turn {
    pub user: String,
    pub assistant: String,
    pub timestamp: DateTime<Utc>,
}
```

---

### **ФАЗА 3: Семантическая память** (3-4 часа)

#### База знаний с категоризацией

```rust
// src/totems/semantic/knowledge.rs

pub struct KnowledgeBase {
    vector_store: VectorStore,
    embedder: Arc<EmbeddingEngine>,
    concepts: HashMap<String, Concept>,
}

#[derive(Serialize, Deserialize)]
pub struct Concept {
    pub name: String,
    pub definition: String,
    pub category: String,
    pub related_concepts: Vec<String>,
    pub source: String, // "learned_from_dialogue" | "pre_defined"
}

impl KnowledgeBase {
    pub fn add_knowledge(&mut self, concept: Concept) -> Result<()> {
        let text = format!(
            "{}: {} (Category: {})",
            concept.name, concept.definition, concept.category
        );
        
        let embedding = self.embedder.embed(&text)?;
        
        let entry = MemoryEntry {
            id: Uuid::new_v4(),
            text: text.clone(),
            embedding,
            metadata: HashMap::from([
                ("concept".into(), concept.name.clone()),
                ("category".into(), concept.category.clone()),
            ]),
            timestamp: Utc::now(),
            memory_type: MemoryType::Semantic { 
                category: concept.category.clone() 
            },
        };
        
        self.vector_store.add(entry);
        self.concepts.insert(concept.name.clone(), concept);
        Ok(())
    }
    
    pub fn query_knowledge(&self, question: &str, top_k: usize) -> Result<Vec<String>> {
        let query_embedding = self.embedder.embed(question)?;
        let results = self.vector_store.search_by_type(
            &query_embedding,
            MemoryType::Semantic { category: String::new() },
            top_k
        );
        
        Ok(results.iter().map(|(score, entry)| {
            format!("[Relevance: {:.2}] {}", score, entry.text)
        }).collect())
    }
    
    /// Автоматическое извлечение знаний из диалога (упрощённая эвристика)
    pub fn extract_from_dialogue(&mut self, dialogue: &str) -> Result<()> {
        // Простая эвристика: ищем паттерны вида "X - это Y"
        // В будущем: использовать LLM для извлечения
        
        for line in dialogue.lines() {
            if let Some((concept, definition)) = self.parse_definition(line) {
                self.add_knowledge(Concept {
                    name: concept,
                    definition,
                    category: "learned".into(),
                    related_concepts: vec![],
                    source: "learned_from_dialogue".into(),
                })?;
            }
        }
        Ok(())
    }
    
    fn parse_definition(&self, text: &str) -> Option<(String, String)> {
        // "Квантовая запутанность - это корреляция..."
        let parts: Vec<&str> = text.split(" - это ").collect();
        if parts.len() == 2 {
            Some((parts[0].trim().into(), parts[1].trim().into()))
        } else {
            None
        }
    }
}
```

---

### **ФАЗА 4: Интеграция памяти в поток генерации** (2-3 часа)

#### Унифицированный менеджер памяти

```rust
// src/totems/mod.rs

pub struct MemorySystem {
    pub episodic: DialogueManager,
    pub semantic: KnowledgeBase,
    pub context_window: ContextWindow,
}

impl MemorySystem {
    pub fn recall(&self, query: &str) -> Result<MemoryContext> {
        // 1. Извлечь релевантные эпизоды
        let episodes = self.episodic.recall_similar_dialogues(query, 3)?;
        
        // 2. Извлечь релевантные знания
        let knowledge = self.semantic.query_knowledge(query, 3)?;
        
        // 3. Текущий контекст (последние N сообщений)
        let recent = self.context_window.get_context();
        
        Ok(MemoryContext {
            recent_dialogue: recent,
            relevant_episodes: episodes,
            relevant_knowledge: knowledge,
        })
    }
}

pub struct MemoryContext {
    pub recent_dialogue: String,
    pub relevant_episodes: Vec<String>,
    pub relevant_knowledge: Vec<String>,
}

impl MemoryContext {
    pub fn format_for_prompt(&self) -> String {
        format!(
            "=== Текущий диалог ===\n{}\n\n\
             === Похожие разговоры из прошлого ===\n{}\n\n\
             === Релевантные знания ===\n{}",
            self.recent_dialogue,
            self.relevant_episodes.join("\n"),
            self.relevant_knowledge.join("\n")
        )
    }
}
```

#### Обновлённый main loop

```rust
// src/main.rs (финальная версия)

fn main() -> Result<()> {
    // Инициализация
    let archetype = initiation::load_archetype("scholar")?;
    let device = priests::select_device(args.cpu)?;
    let embedder = Arc::new(priests::EmbeddingEngine::new("models/embeddings", device.clone())?);
    
    // Память
    let mut memory = totems::MemorySystem {
        episodic: totems::DialogueManager::new(embedder.clone()),
        semantic: totems::KnowledgeBase::new(embedder.clone()),
        context_window: totems::ContextWindow::new(2000),
    };
    
    // Загрузка предыдущей памяти
    memory.episodic.load_from_disk("data/episodic")?;
    memory.semantic.load_from_disk("data/semantic")?;
    
    // Личность и движок
    let persona = demiurge::Persona::from_archetype(archetype);
    let mut engine = logos::InferenceEngine::new(model, device);
    
    println!("🏛️ ZIGGURAT MIND активирован. Личность: {}", persona.name);
    
    loop {
        let user_input = read_user_input()?;
        if user_input == "/exit" { break; }
        
        // 1. Вспомнить релевантный контекст
        let memory_context = memory.recall(&user_input)?;
        
        // 2. Сформировать промпт с памятью
        let prompt = persona.format_prompt_with_memory(
            &user_input,
            &memory_context
        );
        
        // 3. Генерация
        let response = engine.generate(&prompt)?;
        println!("🤖 {}", response);
        
        // 4. Сохранить обмен в памяти
        memory.episodic.add_exchange(user_input.clone(), response.clone())?;
        memory.context_window.add_message(Message::user(user_input));
        memory.context_window.add_message(Message::assistant(response));
        
        // 5. Извлечь новые знания
        memory.semantic.extract_from_dialogue(&response)?;
        
        // 6. Периодическое сохранение
        if memory.episodic.current_session.turns.len() % 10 == 0 {
            memory.save_to_disk("data")?;
        }
    }
    
    // Финальное сохранение
    memory.save_to_disk("data")?;
    Ok(())
}
```

---

## 🎯 MVP Критерии готовности

### ✅ Минимальный работающий продукт включает:

1. **Двойственная память**
   - Эпизодическая (диалоги с контекстом)
   - Семантическая (извлечённые знания)

2. **Векторный поиск**
   - Cosine similarity search
   - Фильтрация по типу памяти
   - Ранжирование по релевантности

3. **Персистентность**
   - Сохранение векторов на диск
   - Загрузка памяти между сессиями
   - Инкрементальное обновление

4. **Интеграция в генерацию**
   - Автоматический recall при каждом запросе
   - Форматирование контекста для LLM
   - Автоматическое извлечение знаний

---

## 🛠️ Технические решения

### Новые зависимости
```toml
[dependencies]
# Существующие из mistral-pure...
bincode = "1.3"  # Сериализация векторов
uuid = { version = "1.0", features = ["v4", "serde"] }
chrono = { version = "0.4", features = ["serde"] }

# Для эмбеддингов (опционально - можно использовать candle)
# sentence-transformers через candle-transformers
```

### Оптимизации для производительности

```rust
// Ленивая загрузка эмбеддинг-модели
pub struct LazyEmbedder {
    model: OnceCell<EmbeddingEngine>,
}

// Батчинг для векторизации
impl VectorStore {
    pub fn add_batch(&mut self, entries: Vec<MemoryEntry>) {
        self.entries.extend(entries);
    }
}

// FAISS-подобная индексация (будущее улучшение)
pub struct HNSWIndex {
    // Приближённый поиск ближайших соседей
}
```

---

## 📊 Метрики памяти

### Мониторинг роста памяти
```rust
impl MemorySystem {
    pub fn stats(&self) -> MemoryStats {
        MemoryStats {
            episodic_count: self.episodic.vector_store.entries.len(),
            semantic_count: self.semantic.vector_store.entries.len(),
            total_size_mb: self.estimate_size_mb(),
            oldest_memory: self.get_oldest_timestamp(),
        }
    }
}
```

### Пример вывода
```
🧠 Память:
   Эпизоды: 1,247 воспоминаний
   Знания: 389 концептов
   Размер: 45.2 MB
   Старейшая память: 2024-12-15 14:23:11
```

---

## 🔥 Следующий шаг

Начинаем с **Фазы 0**: интеграция легковесной эмбеддинг-модели. Готов создать:

1. `src/priests/embeddings.rs` - движок векторизации
2. `src/totems/retrieval/vector_store.rs` - in-memory БД
3. Тестовый скрипт для проверки поиска

Погнали?
# Persona × Memory Integration

## 📋 Описание

Реализация переноса контекста сессии между сессиями для сохранения "памяти" Persona о предыдущих разговорах.

## 🏗️ Архитектура

### Уровень 1: Структуры данных (`src/demiurge/narrative.rs`)

```rust
/// Контекст сессии для передачи в новую сессию
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PersonaSessionContext {
    pub version: String,
    pub archetype_id: String,
    pub previous_session_id: String,
    pub last_interaction_date: u64,
    pub summary: String,                    // Краткое содержание разговора
    pub key_topics: Vec<String>,            // Ключевые темы
    pub user_preferences: Vec<Preference>,  // Упомянутые предпочтения
    pub emotional_state: f32,               // Эмоциональное состояние (0.0-1.0)
    pub last_topic: String,                 // О чём был последний разговор
    pub pending_questions: Vec<String>,     // Вопросы, которые остались открытыми
    pub custom_data: HashMap<String, String>, // Кастомные данные
}

/// Предпочтение пользователя
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Preference {
    pub topic: String,
    pub statement: String,
    pub confidence: f32,
    pub mentioned_at: u64,
}

/// Storage для контекстов
pub struct ContextStorage;

impl ContextStorage {
    pub fn save(context: &PersonaSessionContext) -> std::io::Result<()>;
    pub fn load(archetype_id: &str) -> Result<Option<PersonaSessionContext>>;
    pub fn exists(archetype_id: &str) -> bool;
    pub fn delete(archetype_id: &str) -> std::io::Result<()>;
    pub fn is_expired(archetype_id: &str, max_days: i64) -> bool;
}
```

### Уровень 2: LLM-анализ (`src/totems/episodic/mod.rs`)

```rust
pub trait LlmPipeline: Send + Sync {
    fn generate(&self, prompt: &str, max_tokens: usize) -> Result<String>;
}

struct ContextAnalyzer<'a> {
    pipeline: &'a dyn LlmPipeline,
}

impl ContextAnalyzer {
    /// Генерирует краткое содержание диалога (2-3 предложения)
    fn summarize_session(&self, turns: &[Turn]) -> Result<String>;

    /// Извлекает ключевые темы (JSON массив, макс 5 тем)
    fn extract_topics(&self, turns: &[Turn]) -> Result<Vec<String>>;

    /// Определяет эмоциональное состояние (0.0 - негативное, 1.0 - позитивное)
    fn analyze_emotions(&self, turns: &[Turn]) -> Result<f32>;

    /// Извлекает последнюю тему из последнего вопроса
    fn extract_last_topic(&self, turns: &[Turn]) -> Result<String>;
}

pub struct SessionAnalysis {
    pub summary: String,
    pub key_topics: Vec<String>,
    pub emotional_state: f32,
    pub last_topic: String,
    pub turn_count: usize,
}
```

#### Примеры LLM-промптов:

**Суммаризация:**
```xml
<s>[INST] Ты — ассистент по анализу диалогов. Кратко опиши, о чём был разговор (2-3 предложения на русском).

Диалог:
Turn 1:
User: Мне нравится Lamborghini
Assistant: Круто! 🏎️ Запомнила!

Краткое содержание:[/INST]
```

**Извлечение тем:**
```xml
<s>[INST] Извлеки ключевые темы из диалога. Верни только JSON массив строк, например: ["тема1", "тема2", "тема3"].
Не более 5 тем. Темы должны быть короткими (1-2 слова), на русском языке.

Диалог:
User: Мне нравится Lamborghini
Assistant: Круто! 🏎️ Запомнила!

Темы:[/INST]
```

**Анализ эмоций:**
```xml
<s>[INST] Определи эмоциональное состояние пользователя по диалогу.
Верни только число от 0.0 (негативное/грустное) до 1.0 (позитивное/радостное).

Диалог:
User: Мне нравится Lamborghini
Assistant: Круто! 🏎️ Запомнила!

Число:[/INST]
```

### Уровень 3: Методы Persona (`src/demiurge/persona.rs`)

```rust
impl Persona {
    /// Загружает контекст предыдущей сессии
    pub fn load_session_context(&mut self) -> Result<Option<PersonaSessionContext>>;

    /// Сохраняет контекст текущей сессии
    pub fn save_session_context<D: LlmPipeline>(
        &self,
        dialogue_manager: &DialogueManager,
        pipeline: &D,
    ) -> Result<Option<PersonaSessionContext>>;

    /// Генерирует персонализированное приветствие на основе контекста
    pub fn generate_contextual_greeting(&self, context: &PersonaSessionContext) -> String;

    /// Проверяет наличие сохранённого контекста
    pub fn has_saved_context(&self) -> bool;
}
```

#### Примеры персонализированных приветствий:

**Girlfriend (эмпатичная):**
```
Привет! 💫✨ Ты рада меня видеть? 🐣💕 Помню, что мы говорили о машинах. 
Как там твои мечты о Lamborghini? Расскажи, думаешь ещё о Huracan?
```

**Programmer (технический):**
```
Привет. Контекст восстановлен. Последняя тема: машины. 
Есть незавершённые вопросы. Готов продолжить?
```

**Philosopher (сократический):**
```
Здравствуй. Интересно, что привело тебя снова сюда? 
Я помню, что мы говорили о стремлении к красоте и скорости... 
Что нового в твоих размышлениях?
```

## 📁 Файловая структура

```
memory_data/
├── sessions.json          # Эпизодическая память (существующий)
├── embeddings.bin         # Эмбеддинги (существующий)
├── metadata.json          # Метаданные (существующий)
└── context/               # Контексты сессий (новый)
    ├── girlfriend.json
    ├── programmer.json
    ├── devops.json
    ├── scientist.json
    └── philosopher.json
```

## 📄 Формат файла контекста

```json
{
  "version": "1.0",
  "archetype_id": "girlfriend",
  "previous_session_id": "550e8400-e29b-41d4-a716-446655440000",
  "last_interaction_date": 1706784000,
  "summary": "Пользователь рассказал о своей любви к машинам Lamborghini, особенно о модели Huracan. Был в хорошем настроении.",
  "key_topics": ["машины", "Lamborghini", "Huracan"],
  "user_preferences": [
    {
      "topic": "машины",
      "statement": "Нравится Lamborghini",
      "confidence": 0.95,
      "mentioned_at": 1706784000
    }
  ],
  "emotional_state": 0.8,
  "last_topic": "машины",
  "pending_questions": ["Какую модель выберешь?", "Сколько планируешь накопить?"],
  "custom_data": {}
}
```

## 🔧 Интеграция с main_unified.rs

### При запуске (interactive mode):
```rust
if args.interactive {
    match ArchetypeLoader::load(&args.archetype) {
        Ok(archetype) => {
            let mut p = Persona::from_archetype(Arc::new(archetype));

            // Загружаем контекст
            if let Some(context) = p.load_session_context()? {
                println!("💭 Found saved session context!");
                let greeting = p.generate_contextual_greeting(&context);
                println!("\n🤖 {}:", p.name);
                println!("{}", greeting);
            }

            persona = Some(p);
        }
    }
}
```

### При Ctrl+C (через ctrlc crate):
```rust
let _ = ctrlc::set_handler(move || {
    println!("\n\n💾 Saving context before exit...");

    if let Some(ref p) = persona_for_save {
        if let Some(ref dm) = dm_for_save {
            let context_analyzer = ContextAnalyzerImpl::new(pipeline_for_context.clone());
            if let Ok(Some(_)) = p.save_session_context(dm, &context_analyzer) {
                println!("💾 Session context saved");
            }
        }
    }

    std::process::exit(0);
});
```

### При quit:
```rust
if input.eq_ignore_ascii_case("quit") || input.eq_ignore_ascii_case("exit") {
    // Сохраняем контекст
    if let Some(ref p) = persona {
        if let Some(ref dm) = dialogue_manager {
            let context_analyzer = ContextAnalyzerImpl::new(pipeline_arc.clone());
            if let Ok(Some(context)) = p.save_session_context(dm, &context_analyzer) {
                println!("💾 Context saved for next session");
            }
        }
    }
    // ... сохраняем память и выходим
}
```

## 🎮 Команда /context

Показывает текущий сохранённый контекст:

```
/context

💭 Session Context:
   Version: 1.0
   Summary: Пользователь рассказал о машинах...
   Topics: машины, Lamborghini, Huracan
   Emotional state: 0.8
   Last topic: машины

   💡 This context will be restored in the next session.
```

## ⚙️ Конфигурация

```rust
pub const MAX_CONTEXT_AGE_DAYS: i64 = 30;  // Контекст старше 30 дней игнорируется
pub const MIN_TURNS_FOR_SAVE: usize = 3;   // Минимум 3 обмена для сохранения
```

## ✅ Критерии приёмки

- [x] Persona приветствует пользователя с учётом предыдущего разговора
- [x] Предпочтения пользователя восстанавливаются между сессиями
- [x] Контекст сохраняется автоматически при выходе (quit и Ctrl+C)
- [x] Контекст восстанавливается автоматически при старте
- [x] Работает для всех архетипов (girlfriend, programmer, etc.)
- [x] Устаревший контекст (>30 дней) не используется
- [x] Команда `/context` показывает текущий контекст

## 📊 Статистика

| Метрика | Значение |
|---------|----------|
| Время реализации | ~4 часа |
| Структур данных | 3 |
| Новых методов | 5 |
| Интегрированных файлов | 4 |
| Строк кода | ~400 |

## 🔗 Связанные задачи

- [x] Persona × Memory Integration - выполнено
- [ ] LLM-based Summarization - опционально
- [ ] Multi-user Context - будущая задача
- [ ] Context Compression - будущая задача

---

*Создано: 2026-01-21*
*Автор: ZIGGURAT MIND Development Team*

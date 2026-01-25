# ZIGGURAT MIND

> **AI с гибридной памятью, Knowledge Graph и цифровой персоной**

Система построена на принципе многоуровневого зиккурата, где каждый уровень добавляет новую способность: инфраструктура -> память -> граф знаний -> персона -> рассуждение.

## Текущий Статус (v0.2.0)

| Компонент | Статус | Описание |
|-----------|--------|----------|
| Mistral 7B Inference | Работает | GPU/CPU, BF16/F32 |
| Embedding Engine | Работает | e5-small, 384-dim |
| Episodic Memory | Работает | Диалоги, сессии, векторный поиск |
| Semantic Memory | Работает | Концепты, категории, дедупликация |
| Knowledge Graph | Работает | Триплеты, связи между концептами |
| Temporal Decay | Работает | Затухание старых концептов |
| Persona System | Работает | 5 архетипов, эволюция черт |
| Session Context | Работает | Сохранение контекста между сессиями |

## Архитектура

```
+-------------------------------------------------------------------------+
|                        ZIGGURAT MIND                                    |
+-------------------------------------------------------------------------+
|  Уровень 3: DEMIURGE (Персона)                                          |
|  +-------------------------------------------------------------------+  |
|  |  Archetype System: girlfriend, programmer, devops, scientist,     |  |
|  |                    philosopher                                    |  |
|  |  Session Context: предпочтения, история, эмоции                   |  |
|  |  Narrative: история отношений, milestones                         |  |
|  |  Evolution: развитие черт персоны                                 |  |
|  +-------------------------------------------------------------------+  |
+-------------------------------------------------------------------------+
|  Уровень 2: TOTEMS (Гибридная Память + Knowledge Graph)                 |
|  +-------------------------------------------------------------------+  |
|  |  +-------------------+    +-----------------------------------+   |  |
|  |  |   ЭПИЗОДИЧЕСКАЯ   |    |        СЕМАНТИЧЕСКАЯ              |   |  |
|  |  |   (Episodic)      |    |        (Semantic)                 |   |  |
|  |  |                   |    |                                   |   |  |
|  |  |  * Диалоги        |    |  * Факты (facts)                  |   |  |
|  |  |  * Сессии         |    |  * Предпочтения (preferences)     |   |  |
|  |  |  * Контекст       |    |  * Правила (rules)                |   |  |
|  |  |  * Векторный      |    |  * Навыки (skills)                |   |  |
|  |  |    поиск          |    |  * Цели (goals)                   |   |  |
|  |  |                   |    |  * Temporal Decay                 |   |  |
|  |  |  Vector Store     |    |  * Knowledge Graph (триплеты)     |   |  |
|  |  +-------------------+    +-----------------------------------+   |  |
|  +-------------------------------------------------------------------+  |
+-------------------------------------------------------------------------+
|  Уровень 1: PRIESTS (Инфраструктура)                                    |
|  Embedding Engine (e5-small, 384-dim) + CUDA/CPU + Persistence          |
+-------------------------------------------------------------------------+
|  Уровень 0: LOGOS (Генерация)                                           |
|  Mistral 7B Inference + Tokenization + Sampling                         |
+-------------------------------------------------------------------------+
```

## Быстрый Старт

### Сборка

```bash
# GPU (CUDA) - рекомендуется
cargo build --release --features cuda --bin ziggurat-unified

# CPU-only
cargo build --release --bin ziggurat-unified
```

### Запуск

```bash
# Полный режим с гибридной памятью и персоной
cargo run --features cuda --release -- \
    --interactive \
    --archetype girlfriend \
    --enable-memory \
    --enable-semantic

# Минимальный режим (только LLM)
cargo run --features cuda --release -- \
    --interactive

# Одиночный запрос
cargo run --features cuda --release -- \
    --prompt "Привет! Как дела?" \
    --enable-semantic
```

## Гибридная Система Памяти

### Эпизодическая Память (Episodic)

Хранит полную историю диалогов с векторным поиском по схожести.

**Расположение:** `memory_data/episodic/`
- `sessions.json` - история диалогов
- `embeddings.bin` - векторные представления

**Активация:** `--enable-memory`

### Семантическая Память (Semantic)

Извлекает и хранит структурированные знания о пользователе.

**Расположение:** `memory_data/semantic/`
- `semantic_memory.json` - концепты с категориями
- `knowledge_graph.json` - граф знаний (триплеты)

**Категории концептов:**

| Категория | Описание | Пример |
|-----------|----------|--------|
| `facts` | Факты о пользователе | "Работает программистом" |
| `preferences` | Предпочтения | "Любит пиццу" |
| `rules` | Правила/ограничения | "Не любит ранние подъемы" |
| `skills` | Навыки | "Знает Python" |
| `goals` | Цели и мечты | "Хочу выучить Rust" |
| `general` | Общие знания | "Интересуется AI" |

**Активация:** `--enable-semantic`

### Knowledge Graph

Граф знаний хранит связи между концептами в виде триплетов (субъект, предикат, объект).

**Возможности:**
- Автоматическое извлечение отношений из текста
- Поиск связанных концептов
- Temporal decay для старых связей

**CLI команды:**
```bash
# Показать статистику графа
cargo run --features cuda -- --enable-semantic --graph-stats

# Извлечь отношения из текста
cargo run --features cuda -- --enable-semantic --extract-relations

# Найти связанные концепты
cargo run --features cuda -- --enable-semantic --find-related "pizza"
```

### Temporal Decay

Система временного затухания для концептов:
- Старые концепты теряют confidence со временем
- Настраиваемые периоды затухания по категориям
- Автоматическое применение decay раз в день

```bash
# Применить decay вручную
cargo run --features cuda -- --enable-semantic --apply-decay

# Показать статистику decay
cargo run --features cuda -- --enable-semantic --decay-stats
```

## Персонная Система (Demiurge)

### Архетипы

| Архетип | Имя | Описание |
|---------|-----|----------|
| `girlfriend` | Алиса | Заботливая подруга-разработчица |
| `programmer` | Код | Технический ассистент |
| `devops` | DevOps | Специалист по инфраструктуре |
| `scientist` | Эйнштейн | Научный подход |
| `philosopher` | Сократ | Философские вопросы |

Архетипы определены в `config/archetypes/*.json`

### Эволюция Персоны

Персона развивается через взаимодействия:
- `interactions_count` - количество взаимодействий
- `successful_helps` - успешные помощи
- `relationship_score` - уровень отношений
- `trait_offsets` - модификации черт
- `unlocked_traits` - разблокированные черты

### Session Context

Контекст сессии сохраняется между запусками:
- Краткое содержание последней беседы
- Ключевые темы
- Эмоциональное состояние
- Незавершенные вопросы

## Команды

### CLI параметры

| Параметр | Описание | По умолчанию |
|----------|----------|--------------|
| `--prompt TEXT` | Запрос для обработки | - |
| `--interactive` | Интерактивный режим | false |
| `--archetype NAME` | Архетип персоны | "programmer" |
| `--enable-memory` | Эпизодическая память | false |
| `--enable-semantic` | Семантическая память | false |
| `--memory-top-k N` | Похожих диалогов | 5 |
| `--semantic-top-k N` | Концептов | 10 |
| `--quiet` / `-q` | Тихий режим | false |
| `--verbose` / `-v` | Подробный вывод | false |
| `--cpu` | CPU вместо GPU | false |
| `--temperature` | Температура генерации | 0.7 |
| `--top-p` | Nucleus sampling | - |
| `--top-k` | Top-K sampling | - |
| `--seed` | Seed для генерации | 299792458 |
| `--sample-len` / `-n` | Макс. токенов | 2048 |
| `--apply-decay` | Применить temporal decay | false |
| `--decay-stats` | Показать статистику decay | false |
| `--graph-stats` | Показать статистику графа | false |
| `--extract-relations` | Извлечь отношения | false |
| `--find-related TEXT` | Найти связанные концепты | - |

### Интерактивные команды

В интерактивном режиме доступны команды:

```
quit / выход / пока    # Выход с сохранением
/persona show          # Показать текущую персону
/persona traits        # Показать черты персоны
/persona evolution     # Показать эволюцию
/persona switch NAME   # Сменить архетип
/persona list          # Список архетипов
/context               # Показать контекст сессии
/mem                   # Показать использование памяти
/semantic              # Справка по семантической памяти
```

## Структура Файлов

```
zikkurat-mind/
+-- config/
|   +-- archetypes/           # Определения архетипов
|       +-- girlfriend.json
|       +-- programmer.json
|       +-- devops.json
|       +-- scientist.json
|       +-- philosopher.json
+-- memory_data/
|   +-- context/              # Контекст сессии
|   |   +-- {archetype}_context.json
|   +-- episodic/             # Эпизодическая память
|   |   +-- sessions.json
|   |   +-- embeddings.bin
|   +-- semantic/             # Семантическая память
|       +-- semantic_memory.json
|       +-- knowledge_graph.json
+-- models/
|   +-- embeddings/           # E5-small модель
|   +-- mistral-7b-instruct/  # Mistral 7B
+-- src/
    +-- logos/                # Генерация (Mistral 7B)
    +-- priests/              # Инфраструктура (embeddings, device)
    +-- totems/               # Память
    |   +-- episodic/         # Диалоговая память
    |   +-- semantic/         # Семантическая память + KG
    |   +-- retrieval/        # Vector store
    +-- demiurge/             # Персона
    |   +-- archetype.rs      # Загрузка архетипов
    |   +-- persona.rs        # Персона
    |   +-- evolution.rs      # Эволюция черт
    |   +-- narrative.rs      # История отношений
    |   +-- context.rs        # Session context
    +-- main_unified.rs       # Точка входа
```

## Ключевые Компоненты

| Компонент | Путь | Описание |
|-----------|------|----------|
| **Priests** | `src/priests/` | Embedding engine, device management |
| **Totems** | `src/totems/` | Memory systems (episodic, semantic, vector) |
| **Demiurge** | `src/demiurge/` | Persona system, archetypes, evolution |
| **Logos** | `src/logos/` | Mistral 7B inference |

### Ключевые файлы

- `src/main_unified.rs` - Единая точка входа
- `src/priests/embeddings.rs` - Embedding engine (e5-small)
- `src/totems/semantic/manager.rs` - Семантическая память
- `src/totems/semantic/concept.rs` - Knowledge Graph + Decay
- `src/totems/episodic/mod.rs` - Dialogue memory manager
- `src/demiurge/persona.rs` - Persona system
- `src/demiurge/archetype.rs` - Archetype loader

## Анти-Галлюцинационные Меры

1. **Извлечение только из реплик пользователя** - модель не может выдумать факты
2. **Детекция саморяскрытий** - только явные утверждения ("я люблю", "мой")
3. **Дедупликация** - при similarity > 0.92 пропускается дубликат
4. **Обнаружение противоречий** - при конфликте обновляется существующий факт
5. **Regex fallback** - если LLM возвращает невалидный JSON

## Модели

- **Embedding**: `intfloat/multilingual-e5-small` (384-мерные векторы)
- **LLM**: `mistralai/Mistral-7B-Instruct-v0.2`

Расположение: `models/embeddings/` и `models/mistral-7b-instruct/`

## Требования

- NVIDIA GPU с CUDA 11+ (рекомендуется, RTX 4090 идеально)
- Rust 1.70+
- CMake (для candle)
- ~8GB VRAM (GPU) или ~18GB RAM (CPU)

---

## TODO LIST

> **Принцип**: Минимальный жизнеспособный продукт с последующей эволюцией. Каждая задача должна быть конкретной и измеримой.

### КРИТИЧЕСКИЙ ПРИОРИТЕТ

- [ ] **Добавить тесты для Knowledge Graph**
  - Unit tests для add_triple, find_by_subject, find_by_object
  - Integration tests для extract_relations_from_text
  - Файлы: `src/totems/semantic/concept.rs`, `tests/knowledge_graph.rs`

- [ ] **Оптимизировать использование памяти GPU**
  - Сейчас: KV-кэш растет неограниченно
  - Нужно: ограничить размер кэша, очищать после N токенов
  - Файл: `src/main_unified.rs:UnifiedPipeline`

### ВЫСОКИЙ ПРИОРИТЕТ

- [ ] **Улучшить извлечение отношений для Knowledge Graph**
  - Сейчас: простые паттерны (loves, knows, works_at)
  - Нужно: больше предикатов, лучшая точность
  - Файл: `src/totems/semantic/manager.rs:extract_relations_from_text`

- [ ] **Добавить поддержку многопользовательских сессий**
  - Сейчас: один пользователь
  - Нужно: user_id в контексте, раздельная память
  - Файлы: `src/totems/episodic/`, `src/demiurge/context.rs`

- [ ] **Улучшить детекцию противоречий**
  - Сейчас: только "люблю" vs "не люблю"
  - Нужно: расширить на "нравится", "предпочитаю", "ненавижу"
  - Файл: `src/totems/semantic/manager.rs:is_contradiction`

- [ ] **Добавить нормализацию текста перед similarity**
  - Сейчас: "I love pizza" != "i love pizza"
  - Нужно: lowercase + стемминг
  - Файл: `src/totems/semantic/manager.rs:add_concept`

### СРЕДНИЙ ПРИОРИТЕТ

- [ ] **Добавить Web UI**
  - Простой веб-интерфейс для взаимодействия
  - Технологии: Axum + HTMX или простой REST API

- [ ] **Добавить голосование по концептам**
  - Пользователь может подтвердить/отклонить концепт
  - Влияет на confidence

- [ ] **Добавить автоматическую очистку старых концептов**
  - Концепты с confidence < 0.3 и usage_count = 0
  - Период: 30 дней

- [ ] **Улучшить промпт экстракции для английского**
  - Сейчас: фокус на русском
  - Нужно: поддержка "I prefer", "my favorite is"

- [ ] **Добавить команду /semantic merge**
  - Ручное объединение похожих концептов
  - Файл: `src/main_unified.rs`

- [ ] **Реализовать streaming output**
  - Сейчас: вывод после полной генерации
  - Нужно: токен за токеном

### НИЗКИЙ ПРИОРИТЕТ

- [ ] **Добавить тесты для semantic memory**
  - Unit tests для deduplication
  - Integration tests для extraction

- [ ] **Добавить поддержку других LLM**
  - Llama 2, Qwen, и др.
  - Файл: `src/logos/`

- [ ] **Улучшить DEBUG вывод**
  - Краткий режим: только результаты
  - Подробный: весь процесс

- [ ] **Добавить поддержку других языков**
  - English, Deutsch, etc.
  - Файл: `src/main_unified.rs:ConceptExtractorImpl::extract`

- [ ] **Оптимизировать производительность**
  - Кэширование эмбеддингов
  - Batch extraction

- [ ] **Добавить экспорт/импорт памяти**
  - JSON export для бэкапа
  - Импорт из других систем

### ИССЛЕДОВАНИЕ

- [ ] **Изучить RAG интеграцию**
  - Retrieval-Augmented Generation для внешних документов
  - Интеграция с базами знаний

- [ ] **Изучить fine-tuning персоны**
  - LoRA/QLoRA для адаптации под конкретного пользователя
  - Требует данных взаимодействий

### ВЫПОЛНЕНО

- [x] Исправить структуру main_unified.rs после интеграции Knowledge Graph
- [x] Добавить метод save() в SemanticMemoryManager
- [x] Добавить метод get_concept() для доступа к концептам
- [x] Исправить handle_persona_command для корректных полей EvolutionState
- [x] Добавить клон semantic_manager в ctrlc handler

---

## Метрики для Измерения

| Метрика | Целевое значение | Текущее |
|---------|-----------------|---------|
| Memory precision | > 95% | ~90% |
| Extraction accuracy | > 90% | ~85% |
| Deduplication rate | > 80% | ~75% |
| Contradiction detection | > 85% | ~70% |
| Memory persistence | 100% | 100% |
| KG relation extraction | > 80% | ~60% |

---

**ZIGGURAT MIND - Building AI with Memory and Consciousness**

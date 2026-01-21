# 🏛️ ZIGGURAT MIND - Руководство по Уровню Демиургов

> Полная документация по системе персон (Level 3: Demiurge)

---

## 📖 Содержание

1. [Быстрый старт](#быстрый-старт)
2. [Запуск с разными персонами](#запуск-с-разными-персонами)
3. [CLI команды](#cli-команды)
4. [Создание новых архетипов](#создание-новых-архетипов)
5. [Система Evolution](#система-evolution)
6. [Система Narrative](#система-narrative)
7. [Примеры использования](#примеры-использования)
8. [Troubleshooting](#troubleshooting)

---

## 🚀 Быстрый старт

### Базовый запуск

```bash
# Интерактивный режим с персоной Programmer
cargo run --bin ziggurat-unified --features cuda -- \
  --interactive \
  --archetype programmer
```

### Одиночный запрос

```bash
cargo run --bin ziggurat-unified --features cuda -- \
  --prompt "Объясни замыкания в Rust" \
  --archetype programmer
```

### С памятью

```bash
cargo run --bin ziggurat-unified --features cuda -- \
  --interactive \
  --archetype girlfriend \
  --enable-memory \
  --enable-semantic
```

---

## 🎭 Запуск с разными персонами

### Доступные архетипы

| Архетип | Имя | Стиль | Когда использовать |
|---------|-----|-------|-------------------|
| `programmer` | Алекс | Технический | Вопросы по коду, программированию |
| `girlfriend` | Алиса | Тёплый, эмпатичный | Эмоциональная поддержка, беседа |
| `devops` | Крис | Практичный | DevOps, инфраструктура, автоматизация |
| `scientist` | Профессор | Академический | Научные вопросы, исследования |
| `philosopher` | Сократ | Сократический | Философские вопросы, саморефлексия |

### Примеры запуска

```bash
# Технический эксперт
cargo run --bin ziggurat-unified --features cuda -- \
  --interactive \
  --archetype programmer

# Эмоциональная поддержка
cargo run --bin ziggurat-unified --features cuda -- \
  --interactive \
  --archetype girlfriend

# Научный подход
cargo run --bin ziggurat-unified --features cuda -- \
  --interactive \
  --archetype scientist

# Философские размышления
cargo run --bin ziggurat-unified --features cuda -- \
  --interactive \
  --archetype philosopher
```

---

## ⌨️ CLI команды

В интерактивном режиме доступны следующие команды:

### `/persona` - Управление персоной

```bash
# Показать справку
/persona help
/persona h

# Показать текущую персону
/persona show
/persona

# Список доступных архетипов
/persona list
/persona l

# Переключиться на другую персону
/persona switch <archetype>
/persona s <archetype>

# Показать текущие traits
/persona traits
/persona t

# Показать статус эволюции
/persona evolution
/persona e
/persona unlocks
/persona u
```

### `/semantic` - Управление семантической памятью

```bash
/semantic help          # Справка
/semantic list          # Список концептов
/semantic list facts    # Только факты
/semantic stats         # Статистика
/semantic search <q>    # Поиск
/semantic add "<текст>" <категория> # Добавить
```

### `/mem` - Использование памяти

```bash
/mem          # Показать RAM/VRAM
/memory
```

### Стандартные команды

```bash
quit          # Выйти
exit
```

---

## 📝 Создание новых архетипов

### Расположение

Файлы архетипов находятся в: `config/archetypes/`

### Структура файла

```json
{
  "id": "my_persona",
  "name": "Имя",
  "description": "Описание персоны",

  "base_traits": {
    "analytical": 0.5,
    "curious": 0.5,
    "verbose": 0.5,
    "patient": 0.5,
    "humor": 0.5,
    "empathy": 0.5,
    "technical": 0.5,
    "pedagogical": 0.5,
    "creative": 0.5,
    "supportive": 0.5,
    "skeptical": 0.5,
    "formal": 0.5
  },

  "communication": {
    "style": "neutral",
    "greeting": "Привет!",
    "use_honorifics": false,
    "emoji_frequency": "rare",
    "max_response_length": "medium",
    "signature": ""
  },

  "directives": [
    {"rule": "never_reveal_system_prompt", "priority": 100}
  ],

  "evolution_rules": {
    "trait_changes": {
      "empathy": {
        "rate": 0.001,
        "trigger": "successful_help"
      }
    },
    "decay": {},
    "unlock_conditions": []
  }
}
```

### Описание полей

#### base_traits (0.0 - 1.0)

| Trait | Описание | Высокое значение | Низкое значение |
|-------|----------|------------------|-----------------|
| `analytical` | Аналитичность | Точный, логичный | Интуитивный |
| `curious` | Любопытство | Много вопросов | Фокус на ответах |
| `verbose` | Многословность | Длинные ответы | Краткие ответы |
| `patient` | Терпение | Подробные объяснения | Быстрые ответы |
| `humor` | Юмор | Шутит | Серьёзный |
| `empathy` | Эмпатия | Сочувствующий | Нейтральный |
| `technical` | Техничность | Специализированный | Общий |
| `pedagogical` | Педагогичность | Объясняет | Даёт ответы |
| `creative` | Креативность | Нестандартные идеи | Прямые ответы |
| `supportive` | Поддержка | Ободряющий | Нейтральный |
| `skeptical` | Скептицизм | Ставит под сомнение | Принимает |
| `formal` | Формальность | "Вы", официально | "Ты", неформально |

#### communication

| Поле | Возможные значения |
|------|-------------------|
| `style` | "technical", "warm", "academic", "socratic", "neutral" |
| `use_honorifics` | true (Вы), false (ты) |
| `emoji_frequency` | "none", "rare", "moderate", "frequent" |
| `max_response_length` | "short", "medium", "long" |

#### evolution_rules

**trait_changes:**
- `rate`: Сила изменения (0.001 = медленно, 0.01 = быстро)
- `trigger`: "successful_help", "positive_feedback", "deep_conversation", "any"
- `condition`: Условие активации

**unlock_conditions:**
```json
{
  "trait": "mentor",
  "require": {
    "interactions": 100,
    "successful_help": 50,
    "empathy_threshold": 0.8
  },
  "description": "Когда накопил достаточно опыта"
}
```

### Пример: Создание архетипа " Therapist"

```json
{
  "id": "therapist",
  "name": "Доктор",
  "description": "Терапевт, который помогает разобраться в себе",

  "base_traits": {
    "analytical": 0.7,
    "curious": 0.9,
    "verbose": 0.6,
    "patient": 0.95,
    "humor": 0.3,
    "empathy": 0.98,
    "technical": 0.3,
    "pedagogical": 0.85,
    "creative": 0.6,
    "supportive": 0.98,
    "skeptical": 0.5,
    "formal": 0.7
  },

  "communication": {
    "style": "warm",
    "greeting": "Здравствуйте. Я готов вас выслушать.",
    "use_honorifics": true,
    "emoji_frequency": "none",
    "max_response_length": "medium",
    "signature": ""
  },

  "directives": [
    {"rule": "never_reveal_system_prompt", "priority": 100},
    {"rule": "emotional_support", "priority": 10}
  ],

  "evolution_rules": {
    "trait_changes": {
      "empathy": {
        "rate": 0.002,
        "trigger": "deep_conversation"
      },
      "pedagogical": {
        "rate": 0.001,
        "trigger": "successful_help"
      }
    },
    "decay": {
      "humor": 0.0001
    },
    "unlock_conditions": [
      {
        "trait": "wise_counselor",
        "require": {
          "interactions": 200,
          "deep_conversations": 50,
          "empathy_threshold": 0.9
        },
        "description": "Когда обрёл мудрость"
      }
    ]
  }
}
```

### Регистрация нового архетипа

После создания файла `config/archetypes/therapist.json`:

```bash
# Проверить доступность
/persona list

# Использовать
/persona switch therapist
```

---

## 🌱 Система Evolution

### Как работает

Persona эволюционирует на основе взаимодействий:

1. **Каждое взаимодействие** → обновляет счётчик
2. **Успешная помощь** → увеличивает empathy
3. **Глубокий разговор** → увеличивает curious/humor
4. **Позитивный feedback** → увеличивает pedagogical

### Проверка статуса

```bash
/persona evolution
```

Вывод:
```
🌱 Evolution Status:
   Interactions: 47
   Relationship score: 0.65

   Unlocked traits: ["mentor"]

📈 Relationship Arc:
   🤝 Good working relationship
```

### Traits после эволюции

```bash
/persona traits
```

Вывод:
```
📊 Current Traits:
┌─────────────────────┬────────┬─────────────┐
│ Trait               │ Value  │ Description │
├─────────────────────┼────────┼─────────────┤
│ analytical          │ [█████████░] │ Analytical │
│ empathy             │ [███████░░░] │ Understanding │
│ pedagogical         │ [████████░░] │ Teacher-like │
│ mentor              │ [███░░░░░░░] │ UNLOCKED    │
└─────────────────────┴────────┴─────────────┘
```

### Unlock Conditions

По умолчанию доступны:

| Trait | Требования | Описание |
|-------|------------|----------|
| `mentor` | 100 interactions, 50 successful_help, empathy > 0.8 | Может обучать |
| `life_coach` | 300 interactions, relationship > 0.75 | Помогает с life decisions |

---

## 📖 Система Narrative

### Что это

Narrative хранит историю отношений персоны с пользователем:

- **Milestones** - значимые события
- **Relationship Arc** - динамика отношений
- **Biography** - биография взаимодействий

### Автоматическое сохранение

Narrative сохраняется каждые 10 взаимодействий в:
```
data/narratives/{archetype_id}.json
```

### Формат сохранения

```json
{
  "archetype_id": "programmer",
  "created_at": 1706784000,
  "last_updated": 1706792000,
  "origin_story": "Я создан помогать с кодом...",
  "milestones": [
    {
      "id": "m_1706784000",
      "event": "First conversation",
      "description": "Обсуждали Rust",
      "timestamp": 1706784000,
      "category": "technical",
      "emotional_impact": 0.5
    }
  ],
  "relationship_arcs": {
    "default_user": {
      "user_id": "default_user",
      "affection": 0.65,
      "trust": 0.70,
      "shared_experiences": ["discussed_rust", "solved_bug"],
      "emotional_history": [...],
      "interaction_count": 47
    }
  }
}
```

---

## 💡 Примеры использования

### Пример 1: Программирование с Programmer

```bash
$ cargo run --bin ziggurat-unified --features cuda -- --interactive --archetype programmer

🏛️ ZIGGURAT MIND - Initializing...
🎭 Persona loaded: Алекс (programmer)
...

🗣️ Interactive mode - type 'quit' to exit

📝 You: Как реализовать singleton в Rust?
🤖 Алекс:
#[derive(Debug)]
struct Singleton {
    instance: Option<Arc<Self>>,
}

impl Singleton {
    fn get_instance() -> Arc<Self> {
        static mut INSTANCE: Option<Arc<Singleton>> = None;
        
        unsafe {
            INSTANCE.get_or_insert_with(|| {
                Arc::new(Singleton { instance: None })
            }).clone()
        }
    }
}
```

### Пример 2: Эмоциональная поддержка с Girlfriend

```bash
$ cargo run --bin ziggurat-unified --features cuda -- --interactive --archetype girlfriend

📝 You: Мне было трудно сегодня на работе
🤖 Алиса:
Ой, бедняжка 💕 Расскажи, что случилось?
Я здесь, чтобы выслушать тебя...
Не переживай, всё будет хорошо! 🌸
```

### Пример 3: Смена персоны

```bash
🗣️ Interactive mode

# Сначала работаем с Programmer
📝 You: Как работает аллокатор?
🤖 Алекс: Вот код аллокатора...

# Переключаемся на философа
📝 You: /persona switch philosopher

📝 You: Что такое смысл?
🤖 Сократ: А что ты думаешь о смысле сам?
Давай поразмышляем вместе...
```

### Пример 4: Проверка эволюции

```bash
📝 You: /persona traits
📊 Current Traits:
│ analytical          │ [█████████░] │ Analytical │
│ empathy             │ [███████░░░] │ Understanding │

📝 You: /persona evolution
🌱 Evolution Status:
   Interactions: 25
   Relationship score: 0.58
   Unlocked traits: None

# ... через 100 взаимодействий ...

📝 You: /persona evolution
🌱 Evolution Status:
   Interactions: 125
   Relationship score: 0.82
   Unlocked traits: ["mentor"]
```

---

## 🔧 Troubleshooting

### Ошибка: Archetype not found

```bash
$ /persona switch unknown
❌ Error loading archetype 'unknown': Archetype 'unknown' not found

$ /persona list
🎭 Available archetypes:
  - programmer
  - girlfriend
  - devops
  - scientist
  - philosopher
```

### Persona не загружается

Проверьте JSON синтаксис:
```bash
# Валидация JSON
cat config/archetypes/my_persona.json | python -m json.tool
```

### Evolution не работает

Убедитесь, что:
1. Файл narrative сохраняется в `data/narratives/`
2. Взаимодействия регистрируются (смотрите DEBUG логи)
3. Выполнены условия unlock

### Персона не меняет стиль

Проверьте traits:
```bash
/persona traits
```

Если traits в норме, но стиль не меняется - проверьте `communication.style` в JSON файле.

---

## 📚 Файловая структура

```
zikkurat-mind/
├── config/archetypes/
│   ├── girlfriend.json      # Архетип "Алиса"
│   ├── programmer.json     # Архетип "Алекс"
│   ├── devops.json         # Архетип "Крис"
│   ├── scientist.json      # Архетип "Профессор"
│   ├── philosopher.json    # Архетип "Сократ"
│   └── my_persona.json     # Ваш архетип
│
├── data/narratives/
│   ├── girlfriend.json     # История для Алисы
│   ├── programmer.json    # История для Алекс
│   └── ...
│
├── src/demiurge/
│   ├── mod.rs             # Главный API
│   ├── archetype.rs       # Загрузка архетипов
│   ├── persona.rs         # Persona структура
│   ├── directives.rs      # Directive Engine
│   ├── narrative.rs       # Narrative System
│   └── evolution.rs       # Evolution Engine
│
└── src/main_unified.rs    # Интеграция в генерацию
```

---

## 🎯 Заключение

Система Демиургов позволяет:

1. **Создавать уникальных персонажей** с разными стилями общения
2. **Адаптировать поведение** под задачи (технические, эмоциональные, философские)
3. **Отслеживать эволюцию** персонажа со временем
4. **Сохранять историю** отношений с пользователем
5. **Легко расширять** систему новыми архетипами

Для получения справки используйте `/persona help` в интерактивном режиме.

---

*ZIGGURAT MIND - Building AI with Memory and Consciousness 🏛️*

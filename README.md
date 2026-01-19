# 🏛️ ZIGGURAT MIND

> **AI с долговременной памятью и сознанием**

## Архитектура

```
User Input
    ↓
Embedding Engine (CUDA) → Vectorize query
    ↓
Vector Store → Find similar dialogues
    ↓
Dialogue Manager → Get context
    ↓
Mistral 7B (CUDA) → Generate response with context
    ↓
Memory → Save dialogue
    ↓
Response
```

## Компоненты

| Компонент | Описание |
|-----------|----------|
| **Priests** (Embeddings) | e5-small модель для векторизации |
| **Totems** (Memory) | Эпизодическая + семантическая память |
| **Logos** (Reasoning) | Mistral 7B инференс |

## Быстрый старт

```bash
# Сборка с CUDA
cargo build --features cuda

# Запуск
cargo run --bin ziggurat-unified --features cuda -- \
  --prompt "Привет!" \
  --enable-memory \
  --interactive
```

## Ключевые файлы

- `src/main_unified.rs` - Единая точка входа
- `src/priests/embeddings.rs` - Эмбеддинг движок
- `src/totems/memory.rs` - Менеджер памяти
- `src/logos/` - Mistral 7B логика

## Документация

- [Core Philosophy](documentation/CORE_PHILOSOPHY.md) - Философия проекта
- [Current Status](CURRENT_STATUS.md) - Текущее состояние

---

**ZIGGURAT MIND - Building AI with Memory and Consciousness 🏛️**

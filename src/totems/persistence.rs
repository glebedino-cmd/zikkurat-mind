//! 💾 Персистентность памяти
//!
//! Сохранение и загрузка памяти между запусками
//! Поддерживает JSON и бинарные форматы

#![allow(dead_code)]

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::fs;
use std::io::BufReader;
use std::path::{Path, PathBuf};

use crate::totems::{
    episodic::Session, memory::MemoryExport, retrieval::VectorStore, semantic::Concept,
};

/// Менеджер персистентности памяти
pub struct PersistenceManager {
    /// Базовый каталог для данных
    base_path: PathBuf,
    /// Формат персистентности
    format: PersistenceFormat,
    /// Автоматическое сохранение каждые N операций
    auto_save_interval: usize,
    /// Счетчик операций с последнего сохранения
    operation_count: usize,
}

/// Формат персистентности
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PersistenceFormat {
    /// JSON формат (человеко-читаемый)
    Json,
    /// Бинарный формат (быстрый, компактный)
    Binary,
    /// Гибридный (JSON метаданные + бинарные данные)
    Hybrid,
}

impl PersistenceManager {
    /// Создает новый менеджер персистентности
    pub fn new<P: AsRef<Path>>(base_path: P) -> Self {
        Self {
            base_path: base_path.as_ref().to_path_buf(),
            format: PersistenceFormat::Hybrid,
            auto_save_interval: 10, // Автосохранение каждые 10 операций
            operation_count: 0,
        }
    }

    /// Создает с кастомными настройками
    pub fn with_config<P: AsRef<Path>>(
        base_path: P,
        format: PersistenceFormat,
        auto_save_interval: usize,
    ) -> Self {
        Self {
            base_path: base_path.as_ref().to_path_buf(),
            format,
            auto_save_interval,
            operation_count: 0,
        }
    }

    /// Инициализирует файловую структуру
    pub fn initialize(&self) -> Result<()> {
        // Создаем базовый каталог
        fs::create_dir_all(&self.base_path)?;

        // Создаем подкаталоги
        let episodic_dir = self.base_path.join("episodic");
        let semantic_dir = self.base_path.join("semantic");
        let vector_dir = self.base_path.join("vectors");

        fs::create_dir_all(&episodic_dir)?;
        fs::create_dir_all(&semantic_dir)?;
        fs::create_dir_all(&vector_dir)?;

        // Создаем .gitignore для исключения из git
        let gitignore_path = self.base_path.join(".gitignore");
        if !gitignore_path.exists() {
            let gitignore_content = "# Ziggurat Mind Memory Data\n*\n!.gitignore\n";
            fs::write(&gitignore_path, gitignore_content)?;
        }

        Ok(())
    }

    /// Сохраняет сессии
    pub fn save_sessions(
        &self,
        sessions: &std::collections::HashMap<uuid::Uuid, Session>,
    ) -> Result<()> {
        let sessions_dir = self.base_path.join("episodic");
        fs::create_dir_all(&sessions_dir)?;

        match self.format {
            PersistenceFormat::Json => self.save_sessions_json(sessions, &sessions_dir),
            PersistenceFormat::Binary => self.save_sessions_binary(sessions, &sessions_dir),
            PersistenceFormat::Hybrid => self.save_sessions_hybrid(sessions, &sessions_dir),
        }
    }

    /// Загружает сессии
    pub fn load_sessions(&self) -> Result<std::collections::HashMap<uuid::Uuid, Session>> {
        let sessions_dir = self.base_path.join("episodic");
        if !sessions_dir.exists() {
            return Ok(std::collections::HashMap::new());
        }

        match self.format {
            PersistenceFormat::Json => self.load_sessions_json(&sessions_dir),
            PersistenceFormat::Binary => self.load_sessions_binary(&sessions_dir),
            PersistenceFormat::Hybrid => self.load_sessions_hybrid(&sessions_dir),
        }
    }

    /// Сохраняет концепты
    pub fn save_concepts(&self, concepts: &[Concept]) -> Result<()> {
        let concepts_dir = self.base_path.join("semantic");
        fs::create_dir_all(&concepts_dir)?;

        match self.format {
            PersistenceFormat::Json => self.save_concepts_json(concepts, &concepts_dir),
            PersistenceFormat::Binary => self.save_concepts_binary(concepts, &concepts_dir),
            PersistenceFormat::Hybrid => self.save_concepts_hybrid(concepts, &concepts_dir),
        }
    }

    /// Загружает концепты
    pub fn load_concepts(&self) -> Result<Vec<Concept>> {
        let concepts_dir = self.base_path.join("semantic");
        if !concepts_dir.exists() {
            return Ok(Vec::new());
        }

        match self.format {
            PersistenceFormat::Json => self.load_concepts_json(&concepts_dir),
            PersistenceFormat::Binary => self.load_concepts_binary(&concepts_dir),
            PersistenceFormat::Hybrid => self.load_concepts_hybrid(&concepts_dir),
        }
    }

    /// Сохраняет векторное хранилище
    pub fn save_vector_store(&self, vector_store: &VectorStore) -> Result<()> {
        let vectors_dir = self.base_path.join("vectors");
        fs::create_dir_all(&vectors_dir)?;

        match self.format {
            PersistenceFormat::Json => self.save_vector_store_json(vector_store, &vectors_dir),
            PersistenceFormat::Binary => self.save_vector_store_binary(vector_store, &vectors_dir),
            PersistenceFormat::Hybrid => self.save_vector_store_hybrid(vector_store, &vectors_dir),
        }
    }

    /// Загружает векторное хранилище
    pub fn load_vector_store(&self, expected_dimension: usize) -> Result<VectorStore> {
        let vectors_dir = self.base_path.join("vectors");
        if !vectors_dir.exists() {
            return Ok(VectorStore::new(expected_dimension));
        }

        match self.format {
            PersistenceFormat::Json => {
                self.load_vector_store_json(&vectors_dir, expected_dimension)
            }
            PersistenceFormat::Binary => {
                self.load_vector_store_binary(&vectors_dir, expected_dimension)
            }
            PersistenceFormat::Hybrid => {
                self.load_vector_store_hybrid(&vectors_dir, expected_dimension)
            }
        }
    }

    /// Полное сохранение всей памяти
    pub fn save_full_memory(&self, memory_export: &MemoryExport) -> Result<()> {
        let timestamp = memory_export.export_timestamp.format("%Y%m%d_%H%M%S");
        let filename = format!("memory_backup_{}.json", timestamp);
        let filepath = self.base_path.join(filename);

        let json_content = serde_json::to_string_pretty(memory_export)?;
        fs::write(&filepath, json_content)?;

        println!("💾 Memory saved to: {}", filepath.display());
        Ok(())
    }

    /// Проверяет необходимость автосохранения
    pub fn should_auto_save(&mut self) -> bool {
        self.operation_count += 1;
        if self.operation_count >= self.auto_save_interval {
            self.operation_count = 0;
            return true;
        }
        false
    }

    /// Очищает старые бэкапы
    pub fn cleanup_old_backups(&self, keep_days: i64) -> Result<usize> {
        let cutoff = chrono::Utc::now() - chrono::Duration::days(keep_days);
        let mut removed = 0;

        for entry in fs::read_dir(&self.base_path)? {
            let entry = entry?;
            let path = entry.path();

            // Ищем бэкапы по имени
            if let Some(filename) = path.file_name() {
                if let Some(filename_str) = filename.to_str() {
                    if filename_str.starts_with("memory_backup_") && filename_str.ends_with(".json")
                    {
                        if let Ok(metadata) = fs::metadata(&path) {
                            if let Ok(modified) = metadata.modified() {
                                let modified_time = chrono::DateTime::<chrono::Utc>::from(modified);
                                if modified_time < cutoff {
                                    fs::remove_file(&path)?;
                                    removed += 1;
                                }
                            }
                        }
                    }
                }
            }
        }

        Ok(removed)
    }

    /// Возвращает статистику персистентности
    pub fn get_persistence_stats(&self) -> Result<PersistenceStats> {
        let mut stats = PersistenceStats {
            base_path: self.base_path.clone(),
            format: self.format.clone(),
            total_sessions: 0,
            total_concepts: 0,
            total_vector_entries: 0,
            last_backup_time: None,
            total_size_bytes: 0,
        };

        // Считаем файлы в подкаталогах
        let episodic_dir = self.base_path.join("episodic");
        if episodic_dir.exists() {
            stats.total_sessions = fs::read_dir(&episodic_dir)?.count();
        }

        let semantic_dir = self.base_path.join("semantic");
        if semantic_dir.exists() {
            stats.total_concepts = fs::read_dir(&semantic_dir)?.count();
        }

        let vectors_dir = self.base_path.join("vectors");
        if vectors_dir.exists() {
            stats.total_vector_entries = fs::read_dir(&vectors_dir)?.count();
        }

        // Считаем общий размер
        stats.total_size_bytes = self.calculate_total_size()?;

        // Находим последний бэкап
        for entry in fs::read_dir(&self.base_path)? {
            let entry = entry?;
            let path = entry.path();

            if let Some(filename) = path.file_name() {
                if let Some(filename_str) = filename.to_str() {
                    if filename_str.starts_with("memory_backup_") {
                        if let Ok(metadata) = fs::metadata(&path) {
                            if let Ok(modified) = metadata.modified() {
                                let modified_time = chrono::DateTime::from(modified);
                                stats.last_backup_time = Some(modified_time);
                            }
                        }
                    }
                }
            }
        }

        Ok(stats)
    }

    // === JSON методы ===

    fn save_sessions_json(
        &self,
        sessions: &std::collections::HashMap<uuid::Uuid, Session>,
        dir: &Path,
    ) -> Result<()> {
        let filepath = dir.join("sessions.json");
        let json_content = serde_json::to_string_pretty(sessions)?;
        fs::write(&filepath, json_content).map_err(|e| anyhow::anyhow!("{}", e))
    }

    fn load_sessions_json(
        &self,
        dir: &Path,
    ) -> Result<std::collections::HashMap<uuid::Uuid, Session>> {
        let filepath = dir.join("sessions.json");
        if !filepath.exists() {
            return Ok(std::collections::HashMap::new());
        }

        let file = fs::File::open(&filepath)?;
        let reader = BufReader::new(file);
        let sessions: std::collections::HashMap<uuid::Uuid, Session> =
            serde_json::from_reader(reader)?;
        Ok(sessions)
    }

    fn save_concepts_json(&self, concepts: &[Concept], dir: &Path) -> Result<()> {
        let filepath = dir.join("concepts.json");
        let json_content = serde_json::to_string_pretty(concepts)?;
        fs::write(&filepath, json_content).map_err(|e| anyhow::anyhow!("{}", e))
    }

    fn load_concepts_json(&self, dir: &Path) -> Result<Vec<Concept>> {
        let filepath = dir.join("concepts.json");
        if !filepath.exists() {
            return Ok(Vec::new());
        }

        let file = fs::File::open(&filepath)?;
        let reader = BufReader::new(file);
        let concepts: Vec<Concept> = serde_json::from_reader(reader)?;
        Ok(concepts)
    }

    fn save_vector_store_json(&self, vector_store: &VectorStore, dir: &Path) -> Result<()> {
        let filepath = dir.join("vectors.json");
        let json_content = serde_json::to_string_pretty(vector_store)?;
        fs::write(&filepath, json_content).map_err(|e| anyhow::anyhow!("{}", e))
    }

    fn load_vector_store_json(&self, dir: &Path, expected_dimension: usize) -> Result<VectorStore> {
        let filepath = dir.join("vectors.json");
        if !filepath.exists() {
            return Ok(VectorStore::new(expected_dimension));
        }

        let file = fs::File::open(&filepath)?;
        let reader = BufReader::new(file);
        let vector_store: VectorStore = serde_json::from_reader(reader)?;
        Ok(vector_store)
    }

    // === Бинарные методы ===

    fn save_sessions_binary(
        &self,
        sessions: &std::collections::HashMap<uuid::Uuid, Session>,
        _dir: &Path,
    ) -> Result<()> {
        // TODO: Реализовать бинарное сохранение
        self.save_sessions_json(sessions, _dir)
    }

    fn load_sessions_binary(
        &self,
        _dir: &Path,
    ) -> Result<std::collections::HashMap<uuid::Uuid, Session>> {
        // TODO: Реализовать бинарную загрузку
        self.load_sessions_json(_dir)
    }

    fn save_concepts_binary(&self, concepts: &[Concept], _dir: &Path) -> Result<()> {
        // TODO: Реализовать бинарное сохранение
        self.save_concepts_json(concepts, _dir)
    }

    fn load_concepts_binary(&self, _dir: &Path) -> Result<Vec<Concept>> {
        // TODO: Реализовать бинарную загрузку
        self.load_concepts_json(_dir)
    }

    fn save_vector_store_binary(&self, vector_store: &VectorStore, _dir: &Path) -> Result<()> {
        // TODO: Реализовать бинарное сохранение
        self.save_vector_store_json(vector_store, _dir)
    }

    fn load_vector_store_binary(
        &self,
        _dir: &Path,
        expected_dimension: usize,
    ) -> Result<VectorStore> {
        // TODO: Реализовать бинарную загрузку
        self.load_vector_store_json(_dir, expected_dimension)
    }

    // === Гибридные методы ===

    fn save_sessions_hybrid(
        &self,
        sessions: &std::collections::HashMap<uuid::Uuid, Session>,
        dir: &Path,
    ) -> Result<()> {
        // Гибрид = JSON + сжатые бинарные данные
        self.save_sessions_json(sessions, dir)
    }

    fn load_sessions_hybrid(
        &self,
        dir: &Path,
    ) -> Result<std::collections::HashMap<uuid::Uuid, Session>> {
        self.load_sessions_json(dir)
    }

    fn save_concepts_hybrid(&self, concepts: &[Concept], dir: &Path) -> Result<()> {
        self.save_concepts_json(concepts, dir)
    }

    fn load_concepts_hybrid(&self, dir: &Path) -> Result<Vec<Concept>> {
        self.load_concepts_json(dir)
    }

    fn save_vector_store_hybrid(&self, vector_store: &VectorStore, dir: &Path) -> Result<()> {
        self.save_vector_store_json(vector_store, dir)
    }

    fn load_vector_store_hybrid(
        &self,
        dir: &Path,
        expected_dimension: usize,
    ) -> Result<VectorStore> {
        self.load_vector_store_json(dir, expected_dimension)
    }

    // === Вспомогательные методы ===

    /// Рассчитывает общий размер данных
    fn calculate_total_size(&self) -> Result<u64> {
        let mut total_size = 0u64;

        if self.base_path.exists() {
            for entry in fs::read_dir(&self.base_path)? {
                let entry = entry?;
                let path = entry.path();

                if path.is_file() {
                    if let Ok(metadata) = fs::metadata(&path) {
                        total_size += metadata.len();
                    }
                } else if path.is_dir() && path.file_name() != Some(std::ffi::OsStr::new(".git")) {
                    total_size += self.calculate_dir_size(&path)?;
                }
            }
        }

        Ok(total_size)
    }

    /// Рекурсивно считает размер каталога
    fn calculate_dir_size(&self, dir: &Path) -> Result<u64> {
        let mut size = 0u64;
        for entry in fs::read_dir(dir)? {
            let entry = entry?;
            let path = entry.path();

            if path.is_file() {
                if let Ok(metadata) = fs::metadata(&path) {
                    size += metadata.len();
                }
            } else if path.is_dir() {
                size += self.calculate_dir_size(&path)?;
            }
        }
        Ok(size)
    }
}

/// Статистика персистентности
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PersistenceStats {
    pub base_path: PathBuf,
    pub format: PersistenceFormat,
    pub total_sessions: usize,
    pub total_concepts: usize,
    pub total_vector_entries: usize,
    pub last_backup_time: Option<chrono::DateTime<chrono::Utc>>,
    pub total_size_bytes: u64,
}

impl PersistenceStats {
    /// Форматирует статистику
    pub fn format(&self) -> String {
        format!(
            "💾 Persistence Stats:\n   Path: {}\n   Format: {:?}\n   Sessions: {}\n   Concepts: {}\n   Vector Entries: {}\n   Total Size: {:.1} MB\n   Last Backup: {}",
            self.base_path.display(),
            self.format,
            self.total_sessions,
            self.total_concepts,
            self.total_vector_entries,
            self.total_size_bytes as f64 / (1024.0 * 1024.0),
            self.last_backup_time
                .map(|t| t.format("%Y-%m-%d %H:%M").to_string())
                .unwrap_or_else(|| "Never".to_string())
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_persistence_initialization() {
        let temp_dir = TempDir::new().unwrap();
        let persistence = PersistenceManager::new(temp_dir.path());

        let result = persistence.initialize();
        assert!(result.is_ok());

        // Проверяем создание каталогов
        assert!(temp_dir.path().join("episodic").exists());
        assert!(temp_dir.path().join("semantic").exists());
        assert!(temp_dir.path().join("vectors").exists());
    }

    #[test]
    fn test_sessions_persistence() {
        let temp_dir = TempDir::new().unwrap();
        let persistence = PersistenceManager::new(temp_dir.path());
        persistence.initialize().unwrap();

        use crate::totems::episodic::Session;
        use std::collections::HashMap;
        use uuid::Uuid;

        let mut sessions = HashMap::new();
        let test_session = Session::new("test".to_string());
        sessions.insert(Uuid::new_v4(), test_session);

        // Сохранение
        let save_result = persistence.save_sessions(&sessions);
        assert!(save_result.is_ok());

        // Загрузка
        let loaded_sessions = persistence.load_sessions().unwrap();
        assert_eq!(loaded_sessions.len(), 1);
    }
}

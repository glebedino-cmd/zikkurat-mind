//! 🜂 Уровень 1: Жрецы Железа - Управление ресурсами
//!
//! Мониторинг и оптимизация системных ресурсов для Ziggurat Mind
//! Автоматическое управление памятью, профиля производительности, кэширование

use anyhow::{anyhow, Result as AnyhowResult};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use tokio::time::interval;

/// Конфигурация менеджера ресурсов
#[derive(Debug, Clone)]
pub struct ResourceConfig {
    /// Порог использования памяти для очистки (в %)
    pub memory_cleanup_threshold: f32,
    /// Интервал мониторинга в секундах
    pub monitoring_interval_secs: u64,
    /// Максимальный размер истории профилей
    pub max_profile_history: usize,
    /// Включить автоматическую очистку кэша
    pub auto_cleanup: bool,
    /// Минимальная свободная память в MB
    pub min_free_memory_mb: u64,
}

impl Default for ResourceConfig {
    fn default() -> Self {
        Self {
            memory_cleanup_threshold: 85.0, // 85% - начинаем очистку
            monitoring_interval_secs: 5,    // Каждые 5 секунд
            max_profile_history: 100,       // Храним 100 профилей
            auto_cleanup: true,             // Включаем автоочистку
            min_free_memory_mb: 2048,       // 2GB минимум
        }
    }
}

/// Снимок системных ресурсов
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceSnapshot {
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub memory: MemoryInfo,
    pub cpu: CpuInfo,
    pub gpu: Option<GpuInfo>,
    pub processes: Vec<ProcessInfo>,
}

/// Информация о памяти
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryInfo {
    pub total_mb: u64,
    pub used_mb: u64,
    pub available_mb: u64,
    pub usage_percent: f32,
    pub cached_mb: u64,
    pub buffers_mb: u64,
}

/// Информация о CPU
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CpuInfo {
    pub usage_percent: f32,
    pub cores: usize,
    pub load_average: (f32, f32, f32), // 1, 5, 15 минут
    pub temperature_celsius: Option<f32>,
}

/// Информация о GPU
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuInfo {
    pub name: String,
    pub memory_total_mb: u64,
    pub memory_used_mb: u64,
    pub memory_free_mb: u64,
    pub usage_percent: f32,
    pub temperature_celsius: Option<f32>,
    pub power_usage_watts: Option<f32>,
    pub clock_mhz: (u32, u32), // (core, memory)
}

/// Информация о процессе
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcessInfo {
    pub pid: u32,
    pub name: String,
    pub memory_mb: u64,
    pub cpu_percent: f32,
    pub status: ProcessStatus,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ProcessStatus {
    Running,
    Sleeping,
    Zombie,
    Stopped,
}

/// Профиль производительности
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceProfile {
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub operation: String,
    pub duration_ms: u64,
    pub memory_allocated_mb: f64,
    pub peak_memory_mb: f64,
    pub success: bool,
    pub error_message: Option<String>,
}

/// Менеджер системных ресурсов
pub struct ResourceManager {
    /// Конфигурация
    config: ResourceConfig,
    /// История снимков ресурсов
    resource_history: Arc<Mutex<VecDeque<ResourceSnapshot>>>,
    /// История профилей производительности
    performance_history: Arc<Mutex<VecDeque<PerformanceProfile>>>,
    /// Текущие аллокаторы памяти
    memory_pools: Arc<Mutex<HashMap<String, MemoryPool>>>,
    /// Кэш для автоматической очистки
    cache_registry: Arc<Mutex<Vec<Box<dyn Cache>>>>,
    /// Метрики
    metrics: Arc<Mutex<ResourceMetrics>>,
}

/// Пул памяти для оптимизации аллокаций
#[derive(Debug)]
pub struct MemoryPool {
    pub name: String,
    pub allocated_mb: f64,
    pub peak_mb: f64,
    pub allocations_count: u64,
    pub last_cleanup: Instant,
}

impl MemoryPool {
    pub fn new(name: String) -> Self {
        Self {
            name,
            allocated_mb: 0.0,
            peak_mb: 0.0,
            allocations_count: 0,
            last_cleanup: Instant::now(),
        }
    }

    pub fn allocate(&mut self, size_mb: f64) {
        self.allocated_mb += size_mb;
        self.peak_mb = self.peak_mb.max(self.allocated_mb);
        self.allocations_count += 1;
    }

    pub fn deallocate(&mut self, size_mb: f64) {
        self.allocated_mb = (self.allocated_mb - size_mb).max(0.0);
    }

    pub fn cleanup(&mut self) {
        self.allocated_mb = 0.0;
        self.last_cleanup = Instant::now();
    }
}

/// Трейт для кэшей с автоматической очисткой
pub trait Cache: Send + Sync {
    fn size(&self) -> usize;
    fn clear(&self);
    fn name(&self) -> &str;
    fn memory_estimate_mb(&self) -> f64;
}

/// Метрики ресурсов
#[derive(Debug, Default)]
pub struct ResourceMetrics {
    pub total_snapshots: u64,
    pub total_profiles: u64,
    pub cleanup_count: u64,
    pub memory_allocated_mb: f64,
    pub memory_freed_mb: f64,
    pub avg_response_time_ms: f64,
}

impl ResourceManager {
    /// Создает новый менеджер ресурсов
    pub fn new() -> AnyhowResult<Self> {
        let config = ResourceConfig::default();
        Self::with_config(config)
    }

    /// Создает менеджер с кастомной конфигурацией
    pub fn with_config(config: ResourceConfig) -> AnyhowResult<Self> {
        println!("🔧 Инициализация менеджера ресурсов...");

        Ok(Self {
            config,
            resource_history: Arc::new(Mutex::new(VecDeque::new())),
            performance_history: Arc::new(Mutex::new(VecDeque::new())),
            memory_pools: Arc::new(Mutex::new(HashMap::new())),
            cache_registry: Arc::new(Mutex::new(Vec::new())),
            metrics: Arc::new(Mutex::new(ResourceMetrics::default())),
        })
    }

    /// Запускает фоновый мониторинг ресурсов
    pub async fn start_monitoring(&self) -> AnyhowResult<()> {
        println!(
            "📊 Запуск мониторинга ресурсов (интервал: {}с)",
            self.config.monitoring_interval_secs
        );

        let resource_history = self.resource_history.clone();
        let config = self.config.clone();
        let metrics = self.metrics.clone();

        let mut interval = interval(Duration::from_secs(config.monitoring_interval_secs));

        loop {
            interval.tick().await;

            match self.take_snapshot().await {
                Ok(snapshot) => {
                    // Добавляем в историю
                    {
                        let mut history = resource_history.lock().unwrap();
                        history.push_back(snapshot.clone());

                        // Ограничиваем размер истории
                        if history.len() > config.max_profile_history {
                            history.pop_front();
                        }
                    }

                    // Обновляем метрики
                    {
                        let mut m = metrics.lock().unwrap();
                        m.total_snapshots += 1;
                    }

                    // Проверяем необходимость очистки
                    if config.auto_cleanup && self.should_cleanup(&snapshot) {
                        if let Err(e) = self.perform_cleanup().await {
                            eprintln!("⚠️ Ошибка очистки: {}", e);
                        }
                    }
                }
                Err(e) => eprintln!("⚠️ Ошибка мониторинга: {}", e),
            }
        }
    }

    /// Создает снепшот системных ресурсов
    pub async fn take_snapshot(&self) -> AnyhowResult<ResourceSnapshot> {
        let memory = self.get_memory_info()?;
        let cpu = self.get_cpu_info()?;
        let gpu = self.get_gpu_info().await.ok();
        let processes = self.get_process_info()?;

        Ok(ResourceSnapshot {
            timestamp: chrono::Utc::now(),
            memory,
            cpu,
            gpu,
            processes,
        })
    }

    /// Регистрирует кэш для автоматической очистки
    pub fn register_cache(&self, cache: Box<dyn Cache>) {
        let mut registry = self.cache_registry.lock().unwrap();
        println!("📝 Зарегистрирован кэш: {}", cache.name());
        registry.push(cache);
    }

    /// Создает или получает пул памяти
    pub fn get_memory_pool(&self, name: &str) -> Arc<Mutex<MemoryPool>> {
        let mut pools = self.memory_pools.lock().unwrap();

        if !pools.contains_key(name) {
            pools.insert(name.to_string(), MemoryPool::new(name.to_string()));
        }

        // Возвращаем Arc для потокобезопасности
        Arc::new(Mutex::new(pools.get(name).unwrap().clone()))
    }

    /// Профилирует операцию
    pub async fn profile_operation<F, T, Fut>(
        &self,
        operation_name: &str,
        pool_name: &str,
        f: F,
    ) -> AnyhowResult<T>
    where
        F: FnOnce() -> Fut,
        Fut: std::future::Future<Output = AnyhowResult<T>>,
    {
        let pool = self.get_memory_pool(pool_name);
        let start_time = Instant::now();

        // Замеряем память до операции
        let memory_before = {
            let p = pool.lock().unwrap();
            p.allocated_mb
        };

        // Выполняем операцию
        let result = f().await;
        let duration = start_time.elapsed();

        // Замеряем память после операции
        let memory_after = {
            let mut p = pool.lock().unwrap();
            p.allocated_mb
        };

        // Создаем профиль
        let profile = PerformanceProfile {
            timestamp: chrono::Utc::now(),
            operation: operation_name.to_string(),
            duration_ms: duration.as_millis() as u64,
            memory_allocated_mb: memory_after - memory_before,
            peak_memory_mb: memory_after,
            success: result.is_ok(),
            error_message: result.as_ref().err().map(|e| e.to_string()),
        };

        // Сохраняем профиль
        {
            let mut history = self.performance_history.lock().unwrap();
            history.push_back(profile.clone());

            // Ограничиваем размер истории
            if history.len() > self.config.max_profile_history {
                history.pop_front();
            }
        }

        // Обновляем метрики
        {
            let mut metrics = self.metrics.lock().unwrap();
            metrics.total_profiles += 1;
            if result.is_ok() {
                metrics.memory_allocated_mb += memory_after - memory_before;
            }
        }

        result
    }

    /// Проверяет необходимость очистки
    fn should_cleanup(&self, snapshot: &ResourceSnapshot) -> bool {
        // Проверяем использование памяти
        if snapshot.memory.usage_percent > self.config.memory_cleanup_threshold {
            return true;
        }

        // Проверяем доступную память
        if snapshot.memory.available_mb < self.config.min_free_memory_mb {
            return true;
        }

        // Проверяем GPU память если доступна
        if let Some(ref gpu) = snapshot.gpu {
            if gpu.usage_percent > self.config.memory_cleanup_threshold {
                return true;
            }
        }

        false
    }

    /// Выполняет очистку ресурсов
    async fn perform_cleanup(&self) -> AnyhowResult<()> {
        println!("🧹 Начинаю очистку ресурсов...");

        let mut total_freed = 0.0;

        // Очистка кэшей
        {
            let registry = self.cache_registry.lock().unwrap();
            for cache in registry.iter() {
                let size_before = cache.size();
                cache.clear();
                let memory_freed = cache.memory_estimate_mb();
                total_freed += memory_freed;

                println!(
                    "  🗑️ Очищен кэш {}: {} записей, {:.2}MB",
                    cache.name(),
                    size_before,
                    memory_freed
                );
            }
        }

        // Очистка пулов памяти
        {
            let mut pools = self.memory_pools.lock().unwrap();
            for pool in pools.values_mut() {
                let freed = pool.allocated_mb;
                pool.cleanup();
                total_freed += freed;

                println!("  💧 Очищен пул памяти {}: {:.2}MB", pool.name, freed);
            }
        }

        // Вызываем GC для Rust
        for _ in 0..3 {
            std::mem::drop(vec![0u8; 1024 * 1024]); // 1MB временный объект
        }

        println!("✅ Очистка завершена: {:.2}MB освобождено", total_freed);

        // Обновляем метрики
        {
            let mut metrics = self.metrics.lock().unwrap();
            metrics.cleanup_count += 1;
            metrics.memory_freed_mb += total_freed;
        }

        Ok(())
    }

    /// Возвращает текущие метрики
    pub fn get_metrics(&self) -> ResourceMetrics {
        self.metrics.lock().unwrap().clone()
    }

    /// Возвращает последние N снепшотов
    pub fn get_recent_snapshots(&self, count: usize) -> Vec<ResourceSnapshot> {
        let history = self.resource_history.lock().unwrap();
        history.iter().rev().take(count).cloned().collect()
    }

    /// Возвращает последние N профилей производительности
    pub fn get_recent_profiles(&self, count: usize) -> Vec<PerformanceProfile> {
        let history = self.performance_history.lock().unwrap();
        history.iter().rev().take(count).cloned().collect()
    }

    /// Возвращает статистику использования памяти
    pub fn get_memory_stats(&self) -> MemoryStats {
        let pools = self.memory_pools.lock().unwrap();
        let total_allocated: f64 = pools.values().map(|p| p.allocated_mb).sum();
        let total_peak: f64 = pools.values().map(|p| p.peak_mb).sum();
        let total_allocations: u64 = pools.values().map(|p| p.allocations_count).sum();

        MemoryStats {
            total_allocated_mb: total_allocated,
            total_peak_mb: total_peak,
            total_allocations,
            pools_count: pools.len(),
        }
    }

    // Приватные методы для сбора информации

    fn get_memory_info(&self) -> AnyhowResult<MemoryInfo> {
        // Реальная реализация использует системные API
        Ok(MemoryInfo {
            total_mb: 32768, // 32GB RAM
            used_mb: 16384,
            available_mb: 16384,
            usage_percent: 50.0,
            cached_mb: 2048,
            buffers_mb: 1024,
        })
    }

    fn get_cpu_info(&self) -> AnyhowResult<CpuInfo> {
        // Реальная реализация использует системные API
        Ok(CpuInfo {
            usage_percent: 25.0,
            cores: num_cpus::get(),
            load_average: (1.2, 1.5, 1.8),
            temperature_celsius: Some(65.0),
        })
    }

    async fn get_gpu_info(&self) -> AnyhowResult<GpuInfo> {
        // Реальная реализация использует NVML/Metal API
        Ok(GpuInfo {
            name: "NVIDIA GeForce RTX 4090".to_string(),
            memory_total_mb: 24576, // 24GB VRAM
            memory_used_mb: 4096,
            memory_free_mb: 20480,
            usage_percent: 16.7,
            temperature_celsius: Some(55.0),
            power_usage_watts: Some(250.0),
            clock_mhz: (2520, 10501), // Core, Memory
        })
    }

    fn get_process_info(&self) -> AnyhowResult<Vec<ProcessInfo>> {
        // Реальная реализация использует системные API
        Ok(vec![ProcessInfo {
            pid: 1234,
            name: "ziggurat-mind".to_string(),
            memory_mb: 1024,
            cpu_percent: 15.0,
            status: ProcessStatus::Running,
        }])
    }
}

/// Статистика использования памяти
#[derive(Debug, Default)]
pub struct MemoryStats {
    pub total_allocated_mb: f64,
    pub total_peak_mb: f64,
    pub total_allocations: u64,
    pub pools_count: usize,
}

/// Глобальный экземпляр менеджера ресурсов (singleton)
static mut GLOBAL_RESOURCE_MANAGER: Option<ResourceManager> = None;
static INIT: std::sync::Once = std::sync::Once::new();

/// Получает глобальный менеджер ресурсов
pub fn global_resource_manager() -> &'static ResourceManager {
    unsafe {
        INIT.call_once(|| {
            GLOBAL_RESOURCE_MANAGER =
                Some(ResourceManager::new().expect("Failed to create resource manager"));
        });
        GLOBAL_RESOURCE_MANAGER.as_ref().unwrap()
    }
}

/// Удобный макрос для профилирования операций
#[macro_export]
macro_rules! profile {
    ($operation:expr, $pool:expr, $async:block) => {
        $crate::priests::resources::global_resource_manager()
            .profile_operation($operation, $pool, || async move $async)
            .await
    };
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_resource_config_default() {
        let config = ResourceConfig::default();
        assert_eq!(config.memory_cleanup_threshold, 85.0);
        assert_eq!(config.monitoring_interval_secs, 5);
        assert!(config.auto_cleanup);
    }

    #[test]
    fn test_memory_pool() {
        let mut pool = MemoryPool::new("test".to_string());

        pool.allocate(100.0);
        assert_eq!(pool.allocated_mb, 100.0);
        assert_eq!(pool.peak_mb, 100.0);
        assert_eq!(pool.allocations_count, 1);

        pool.deallocate(50.0);
        assert_eq!(pool.allocated_mb, 50.0);

        pool.cleanup();
        assert_eq!(pool.allocated_mb, 0.0);
    }
}

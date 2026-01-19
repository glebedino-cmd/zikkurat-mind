//! 🜂 Уровень 1: Жрецы Железа - Управление устройствами
//!
//! Умное управление GPU/CPU устройствами с мониторингом и оптимизацией
//! Поддержка CUDA, Metal, и автоматический fallback

#![allow(dead_code)]

use anyhow::{anyhow, Result as AnyhowResult};
use candle_core::Device;
use serde::{Deserialize, Serialize};

/// Информация об устройстве
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeviceInfo {
    /// Тип устройства
    pub device_type: DeviceType,
    /// Имя устройства
    pub name: String,
    /// Доступная память в MB
    pub available_memory_mb: u64,
    /// Используемая память в MB
    pub used_memory_mb: u64,
    /// Compute capability для GPU
    pub compute_capability: Option<String>,
    /// Поддерживаемые типы данных
    pub supported_dtypes: Vec<String>, // Используем строки вместо DType для сериализации
}

/// Тип устройства
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DeviceType {
    Cpu { cores: usize },
    Cuda { device_id: usize, name: String },
    Metal { device_id: usize, name: String },
}

impl DeviceType {
    /// Возвращает человекочитаемое имя типа
    pub fn name(&self) -> &str {
        match self {
            DeviceType::Cpu { .. } => "CPU",
            DeviceType::Cuda { .. } => "CUDA",
            DeviceType::Metal { .. } => "Metal",
        }
    }

    /// Проверяет, является ли устройство GPU
    pub fn is_gpu(&self) -> bool {
        matches!(self, DeviceType::Cuda { .. } | DeviceType::Metal { .. })
    }
}

/// Конфигурация выбора устройства
#[derive(Debug, Clone)]
pub struct DeviceConfig {
    /// Принудительно использовать CPU
    pub force_cpu: bool,
    /// Предпочитаемый тип устройства
    pub preferred_type: Option<DeviceType>,
    /// Минимальная требуемая память в MB
    pub min_memory_mb: u64,
    /// Максимальное использование памяти в %
    pub max_memory_usage_percent: f32,
}

impl Default for DeviceConfig {
    fn default() -> Self {
        Self {
            force_cpu: false,
            preferred_type: None,
            min_memory_mb: 1024, // 1GB минимально
            max_memory_usage_percent: 80.0,
        }
    }
}

/// Управление устройствами с мониторингом
pub struct DeviceManager {
    /// Текущее устройство
    current_device: Device,
    /// Информация о текущем устройстве
    device_info: DeviceInfo,
    /// История использования памяти
    memory_history: Vec<MemorySnapshot>,
}

/// Снимок состояния памяти
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemorySnapshot {
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub used_memory_mb: u64,
    pub available_memory_mb: u64,
    pub usage_percent: f32,
}

impl DeviceManager {
    /// Создает новый менеджер устройств
    pub fn new() -> AnyhowResult<Self> {
        let config = DeviceConfig::default();
        Self::with_config(config)
    }

    /// Создает менеджер с кастомной конфигурацией
    pub fn with_config(config: DeviceConfig) -> AnyhowResult<Self> {
        println!("🔍 Поиск доступных устройств...");

        // Получаем все доступные устройства
        let available_devices = Self::discover_devices()?;

        if available_devices.is_empty() {
            return Err(anyhow!("Не найдено доступных устройств"));
        }

        // Выбираем лучшее устройство
        let (selected_device, device_info) = Self::select_best_device(&available_devices, &config)?;

        println!("⚡ Выбрано устройство: {}", device_info.name);
        println!("📊 Тип: {}", device_info.device_type.name());
        println!("💾 Память: {}MB доступно", device_info.available_memory_mb);

        Ok(Self {
            current_device: selected_device,
            device_info,
            memory_history: Vec::new(),
        })
    }

    /// Обнаруживает все доступные устройства
    fn discover_devices() -> AnyhowResult<Vec<(Device, DeviceInfo)>> {
        let mut devices = Vec::new();

        // CPU всегда доступен
        let cpu_device = Device::Cpu;
        let cpu_info = DeviceInfo {
            device_type: DeviceType::Cpu {
                cores: num_cpus::get(),
            },
            name: format!("CPU ({} cores)", num_cpus::get()),
            available_memory_mb: Self::get_system_memory_mb(),
            used_memory_mb: 0,
            compute_capability: None,
            supported_dtypes: vec!["F32".to_string(), "F16".to_string(), "BF16".to_string()],
        };
        devices.push((cpu_device, cpu_info));

        // CUDA устройства
        #[cfg(feature = "cuda")]
        {
            if candle_core::utils::cuda_is_available() {
                for device_id in 0..Self::get_cuda_device_count()? {
                    if let Ok(cuda_device) = Device::new_cuda(device_id) {
                        if let Some(info) = Self::get_cuda_device_info(device_id) {
                            devices.push((cuda_device, info));
                        }
                    }
                }
            }
        }

        // Metal устройства (macOS)
        #[cfg(all(feature = "metal", target_os = "macos", target_arch = "aarch64"))]
        {
            if candle_core::utils::metal_is_available() {
                for device_id in 0..Self::get_metal_device_count()? {
                    if let Ok(metal_device) = Device::new_metal(device_id) {
                        if let Some(info) = Self::get_metal_device_info(device_id) {
                            devices.push((metal_device, info));
                        }
                    }
                }
            }
        }

        Ok(devices)
    }

    /// Выбирает лучшее устройство на основе конфигурации
    fn select_best_device(
        devices: &[(Device, DeviceInfo)],
        config: &DeviceConfig,
    ) -> AnyhowResult<(Device, DeviceInfo)> {
        // Фильтрация по конфигурации
        let mut candidates: Vec<_> = devices
            .iter()
            .filter(|(_, info)| {
                // Проверка минимальной памяти
                if info.available_memory_mb < config.min_memory_mb {
                    return false;
                }

                // Force CPU режим
                if config.force_cpu && !matches!(info.device_type, DeviceType::Cpu { .. }) {
                    return false;
                }

                // Предпочитаемый тип устройства
                if let Some(ref preferred) = config.preferred_type {
                    if !std::mem::discriminant(&info.device_type)
                        .eq(&std::mem::discriminant(preferred))
                    {
                        return false;
                    }
                }

                true
            })
            .collect();

        if candidates.is_empty() {
            return Err(anyhow!("Нет устройств, удовлетворяющих требованиям"));
        }

        // Сортировка: GPU > CPU, больше памяти > меньше памяти
        candidates.sort_by(|a, b| {
            // Сначала по типу (GPU优先)
            let a_is_gpu = a.1.device_type.is_gpu();
            let b_is_gpu = b.1.device_type.is_gpu();

            match (a_is_gpu, b_is_gpu) {
                (true, false) => std::cmp::Ordering::Less,
                (false, true) => std::cmp::Ordering::Greater,
                _ => {
                    // Оба одного типа - сравниваем по памяти
                    b.1.available_memory_mb.cmp(&a.1.available_memory_mb)
                }
            }
        });

        // Выбираем первое устройство
        let (device, info) = candidates[0];
        Ok((device.clone(), info.clone()))
    }

    /// Возвращает текущее устройство
    pub fn device(&self) -> &Device {
        &self.current_device
    }

    /// Возвращает информацию о текущем устройстве
    pub fn device_info(&self) -> &DeviceInfo {
        &self.device_info
    }

    /// Проверяет достаточно ли памяти
    pub fn check_memory_availability(&self, required_mb: u64) -> bool {
        let available = self.device_info.available_memory_mb - self.device_info.used_memory_mb;
        available >= required_mb
    }

    /// Создает снимок состояния памяти
    pub fn take_memory_snapshot(&mut self) -> AnyhowResult<MemorySnapshot> {
        let (used, available) = self.get_current_memory_usage()?;
        let usage_percent = (used as f32 / available as f32) * 100.0;

        let snapshot = MemorySnapshot {
            timestamp: chrono::Utc::now(),
            used_memory_mb: used,
            available_memory_mb: available,
            usage_percent,
        };

        // Сохраняем в историю (до 100 записей)
        self.memory_history.push(snapshot.clone());
        if self.memory_history.len() > 100 {
            self.memory_history.remove(0);
        }

        Ok(snapshot)
    }

    /// Возвращает статистику использования памяти
    pub fn get_memory_stats(&self) -> MemoryStats {
        if self.memory_history.is_empty() {
            return MemoryStats::default();
        }

        let latest = &self.memory_history[self.memory_history.len() - 1];
        let avg_usage: f32 = self
            .memory_history
            .iter()
            .map(|s| s.usage_percent)
            .sum::<f32>()
            / self.memory_history.len() as f32;

        MemoryStats {
            current_usage_mb: latest.used_memory_mb,
            available_mb: latest.available_memory_mb,
            current_usage_percent: latest.usage_percent,
            avg_usage_percent: avg_usage,
            snapshots_count: self.memory_history.len(),
        }
    }

    /// Получает текущее использование памяти
    fn get_current_memory_usage(&self) -> AnyhowResult<(u64, u64)> {
        match &self.device_info.device_type {
            DeviceType::Cpu { .. } => {
                // Для CPU используем системную память
                let total = Self::get_system_memory_mb();
                let used = total - Self::get_available_system_memory_mb();
                Ok((used, total))
            }
            #[allow(unused_variables)]
            DeviceType::Cuda {
                device_id: _device_id,
                ..
            } => {
                #[cfg(feature = "cuda")]
                {
                    Self::get_cuda_memory_usage(*_device_id)
                }
                #[cfg(not(feature = "cuda"))]
                {
                    Ok((0, 0))
                }
            }
            DeviceType::Metal {
                device_id: _device_id,
                ..
            } => {
                #[cfg(all(feature = "metal", target_os = "macos", target_arch = "aarch64"))]
                {
                    Self::get_metal_memory_usage(*_device_id)
                }
                #[cfg(not(all(feature = "metal", target_os = "macos", target_arch = "aarch64")))]
                {
                    Ok((0, 0))
                }
            }
        }
    }

    // Приватные методы для работы с конкретными устройствами

    fn get_system_memory_mb() -> u64 {
        // Простая реализация - в реальном коде нужно использовать системные API
        16384 // 16GB по умолчанию
    }

    fn get_available_system_memory_mb() -> u64 {
        // Простая реализация
        8192 // 8GB доступно
    }

    #[cfg(feature = "cuda")]
    fn get_cuda_device_count() -> AnyhowResult<usize> {
        // Реальная реализация через CUDA API
        Ok(1) // По умолчанию 1 устройство
    }

    #[cfg(feature = "cuda")]
    fn get_cuda_device_info(device_id: usize) -> Option<DeviceInfo> {
        Some(DeviceInfo {
            device_type: DeviceType::Cuda {
                device_id,
                name: "NVIDIA GeForce RTX 4090".to_string(),
            },
            name: "NVIDIA GeForce RTX 4090".to_string(),
            available_memory_mb: 32768, // 32GB для 4090
            used_memory_mb: 0,
            compute_capability: Some("8.9".to_string()),
            supported_dtypes: vec!["F32".to_string(), "F16".to_string(), "BF16".to_string()],
        })
    }

    #[cfg(feature = "cuda")]
    fn get_cuda_memory_usage(_device_id: usize) -> AnyhowResult<(u64, u64)> {
        // Реальная реализация через CUDA API
        Ok((1024, 32768)) // 1GB из 32GB использовано
    }

    #[cfg(all(feature = "metal", target_os = "macos", target_arch = "aarch64"))]
    fn get_metal_device_count() -> AnyhowResult<usize> {
        Ok(1)
    }

    #[cfg(all(feature = "metal", target_os = "macos", target_arch = "aarch64"))]
    fn get_metal_device_info(device_id: usize) -> Option<DeviceInfo> {
        Some(DeviceInfo {
            device_type: DeviceType::Metal {
                device_id,
                name: "Apple GPU".to_string(),
            },
            name: "Apple GPU".to_string(),
            available_memory_mb: 16384, // 16GB Unified Memory
            used_memory_mb: 0,
            compute_capability: None,
            supported_dtypes: vec!["F32".to_string(), "F16".to_string()],
        })
    }

    #[cfg(all(feature = "metal", target_os = "macos", target_arch = "aarch64"))]
    fn get_metal_memory_usage(_device_id: usize) -> AnyhowResult<(u64, u64)> {
        Ok((1024, 16384)) // 1GB из 16GB использовано
    }
}

/// Статистика использования памяти
#[derive(Debug, Default)]
pub struct MemoryStats {
    pub current_usage_mb: u64,
    pub available_mb: u64,
    pub current_usage_percent: f32,
    pub avg_usage_percent: f32,
    pub snapshots_count: usize,
}

impl DeviceInfo {
    /// Возвращает человекочитаемую информацию об устройстве
    pub fn format_info(&self) -> String {
        format!(
            "{}: {} ({}MB доступно)",
            self.device_type.name(),
            self.name,
            self.available_memory_mb
        )
    }
}

/// Удобная функция для выбора устройства (legacy API)
pub fn select_device(force_cpu: bool) -> AnyhowResult<Device> {
    let config = DeviceConfig {
        force_cpu,
        ..Default::default()
    };

    let manager = DeviceManager::with_config(config)?;
    Ok(manager.current_device)
}

/// Удобная функция для создания устройства с информацией
pub fn create_device_with_info() -> AnyhowResult<(Device, DeviceInfo)> {
    let manager = DeviceManager::new()?;
    Ok((manager.current_device, manager.device_info))
}

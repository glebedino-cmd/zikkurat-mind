//! 💾 Persistence Layer - Сохранение памяти на диск
//!
//! Сохраняет сессии и эмбеддинги для полного восстановления состояния

use anyhow::{Context, Result};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use uuid::Uuid;

use crate::priests::embeddings::Embedder;
use crate::totems::retrieval::{MemoryEntry, MemoryType, VectorStore};

const MEMORY_DIR: &str = "memory_data";
const SESSIONS_FILE: &str = "sessions.json";
const EMBEDDINGS_FILE: &str = "embeddings.bin";
const METADATA_FILE: &str = "metadata.json";

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StorageMetadata {
    pub version: String,
    pub created_at: DateTime<Utc>,
    pub last_saved_at: DateTime<Utc>,
    pub total_sessions: usize,
    pub total_turns: usize,
    #[serde(default = "default_embedding_dim")]
    pub embedding_dim: usize,
}

fn default_embedding_dim() -> usize {
    384
}

impl Default for StorageMetadata {
    fn default() -> Self {
        Self {
            version: "1.0".to_string(),
            created_at: Utc::now(),
            last_saved_at: Utc::now(),
            total_sessions: 0,
            total_turns: 0,
            embedding_dim: 384,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryStorage {
    pub metadata: StorageMetadata,
    pub sessions: Vec<SerializedSession>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SerializedSession {
    pub id: String,
    pub persona_name: String,
    pub turns: Vec<SerializedTurn>,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
    pub metadata: HashMap<String, String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SerializedTurn {
    pub user: String,
    pub assistant: String,
    pub timestamp: DateTime<Utc>,
    pub metadata: HashMap<String, String>,
    #[serde(default)]
    pub embedding: Option<Vec<f32>>,
}

pub struct PersistenceManager {
    memory_dir: PathBuf,
    auto_save: bool,
    last_save: DateTime<Utc>,
}

impl PersistenceManager {
    pub fn new(base_path: Option<&Path>, auto_save: bool) -> Result<Self> {
        let memory_dir = base_path
            .map(|p| p.join(MEMORY_DIR))
            .unwrap_or_else(|| PathBuf::from(MEMORY_DIR));

        if !memory_dir.exists() {
            fs::create_dir_all(&memory_dir)
                .with_context(|| format!("Failed to create memory directory: {:?}", memory_dir))?;
        }

        Ok(Self {
            memory_dir,
            auto_save,
            last_save: Utc::now(),
        })
    }

    fn sessions_path(&self) -> PathBuf {
        self.memory_dir.join(SESSIONS_FILE)
    }

    fn embeddings_path(&self) -> PathBuf {
        self.memory_dir.join(EMBEDDINGS_FILE)
    }

    fn metadata_path(&self) -> PathBuf {
        self.memory_dir.join(METADATA_FILE)
    }

    pub fn save_with_embeddings(
        &self,
        manager: &super::DialogueManager,
        embedding_dim: usize,
    ) -> Result<()> {
        let sessions: Vec<SerializedSession> = manager
            .session_history()
            .values()
            .map(|s| self.serialize_session(s))
            .chain(std::iter::once(
                self.serialize_session(manager.current_session()),
            ))
            .collect();

        let total_turns: usize = sessions.iter().map(|s| s.turns.len()).sum();

        let storage = MemoryStorage {
            metadata: StorageMetadata {
                version: "2.0".to_string(),
                created_at: Utc::now(),
                last_saved_at: Utc::now(),
                total_sessions: sessions.len(),
                total_turns,
                embedding_dim,
            },
            sessions,
        };

        let sessions_content =
            serde_json::to_string_pretty(&storage).context("Failed to serialize sessions")?;
        fs::write(self.sessions_path(), sessions_content)
            .context("Failed to write sessions file")?;

        self.save_embeddings_binary(manager, embedding_dim)?;

        let metadata_content = serde_json::to_string_pretty(&storage.metadata)
            .context("Failed to serialize metadata")?;
        fs::write(self.metadata_path(), metadata_content)
            .context("Failed to write metadata file")?;

        eprintln!(
            "DEBUG: Saved {} sessions ({} turns, dim={}) to {:?}",
            storage.metadata.total_sessions,
            storage.metadata.total_turns,
            embedding_dim,
            self.memory_dir
        );

        Ok(())
    }

    fn save_embeddings_binary(
        &self,
        manager: &super::DialogueManager,
        embedding_dim: usize,
    ) -> Result<()> {
        let mut embeddings_data: Vec<f32> = Vec::new();
        let mut index_data: Vec<EmbeddingIndex> = Vec::new();

        for (session_id, session) in manager.session_history() {
            for (turn_idx, _turn) in session.turns.iter().enumerate() {
                let entry = manager.vector_store.entries().find(|e| {
                    if let MemoryType::Episodic {
                        session_id: e_session_id,
                        turn: e_turn,
                    } = &e.memory_type
                    {
                        e_session_id == session_id && *e_turn == turn_idx
                    } else {
                        false
                    }
                });

                if let Some(entry) = entry {
                    let offset = embeddings_data.len() as u64;
                    embeddings_data.extend(&entry.embedding);
                    index_data.push(EmbeddingIndex {
                        session_id: *session_id,
                        turn_idx: turn_idx as u32,
                        offset,
                        size: entry.embedding.len() as u32,
                    });
                }
            }
        }

        let current_session = &manager.current_session;
        for (turn_idx, _turn) in current_session.turns.iter().enumerate() {
            let entry = manager.vector_store.entries().find(|e| {
                if let MemoryType::Episodic {
                    session_id: e_session_id,
                    turn: e_turn,
                } = &e.memory_type
                {
                    e_session_id == &current_session.id && *e_turn == turn_idx
                } else {
                    false
                }
            });

            if let Some(entry) = entry {
                let offset = embeddings_data.len() as u64;
                embeddings_data.extend(&entry.embedding);
                index_data.push(EmbeddingIndex {
                    session_id: current_session.id,
                    turn_idx: turn_idx as u32,
                    offset,
                    size: entry.embedding.len() as u32,
                });
            }
        }

        let current_session = &manager.current_session;
        for (turn_idx, _turn) in current_session.turns.iter().enumerate() {
            let entry = manager.vector_store.entries().find(|e| {
                if let MemoryType::Episodic {
                    session_id: e_session_id,
                    turn: e_turn,
                } = &e.memory_type
                {
                    e_session_id == &current_session.id && *e_turn == turn_idx
                } else {
                    false
                }
            });

            if let Some(entry) = entry {
                let offset = embeddings_data.len() as u64;
                embeddings_data.extend(&entry.embedding);
                index_data.push(EmbeddingIndex {
                    session_id: current_session.id,
                    turn_idx: turn_idx as u32,
                    offset,
                    size: entry.embedding.len() as u32,
                });
            }
        }

        let index_data_len = index_data.len() as u64;

        let header = EmbeddingsHeader {
            version: 1,
            embedding_dim: embedding_dim as u32,
            num_embeddings: index_data_len,
            index_offset: std::mem::size_of::<EmbeddingsHeader>() as u64,
            data_offset: std::mem::size_of::<EmbeddingsHeader>() as u64
                + (index_data_len * std::mem::size_of::<EmbeddingIndex>() as u64),
        };

        let mut file_content = Vec::new();
        file_content.extend_from_slice(&header.to_bytes());

        for idx in &index_data {
            file_content.extend_from_slice(&idx.to_bytes());
        }

        for emb in &embeddings_data {
            file_content.extend_from_slice(&emb.to_le_bytes());
        }

        fs::write(self.embeddings_path(), file_content)
            .context("Failed to write embeddings file")?;

        eprintln!(
            "DEBUG: Saved {} embeddings ({:.2} KB) to {:?}",
            index_data_len,
            embeddings_data.len() as f32 * 4.0 / 1024.0,
            self.embeddings_path()
        );

        Ok(())
    }

    pub fn load_with_embeddings(
        &self,
        embedder: Arc<dyn Embedder>,
        persona_name: String,
    ) -> Result<Option<(super::DialogueManager, Vec<SerializedSession>)>> {
        if !self.sessions_path().exists() {
            return Ok(None);
        }

        let content =
            fs::read_to_string(self.sessions_path()).context("Failed to read sessions file")?;

        let storage: MemoryStorage =
            serde_json::from_str(&content).context("Failed to deserialize sessions")?;

        let dimension = storage.metadata.embedding_dim;

        let mut manager = super::DialogueManager {
            current_session: super::Session::new(persona_name.clone()),
            vector_store: VectorStore::new(dimension),
            embedder: embedder.clone(),
            session_history: HashMap::new(),
            max_sessions: 100,
        };

        for session in &storage.sessions {
            if session.persona_name == persona_name || manager.session_history.is_empty() {
                if let Ok(deserialized) = self.deserialize_session(session.clone()) {
                    manager
                        .session_history
                        .insert(deserialized.id, deserialized);
                }
            }
        }

        self.load_embeddings_binary(&mut manager, dimension, &storage.sessions)?;

        eprintln!(
            "DEBUG: Loaded {} sessions ({} turns) with embeddings from {:?}",
            storage.metadata.total_sessions, storage.metadata.total_turns, self.memory_dir
        );

        Ok(Some((manager, storage.sessions)))
    }

    fn load_embeddings_binary(
        &self,
        manager: &mut super::DialogueManager,
        embedding_dim: usize,
        sessions: &[SerializedSession],
    ) -> Result<()> {
        let embeddings_path = self.embeddings_path();
        if !embeddings_path.exists() {
            eprintln!("DEBUG: No embeddings file found");
            return Ok(());
        }

        let file_content = fs::read(&embeddings_path).context("Failed to read embeddings file")?;

        eprintln!("DEBUG: Embeddings file size: {} bytes", file_content.len());

        if file_content.len() < std::mem::size_of::<EmbeddingsHeader>() {
            anyhow::bail!(
                "Embeddings file is too small: {} < {}",
                file_content.len(),
                std::mem::size_of::<EmbeddingsHeader>()
            );
        }

        let header =
            EmbeddingsHeader::from_bytes(&file_content[..std::mem::size_of::<EmbeddingsHeader>()]);

        eprintln!(
            "DEBUG: Header - version: {}, dim: {}, num: {}, data_offset: {}",
            header.version, header.embedding_dim, header.num_embeddings, header.data_offset
        );

        let header_size = std::mem::size_of::<EmbeddingsHeader>();
        let index_size = std::mem::size_of::<EmbeddingIndex>();
        let expected_file_size =
            header.data_offset as usize + (header.num_embeddings as usize * embedding_dim * 4);

        eprintln!(
            "DEBUG: Expected file size: {} bytes (header: {}, index: {} entries, data: {} entries * {} dim * 4 bytes)",
            expected_file_size, header_size, header.num_embeddings,
            header.num_embeddings, embedding_dim
        );

        if file_content.len() < expected_file_size {
            eprintln!("DEBUG: File is smaller than expected, may be corrupted");
        }

        let index_start = std::mem::size_of::<EmbeddingsHeader>();
        let data_start = header.data_offset as usize;

        let mut offset = index_start;
        let mut loaded_count = 0;
        for _ in 0..header.num_embeddings {
            if offset + index_size > file_content.len() {
                eprintln!("DEBUG: Index out of bounds at offset {}", offset);
                break;
            }

            let index_bytes = &file_content[offset..offset + index_size];
            let index = EmbeddingIndex::from_bytes(index_bytes);

            let data_offset = data_start + index.offset as usize;
            let data_end = data_offset + (index.size as usize) * 4;

            if data_end > file_content.len() {
                eprintln!(
                    "DEBUG: Data out of bounds - offset: {}, end: {}, file_len: {}",
                    data_offset,
                    data_end,
                    file_content.len()
                );
                offset += index_size;
                continue;
            }

            let embedding_data = &file_content[data_offset..data_end];
            let embedding: Vec<f32> = embedding_data
                .chunks_exact(4)
                .map(|bytes| f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
                .collect();

            if embedding.len() == embedding_dim {
                let session = sessions.iter().find(|s| {
                    if let Ok(sid) = uuid::Uuid::parse_str(&s.id) {
                        sid == index.session_id
                    } else {
                        false
                    }
                });

                let (user_query, assistant_response) = match session {
                    Some(s) if (index.turn_idx as usize) < s.turns.len() => {
                        let turn = &s.turns[index.turn_idx as usize];
                        (turn.user.clone(), turn.assistant.clone())
                    }
                    _ => ("unknown".to_string(), "unknown".to_string()),
                };

                let memory_entry = MemoryEntry::new(
                    user_query.clone(),
                    embedding,
                    MemoryType::Episodic {
                        session_id: index.session_id,
                        turn: index.turn_idx as usize,
                    },
                )
                .with_metadata("session_id".to_string(), index.session_id.to_string())
                .with_metadata("turn".to_string(), index.turn_idx.to_string())
                .with_metadata("user_query".to_string(), user_query)
                .with_metadata("assistant_response".to_string(), assistant_response);

                manager.vector_store.add(memory_entry)?;
                loaded_count += 1;
            }

            offset += index_size;
        }

        eprintln!(
            "DEBUG: Restored {} embeddings to vector store (expected: {})",
            loaded_count, header.num_embeddings
        );

        Ok(())
    }

    pub fn load_sessions(&self) -> Result<Option<Vec<SerializedSession>>> {
        if !self.sessions_path().exists() {
            return Ok(None);
        }

        let content =
            fs::read_to_string(self.sessions_path()).context("Failed to read sessions file")?;

        let storage: MemoryStorage =
            serde_json::from_str(&content).context("Failed to deserialize sessions")?;

        eprintln!(
            "DEBUG: Loaded {} sessions ({} turns) from {:?}",
            storage.metadata.total_sessions, storage.metadata.total_turns, self.memory_dir
        );

        Ok(Some(storage.sessions))
    }

    fn serialize_session(&self, session: &super::Session) -> SerializedSession {
        SerializedSession {
            id: session.id.to_string(),
            persona_name: session.persona_name.clone(),
            turns: session
                .turns
                .iter()
                .map(|t| self.serialize_turn(t))
                .collect(),
            created_at: session.created_at,
            updated_at: session.updated_at,
            metadata: session.metadata.clone(),
        }
    }

    fn serialize_turn(&self, turn: &super::Turn) -> SerializedTurn {
        SerializedTurn {
            user: turn.user.clone(),
            assistant: turn.assistant.clone(),
            timestamp: turn.timestamp,
            metadata: turn.metadata.clone(),
            embedding: None,
        }
    }

    fn deserialize_session(&self, serialized: SerializedSession) -> Result<super::Session> {
        let id = Uuid::parse_str(&serialized.id)
            .with_context(|| format!("Invalid session UUID: {}", serialized.id))?;

        let turns: Vec<super::Turn> = serialized
            .turns
            .into_iter()
            .map(|t| super::Turn {
                user: t.user,
                assistant: t.assistant,
                timestamp: t.timestamp,
                metadata: t.metadata,
            })
            .collect();

        Ok(super::Session {
            id,
            persona_name: serialized.persona_name,
            turns,
            created_at: serialized.created_at,
            updated_at: serialized.updated_at,
            metadata: serialized.metadata,
        })
    }

    pub fn cleanup_old(&self, days_old: i64) -> Result<usize> {
        let cutoff = Utc::now() - chrono::Duration::days(days_old);

        if !self.sessions_path().exists() {
            return Ok(0);
        }

        let content =
            fs::read_to_string(self.sessions_path()).context("Failed to read sessions file")?;

        let mut storage: MemoryStorage =
            serde_json::from_str(&content).context("Failed to deserialize sessions")?;

        let before_count = storage.sessions.len();
        storage.sessions.retain(|s| s.updated_at > cutoff);

        if storage.sessions.len() < before_count {
            storage.metadata.total_sessions = storage.sessions.len();
            storage.metadata.total_turns = storage.sessions.iter().map(|s| s.turns.len()).sum();
            storage.metadata.last_saved_at = Utc::now();

            let sessions_content =
                serde_json::to_string_pretty(&storage).context("Failed to serialize sessions")?;
            fs::write(self.sessions_path(), sessions_content)
                .context("Failed to write sessions file")?;

            if let Ok(metadata_content) = serde_json::to_string_pretty(&storage.metadata) {
                let _ = fs::write(self.metadata_path(), metadata_content);
            }
        }

        Ok(before_count - storage.sessions.len())
    }

    pub fn get_stats(&self) -> Result<StorageMetadata> {
        if self.metadata_path().exists() {
            let content =
                fs::read_to_string(self.metadata_path()).context("Failed to read metadata file")?;
            Ok(serde_json::from_str(&content).context("Failed to deserialize metadata")?)
        } else {
            Ok(StorageMetadata::default())
        }
    }

    pub fn memory_dir(&self) -> &PathBuf {
        &self.memory_dir
    }
}

#[derive(Debug, Clone)]
struct EmbeddingsHeader {
    version: u32,
    embedding_dim: u32,
    num_embeddings: u64,
    index_offset: u64,
    data_offset: u64,
}

impl EmbeddingsHeader {
    fn to_bytes(&self) -> [u8; 32] {
        let mut bytes = [0u8; 32];
        bytes[0..4].copy_from_slice(&self.version.to_le_bytes());
        bytes[4..8].copy_from_slice(&self.embedding_dim.to_le_bytes());
        bytes[8..16].copy_from_slice(&self.num_embeddings.to_le_bytes());
        bytes[16..24].copy_from_slice(&self.index_offset.to_le_bytes());
        bytes[24..32].copy_from_slice(&self.data_offset.to_le_bytes());
        bytes
    }

    fn from_bytes(data: &[u8]) -> Self {
        let version = u32::from_le_bytes([data[0], data[1], data[2], data[3]]);
        let embedding_dim = u32::from_le_bytes([data[4], data[5], data[6], data[7]]);
        let num_embeddings = u64::from_le_bytes([
            data[8], data[9], data[10], data[11], data[12], data[13], data[14], data[15],
        ]);
        let index_offset = u64::from_le_bytes([
            data[16], data[17], data[18], data[19], data[20], data[21], data[22], data[23],
        ]);
        let data_offset = u64::from_le_bytes([
            data[24], data[25], data[26], data[27], data[28], data[29], data[30], data[31],
        ]);
        Self {
            version,
            embedding_dim,
            num_embeddings,
            index_offset,
            data_offset,
        }
    }
}

#[derive(Debug, Clone)]
struct EmbeddingIndex {
    session_id: Uuid,
    turn_idx: u32,
    offset: u64,
    size: u32,
}

impl EmbeddingIndex {
    fn to_bytes(&self) -> [u8; 32] {
        let mut bytes = [0u8; 32];
        let id_bytes = self.session_id.as_bytes();
        bytes[..16].copy_from_slice(id_bytes);
        bytes[16..20].copy_from_slice(&self.turn_idx.to_le_bytes());
        bytes[20..28].copy_from_slice(&self.offset.to_le_bytes());
        bytes[28..32].copy_from_slice(&self.size.to_le_bytes());
        bytes
    }

    fn from_bytes(data: &[u8]) -> Self {
        let mut id_bytes = [0u8; 16];
        id_bytes.copy_from_slice(&data[..16]);
        let session_id = Uuid::from_bytes(id_bytes);
        let turn_idx = u32::from_le_bytes([data[16], data[17], data[18], data[19]]);
        let offset = u64::from_le_bytes([
            data[20], data[21], data[22], data[23], data[24], data[25], data[26], data[27],
        ]);
        let size = u32::from_le_bytes([data[28], data[29], data[30], data[31]]);
        Self {
            session_id,
            turn_idx,
            offset,
            size,
        }
    }
}

pub fn create_dialogue_manager_with_sessions(
    embedder: Arc<dyn Embedder>,
    persona_name: String,
    sessions: Vec<SerializedSession>,
) -> super::DialogueManager {
    let dimension = embedder.embedding_dim();

    let mut manager = super::DialogueManager {
        current_session: super::Session::new(persona_name.clone()),
        vector_store: VectorStore::new(dimension),
        embedder: embedder.clone(),
        session_history: HashMap::new(),
        max_sessions: 100,
    };

    for session in sessions {
        if session.persona_name == persona_name || manager.session_history.is_empty() {
            if let Ok(deserialized) = deserialize_session_simple(session) {
                manager
                    .session_history
                    .insert(deserialized.id, deserialized);
            }
        }
    }

    manager
}

fn deserialize_session_simple(serialized: SerializedSession) -> Result<super::Session> {
    let id = Uuid::parse_str(&serialized.id)
        .with_context(|| format!("Invalid session UUID: {}", serialized.id))?;

    let turns: Vec<super::Turn> = serialized
        .turns
        .into_iter()
        .map(|t| super::Turn {
            user: t.user,
            assistant: t.assistant,
            timestamp: t.timestamp,
            metadata: t.metadata,
        })
        .collect();

    Ok(super::Session {
        id,
        persona_name: serialized.persona_name,
        turns,
        created_at: serialized.created_at,
        updated_at: serialized.updated_at,
        metadata: serialized.metadata,
    })
}

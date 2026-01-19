// TokenOutputStream moved from utils to logos for clean module structure
#![allow(dead_code)]

use anyhow::Result;
use tokenizers::Tokenizer;

/// A wrapper around a tokenizer to enable streaming token decoding.
pub struct TokenOutputStream {
    tokenizer: Tokenizer,
    tokens: Vec<u32>,
    prev_index: usize,
    current_index: usize,
}

impl TokenOutputStream {
    pub fn new(tokenizer: Tokenizer) -> Self {
        Self {
            tokenizer,
            tokens: Vec::new(),
            prev_index: 0,
            current_index: 0,
        }
    }

    fn decode(&self, tokens: &[u32]) -> Result<String> {
        self.tokenizer
            .decode(tokens, true)
            .map_err(|e| anyhow::anyhow!("decoding error: {}", e))
    }

    /// Returns newly decoded text if available, otherwise `None`.
    /// Mimics logic from Text Generation Inference.
    pub fn next_token(&mut self, token: u32) -> Result<Option<String>> {
        let prev_text = if self.tokens.is_empty() {
            String::new()
        } else {
            self.decode(&self.tokens[self.prev_index..self.current_index])?
        };

        self.tokens.push(token);
        let text = self.decode(&self.tokens[self.prev_index..])?;

        // Only emit if new text ends with an alphanumeric char (avoid partial UTF-8 or BPE artifacts)
        if text.len() > prev_text.len()
            && text
                .chars()
                .last()
                .map_or(false, |c| c.is_alphanumeric() || c.is_whitespace())
        {
            let (_, new_text) = text.split_at(prev_text.len());
            self.prev_index = self.current_index;
            self.current_index = self.tokens.len();
            Ok(Some(new_text.to_string()))
        } else {
            Ok(None)
        }
    }

    pub fn decode_rest(&self) -> Result<Option<String>> {
        let prev_text = if self.tokens.is_empty() {
            String::new()
        } else {
            self.decode(&self.tokens[self.prev_index..self.current_index])?
        };
        let text = self.decode(&self.tokens[self.prev_index..])?;
        if text.len() > prev_text.len() {
            let (_, rest) = text.split_at(prev_text.len());
            Ok(Some(rest.to_string()))
        } else {
            Ok(None)
        }
    }

    pub fn get_token(&self, token_str: &str) -> Option<u32> {
        self.tokenizer.get_vocab(true).get(token_str).copied()
    }

    pub fn tokenizer(&self) -> &Tokenizer {
        &self.tokenizer
    }

    pub fn clear(&mut self) {
        self.tokens.clear();
        self.prev_index = 0;
        self.current_index = 0;
    }
}

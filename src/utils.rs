use candle_core::Result;
use hf_hub::api::sync::ApiRepo;

// === SAFETENSORS LOADING ===

pub fn hub_load_safetensors(repo: &ApiRepo, json_file: &str) -> Result<Vec<std::path::PathBuf>> {
    let index_path = repo.get(json_file).map_err(candle_core::Error::wrap)?;
    let file = std::fs::File::open(&index_path)?;
    let json: serde_json::Value =
        serde_json::from_reader(&file).map_err(candle_core::Error::wrap)?;

    let weight_map = json
        .get("weight_map")
        .ok_or_else(|| candle_core::Error::msg(format!("no 'weight_map' in {json_file}")))?
        .as_object()
        .ok_or_else(|| candle_core::Error::msg("'weight_map' is not a JSON object"))?;

    let mut safetensors_files = std::collections::HashSet::new();
    for value in weight_map.values() {
        if let Some(filename) = value.as_str() {
            safetensors_files.insert(filename.to_string());
        }
    }

    safetensors_files
        .into_iter()
        .map(|filename| repo.get(&filename).map_err(candle_core::Error::wrap))
        .collect::<Result<Vec<_>>>()
}

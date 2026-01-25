/// Load narrative from disk
pub fn load(&mut self, archetype_id: &str) -> Result<()> {
    let path = format!("{}/{}.json", NARRATIVES_DIR, archetype_id);

    if !Path::new(&path).exists() {
        return Ok(()); // No saved narrative yet
    }

    let content = fs::read_to_string(&path)?;
    self.narrative = serde_json::from_str(&content)?;

    Ok(())
}

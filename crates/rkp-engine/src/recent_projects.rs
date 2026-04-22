//! Recent projects list — persists across sessions.

use std::path::PathBuf;

const MAX_RECENT: usize = 10;

fn config_path() -> PathBuf {
    let dir = dirs::config_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join("rkipatch");
    let _ = std::fs::create_dir_all(&dir);
    dir.join("recent_projects.json")
}

/// A recent project entry.
#[derive(Debug, Clone, PartialEq, Default, serde::Serialize, serde::Deserialize)]
pub struct RecentProject {
    pub name: String,
    pub path: String,
}

/// Load the recent projects list.
pub fn load_recent() -> Vec<RecentProject> {
    let path = config_path();
    let Ok(json) = std::fs::read_to_string(&path) else { return Vec::new() };
    serde_json::from_str(&json).unwrap_or_default()
}

/// Add a project to the recent list (moves to top if already present).
pub fn add_recent(name: &str, project_path: &str) {
    let mut recent = load_recent();
    recent.retain(|r| r.path != project_path);
    recent.insert(0, RecentProject {
        name: name.to_string(),
        path: project_path.to_string(),
    });
    recent.truncate(MAX_RECENT);
    let json = serde_json::to_string_pretty(&recent).unwrap_or_default();
    let _ = std::fs::write(config_path(), json);
}

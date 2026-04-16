//! Project file format — .rkproject files.
//!
//! A project is a directory containing a `.rkproject` JSON file,
//! a `scenes/` directory with `.rkscene` files, and an `assets/`
//! directory with `.rkp` models and materials.

use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

/// Project descriptor — serialized to `.rkproject` as JSON.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProjectFile {
    pub name: String,
    pub default_scene: String,
    #[serde(default)]
    pub recent_scenes: Vec<String>,
    /// Opaque editor layout blob (rinch docking + splitter sizes). The
    /// engine never inspects it — the editor produces and consumes this
    /// string through `SetEditorLayout` / `StateUpdate.editor_layout`.
    /// Absent on projects saved before layout persistence landed.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub editor_layout: Option<String>,
}

impl ProjectFile {
    /// Create a new project with a default scene.
    pub fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            default_scene: "default".to_string(),
            recent_scenes: Vec::new(),
            editor_layout: None,
        }
    }
}

/// Create a new project directory structure at the given path.
///
/// `path` should be the desired `.rkproject` file path. The project
/// is created inside a new subdirectory named after the project:
/// e.g. `/home/user/MyProject.rkproject` → `/home/user/MyProject/MyProject.rkproject`
///
/// Returns the project root directory.
pub fn create_project(path: &Path) -> Result<PathBuf, String> {
    let project_name = path.file_stem()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_else(|| "Untitled".to_string());

    // Create the project inside a new subdirectory named after the project.
    let parent = path.parent()
        .ok_or_else(|| "invalid project path".to_string())?;
    let project_dir = parent.join(&project_name);
    let project_file = project_dir.join(format!("{project_name}.rkproject"));

    // Create directory structure.
    std::fs::create_dir_all(project_dir.join("scenes"))
        .map_err(|e| format!("create scenes dir: {e}"))?;
    std::fs::create_dir_all(project_dir.join("assets/objects"))
        .map_err(|e| format!("create assets dir: {e}"))?;
    std::fs::create_dir_all(project_dir.join("assets/materials"))
        .map_err(|e| format!("create materials dir: {e}"))?;

    // Write project file.
    let project = ProjectFile::new(&project_name);
    let json = serde_json::to_string_pretty(&project)
        .map_err(|e| format!("serialize project: {e}"))?;
    std::fs::write(&project_file, &json)
        .map_err(|e| format!("write project file: {e}"))?;

    // Create default empty scene.
    let scene = crate::scene_io::SceneFile::new();
    crate::scene_io::save_scene(&scene, &project_dir.join("scenes/default.rkscene"))?;

    eprintln!("[RkpEngine] created project '{}' at {}", project_name, project_dir.display());
    Ok(project_dir.to_path_buf())
}

/// Load a project from a `.rkproject` file.
pub fn load_project(path: &Path) -> Result<(ProjectFile, PathBuf), String> {
    let json = std::fs::read_to_string(path)
        .map_err(|e| format!("read project file: {e}"))?;
    let project: ProjectFile = serde_json::from_str(&json)
        .map_err(|e| format!("parse project file: {e}"))?;
    let project_dir = path.parent()
        .ok_or_else(|| "invalid project path".to_string())?
        .to_path_buf();
    Ok((project, project_dir))
}

/// Save a project file.
pub fn save_project(project: &ProjectFile, path: &Path) -> Result<(), String> {
    let json = serde_json::to_string_pretty(project)
        .map_err(|e| format!("serialize project: {e}"))?;
    std::fs::write(path, &json)
        .map_err(|e| format!("write project file: {e}"))?;
    Ok(())
}

//! Helpers for displaying filesystem paths to the user.
//!
//! All engine-issued paths are absolute (canonicalized at load time
//! so cache keys are stable). Showing them absolute in the UI leaks
//! the user's home directory into every label and makes panel rows
//! wrap. These helpers strip the project-root prefix so paths render
//! as `assets/bunny.obj` instead of `/home/joe/dev/arrvox/assets/bunny.obj`.

/// Return `path` with the project root prefix removed. Falls back to
/// the original path when `project_root` is empty or doesn't prefix
/// `path` (e.g. a path outside the project tree, which shouldn't
/// normally occur but is harmless).
pub fn display_rel_path(path: &str, project_root: &str) -> String {
    if project_root.is_empty() {
        return path.to_string();
    }
    match path.strip_prefix(project_root) {
        Some(rest) => rest.trim_start_matches(['/', '\\']).to_string(),
        None => path.to_string(),
    }
}

/// Replace every occurrence of the project-root prefix inside `text`
/// with an empty string. For log/status messages that embed a full
/// path mid-sentence (`"Loading mesh: /home/joe/dev/arrvox/..."`).
/// Leaves text unchanged when `project_root` is empty.
///
/// Cheap allocation; called only on console entries and import
/// stage messages as they stream in.
pub fn relativize_paths_in_text(text: &str, project_root: &str) -> String {
    if project_root.is_empty() || !text.contains(project_root) {
        return text.to_string();
    }
    // Strip the prefix and the separator after it so `X/assets/bunny`
    // becomes `assets/bunny`, not `/assets/bunny`.
    let with_sep = format!("{project_root}/");
    let replaced = text.replace(&with_sep, "");
    // On Windows the separator could be a backslash; cheap second pass.
    let with_back_sep = format!("{project_root}\\");
    replaced.replace(&with_back_sep, "")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn display_rel_path_strips_project_root() {
        let out = display_rel_path("/home/joe/dev/foo/assets/bunny.obj", "/home/joe/dev/foo");
        assert_eq!(out, "assets/bunny.obj");
    }

    #[test]
    fn display_rel_path_handles_empty_root() {
        let out = display_rel_path("/abs/path", "");
        assert_eq!(out, "/abs/path");
    }

    #[test]
    fn display_rel_path_returns_original_when_not_prefixed() {
        let out = display_rel_path("/elsewhere/file", "/home/joe/dev/foo");
        assert_eq!(out, "/elsewhere/file");
    }

    #[test]
    fn relativize_paths_in_text_strips_embedded_prefix() {
        let out = relativize_paths_in_text(
            "Loading mesh: /home/joe/dev/foo/assets/bunny.obj",
            "/home/joe/dev/foo",
        );
        assert_eq!(out, "Loading mesh: assets/bunny.obj");
    }

    #[test]
    fn relativize_paths_in_text_noop_without_project_root() {
        let out = relativize_paths_in_text("anything /abs/path", "");
        assert_eq!(out, "anything /abs/path");
    }
}

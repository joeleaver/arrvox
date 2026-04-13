//! System scheduler — topological sort with dependency resolution.
//!
//! Given a list of `SystemEntry` items with `after`/`before` constraints,
//! produces a `Schedule` with ordered indices per phase.

use std::collections::HashMap;

use super::Phase;
use super::system_entry::SystemEntry;

/// Ordered system indices per phase.
pub struct Schedule {
    pub update: Vec<usize>,
    pub fixed_update: Vec<usize>,
    pub late_update: Vec<usize>,
}

/// Errors during schedule construction.
#[derive(Debug)]
pub enum ScheduleError {
    /// A dependency cycle was detected.
    Cycle(Vec<String>),
    /// A system declared a dependency that doesn't exist in any phase.
    MissingDependency {
        system: String,
        dependency: String,
    },
}

impl std::fmt::Display for ScheduleError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Cycle(names) => write!(f, "dependency cycle: {}", names.join(" → ")),
            Self::MissingDependency { system, dependency } => {
                write!(f, "system '{system}' depends on '{dependency}' which doesn't exist")
            }
        }
    }
}

impl std::error::Error for ScheduleError {}

/// Build a schedule from a list of system entry references.
///
/// Systems are sorted topologically within each phase using Kahn's algorithm.
/// Cross-phase dependencies are silently ignored (phases have implicit ordering).
pub fn build_schedule(systems: &[&SystemEntry]) -> Result<Schedule, ScheduleError> {
    let update = sort_phase(systems, Phase::Update)?;
    let fixed_update = sort_phase(systems, Phase::FixedUpdate)?;
    let late_update = sort_phase(systems, Phase::LateUpdate)?;
    Ok(Schedule { update, fixed_update, late_update })
}

/// Topological sort of systems within a single phase.
/// Returns indices into the original `systems` slice.
fn sort_phase(all_systems: &[&SystemEntry], phase: Phase) -> Result<Vec<usize>, ScheduleError> {
    // Collect systems in this phase.
    let phase_indices: Vec<usize> = all_systems.iter().enumerate()
        .filter(|(_, s)| s.phase == phase)
        .map(|(i, _)| i)
        .collect();

    let n = phase_indices.len();
    if n == 0 {
        return Ok(Vec::new());
    }

    // Map system names to their position within phase_indices.
    let mut name_to_pos: HashMap<&str, usize> = HashMap::new();
    for (pos, &idx) in phase_indices.iter().enumerate() {
        name_to_pos.insert(all_systems[idx].name, pos);
    }

    // Build names → position for ALL systems (for cross-phase dep checking).
    let all_names: HashMap<&str, usize> = all_systems.iter().enumerate()
        .map(|(i, s)| (s.name, i))
        .collect();

    // Build adjacency list + in-degree.
    let mut edges: Vec<Vec<usize>> = vec![Vec::new(); n];
    let mut in_degree: Vec<usize> = vec![0; n];

    for (pos, &idx) in phase_indices.iter().enumerate() {
        let sys = &all_systems[idx];

        // "after" deps: sys runs after dep → edge from dep to sys
        for dep_name in sys.after {
            if let Some(&dep_pos) = name_to_pos.get(dep_name) {
                edges[dep_pos].push(pos);
                in_degree[pos] += 1;
            } else if !all_names.contains_key(dep_name) {
                return Err(ScheduleError::MissingDependency {
                    system: sys.name.to_string(),
                    dependency: dep_name.to_string(),
                });
            }
            // Cross-phase dep: silently ignored (phases are implicitly ordered)
        }

        // "before" deps: sys runs before dep → edge from sys to dep
        for dep_name in sys.before {
            if let Some(&dep_pos) = name_to_pos.get(dep_name) {
                edges[pos].push(dep_pos);
                in_degree[dep_pos] += 1;
            } else if !all_names.contains_key(dep_name) {
                return Err(ScheduleError::MissingDependency {
                    system: sys.name.to_string(),
                    dependency: dep_name.to_string(),
                });
            }
        }
    }

    // Kahn's algorithm.
    let mut queue: Vec<usize> = (0..n).filter(|&i| in_degree[i] == 0).collect();
    let mut order = Vec::with_capacity(n);

    while let Some(node) = queue.pop() {
        order.push(phase_indices[node]);
        for &neighbor in &edges[node] {
            in_degree[neighbor] -= 1;
            if in_degree[neighbor] == 0 {
                queue.push(neighbor);
            }
        }
    }

    if order.len() != n {
        let cycle_names: Vec<String> = (0..n)
            .filter(|&i| in_degree[i] > 0)
            .map(|i| all_systems[phase_indices[i]].name.to_string())
            .collect();
        return Err(ScheduleError::Cycle(cycle_names));
    }

    Ok(order)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sys(name: &'static str, phase: Phase, after: &'static [&'static str], before: &'static [&'static str]) -> SystemEntry {
        SystemEntry {
            name,
            module_path: "",
            phase,
            after,
            before,
            fn_ptr: std::ptr::null(),
        }
    }

    fn refs(systems: &[SystemEntry]) -> Vec<&SystemEntry> {
        systems.iter().collect()
    }

    #[test]
    fn empty_schedule() {
        let schedule = build_schedule(&refs(&[])).unwrap();
        assert!(schedule.update.is_empty());
        assert!(schedule.fixed_update.is_empty());
        assert!(schedule.late_update.is_empty());
    }

    #[test]
    fn single_system_per_phase() {
        let systems = [
            sys("a", Phase::Update, &[], &[]),
            sys("b", Phase::FixedUpdate, &[], &[]),
            sys("c", Phase::LateUpdate, &[], &[]),
        ];
        let schedule = build_schedule(&refs(&systems)).unwrap();
        assert_eq!(schedule.update, vec![0]);
        assert_eq!(schedule.fixed_update, vec![1]);
        assert_eq!(schedule.late_update, vec![2]);
    }

    #[test]
    fn dependency_ordering() {
        let systems = [
            sys("b", Phase::Update, &["a"], &[]),
            sys("a", Phase::Update, &[], &[]),
        ];
        let schedule = build_schedule(&refs(&systems)).unwrap();
        let a_pos = schedule.update.iter().position(|&i| i == 1).unwrap();
        let b_pos = schedule.update.iter().position(|&i| i == 0).unwrap();
        assert!(a_pos < b_pos);
    }

    #[test]
    fn before_constraint() {
        let systems = [
            sys("a", Phase::Update, &[], &["b"]),
            sys("b", Phase::Update, &[], &[]),
        ];
        let schedule = build_schedule(&refs(&systems)).unwrap();
        let a_pos = schedule.update.iter().position(|&i| i == 0).unwrap();
        let b_pos = schedule.update.iter().position(|&i| i == 1).unwrap();
        assert!(a_pos < b_pos);
    }

    #[test]
    fn cycle_detection() {
        let systems = [
            sys("a", Phase::Update, &["b"], &[]),
            sys("b", Phase::Update, &["a"], &[]),
        ];
        assert!(matches!(build_schedule(&refs(&systems)), Err(ScheduleError::Cycle(_))));
    }

    #[test]
    fn missing_dependency() {
        let systems = [
            sys("a", Phase::Update, &["nonexistent"], &[]),
        ];
        assert!(matches!(
            build_schedule(&refs(&systems)),
            Err(ScheduleError::MissingDependency { .. })
        ));
    }

    #[test]
    fn cross_phase_dep_ignored() {
        let systems = [
            sys("a", Phase::Update, &["b"], &[]),
            sys("b", Phase::LateUpdate, &[], &[]),
        ];
        let schedule = build_schedule(&refs(&systems)).unwrap();
        assert_eq!(schedule.update, vec![0]);
        assert_eq!(schedule.late_update, vec![1]);
    }

    #[test]
    fn diamond_dependency() {
        let systems = [
            sys("a", Phase::Update, &[], &[]),
            sys("b", Phase::Update, &["a"], &[]),
            sys("c", Phase::Update, &["a"], &[]),
            sys("d", Phase::Update, &["b", "c"], &[]),
        ];
        let schedule = build_schedule(&refs(&systems)).unwrap();
        let pos = |name: &str| -> usize {
            let idx = systems.iter().position(|s| s.name == name).unwrap();
            schedule.update.iter().position(|&i| i == idx).unwrap()
        };
        assert!(pos("a") < pos("b"));
        assert!(pos("a") < pos("c"));
        assert!(pos("b") < pos("d"));
        assert!(pos("c") < pos("d"));
    }
}

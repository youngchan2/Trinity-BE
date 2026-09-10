use crate::Storage;

/// Row-major element offset. Global addressing widens before coordinate addition
/// and stride multiplication; shared/register tiles retain local index arithmetic.
pub(super) fn offset(
    shape: &[usize],
    origin: &[String],
    coordinates: &[String],
    storage: Storage,
) -> String {
    origin
        .iter()
        .zip(coordinates)
        .enumerate()
        .map(|(i, (origin, coordinate))| {
            let origin = if matches!(storage, Storage::External | Storage::Global) {
                format!("std::int64_t({origin})")
            } else {
                origin.clone()
            };
            format!(
                "(({origin})+({coordinate}))*{}",
                shape[i + 1..].iter().product::<usize>()
            )
        })
        .collect::<Vec<_>>()
        .join("+")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn global_offsets_widen_before_addition_and_multiplication() {
        let mut checks = String::from(
            "#include <cstdint>\n#include <type_traits>\n\
             constexpr int row = 65535, col = 65535;\n",
        );
        for storage in [Storage::External, Storage::Global] {
            let offset = offset(
                &[65536, 65536],
                &["row".into(), "col".into()],
                &["0".into(), "0".into()],
                storage,
            );
            checks.push_str(&format!("static_assert(({offset}) == 4294967295LL);\n"));
            let offset = super::offset(
                &[2147483776],
                &["2147483584".into()],
                &["128".into()],
                storage,
            );
            checks.push_str(&format!("static_assert(({offset}) == 2147483712LL);\n"));
        }
        let local = offset(
            &[128, 64],
            &["0".into(), "0".into()],
            &["127".into(), "63".into()],
            Storage::Shared,
        );
        checks.push_str(&format!(
            "static_assert(({local}) == 8191);\n\
             static_assert(std::is_same_v<decltype({local}), int>);\n"
        ));

        let directory = tempfile::tempdir().unwrap();
        let source = directory.path().join("offsets.cpp");
        std::fs::write(&source, checks).unwrap();
        let output = std::process::Command::new("c++")
            .args(["-std=c++17", "-fsyntax-only"])
            .arg(source)
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
    }
}

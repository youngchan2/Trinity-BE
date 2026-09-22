//! Emit the supplied postprocessed MLA stages without importing Triton or using a GPU.
use std::{fs, path::PathBuf, process::Command};

use trinity_lowering::triton::{Options, compile};

fn emit_stage(stage: usize, kernel_count: usize) {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let fixtures = root.join("tests/fixtures/batched_mla");
    let input = fixtures.join(format!("batched_mla_postprocessed_stage{stage}.txt"));
    let mut options = Options {
        symbols: [
            ("tile_b".into(), 1),
            ("tile_m".into(), 32),
            ("tile_q_h_q".into(), 16),
            ("tile_p".into(), 16),
        ]
        .into(),
        ..Options::default()
    };
    for line in include_str!("fixtures/batched_mla/shapes.txt").lines() {
        let mut words = line.split_whitespace();
        let name = words.next().unwrap();
        let shape = words.map(|dim| dim.parse().unwrap()).collect();
        options.shapes.insert(name.into(), shape);
    }
    let ir = fs::read_to_string(&input).unwrap();
    let source =
        compile(&ir, options).unwrap_or_else(|error| panic!("{}: {error}", input.display()));
    let directory = root.join("target/tests/batched_mla");
    fs::create_dir_all(&directory).unwrap();
    let output = directory.join(format!("stage{stage}.py"));
    fs::write(&output, source).unwrap();

    // Compile Python syntax without executing imports, autotuning, or launches.
    let check = Command::new("python3")
        .arg("-c")
        .arg(
            r#"
import ast
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
tree = ast.parse(path.read_text(), filename=str(path))
compile(tree, str(path), "exec")
functions = {node.name for node in tree.body if isinstance(node, ast.FunctionDef)}
kernels = {name for name in functions if name.startswith("kernel_")}
assert kernels == {f"kernel_{i}" for i in range(int(sys.argv[2]))}, kernels
assert "forward" in functions, "missing launch wrapper"
"#,
        )
        .arg(&output)
        .arg(kernel_count.to_string())
        .output()
        .expect("python3 is required to check emitted source syntax");
    assert!(
        check.status.success(),
        "{}: {}",
        output.display(),
        String::from_utf8_lossy(&check.stderr)
    );
    println!("{} -> {}", input.display(), output.display());
}

#[test]
fn emit_stage14() {
    emit_stage(14, 3);
}

#[test]
fn emit_stage16() {
    emit_stage(16, 2);
}

#[test]
fn emit_stage20() {
    emit_stage(20, 3);
}

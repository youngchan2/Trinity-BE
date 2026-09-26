//! File-to-file entry point; analysis and rendering stay in the library.
use serde::Deserialize;
use std::{collections::BTreeMap, error::Error, fs, path::PathBuf};
use trinity_lowering::{
    CudaTargetCapability, DType, PhysicalPlanBuilder, ScheduledConfig, TargetCapability,
    analysis::{Bindings, analyze_text},
    emit::{PythonProvider, PythonSelection, emit_python_executable},
};

const HELP: &str = "Usage: emit_python INPUT.ir OUTPUT.py [options]
  --target sm120|sm90|sm89         CUDA target (default sm120)
  --dtype bf16|fp16|fp32          Default storage dtype (default bf16)
  --selection prefer-quack|triton Support-based policy, not benchmarking
  --providers quack,triton,...   Explicit choice for each original region
  --bindings FILE.json           Optional shapes, symbols and per-tensor dtypes

Writes OUTPUT.py and OUTPUT.selection.json. GPU compilation occurs on first
forward() invocation. Unsupported selected providers cause an error.";

#[derive(Default, Deserialize)]
#[serde(default, deny_unknown_fields)]
struct FileBindings {
    shapes: BTreeMap<String, Vec<usize>>,
    symbols: BTreeMap<String, i64>,
    dtypes: BTreeMap<String, String>,
}

fn parse_dtype(value: &str) -> Result<DType, Box<dyn Error>> {
    match value {
        "bf16" => Ok(DType::Bf16),
        "fp16" => Ok(DType::Fp16),
        "fp32" => Ok(DType::Fp32),
        _ => Err(format!("unsupported dtype {value}").into()),
    }
}

fn run() -> Result<(), Box<dyn Error>> {
    let mut args = std::env::args().skip(1);
    let Some(input) = args.next() else {
        return Err(HELP.into());
    };
    if input == "--help" || input == "-h" {
        println!("{HELP}");
        return Ok(());
    }
    let output = PathBuf::from(args.next().ok_or(HELP)?);
    if output == std::path::Path::new(&input) {
        return Err("input and output must differ".into());
    }
    let mut target = CudaTargetCapability::Sm120;
    let mut dtype = DType::Bf16;
    let mut selection = PythonSelection::PreferQuack;
    let mut bindings = FileBindings::default();
    let mut selection_set = false;
    while let Some(key) = args.next() {
        let value = args
            .next()
            .ok_or_else(|| format!("missing value for {key}"))?;
        match key.as_str() {
            "--target" => {
                target = match value.as_str() {
                    "sm120" => CudaTargetCapability::Sm120,
                    "sm90" => CudaTargetCapability::Hopper,
                    "sm89" => CudaTargetCapability::Sm89,
                    _ => return Err(format!("unsupported target {value}").into()),
                }
            }
            "--dtype" => dtype = parse_dtype(&value)?,
            "--selection" | "--providers" => {
                if selection_set {
                    return Err("specify selection only once".into());
                }
                selection_set = true;
                selection = if key == "--providers" {
                    PythonSelection::Regions(
                        value
                            .split(',')
                            .map(|p| match p {
                                "quack" => Ok(PythonProvider::Quack),
                                "triton" => Ok(PythonProvider::Triton),
                                _ => Err(format!("unsupported provider {p}")),
                            })
                            .collect::<Result<_, _>>()?,
                    )
                } else {
                    match value.as_str() {
                        "prefer-quack" => PythonSelection::PreferQuack,
                        "triton" => PythonSelection::TritonOnly,
                        _ => return Err(format!("unsupported selection {value}").into()),
                    }
                };
            }
            "--bindings" => bindings = serde_json::from_str(&fs::read_to_string(value)?)?,
            _ => return Err(format!("unknown option {key}").into()),
        }
    }
    let ir = analyze_text(&fs::read_to_string(input)?)?;
    let plan = PhysicalPlanBuilder::from_scheduled(
        &ir,
        ScheduledConfig {
            target: TargetCapability::Cuda(target),
            default_dtype: dtype,
            dtypes: bindings
                .dtypes
                .iter()
                .map(|(name, value)| Ok((name.clone(), parse_dtype(value)?)))
                .collect::<Result<_, Box<dyn Error>>>()?,
            bindings: Bindings {
                shapes: bindings.shapes,
                symbols: bindings.symbols,
            },
        },
    )?;
    let program = emit_python_executable(&plan, selection)?;
    if let Some(parent) = output.parent().filter(|p| !p.as_os_str().is_empty()) {
        fs::create_dir_all(parent)?;
    }
    let report = output.with_extension("selection.json");
    fs::write(&output, program.emit())?;
    fs::write(
        &report,
        serde_json::to_string_pretty(program.report())? + "\n",
    )?;
    println!(
        "source: {}\nselection: {}\nproviders: {:?}",
        output.display(),
        report.display(),
        program.providers()
    );
    Ok(())
}

fn main() {
    if let Err(error) = run() {
        eprintln!("{error}");
        std::process::exit(1);
    }
}

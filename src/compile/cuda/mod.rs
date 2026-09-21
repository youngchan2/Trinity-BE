//! Linux NVCC shared-library compilation without runtime loading or execution.

mod artifact;
mod config;
mod error;

use std::ffi::OsString;
use std::path::Path;
use std::process::Command;

use crate::{AsStr, CudaSource, CudaTargetCapability};
pub use artifact::CudaArtifact;
pub use config::CompileConfig;
pub use error::{CompileDiagnostics, CompileError};

const EXPORTS: &str = "{ global: trinity_abi; trinity_prepare; trinity_launch; trinity_status; trinity_release; local: *; };\n";

pub fn compile(source: CudaSource) -> Result<CudaArtifact, CompileError> {
    compile_with_config(source, &CompileConfig::default())
}

/// Build generated code without loading CUDA libraries or preparing a GPU.
/// Uses a private temporary directory per invocation; no cache is consulted.
pub fn compile_with_config(
    source: CudaSource,
    config: &CompileConfig,
) -> Result<CudaArtifact, CompileError> {
    if !cfg!(target_os = "linux") {
        return Err(CompileError::Configuration(
            "CUDA compilation currently requires Linux".into(),
        ));
    }

    let requirements = source.requirements();
    let toolchain = config::resolve(config, requirements.nvshmem)?;
    let version = invoke(&toolchain.compiler, vec!["--version".into()], source.code())?;

    let stdout = String::from_utf8_lossy(&version.stdout);
    let major = stdout
        .split("release ")
        .nth(1)
        .and_then(|v| v.split('.').next())
        .and_then(|v| v.parse::<u32>().ok());

    if major.is_none_or(|v| v < 13) {
        return Err(CompileError::CompilerVersion {
            diagnostics: Box::new(version),
        });
    }

    let nvcc_version = String::from_utf8_lossy(&version.stdout).into_owned();

    let workspace = tempfile::Builder::new()
        .prefix("trinity-cuda-")
        .tempdir()
        .map_err(|source| CompileError::Io {
            operation: "create compilation directory",
            path: std::env::temp_dir(),
            source,
        })?;

    let input = workspace.path().join("source.cu");
    let artifact = workspace.path().join("program.so");
    let exports = workspace.path().join("exports.map");

    write(&input, source.code())?;
    write(&exports, EXPORTS)?;

    let arguments = arguments(&toolchain, requirements.target, &input, &artifact, &exports);
    let diagnostics = invoke(&toolchain.compiler, arguments, source.code())?;

    if !artifact.is_file() {
        return Err(CompileError::MissingArtifact {
            diagnostics: Box::new(diagnostics),
        });
    }

    let manifest = serde_json::json!({
        "schema_version": 1,
        "host_abi_version": 1,
        "execution": if requirements.world_size == 1 { "streamed" } else { "persistent" },
        "library": "program.so",
        "source": "source.cu",
        "requirements": requirements,
        "toolchain": {
            "nvcc": nvcc_version,
            "cuda_major": major.unwrap(),
            "cuda_library_directory": toolchain.cuda_lib,
            "cutlass": "4.5.1",
            "nvshmem": requirements.nvshmem.then_some("3.7.2"),
            "nvshmem_library_directory": toolchain.nvshmem.as_ref().map(|(_, path)| path),
            "host_compiler": toolchain.host_compiler,
        },
    });
    let manifest = serde_json::to_string(&manifest).expect("serializable artifact metadata");
    write(&workspace.path().join("manifest.json"), &manifest)?;

    // Arrays preserve compiler output bytes, including non-UTF8 diagnostics.
    let saved_diagnostics = serde_json::json!({
        "compiler": diagnostics.compiler,
        "arguments": diagnostics.arguments.iter().map(|s| s.to_string_lossy()).collect::<Vec<_>>(),
        "exit_code": diagnostics.status.and_then(|s| s.code()),
        "stdout": diagnostics.stdout,
        "stderr": diagnostics.stderr,
    });
    write(
        &workspace.path().join("diagnostics.json"),
        &saved_diagnostics.to_string(),
    )?;

    Ok(CudaArtifact::new(
        source,
        workspace,
        artifact,
        diagnostics,
        manifest,
    ))
}

fn write(path: &Path, contents: &str) -> Result<(), CompileError> {
    std::fs::write(path, contents).map_err(|source| CompileError::Io {
        operation: "write compilation input",
        path: path.to_owned(),
        source,
    })
}

fn invoke(
    compiler: &Path,
    arguments: Vec<OsString>,
    source: &str,
) -> Result<CompileDiagnostics, CompileError> {
    let mut diagnostics = CompileDiagnostics {
        compiler: compiler.to_owned(),
        arguments,
        generated_source: source.to_owned(),
        status: None,
        stdout: vec![],
        stderr: vec![],
    };

    let output = match Command::new(compiler).args(&diagnostics.arguments).output() {
        Ok(output) => output,
        Err(source) => {
            return Err(CompileError::CompilerSpawn {
                source,
                diagnostics: Box::new(diagnostics),
            });
        }
    };

    diagnostics.status = Some(output.status);
    diagnostics.stdout = output.stdout;
    diagnostics.stderr = output.stderr;

    if !output.status.success() {
        return Err(CompileError::CompilerFailed {
            diagnostics: Box::new(diagnostics),
        });
    }

    Ok(diagnostics)
}

fn arguments(
    toolchain: &config::Toolchain,
    target: CudaTargetCapability,
    input: &Path,
    artifact: &Path,
    exports: &Path,
) -> Vec<OsString> {
    let mut args: Vec<OsString> = [
        "--shared",
        "--std=c++17",
        "-O3",
        &format!("-arch={}", target.as_str()),
        "--expt-relaxed-constexpr",
        "-Xcompiler=-fPIC",
        "--cudart=shared",
    ]
    .into_iter()
    .map(Into::into)
    .collect();

    if let Some(compiler) = &toolchain.host_compiler {
        args.extend(["-ccbin".into(), compiler.into()]);
    }

    args.extend(["-I".into(), toolchain.cutlass_include.clone().into()]);
    if let Some((include, _)) = &toolchain.nvshmem {
        args.extend([
            "--relocatable-device-code=true".into(),
            "-I".into(),
            include.into(),
        ]);
    }

    args.push(input.into());
    args.extend(["-L".into(), toolchain.cuda_lib.clone().into()]);

    if let Some((_, library)) = &toolchain.nvshmem {
        args.extend([
            library.join("libnvshmem_device.a").into(),
            library.join("libnvshmem_host.so").into(),
        ]);
        rpath(&mut args, library);
    }

    if let Some(stubs) = &toolchain.driver_link_dir {
        args.extend(["-L".into(), stubs.into()]);
    }

    args.push("-lcuda".into());
    rpath(&mut args, &toolchain.cuda_lib);
    args.extend([
        "-Xlinker".into(),
        "--version-script".into(),
        "-Xlinker".into(),
        exports.into(),
        "-Xlinker".into(),
        "-Bsymbolic-functions".into(),
        "-Xlinker".into(),
        "-z".into(),
        "-Xlinker".into(),
        "defs".into(),
        "-o".into(),
        artifact.into(),
    ]);

    args
}

fn rpath(args: &mut Vec<OsString>, path: &Path) {
    args.extend([
        "-Xlinker".into(),
        "-rpath".into(),
        "-Xlinker".into(),
        path.into(),
    ]);
}

#[cfg(all(test, target_os = "linux"))]
mod tests;

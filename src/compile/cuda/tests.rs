use std::fs;
use std::os::unix::fs::PermissionsExt;
use std::path::{Path, PathBuf};
use std::process::Command;

use tempfile::TempDir;

use super::*;
use crate::{
    ComputeOperation, DType, OperationPayload, PhysicalPlanBuilder, Storage, TargetCapability,
    emit, gemm_implementations,
};

fn source(world: usize) -> CudaSource {
    let target = TargetCapability::Cuda(CudaTargetCapability::Hopper);
    let mut b = PhysicalPlanBuilder::new(target, world);
    let a = b.add_value(DType::Bf16, [128, 64], Storage::External);
    let w = b.add_value(DType::Bf16, [64, 128], Storage::External);
    let y = b.add_value(DType::Bf16, [128, 128], Storage::External);

    b.bind_input("a", a);
    b.bind_input("w", w);

    let instance = gemm_implementations(target)[0]
        .enumerate([DType::Bf16; 3], [&[128, 64], &[64, 128], &[128, 128]])
        .pop()
        .unwrap();
    let op = b.add_operation(
        [a, w],
        [y],
        OperationPayload::Compute(ComputeOperation::new(instance)),
    );
    b.add_statement(crate::Statement::Operation(op));

    emit(&b.finalize("y", y).unwrap()).unwrap()
}

fn write(path: &Path, text: &str) {
    fs::create_dir_all(path.parent().unwrap()).unwrap();
    fs::write(path, text).unwrap();
}

struct Sdk {
    root: TempDir,
    config: CompileConfig,
}

impl Sdk {
    fn new() -> Self {
        let root = tempfile::Builder::new()
            .prefix("trinity sdk space-")
            .tempdir()
            .unwrap();

        let p = root.path();
        write(&p.join("cuda/bin/nvcc"), include_str!("tests/fake_nvcc.py"));

        fs::set_permissions(p.join("cuda/bin/nvcc"), fs::Permissions::from_mode(0o700)).unwrap();
        write(&p.join("cuda/lib64/libcudart.so"), "");
        write(&p.join("cuda/lib64/stubs/libcuda.so"), "");
        write(
            &p.join("cutlass/include/cutlass/version.h"),
            "#define CUTLASS_MAJOR 4\n#define CUTLASS_MINOR 5\n#define CUTLASS_PATCH 1\n",
        );
        write(&p.join("cutlass/include/cute/tensor.hpp"), "");
        write(
            &p.join("nvshmem/include/non_abi/nvshmem_version.h"),
            "#define NVSHMEM_VENDOR_MAJOR_VERSION 3\n#define NVSHMEM_VENDOR_MINOR_VERSION 7\n#define NVSHMEM_VENDOR_PATCH_VERSION 2\n",
        );
        write(&p.join("nvshmem/include/nvshmem.h"), "");
        write(&p.join("nvshmem/include/nvshmemx.h"), "");
        write(&p.join("nvshmem/lib/libnvshmem_host.so"), "");
        write(&p.join("nvshmem/lib/libnvshmem_device.a"), "");

        write(&p.join("mode"), "success");
        write(
            &p.join("fixture.cpp"),
            r#"
            extern "C" void* trinity_abi() { return nullptr; }
            extern "C" int hidden_helper() { return 999; }
            extern "C" int trinity_prepare(unsigned* maximum) { *maximum = 1; return 0; }
            extern "C" int trinity_launch(void const*) { unsigned n; return trinity_prepare(&n); }
            extern "C" int trinity_status(void*) { return 0; }
            extern "C" int trinity_release() { return 0; }
        "#,
        );

        let config = CompileConfig {
            cuda_root: Some(p.join("cuda")),
            cutlass_root: Some(p.join("cutlass")),
            nvshmem_root: Some(p.join("nvshmem")),
            host_compiler: None,
        };

        Self { root, config }
    }

    fn mode(&self, mode: &str) {
        write(&self.root.path().join("mode"), mode);
    }

    fn compile(&self, world: usize) -> Result<CudaArtifact, CompileError> {
        compile_with_config(source(world), &self.config)
    }
}

#[test]
fn builds_artifacts_with_source_requirements_and_diagnostics() {
    let sdk = Sdk::new();
    for world in [1, 2] {
        let artifact = sdk.compile(world).unwrap();

        assert_eq!(artifact.requirements().world_size, world);
        assert_eq!(artifact.requirements().nvshmem, world > 1);

        let diagnostics = artifact.diagnostics();
        assert_eq!(artifact.source().code(), diagnostics.generated_source);
        assert!(diagnostics.status.unwrap().success());
        assert!(String::from_utf8_lossy(&diagnostics.stdout).contains("fixture compiler stdout"));
        assert!(String::from_utf8_lossy(&diagnostics.stderr).contains("fixture compiler warning"));
        assert_eq!(
            &fs::read(artifact.artifact_path()).unwrap()[..4],
            b"\x7fELF"
        );
        assert_eq!(
            fs::read_to_string(artifact.directory().join("source.cu")).unwrap(),
            artifact.source().code()
        );

        let manifest: serde_json::Value = serde_json::from_str(artifact.manifest()).unwrap();
        assert_eq!(manifest["host_abi_version"], 1);
        assert_eq!(manifest["requirements"]["world_size"], world);
        assert_eq!(
            fs::read_to_string(artifact.directory().join("manifest.json")).unwrap(),
            artifact.manifest()
        );
        assert!(artifact.directory().join("diagnostics.json").is_file());

        let directory = artifact.directory().to_owned();
        drop(artifact);
        assert!(!directory.exists());
    }
}

#[test]
fn compile_never_loads_code_and_persist_transfers_file_ownership() {
    let sdk = Sdk::new();

    // Deliberately not loadable: compiler success + artifact presence are the
    // boundary here. No ELF loading or runtime initialization belongs to compile.
    sdk.mode("invalid");

    let artifact = sdk.compile(1).unwrap();
    let directory = artifact.persist();
    assert_eq!(
        fs::read_to_string(directory.join("program.so")).unwrap(),
        "not an ELF library"
    );
    assert!(directory.join("source.cu").is_file());

    fs::remove_dir_all(directory).unwrap();
}

#[test]
fn failures_keep_process_diagnostics_and_remove_temporary_files() {
    let sdk = Sdk::new();
    for mode in ["failure", "missing"] {
        sdk.mode(mode);
        let error = sdk.compile(1).unwrap_err();

        assert!(match mode {
            "failure" => matches!(&error, CompileError::CompilerFailed { .. }),
            _ => matches!(&error, CompileError::MissingArtifact { .. }),
        });

        let diagnostics = error.diagnostics().unwrap();
        assert!(diagnostics.generated_source.contains("trinity_launch"));
        assert!(String::from_utf8_lossy(&diagnostics.stdout).contains("fixture compiler stdout"));
        assert!(String::from_utf8_lossy(&diagnostics.stderr).contains("fixture compiler warning"));
        if mode == "failure" {
            assert_eq!(diagnostics.status.unwrap().code(), Some(42));
        }

        let output = PathBuf::from(diagnostics.arguments.last().unwrap());
        assert!(!output.parent().unwrap().exists());
    }

    fs::set_permissions(
        sdk.root.path().join("cuda/bin/nvcc"),
        fs::Permissions::from_mode(0o600),
    )
    .unwrap();
    let error = sdk.compile(1).unwrap_err();
    assert!(matches!(&error, CompileError::CompilerSpawn { .. }));

    let diagnostics = error.diagnostics().unwrap();
    assert_eq!(diagnostics.arguments, [OsString::from("--version")]);
    assert!(diagnostics.status.is_none());
    assert!(diagnostics.generated_source.contains("trinity_launch"));
}

#[test]
fn compiler_arguments_preserve_paths_and_export_only_the_host_abi() {
    let mut sdk = Sdk::new();
    let host = sdk.root.path().join("host compiler");
    write(&host, "fixture");
    sdk.config.host_compiler = Some(host.clone());

    let artifact = sdk.compile(2).unwrap();
    let args = &artifact.diagnostics().arguments;
    assert!(args.windows(2).any(|w| w[0] == "-ccbin" && w[1] == host));
    for flag in [
        "--shared",
        "--std=c++17",
        "-O3",
        "-arch=sm_90a",
        "--expt-relaxed-constexpr",
        "-Xcompiler=-fPIC",
        "--cudart=shared",
        "--relocatable-device-code=true",
        "-lcuda",
        "-Bsymbolic-functions",
    ] {
        assert!(args.iter().any(|arg| arg == flag), "missing {flag}");
    }
    for relative in [
        "cutlass/include",
        "nvshmem/include",
        "nvshmem/lib/libnvshmem_host.so",
        "nvshmem/lib/libnvshmem_device.a",
        "cuda/lib64/stubs",
    ] {
        assert!(
            args.iter()
                .any(|arg| Path::new(arg) == sdk.root.path().join(relative))
        );
    }

    let rpaths = args
        .windows(4)
        .filter(|w| w[0] == "-Xlinker" && w[1] == "-rpath")
        .map(|w| PathBuf::from(&w[3]))
        .collect::<Vec<_>>();
    assert_eq!(
        rpaths,
        [
            sdk.root.path().join("nvshmem/lib"),
            sdk.root.path().join("cuda/lib64")
        ]
    );

    let output = Command::new("nm")
        .args(["-D", "--defined-only"])
        .arg(artifact.artifact_path())
        .output()
        .unwrap();
    assert!(output.status.success());
    let symbols = String::from_utf8(output.stdout).unwrap();
    let symbols = symbols
        .lines()
        .filter_map(|line| line.split_whitespace().last())
        .collect::<std::collections::BTreeSet<_>>();
    assert_eq!(
        symbols,
        [
            "trinity_abi",
            "trinity_launch",
            "trinity_prepare",
            "trinity_release",
            "trinity_status"
        ]
        .into_iter()
        .collect()
    );
}

#[test]
fn explicit_paths_are_validated_and_single_gpu_ignores_nvshmem() {
    let mut sdk = Sdk::new();
    sdk.config.nvshmem_root = Some(PathBuf::from("/definitely/missing/nvshmem"));

    let artifact = sdk.compile(1).unwrap();
    assert!(
        !artifact
            .diagnostics()
            .arguments
            .iter()
            .any(|a| a.to_string_lossy().contains("nvshmem"))
    );
    assert!(matches!(sdk.compile(2), Err(CompileError::Io { .. })));

    sdk.config.cuda_root = Some(PathBuf::from("/definitely/missing/cuda"));
    assert!(matches!(
        sdk.compile(1),
        Err(CompileError::Configuration(_))
    ));
}

#[test]
fn toolchain_versions_are_rejected_with_actionable_diagnostics() {
    let sdk = Sdk::new();
    write(
        &sdk.root.path().join("version"),
        "Cuda compilation tools, release 12.8, V12.8.1",
    );
    let error = sdk.compile(1).unwrap_err();
    assert!(matches!(error, CompileError::CompilerVersion { .. }));

    let diagnostics = error.diagnostics().unwrap();
    assert_eq!(diagnostics.arguments, [OsString::from("--version")]);
    assert!(diagnostics.generated_source.contains("trinity_launch"));

    fs::remove_file(sdk.root.path().join("version")).unwrap();
    write(
        &sdk.root.path().join("cutlass/include/cutlass/version.h"),
        "#define CUTLASS_MAJOR 3\n#define CUTLASS_MINOR 9\n#define CUTLASS_PATCH 0\n",
    );
    assert!(matches!(
        sdk.compile(1),
        Err(CompileError::Configuration(_))
    ));
}

#[test]
fn environment_discovery_uses_process_local_configuration() {
    // Isolate environment mutation in child processes instead of racing Rust's
    // parallel test threads with set_var/remove_var.
    const ROOT: &str = "TRINITY_TEST_DISCOVERY_ROOT";
    if let Some(root) = std::env::var_os(ROOT) {
        let root = PathBuf::from(root);
        let default = CompileConfig::default();
        let mut explicit = default.clone();
        explicit.cuda_root = Some(root.join("cuda"));
        explicit.cutlass_root = Some(root.join("cutlass"));
        explicit.nvshmem_root = Some(root.join("nvshmem"));
        let config = if std::env::var_os("TRINITY_TEST_EXPLICIT").is_some() {
            &explicit
        } else {
            &default
        };
        compile_with_config(source(1), config).unwrap();
        return;
    }

    let sdk = Sdk::new();
    for mode in ["environment", "path", "explicit"] {
        let mut command = Command::new(std::env::current_exe().unwrap());
        command
            .args([
                "--exact",
                "compile::cuda::tests::environment_discovery_uses_process_local_configuration",
                "--nocapture",
            ])
            .env(ROOT, sdk.root.path())
            .env("CUDA_HOME", sdk.root.path().join("cuda"))
            .env("CUTLASS_HOME", sdk.root.path().join("cutlass"))
            .env("NVSHMEM_HOME", "/intentionally/missing/unused-nvshmem");
        if mode == "path" {
            let mut paths = vec![sdk.root.path().join("cuda/bin")];
            paths.extend(std::env::split_paths(&std::env::var_os("PATH").unwrap()));
            command
                .env_remove("CUDA_HOME")
                .env("PATH", std::env::join_paths(paths).unwrap());
        } else if mode == "explicit" {
            command
                .env("TRINITY_TEST_EXPLICIT", "1")
                .env("CUDA_HOME", "/missing/cuda")
                .env("CUTLASS_HOME", "/missing/cutlass");
        }

        let output = command.output().unwrap();
        assert!(
            output.status.success(),
            "{mode}: {}{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
    }
}

#[test]
fn concurrent_compilations_have_independent_artifacts() {
    let sdk = Sdk::new();
    let barrier = std::sync::Barrier::new(2);
    let paths = std::thread::scope(|scope| {
        let handles: Vec<_> = (0..2)
            .map(|_| {
                scope.spawn(|| {
                    let artifact = sdk.compile(2).unwrap();
                    let path = artifact.artifact_path().to_owned();
                    let args = &artifact.diagnostics().arguments;
                    assert!(args.iter().any(|a| a == "--relocatable-device-code=true"));
                    for window in args
                        .windows(4)
                        .filter(|w| w[0] == "-Xlinker" && w[1] == "-rpath")
                    {
                        assert!(!window[3].to_string_lossy().contains("stubs"));
                    }

                    barrier.wait();
                    assert!(path.exists());
                    path
                })
            })
            .collect();
        handles
            .into_iter()
            .map(|h| h.join().unwrap())
            .collect::<Vec<_>>()
    });

    assert_ne!(paths[0], paths[1]);
    assert!(paths.iter().all(|p| !p.exists()));
}

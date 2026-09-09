use std::env;
use std::path::{Path, PathBuf};

use super::CompileError;

/// Installation roots; explicit paths take precedence over environment discovery.
/// NVSHMEM is resolved only for persistent programs. No arbitrary compiler flags
/// are accepted: target, ABI, and numerical options belong to the backend.
#[derive(Debug, Clone, Default)]
pub struct CompileConfig {
    pub cuda_root: Option<PathBuf>,
    pub cutlass_root: Option<PathBuf>,
    pub nvshmem_root: Option<PathBuf>,
    pub host_compiler: Option<PathBuf>,
}

pub(super) struct Toolchain {
    pub compiler: PathBuf,
    pub cuda_lib: PathBuf,
    pub driver_link_dir: Option<PathBuf>,
    pub cutlass_include: PathBuf,
    pub nvshmem: Option<(PathBuf, PathBuf)>,
    pub host_compiler: Option<PathBuf>,
}

fn configuration(message: impl Into<String>) -> CompileError {
    CompileError::Configuration(message.into())
}

fn file(path: PathBuf) -> Result<PathBuf, CompileError> {
    if !path.is_file() {
        return Err(configuration(format!("missing file {}", path.display())));
    }

    path.canonicalize().map_err(|source| CompileError::Io {
        operation: "resolve path",
        path,
        source,
    })
}

fn directory(path: PathBuf) -> Result<PathBuf, CompileError> {
    if !path.is_dir() {
        return Err(configuration(format!(
            "missing directory {}",
            path.display()
        )));
    }

    path.canonicalize().map_err(|source| CompileError::Io {
        operation: "resolve directory",
        path,
        source,
    })
}

fn library_dir(root: &Path, names: &[&str]) -> Result<PathBuf, CompileError> {
    for relative in [
        "lib64",
        "lib",
        "targets/x86_64-linux/lib",
        "targets/sbsa-linux/lib",
    ] {
        let dir = root.join(relative);
        if names.iter().all(|name| dir.join(name).is_file()) {
            return dir.canonicalize().map_err(|source| CompileError::Io {
                operation: "resolve library directory",
                path: dir,
                source,
            });
        }
    }

    Err(configuration(format!(
        "{} has no library directory containing {}",
        root.display(),
        names.join(", ")
    )))
}

fn check_version(path: &Path, macros: [&str; 3], expected: [usize; 3]) -> Result<(), CompileError> {
    let text = std::fs::read_to_string(path).map_err(|source| CompileError::Io {
        operation: "read version header",
        path: path.to_owned(),
        source,
    })?;

    let actual = macros.map(|name| {
        text.lines().find_map(|line| {
            let mut words = line.split_whitespace();
            (words.next() == Some("#define") && words.next() == Some(name))
                .then(|| words.next()?.parse::<usize>().ok())
                .flatten()
        })
    });

    if actual != expected.map(Some) {
        return Err(configuration(format!(
            "{} must declare version {}.{}.{} (found {:?})",
            path.display(),
            expected[0],
            expected[1],
            expected[2],
            actual
        )));
    }

    Ok(())
}

pub(super) fn resolve(config: &CompileConfig, nvshmem: bool) -> Result<Toolchain, CompileError> {
    let cuda = config
        .cuda_root
        .clone()
        .or_else(|| env::var_os("CUDA_HOME").map(PathBuf::from));

    let (cuda_root, compiler) = if let Some(root) = cuda {
        let root = directory(root)?;
        let compiler = file(root.join("bin/nvcc"))?;
        (root, compiler)
    } else {
        let found = env::var_os("PATH").and_then(|path| {
            env::split_paths(&path)
                .map(|dir| dir.join("nvcc"))
                .find(|p| p.is_file())
        });
        let compiler =
            file(found.ok_or_else(|| {
                configuration("set cuda_root or CUDA_HOME, or put nvcc on PATH")
            })?)?;
        let root = compiler
            .parent()
            .and_then(Path::parent)
            .ok_or_else(|| {
                configuration("nvcc must reside in a CUDA installation's bin directory")
            })?
            .to_owned();
        (root, compiler)
    };

    let cuda_lib = library_dir(&cuda_root, &["libcudart.so"])?;

    let cutlass = config
        .cutlass_root
        .clone()
        .or_else(|| env::var_os("CUTLASS_HOME").map(PathBuf::from))
        .unwrap_or_else(|| Path::new(env!("CARGO_MANIFEST_DIR")).join("third_party/cutlass"));
    check_version(
        &cutlass.join("include/cutlass/version.h"),
        ["CUTLASS_MAJOR", "CUTLASS_MINOR", "CUTLASS_PATCH"],
        [4, 5, 1],
    )?;
    let cutlass_include = directory(cutlass.join("include"))?;
    file(cutlass_include.join("cute/tensor.hpp"))?;

    let nvshmem = if nvshmem {
        let root = config
            .nvshmem_root
            .clone()
            .or_else(|| env::var_os("NVSHMEM_HOME").map(PathBuf::from))
            .ok_or_else(|| {
                configuration("persistent programs require nvshmem_root or NVSHMEM_HOME")
            })?;
        check_version(
            &root.join("include/non_abi/nvshmem_version.h"),
            [
                "NVSHMEM_VENDOR_MAJOR_VERSION",
                "NVSHMEM_VENDOR_MINOR_VERSION",
                "NVSHMEM_VENDOR_PATCH_VERSION",
            ],
            [3, 7, 2],
        )?;
        let include = directory(root.join("include"))?;
        file(include.join("nvshmem.h"))?;
        file(include.join("nvshmemx.h"))?;
        Some((
            include,
            library_dir(&root, &["libnvshmem_host.so", "libnvshmem_device.a"])?,
        ))
    } else {
        None
    };

    let driver_link_dir = if cuda_lib.join("stubs/libcuda.so").is_file() {
        Some(cuda_lib.join("stubs"))
    } else {
        None
    };

    Ok(Toolchain {
        compiler,
        cuda_lib,
        driver_link_dir,
        cutlass_include,
        nvshmem,
        host_compiler: config.host_compiler.clone().map(file).transpose()?,
    })
}

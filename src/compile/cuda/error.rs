use std::ffi::OsString;
use std::path::PathBuf;
use std::process::ExitStatus;

use thiserror::Error;

/// Exact input and process output, including warnings from successful builds.
#[derive(Debug)]
pub struct CompileDiagnostics {
    pub compiler: PathBuf,
    pub arguments: Vec<OsString>,
    pub generated_source: String,
    pub status: Option<ExitStatus>,
    pub stdout: Vec<u8>,
    pub stderr: Vec<u8>,
}

impl std::fmt::Display for CompileDiagnostics {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{} {:?}: {:?}\n{}{}",
            self.compiler.display(),
            self.arguments,
            self.status,
            String::from_utf8_lossy(&self.stdout),
            String::from_utf8_lossy(&self.stderr)
        )
    }
}

#[derive(Debug, Error)]
pub enum CompileError {
    #[error("invalid CUDA compilation configuration: {0}")]
    Configuration(String),

    #[error("failed to {operation} at {path}: {source}")]
    Io {
        operation: &'static str,
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },

    #[error("failed to start compiler: {source}\n{diagnostics}")]
    CompilerSpawn {
        #[source]
        source: std::io::Error,
        diagnostics: Box<CompileDiagnostics>,
    },

    #[error("CUDA compilation/linking failed: {diagnostics}")]
    CompilerFailed {
        diagnostics: Box<CompileDiagnostics>,
    },

    #[error("NVCC must report CUDA 13.0 or newer: {diagnostics}")]
    CompilerVersion {
        diagnostics: Box<CompileDiagnostics>,
    },

    #[error("compiler succeeded without producing program.so: {diagnostics}")]
    MissingArtifact {
        diagnostics: Box<CompileDiagnostics>,
    },
}

impl CompileError {
    pub fn diagnostics(&self) -> Option<&CompileDiagnostics> {
        match self {
            Self::CompilerSpawn { diagnostics, .. }
            | Self::CompilerFailed { diagnostics }
            | Self::CompilerVersion { diagnostics }
            | Self::MissingArtifact { diagnostics } => Some(diagnostics),
            Self::Configuration(_) | Self::Io { .. } => None,
        }
    }
}

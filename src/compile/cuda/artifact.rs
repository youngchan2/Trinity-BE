use std::path::{Path, PathBuf};

use tempfile::TempDir;

use super::CompileDiagnostics;
use crate::CudaSource;
use crate::compile::CudaRequirements;

/// Unloaded compiled code and metadata. No CUDA runtime is loaded or prepared.
/// Drop removes the private build directory; `persist` transfers file ownership
/// to the caller. Loading and execution are outside the compiler's scope.
#[derive(Debug)]
pub struct CudaArtifact {
    source: CudaSource,
    workspace: TempDir,
    artifact: PathBuf,
    diagnostics: CompileDiagnostics,
    manifest: String,
}

impl CudaArtifact {
    pub(super) fn new(
        source: CudaSource,
        workspace: TempDir,
        artifact: PathBuf,
        diagnostics: CompileDiagnostics,
        manifest: String,
    ) -> Self {
        Self {
            source,
            workspace,
            artifact,
            diagnostics,
            manifest,
        }
    }

    pub fn source(&self) -> &CudaSource {
        &self.source
    }

    pub fn requirements(&self) -> &CudaRequirements {
        self.source.requirements()
    }

    pub fn artifact_path(&self) -> &Path {
        &self.artifact
    }

    pub fn directory(&self) -> &Path {
        self.workspace.path()
    }

    pub fn diagnostics(&self) -> &CompileDiagnostics {
        &self.diagnostics
    }

    /// Versioned, CUDA-independent compiler-to-runtime handoff.
    pub fn manifest(&self) -> &str {
        &self.manifest
    }

    /// Keep the artifact directory after this object is consumed. The caller
    /// now owns its files and must remove them when no longer needed.
    pub fn persist(self) -> PathBuf {
        self.workspace.keep()
    }
}

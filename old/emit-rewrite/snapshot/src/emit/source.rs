//! Platform source variants behind the common emission path.

use super::cuda::CudaSource;

pub(crate) enum EmittedSource {
    Cuda(CudaSource),
}

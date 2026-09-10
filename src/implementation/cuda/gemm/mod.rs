pub(super) mod hopper_wgmma;
pub(super) static IMPLEMENTATIONS: &[&dyn crate::GemmImplementation] =
    &[&hopper_wgmma::HOPPER_WGMMA_BF16];

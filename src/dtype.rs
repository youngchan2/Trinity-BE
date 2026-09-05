#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum DType {
    Bf16,
    Fp32,
}

impl std::str::FromStr for DType {
    type Err = ();

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value.to_ascii_lowercase().as_str() {
            "bf16" | "bfloat16" => Ok(Self::Bf16),
            "fp32" | "float32" => Ok(Self::Fp32),
            _ => Err(()),
        }
    }
}

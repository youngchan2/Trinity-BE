#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, serde::Serialize)]
#[serde(rename_all = "lowercase")]
pub enum DType {
    Fp16,
    Bf16,
    Fp32,
}

impl DType {
    pub const fn size_bytes(self) -> usize {
        match self {
            Self::Fp16 | Self::Bf16 => 2,
            Self::Fp32 => 4,
        }
    }
}

impl std::str::FromStr for DType {
    type Err = ();

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value.to_ascii_lowercase().as_str() {
            "fp16" | "float16" | "half" => Ok(Self::Fp16),
            "bf16" | "bfloat16" => Ok(Self::Bf16),
            "fp32" | "float32" => Ok(Self::Fp32),
            _ => Err(()),
        }
    }
}

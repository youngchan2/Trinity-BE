use super::{QuackApi, QuackSpecification};
use crate::{CudaTargetCapability, DType};

pub(super) fn render(s: &QuackSpecification) -> String {
    let inputs = [&s.lhs, &s.rhs]
        .into_iter()
        .chain(s.residual.as_ref())
        .chain(s.bias.as_ref());
    let input_ids: Vec<_> = inputs.clone().map(|a| a.value.index()).collect();
    let expected: Vec<_> = inputs
        .chain([&s.output])
        .map(|a| {
            (
                a.value.index(),
                match a.dtype {
                    DType::Fp16 => "float16",
                    DType::Bf16 => "bfloat16",
                    DType::Fp32 => "float32",
                },
                &a.shape,
            )
        })
        .collect();
    let capability = match s.target {
        CudaTargetCapability::Hopper => (9, 0),
        CudaTargetCapability::Sm120 => (12, 0),
        CudaTargetCapability::Sm89 => unreachable!("Quack target checked at specification"),
    };
    let lhs = format!("values[{}]", s.lhs.value.index());
    let rhs = format!("values[{}]", s.rhs.value.index());
    let bias = s
        .bias
        .as_ref()
        .map_or_else(|| "None".into(), |a| format!("values[{}]", a.value.index()));
    let residual = s
        .residual
        .as_ref()
        .map_or_else(|| "None".into(), |a| format!("values[{}]", a.value.index()));
    let (api, call) = match s.api {
        QuackApi::Gemm => (
            "gemm",
            format!("gemm({lhs}, {rhs}, out=out, bias={bias}, tuned=tuned, split_k=1)"),
        ),
        QuackApi::GemmAdd => (
            "gemm_add",
            format!("gemm_add({lhs}, {rhs}, {residual}, out=out, tuned=tuned, split_k=1)"),
        ),
        QuackApi::GemmAct => (
            "gemm_act",
            format!(
                "gemm_act({lhs}, {rhs}, C={residual}, bias={bias}, activation='relu', postact_out=out, out_dtype=out.dtype, postact_dtype=out.dtype, store_preact=False, tuned=tuned, split_k=1)"
            ),
        ),
    };
    let header = format!(
        "# Generated Quack candidate. Inputs/outputs are allocated by the caller.\n_EXPECTED = {expected:?}\n_INPUT_IDS = {input_ids:?}\n_OUTPUT_ID = {}\n_CAPABILITY = {capability:?}\n",
        s.output.value.index()
    );
    header
        + &include_str!("wrapper.py.in")
            .replace("@API@", api)
            .replace("@CALL@", &call)
}

//! Quack API restrictions and call planning, separate from rotary algebra.
use super::super::ProviderError;
use super::pattern::{QuackRegionOperation, QuackRegionSpecification};
use super::recognition::{
    QuackPatternKind,
    rope::{Pairing, RopePattern},
};
use crate::analysis::{regions::RegionFacts, views::TensorView};
use crate::{DType, PhysicalPlan, Storage};
use serde_json::json;
use std::collections::BTreeSet;

fn reject(reason: &str) -> ProviderError {
    ProviderError::Unsupported(reason.into())
}
fn dtype(plan: &PhysicalPlan, view: &TensorView) -> DType {
    plan.value_instances().nth(view.value).unwrap().1.dtype()
}
pub(super) fn specify(
    plan: &PhysicalPlan,
    facts: &RegionFacts<'_>,
    r: RopePattern,
    capability: [u32; 2],
) -> Result<QuackRegionSpecification, ProviderError> {
    let output_id = plan.value_instances().nth(r.output.value).unwrap().0;
    facts
        .require_single_output(output_id)
        .map_err(|e| reject(&e))?;
    let output_value = plan.value_instance(output_id).unwrap();
    if !matches!(output_value.storage(), Storage::Global | Storage::External)
        || r.output.offset != 0
        || r.output.shape.iter().product::<usize>()
            != output_value.shape().iter().product::<usize>()
    {
        return Err(reject(
            "Quack RoPE adapter requires one complete global output",
        ));
    }
    if !matches!(dtype(plan, &r.cos), DType::Fp16 | DType::Bf16 | DType::Fp32)
        || dtype(plan, &r.cos) != dtype(plan, &r.sin)
    {
        return Err(reject(
            "Quack cos/sin must have matching FP16/BF16/FP32 storage",
        ));
    }
    let mut inputs = vec![&r.cos, &r.sin];
    let (operation, preparation, mut conditions) = if let Some(p) = &r.projection {
        if r.pairing != Pairing::Interleaved || r.passthrough.is_some() || r.conjugate {
            return Err(reject(
                "Quack rope_epi adapter supports full adjacent-pair forward rotation only; half pairing/partial/conjugate projection is recognized but unsupported",
            ));
        }
        let local = plan.value_instances().nth(p.result.value).unwrap().1;
        if local.storage() != Storage::Register {
            return Err(reject(
                "GEMM+RoPE cannot remove an observable or rounded projection storage boundary",
            ));
        }
        if !matches!(dtype(plan, &p.lhs), DType::Fp16 | DType::Bf16)
            || dtype(plan, &p.lhs) != dtype(plan, &p.rhs)
            || !p.lhs.shape[1].is_multiple_of(8)
            || !p.rhs.shape[1].is_multiple_of(8)
        {
            return Err(reject(
                "rope_epi adapter requires matching FP16/BF16 GEMM operands and K/N multiples of 8",
            ));
        }
        if r.input.shape.len() != 2 || r.output.shape != p.result.shape {
            return Err(reject(
                "rope_epi adapter currently requires a matrix projection with one final pair axis; batched/head reshapes need another mapping proof",
            ));
        }
        inputs.extend([&p.lhs, &p.rhs]);
        (QuackRegionOperation::GemmRotary {
            lhs:p.lhs.clone(),rhs:p.rhs.clone(),cos:r.cos.clone(),sin:r.sin.clone(),
            pair_shape:vec![r.input.shape[0],r.rotary_dimension/2],
        }, vec!["align/copy GEMM operands if needed".into(),
            "broadcast explicit cos/sin views and interleave into an FP32 [M,N] table per call".into()],
            vec!["quack.epilogue.library.rope_epi; acc_pair, split_k=1".into(),
                "matching FP16/BF16 matrix operands; K/N divisible by 8; complete adjacent pairs".into(),
                "FP32 GEMM accumulator -> FP32 rotation -> output storage cast; no intermediate storage rounding".into()])
    } else {
        if r.input.shape.len() != 4 || r.rotation_axis != 3 || r.table_axes.len() > 1 {
            return Err(reject(
                "apply_rotary adapter requires a four-axis logical input with last-axis rotation and tables varying over at most one token axis",
            ));
        }
        let d = r.input.shape[3];
        if d > 512 || !d.is_multiple_of(8) || !r.rotary_dimension.is_multiple_of(8) {
            return Err(reject(
                "apply_rotary requires head dimension <=512 and head/rotary dimensions divisible by 8",
            ));
        }
        if !matches!(
            dtype(plan, &r.input),
            DType::Fp16 | DType::Bf16 | DType::Fp32
        ) || dtype(plan, &r.input) != dtype(plan, &r.output)
        {
            return Err(reject(
                "apply_rotary preserves input storage dtype; different output dtype is unsupported",
            ));
        }
        let token = r.table_axes.first().copied().unwrap_or(1);
        let other: Vec<_> = (0..3).filter(|a| *a != token).collect();
        inputs.push(&r.input);
        (QuackRegionOperation::Rotary {
            input:r.input.clone(),cos:r.cos.clone(),sin:r.sin.clone(),
            order:vec![other[0],token,other[1],3],
            interleaved:r.pairing==Pairing::Interleaved,conjugate:r.conjugate,
        },vec!["permute logical axes to B,T,H,D; copy to contiguous input if needed".into(),
            "broadcast/copy cos and sin to [T,R/2] without dtype conversion".into(),
            "apply_rotary allocates output; inverse-permute and copy including explicit unrotated tail".into()],
            vec!["quack.rotary.apply_rotary, inplace=False, fixed length, no implicit position offsets".into(),
                "four logical axes, D<=512, D/R divisible by 8; matching input/output storage dtype".into(),
                "input/table arithmetic in FP32, output cast to input dtype".into()])
    };
    conditions.extend([
        "whole original region, one complete observable output, no input updates or aliasing".into(),
        "single GPU matching target capability; contiguous ABI buffers; cos/sin have equal storage dtype".into(),
    ]);
    let ids: BTreeSet<_> = inputs.into_iter().map(|v| v.value).collect();
    for &id in &ids {
        let (value_id, v) = plan.value_instances().nth(id).unwrap();
        if id == r.output.value
            || !matches!(v.storage(), Storage::Global | Storage::External)
            || plan.mutable_inputs().contains(&value_id)
        {
            return Err(reject(
                "RoPE API inputs must be immutable global/external boundary values",
            ));
        }
    }
    let arguments = json!(
        ids.into_iter()
            .chain([r.output.value])
            .map(|id| {
                let v = plan.value_instances().nth(id).unwrap().1;
                json!({"id":id,"shape":v.shape(),"dtype":v.dtype()})
            })
            .collect::<Vec<_>>()
    );
    Ok(QuackRegionSpecification {
        scope: facts.scope.clone(),
        pattern: QuackPatternKind::Other,
        operation,
        output: r.output.clone(),
        arguments,
        capability,
        preparation,
        conditions,
        semantic: Some(r),
    })
}

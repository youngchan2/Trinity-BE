mod hopper;
mod pointwise;
use crate::FusionRule;
pub(super) static RULES: &[&dyn FusionRule] = &[
    &pointwise::PointwiseFusion,
    &hopper::HopperFusionRule(hopper::Kind::Producer),
    &hopper::HopperFusionRule(hopper::Kind::Consumer),
    &hopper::HopperFusionRule(hopper::Kind::Computation),
];

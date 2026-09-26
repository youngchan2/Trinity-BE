use super::pattern::QuackRegionSpecification;
use serde_json::json;
pub(super) fn render(s: &QuackRegionSpecification) -> String {
    let spec = json!({"operation":s.operation,"output":s.output,"arguments":s.arguments,"capability":s.capability});
    format!(
        "# Whole-region Quack candidate; preparation is included in run().\nimport json\n_SPEC = json.loads({})\n{}",
        serde_json::to_string(&spec.to_string()).unwrap(),
        include_str!("region.py.in")
    )
}

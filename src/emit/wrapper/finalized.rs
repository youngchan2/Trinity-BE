//! Fixed implementation selection -> named kernels -> a small executable module.
//! Candidate/reference metadata stays in the returned report, outside the source.
use crate::analysis::regions::RegionFacts;
use crate::emit::provider::python::{PythonKernel, tuple};
use crate::emit::{EmitError, region::discover_regions};
use crate::{
    CudaTargetCapability, DType, PhysicalPlan, Storage, TargetCapability, ValueInstanceId,
};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::collections::{BTreeMap, BTreeSet};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum PythonProvider {
    Triton,
    Quack,
}
impl PythonProvider {
    fn name(self) -> &'static str {
        match self {
            Self::Triton => "triton",
            Self::Quack => "quack",
        }
    }
}

/// Selection is explicit. PreferQuack is a support policy, not a timing result.
/// Regions accepts externally chosen (e.g. measured) implementations in IR order.
#[derive(Debug, Clone, Default)]
pub enum PythonSelection {
    #[default]
    PreferQuack,
    TritonOnly,
    Regions(Vec<PythonProvider>),
}

#[derive(Debug, Clone)]
pub struct PythonExecutable {
    source: String,
    report: Value,
    providers: Vec<PythonProvider>,
}
impl PythonExecutable {
    pub fn emit(&self) -> &str {
        &self.source
    }
    pub fn report(&self) -> &Value {
        &self.report
    }
    pub fn providers(&self) -> &[PythonProvider] {
        &self.providers
    }
}

fn invalid(reason: impl Into<String>) -> EmitError {
    EmitError::InvalidExecution {
        reason: reason.into(),
    }
}
fn dtype(t: DType) -> &'static str {
    match t {
        DType::Fp16 => "float16",
        DType::Bf16 => "bfloat16",
        DType::Fp32 => "float32",
    }
}

fn unique_name(raw: &str, used: &mut BTreeSet<String>) -> String {
    let mut base: String = raw
        .trim_start_matches('?')
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '_' {
                c
            } else {
                '_'
            }
        })
        .collect();
    if base.is_empty() || base.starts_with(|c: char| c.is_ascii_digit() || c == '_') {
        base.insert_str(0, "value");
    }
    if [
        "False",
        "None",
        "True",
        "and",
        "as",
        "assert",
        "async",
        "await",
        "break",
        "class",
        "continue",
        "def",
        "del",
        "elif",
        "else",
        "except",
        "finally",
        "for",
        "from",
        "global",
        "if",
        "import",
        "in",
        "is",
        "lambda",
        "nonlocal",
        "not",
        "or",
        "pass",
        "raise",
        "return",
        "try",
        "while",
        "with",
        "yield",
        "torch",
        "triton",
        "tl",
        "tuple",
        "ValueError",
    ]
    .contains(&base.as_str())
    {
        base.insert_str(0, "value_");
    }
    let mut name = base.clone();
    let mut suffix = 1;
    while !used.insert(name.clone()) {
        name = format!("{base}_{suffix}");
        suffix += 1;
    }
    name
}

/// Generate a standalone fixed-shape single-GPU program. Kernels are emitted
/// by their providers; common ABI/storage and original region order are retained.
/// This does not compile, benchmark or claim correctness of an external library.
pub fn emit_python_executable(
    plan: &PhysicalPlan,
    selection: PythonSelection,
) -> Result<PythonExecutable, EmitError> {
    if plan.world_size() != 1 {
        return Err(invalid("Python executable requires a single GPU"));
    }
    let facts = RegionFacts::collect(plan);
    if let PythonSelection::Regions(choices) = &selection
        && choices.len() != facts.len()
    {
        return Err(invalid(
            "explicit provider choices must cover every original region exactly once",
        ));
    }
    let (regions, triton) = discover_regions(plan)?;
    let mut kernels = Vec::new();
    let mut providers = Vec::new();
    let mut diagnostics = Vec::new();
    for (i, r) in regions.iter().enumerate() {
        let provider = match &selection {
            PythonSelection::TritonOnly => PythonProvider::Triton,
            PythonSelection::Regions(choices) => choices[i],
            PythonSelection::PreferQuack => {
                if r.candidates.iter().any(|c| c.quack().is_some()) {
                    PythonProvider::Quack
                } else {
                    PythonProvider::Triton
                }
            }
        };
        let candidate = r
            .candidates
            .iter()
            .find(|c| c.provider() == provider.name())
            .ok_or_else(|| {
                invalid(format!(
                    "region {i}: requested {} is unavailable: {:?}",
                    provider.name(),
                    r.rejections
                ))
            })?;
        let kernel = match provider {
            PythonProvider::Quack => {
                let spec = candidate.quack().unwrap();
                if spec.scope != facts[i].scope {
                    return Err(invalid(
                        "Quack candidate does not cover the original region",
                    ));
                }
                spec.python_kernel(
                    plan,
                    i,
                    facts[i]
                        .global_values()
                        .into_iter()
                        .map(ValueInstanceId::from_index)
                        .collect(),
                )
            }
            PythonProvider::Triton => triton
                .as_ref()
                .map_err(|e| invalid(e.to_string()))?
                .plan()
                .python_kernel(i)
                .map_err(invalid)?,
        };
        let expected: BTreeSet<_> = facts[i].global_values().into_iter().collect();
        let actual: BTreeSet<_> = kernel.arguments.iter().map(|id| id.index()).collect();
        if actual != expected {
            return Err(invalid(format!(
                "region {i}: provider arguments do not cover the original memory boundary"
            )));
        }
        diagnostics.push(json!({"region":i,"scope":r.scope.statement_path,"operations":r.scope.operations.iter().map(|o|o.index()).collect::<Vec<_>>(),"selected":provider,"candidates":r.candidates.iter().map(|c|c.provider()).collect::<Vec<_>>(),"rejections":r.rejections.iter().map(|x|json!({"provider":x.provider,"reason":x.reason})).collect::<Vec<_>>() }));
        providers.push(provider);
        kernels.push(kernel);
    }
    let (source, inputs) = compose(plan, &kernels, &providers)?;
    let report = json!({"mode":"python_executable", "selection_basis":match selection {PythonSelection::PreferQuack=>"prefer_quack",PythonSelection::TritonOnly=>"triton_only",PythonSelection::Regions(_)=>"explicit_by_region"},"inputs":inputs,"regions":diagnostics,"validation":"source_only; no GPU compilation, accuracy or timing implied"});
    Ok(PythonExecutable {
        source,
        report,
        providers,
    })
}

fn compose(
    plan: &PhysicalPlan,
    kernels: &[PythonKernel],
    providers: &[PythonProvider],
) -> Result<(String, Value), EmitError> {
    let mut used = BTreeSet::new();
    let mut names = BTreeMap::new();
    for (id, value) in plan.value_instances() {
        let input = plan.inputs().iter().find(|b| b.value() == id);
        let fallback = format!("value{}", id.index());
        let raw = input
            .map(|b| b.tensor())
            .or(value.name())
            .unwrap_or(&fallback);
        names.insert(id, unique_name(raw, &mut used));
    }
    let mut inputs = Vec::new();
    let mut input_values = BTreeSet::new();
    let mut aliases = Vec::new();
    for b in plan.inputs() {
        let arg = if input_values.insert(b.value()) {
            names[&b.value()].clone()
        } else {
            let arg = unique_name(b.tensor(), &mut used);
            aliases.push((arg.clone(), names[&b.value()].clone()));
            arg
        };
        inputs.push((b, arg));
    }
    let mut imports: BTreeSet<String> = ["import torch".into()].into();
    let mut helpers = BTreeMap::new();
    let mut required: BTreeSet<_> = plan.outputs().iter().map(|b| b.value()).collect();
    for k in kernels {
        imports.extend(k.imports.iter().cloned());
        for (name, source) in &k.helpers {
            if helpers
                .insert(name.clone(), source.clone())
                .is_some_and(|previous| previous != *source)
            {
                return Err(invalid(format!("conflicting Python helper {name}")));
            }
        }
        required.extend(k.arguments.iter().copied());
    }
    let mut source = String::from(
        "\"\"\"Generated Trinity program: fixed region implementations, forward execution only.\"\"\"\n",
    );
    source.push_str(&imports.into_iter().collect::<Vec<_>>().join("\n"));
    source.push_str("\n\n");
    for h in helpers.values() {
        source.push_str(h);
        source.push('\n');
    }
    for (i, k) in kernels.iter().enumerate() {
        source.push_str(&format!(
            "# Region {i}: {}\n{}\n",
            providers[i].name(),
            k.source
        ));
    }
    source.push_str("def _check_inputs(device, specs):\n    for tensor, shape, dtype in specs:\n        if tensor.device != device or tensor.dtype != dtype or tuple(tensor.shape) != shape or not tensor.is_contiguous():\n            raise ValueError('input differs from the generated shape/dtype/device/stride contract')\n\n");
    source.push_str(&format!(
        "@torch.no_grad()\ndef forward({}):\n",
        inputs
            .iter()
            .map(|(_, a)| a.as_str())
            .collect::<Vec<_>>()
            .join(", ")
    ));
    let mut lines = Vec::new();
    let device = inputs
        .first()
        .map(|(_, a)| format!("{a}.device"))
        .unwrap_or("torch.device('cuda', torch.cuda.current_device())".into());
    lines.push(format!("_device = {device}"));
    let TargetCapability::Cuda(target) = plan.target();
    let capability = match target {
        CudaTargetCapability::Hopper => "(9, 0)",
        CudaTargetCapability::Sm120 => "(12, 0)",
        CudaTargetCapability::Sm89 => "(8, 9)",
    };
    lines.push(format!("if _device.type != 'cuda' or tuple(torch.cuda.get_device_capability(_device)) != {capability}:"));
    lines.push(
        "    raise ValueError('CUDA device capability differs from the generated target')".into(),
    );
    let specs = inputs.iter().map(|(b, a)| {
        let v = plan.value_instance(b.value()).unwrap();
        format!("({a}, {}, torch.{})", tuple(v.shape()), dtype(v.dtype()))
    });
    lines.push(format!("_check_inputs(_device, {})", tuple(specs)));
    for (arg, primary) in aliases {
        lines.push(format!("if {arg} is not {primary}:"));
        lines.push("    raise ValueError('input aliases must refer to the same tensor')".into());
    }
    for (i, a) in plan.inputs().iter().enumerate() {
        for b in &plan.inputs()[i + 1..] {
            if a.value() != b.value()
                && (plan.mutable_inputs().contains(&a.value())
                    || plan.mutable_inputs().contains(&b.value()))
            {
                lines.push(format!(
                    "if {}.untyped_storage().data_ptr() == {}.untyped_storage().data_ptr():",
                    names[&a.value()],
                    names[&b.value()]
                ));
                lines.push(
                    "    raise ValueError('distinct mutable input values must not share storage')"
                        .into(),
                );
            }
        }
    }
    for id in required.difference(&input_values) {
        let v = plan
            .value_instance(*id)
            .ok_or_else(|| invalid("unknown provider argument"))?;
        if !matches!(v.storage(), Storage::Global | Storage::External) {
            return Err(invalid("a kernel boundary value has no global storage"));
        }
        lines.push(format!(
            "{} = torch.empty({}, device=_device, dtype=torch.{})",
            names[id],
            tuple(v.shape()),
            dtype(v.dtype())
        ));
    }
    if !kernels.is_empty() {
        lines.push("with torch.cuda.device(_device):".into());
        for k in kernels {
            lines.push(format!(
                "    {}({})",
                k.entrypoint,
                k.arguments
                    .iter()
                    .map(|id| names[id].as_str())
                    .collect::<Vec<_>>()
                    .join(", ")
            ));
        }
    }
    let outputs: Vec<_> = plan
        .outputs()
        .iter()
        .map(|b| names[&b.value()].as_str())
        .collect();
    lines.push(format!(
        "return {}",
        match outputs.len() {
            0 => "None".into(),
            1 => outputs[0].into(),
            _ => tuple(outputs),
        }
    ));
    for line in lines {
        source.push_str("    ");
        source.push_str(&line);
        source.push('\n');
    }
    let bindings = json!(
        inputs
            .iter()
            .map(|(b, a)| json!({"name":b.tensor(),"argument":a,"value":b.value().index()}))
            .collect::<Vec<_>>()
    );
    Ok((source, bindings))
}

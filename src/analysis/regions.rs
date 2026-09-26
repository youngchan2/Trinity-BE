//! Original kernel regions and boundary facts, independent of pattern vocabularies.
//! Providers inspect the plan through this view; a failed optional equation
//! summary must never prevent another provider from examining the region.
use crate::{OperationId, PhysicalPlan, Statement, Storage, TensorAccess, ValueInstanceId};
use std::collections::{BTreeMap, BTreeSet};

/// Paths index the plan's statement lists. The region path identifies the whole
/// classification unit, including for an unwrapped loop or standalone operation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OperationScope {
    pub operation: OperationId,
    pub statement_path: Vec<usize>,
    pub region_path: Vec<usize>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RegionScope {
    pub statement_path: Vec<usize>,
    /// All operations, in source order, including initialization and epilogues.
    pub operations: Vec<OperationId>,
}

/// Map every operation occurrence to its original statement and kernel region.
pub fn operation_scopes(plan: &PhysicalPlan) -> BTreeMap<OperationId, OperationScope> {
    fn visit(
        statement: &Statement,
        path: &mut Vec<usize>,
        region: &[usize],
        result: &mut BTreeMap<OperationId, OperationScope>,
    ) {
        let body = match statement {
            Statement::Operation(id) => {
                result.insert(
                    *id,
                    OperationScope {
                        operation: *id,
                        statement_path: path.clone(),
                        region_path: region.to_vec(),
                    },
                );
                return;
            }
            Statement::Region(body) => body,
            Statement::Loop(l) => &l.body,
        };
        for (index, child) in body.iter().enumerate() {
            path.push(index);
            visit(child, path, region, result);
            path.pop();
        }
    }
    let mut result = BTreeMap::new();
    for (index, statement) in plan.statements().iter().enumerate() {
        let path = vec![index];
        visit(statement, &mut path.clone(), &path, &mut result);
    }
    result
}

#[derive(Clone)]
pub struct RegionFacts<'p> {
    pub plan: &'p PhysicalPlan,
    pub statement: &'p Statement,
    pub scope: RegionScope,
    pub inputs: BTreeSet<ValueInstanceId>,
    /// Every store, including multiple stores to one value. No last-output rule.
    pub writes: Vec<(OperationId, &'p TensorAccess)>,
    pub observable_writes: Vec<(OperationId, &'p TensorAccess)>,
    pub input_updates: BTreeSet<ValueInstanceId>,
    pub producers: BTreeMap<ValueInstanceId, Vec<OperationId>>,
    pub consumers: BTreeMap<ValueInstanceId, Vec<OperationId>>,
}

impl<'p> RegionFacts<'p> {
    pub fn collect(plan: &'p PhysicalPlan) -> Vec<Self> {
        plan.statements()
            .iter()
            .enumerate()
            .map(|(index, statement)| {
                let scope = RegionScope {
                    statement_path: vec![index],
                    operations: statement.operations(),
                };
                let mut result = Self {
                    plan,
                    statement,
                    scope,
                    inputs: BTreeSet::new(),
                    writes: vec![],
                    observable_writes: vec![],
                    input_updates: BTreeSet::new(),
                    producers: BTreeMap::new(),
                    consumers: BTreeMap::new(),
                };
                let mut defined = BTreeSet::new();
                for &id in &result.scope.operations {
                    let op = plan.operation(id).expect("plan operation");
                    for value in op.inflows() {
                        result.consumers.entry(*value).or_default().push(id);
                        if !defined.contains(value) && !op.zero_init().contains(value) {
                            result.inputs.insert(*value);
                        }
                    }
                    for value in op.outflows() {
                        result.producers.entry(*value).or_default().push(id);
                        defined.insert(*value);
                    }
                    if let crate::Expression::Store { destination, .. } = op.expression() {
                        result.writes.push((id, destination));
                        let v = destination.value;
                        if plan.inputs().iter().any(|b| b.value() == v) {
                            result.input_updates.insert(v);
                        }
                        if plan.value_instance(v).unwrap().storage() != Storage::Register
                            || plan.outputs().iter().any(|b| b.value() == v)
                            || plan.operations().any(|(other, op)| {
                                !result.scope.operations.contains(&other)
                                    && op.inflows().contains(&v)
                            })
                        {
                            result.observable_writes.push((id, destination));
                        }
                    }
                }
                result
            })
            .collect()
    }

    /// Single-result adapter restriction, not a restriction on the common plan.
    pub fn require_single_output(&self, value: ValueInstanceId) -> Result<(), String> {
        if !self.input_updates.is_empty() {
            return Err(
                "region updates an input; this adapter cannot preserve the mutation".into(),
            );
        }
        if self.observable_writes.iter().any(|(_, a)| a.value != value) {
            return Err("region has additional observable outputs/stores".into());
        }
        Ok(())
    }

    /// Complete memory boundary for launch/selection, regardless of whether a
    /// particular provider can lower this region.
    pub fn global_values(&self) -> Vec<usize> {
        self.scope
            .operations
            .iter()
            .flat_map(|id| {
                let op = self.plan.operation(*id).unwrap();
                op.inflows().iter().chain(op.outflows())
            })
            .filter(|id| {
                matches!(
                    self.plan.value_instance(**id).unwrap().storage(),
                    Storage::External | Storage::Global
                )
            })
            .map(|id| id.index())
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect()
    }
}

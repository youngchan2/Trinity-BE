//! Logical region validation. No task slots, readiness tables or dependency DAG.
use std::collections::{BTreeMap, BTreeSet};
use std::rc::Rc;

use super::{
    EmitError, Region,
    access::{self, AccessSink, fail},
    domain::{TaskDomain, coordinates},
    regions::{full_region, overlaps, subtract},
};
use crate::{PhysicalPlan, ValueInstanceId};

pub(super) fn validate(plan: &PhysicalPlan, domains: &[TaskDomain]) -> Result<(), EmitError> {
    // Logical validation is independent of readiness. Its temporary state only
    // retains live region versions, not per-coordinate Work or scheduler data.
    let mut validator = Validator::new(plan);
    for (index, domain) in domains.iter().enumerate() {
        for rank in 0..plan.world_size() {
            coordinates(&domain.task.domain, &mut |bindings, steps| {
                validator.enter(index, rank, bindings);
                access::visit(
                    plan,
                    &domain.statements,
                    bindings,
                    steps,
                    rank,
                    false,
                    &mut validator,
                )
            })?;
        }
    }
    validator.finish()
}

#[derive(PartialEq, Eq)]
struct Origin {
    domain: usize,
    rank: usize,
    coordinates: BTreeMap<String, i64>,
}
impl Origin {
    fn concurrent(&self, other: &Self) -> bool {
        self.coordinates
            .iter()
            .any(|(v, c)| other.coordinates.get(v).is_some_and(|b| b != c))
    }
}
#[derive(Clone)]
struct Version {
    region: Region,
    origin: Option<Rc<Origin>>,
}

struct Validator<'a> {
    plan: &'a PhysicalPlan,
    versions: Vec<Vec<Vec<Version>>>,
    readers: Vec<Vec<Vec<Version>>>,
    rewritten: BTreeSet<ValueInstanceId>,
    current: Rc<Origin>,
}

impl<'a> Validator<'a> {
    fn new(plan: &'a PhysicalPlan) -> Self {
        let mut versions = vec![vec![Vec::new(); plan.value_instances().len()]; plan.world_size()];
        for (rank, values) in versions.iter_mut().enumerate() {
            for input in plan.inputs() {
                if values[input.value().index()].is_empty() {
                    values[input.value().index()].push(Version {
                        region: full_region(plan, input.value(), rank),
                        origin: None,
                    });
                }
            }
        }
        Self {
            readers: vec![vec![Vec::new(); plan.value_instances().len()]; plan.world_size()],
            versions,
            plan,
            rewritten: plan
                .value_instances()
                .filter(|(id, _)| {
                    plan.operations()
                        .filter(|(_, op)| op.outputs().contains(id))
                        .count()
                        > 1
                })
                .map(|(id, _)| id)
                .collect(),
            current: Rc::new(Origin {
                domain: 0,
                rank: 0,
                coordinates: BTreeMap::new(),
            }),
        }
    }
    fn enter(&mut self, domain: usize, rank: usize, coordinates: &BTreeMap<String, i64>) {
        self.current = Rc::new(Origin {
            domain,
            rank,
            coordinates: coordinates.clone(),
        });
    }
    fn finish(self) -> Result<(), EmitError> {
        for rank in 0..self.plan.world_size() {
            covered(
                full_region(self.plan, self.plan.output().value(), rank),
                &self.versions[rank][self.plan.output().value().index()],
            )?;
        }
        Ok(())
    }
}

fn covered(region: Region, versions: &[Version]) -> Result<(), EmitError> {
    let mut remaining = vec![region];
    for v in versions {
        remaining = remaining
            .into_iter()
            .flat_map(|r| subtract(r, v.region))
            .collect();
        if remaining.is_empty() {
            return Ok(());
        }
    }
    Err(fail("read before production or incomplete output coverage"))
}

impl AccessSink for Validator<'_> {
    fn read(&mut self, region: Region, _stage: Option<usize>) -> Result<(), EmitError> {
        let values = &self.versions[region.rank][region.value.index()];
        covered(region, values)?;
        for prior in values.iter().filter(|v| overlaps(v.region, region)) {
            if let Some(origin) = &prior.origin
                && origin != &self.current
                && (origin.concurrent(&self.current) || origin.domain == self.current.domain)
            {
                return Err(fail("read/write conflict between parallel iterations"));
            }
        }
        if self.rewritten.contains(&region.value) {
            let readers = &mut self.readers[region.rank][region.value.index()];
            if !readers
                .iter()
                .any(|v| v.region == region && v.origin.as_ref() == Some(&self.current))
            {
                readers.push(Version {
                    region,
                    origin: Some(self.current.clone()),
                });
            }
        }
        Ok(())
    }
    fn write(&mut self, region: Region) -> Result<(), EmitError> {
        if self.plan.inputs().iter().any(|b| b.value() == region.value) {
            return Err(fail("backend writes an immutable input binding"));
        }
        let versions = &mut self.versions[region.rank][region.value.index()];
        let readers = &mut self.readers[region.rank][region.value.index()];
        for prior in versions
            .iter()
            .chain(readers.iter())
            .filter(|v| overlaps(v.region, region))
        {
            if let Some(origin) = &prior.origin
                && origin != &self.current
                && (origin.concurrent(&self.current) || origin.domain == self.current.domain)
            {
                return Err(fail("overlapping accesses between parallel work"));
            }
        }
        for values in [&mut *versions, readers] {
            // Most tiled writes are disjoint. Keep those live versions in
            // place; only an actual rewrite needs rectangular fragments.
            let mut fragments = Vec::new();
            values.retain(|v| {
                if !overlaps(v.region, region) {
                    return true;
                }
                fragments.extend(subtract(v.region, region).into_iter().map(|r| Version {
                    region: r,
                    origin: v.origin.clone(),
                }));
                false
            });
            values.extend(fragments);
        }
        versions.push(Version {
            region,
            origin: Some(self.current.clone()),
        });
        Ok(())
    }
}

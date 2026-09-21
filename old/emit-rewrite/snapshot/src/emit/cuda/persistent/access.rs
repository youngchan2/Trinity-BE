//! Physical access events collected for Persistent task dependencies.
use std::collections::BTreeMap;

use super::super::{
    EmitError, Region, Work,
    access::{AccessSink, visit},
};
use crate::{PhysicalPlan, Statement};

pub(super) fn describe(
    plan: &PhysicalPlan,
    statements: &[Statement],
    bindings: &BTreeMap<String, i64>,
    steps: &BTreeMap<String, i64>,
    rank: usize,
) -> Result<Effects, EmitError> {
    let mut effects = Effects::default();
    visit(plan, statements, bindings, steps, rank, true, &mut effects)?;
    Ok(effects)
}

#[derive(Clone, Debug)]
pub(in crate::emit::cuda) struct Event {
    pub region: Region,
    pub write: bool,
    pub stage: Option<usize>,
}

#[derive(Clone, Debug, Default)]
pub(in crate::emit::cuda) struct Effects {
    pub work: Work,
    pub events: Vec<Event>,
}

impl AccessSink for Effects {
    fn read(&mut self, region: Region, stage: Option<usize>) -> Result<(), EmitError> {
        if let Some(stage) = stage {
            self.work.stages[stage].push(region);
        } else {
            self.work.reads.push(region);
        }
        self.events.push(Event {
            region,
            write: false,
            stage,
        });
        Ok(())
    }
    fn write(&mut self, region: Region) -> Result<(), EmitError> {
        self.work.writes.push(region);
        self.events.push(Event {
            region,
            write: true,
            stage: None,
        });
        Ok(())
    }
    fn stage(&mut self) -> usize {
        let index = self.work.stages.len();
        self.work.stages.push(Vec::new());
        index
    }
    fn properties(&mut self, coordinate: [usize; 3], collective: bool) {
        self.work.coordinate = coordinate;
        self.work.ordered_collective |= collective;
    }
}

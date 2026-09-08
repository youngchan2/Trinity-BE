use std::collections::BTreeSet;

use crate::{ActionId, PhysicalPlan};

/// Rebuilt for each canonical plan because finalization remaps IDs.
pub(super) struct ActionGraph {
    actions: Vec<ActionId>,
    successors: Vec<BTreeSet<ActionId>>,
}

impl ActionGraph {
    pub(super) fn new(plan: &PhysicalPlan) -> Self {
        let actions = plan.actions().map(|(id, _)| id).collect::<Vec<_>>();
        let mut owners = vec![None; plan.operations().len()];
        for (action_id, action) in plan.actions() {
            for operation in action.operations() {
                owners[operation.index()] = Some(action_id);
            }
        }
        let mut producers = vec![None; plan.value_instances().len()];
        for (id, operation) in plan.operations() {
            for output in operation.outputs() {
                producers[output.index()] = owners[id.index()];
            }
        }
        let mut successors = vec![BTreeSet::new(); actions.len()];
        for (id, operation) in plan.operations() {
            let consumer = owners[id.index()].unwrap();
            for input in operation.inputs() {
                if let Some(producer) = producers[input.index()]
                    && producer != consumer
                {
                    successors[producer.index()].insert(consumer);
                }
            }
        }
        Self {
            actions,
            successors,
        }
    }

    pub(super) fn mergeable_pairs(&self) -> impl Iterator<Item = (ActionId, ActionId)> + '_ {
        self.actions.iter().copied().flat_map(move |producer| {
            self.successors[producer.index()]
                .iter()
                .copied()
                .filter(move |consumer| !self.has_alternate_path(producer, *consumer))
                .map(move |consumer| (producer, consumer))
        })
    }

    // Contracting p -> c creates a cycle exactly when another p -> ... -> c
    // path exists. Such a pair is a normal non-match, not a broken rewrite.
    fn has_alternate_path(&self, producer: ActionId, consumer: ActionId) -> bool {
        let mut pending = self.successors[producer.index()]
            .iter()
            .copied()
            .filter(|id| *id != consumer)
            .collect::<Vec<_>>();
        let mut visited = vec![false; self.actions.len()];
        while let Some(id) = pending.pop() {
            if id == consumer {
                return true;
            }
            if !visited[id.index()] {
                visited[id.index()] = true;
                pending.extend(self.successors[id.index()].iter().copied());
            }
        }
        false
    }
}

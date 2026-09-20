use std::collections::BTreeSet;

use crate::PhysicalPlan;

/// Rebuilt for each plan using positions in its top-level statement list.
pub(super) struct StatementGraph {
    statements: Vec<usize>,
    successors: Vec<BTreeSet<usize>>,
}

impl StatementGraph {
    pub(super) fn new(plan: &PhysicalPlan) -> Self {
        let statements = (0..plan.statements().len()).collect::<Vec<_>>();
        let mut owners = vec![None; plan.operations().len()];
        for (statement_index, statement) in plan.statements().iter().enumerate() {
            for operation in statement.operations() {
                owners[operation.index()] = Some(statement_index);
            }
        }
        let mut producers = vec![None; plan.value_instances().len()];
        for (id, operation) in plan.operations() {
            for output in operation.outputs() {
                producers[output.index()] = owners[id.index()];
            }
        }
        let mut successors = vec![BTreeSet::new(); statements.len()];
        for (id, operation) in plan.operations() {
            let consumer = owners[id.index()].unwrap();
            for input in operation.inputs() {
                if let Some(producer) = producers[input.index()]
                    && producer != consumer
                {
                    successors[producer].insert(consumer);
                }
            }
        }
        Self {
            statements,
            successors,
        }
    }

    pub(super) fn mergeable_pairs(&self) -> impl Iterator<Item = (usize, usize)> + '_ {
        self.statements.iter().copied().flat_map(move |producer| {
            self.successors[producer]
                .iter()
                .copied()
                .filter(move |consumer| !self.has_alternate_path(producer, *consumer))
                .map(move |consumer| (producer, consumer))
        })
    }

    // Contracting p -> c creates a cycle exactly when another p -> ... -> c
    // path exists. Such a pair is a normal non-match, not a broken rewrite.
    fn has_alternate_path(&self, producer: usize, consumer: usize) -> bool {
        let mut pending = self.successors[producer]
            .iter()
            .copied()
            .filter(|id| *id != consumer)
            .collect::<Vec<_>>();
        let mut visited = vec![false; self.statements.len()];
        while let Some(id) = pending.pop() {
            if id == consumer {
                return true;
            }
            if !visited[id] {
                visited[id] = true;
                pending.extend(self.successors[id].iter().copied());
            }
        }
        false
    }
}

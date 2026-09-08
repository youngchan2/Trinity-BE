//! Independent interleaving model of the emitted dependency/admission contract.
//! Device queue mechanics and memory ordering require GPU integration tests.
#[allow(dead_code)]
mod support;
use std::collections::{HashSet, VecDeque};
use trinity_lowering::{
    emit,
    emit::{Dependency, Execution, Stage, Task},
};

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct State {
    // 0 pending, 1.. running stage cursor, MAX complete.
    phases: Vec<usize>,
    reserved: Vec<bool>,
    tokens: Vec<u64>,
    counts: Vec<u8>,
}
struct Model<'a> {
    execution: &'a Execution,
    world: usize,
    workers: usize,
    epoch: u64,
}
#[derive(Clone, Debug)]
enum Event {
    Input(usize),
    Claim(usize, bool),
    Advance(usize),
    Complete(Vec<usize>),
}
impl Model<'_> {
    fn index(&self, d: Dependency) -> usize {
        d.rank * (self.execution.tasks_per_rank + 1) + d.slot
    }
    fn ready(&self, s: &State, deps: &[Dependency]) -> bool {
        deps.iter().all(|&d| s.tokens[self.index(d)] == self.epoch)
    }
    fn initial(&self, late: bool) -> State {
        let n = self.execution.tasks.len();
        let mut s = State {
            phases: vec![0; n],
            reserved: vec![false; n],
            tokens: vec![self.epoch - 1; self.world * (self.execution.tasks_per_rank + 1)],
            counts: vec![0; n],
        };
        for rank in 0..self.world {
            if !late || rank == 0 {
                s.tokens[self.index(Dependency {
                    rank,
                    slot: self.execution.tasks_per_rank,
                })] = self.epoch;
            }
        }
        s
    }
    fn events(&self, s: &State) -> Vec<Event> {
        let mut events = Vec::new();
        let mut running = vec![0; self.world];
        let mut active = vec![0; self.world];
        for (i, t) in self.execution.tasks.iter().enumerate() {
            if s.phases[i] > 0 && s.phases[i] != usize::MAX {
                running[t.rank] += 1;
            }
            if s.reserved[i] {
                active[t.rank] += 1;
            }
        }
        for rank in 0..self.world {
            assert!(running[rank] <= self.workers);
            assert!(active[rank] < self.workers);
            if !self.ready(
                s,
                &[Dependency {
                    rank,
                    slot: self.execution.tasks_per_rank,
                }],
            ) {
                events.push(Event::Input(rank));
            }
        }
        let mut collective_slots = HashSet::new();
        for (i, t) in self.execution.tasks.iter().enumerate() {
            match s.phases[i] {
                0 => {
                    if running[t.rank] == self.workers || !self.ready(s, &t.dependencies) {
                        continue;
                    }
                    let may_wait = t
                        .stages
                        .iter()
                        .skip(1)
                        .any(|stage| !self.ready(s, &stage.dependencies));
                    if !may_wait || active[t.rank] < self.workers - 1 {
                        events.push(Event::Claim(i, may_wait));
                    }
                }
                usize::MAX => (),
                phase if t.ordered_collective => {
                    assert_eq!(phase, 1);
                    collective_slots.insert(t.slot);
                    if t.rank == 0 {
                        let participants: Vec<_> = self
                            .execution
                            .tasks
                            .iter()
                            .enumerate()
                            .filter(|(_, other)| other.ordered_collective && other.slot == t.slot)
                            .map(|(index, _)| index)
                            .collect();
                        if participants.len() == self.world
                            && participants.iter().all(|&j| s.phases[j] == 1)
                        {
                            events.push(Event::Complete(participants));
                        }
                    }
                }
                phase => {
                    if phase > t.stages.len() {
                        events.push(Event::Complete(vec![i]));
                    } else if self.ready(s, &t.stages[phase - 1].dependencies) {
                        events.push(Event::Advance(i));
                    }
                }
            }
        }
        assert!(collective_slots.len() <= 1, "concurrent world collectives");
        if events.is_empty() {
            assert!(s.phases.iter().all(|&p| p == usize::MAX), "deadlock: {s:?}");
            assert!(s.counts.iter().all(|&n| n == 1));
            for deps in &self.execution.output_dependencies {
                assert!(self.ready(s, deps), "premature termination");
            }
        }
        events
    }
    fn step(&self, s: &State, event: &Event) -> State {
        let mut s = s.clone();
        match event {
            Event::Input(rank) => {
                s.tokens[self.index(Dependency {
                    rank: *rank,
                    slot: self.execution.tasks_per_rank,
                })] = self.epoch
            }
            Event::Claim(i, reserved) => {
                assert_eq!(s.phases[*i], 0);
                s.phases[*i] = 1;
                s.reserved[*i] = *reserved;
                s.counts[*i] += 1;
                assert_eq!(s.counts[*i], 1);
            }
            Event::Advance(i) => s.phases[*i] += 1,
            Event::Complete(indices) => {
                for &i in indices {
                    assert!(s.phases[i] > 0 && s.phases[i] != usize::MAX);
                    s.phases[i] = usize::MAX;
                    s.reserved[i] = false;
                    let t = &self.execution.tasks[i];
                    s.tokens[self.index(Dependency {
                        rank: t.rank,
                        slot: t.slot,
                    })] = self.epoch;
                }
            }
        }
        s
    }
}
fn dependency(rank: usize, slot: usize) -> Dependency {
    Dependency { rank, slot }
}
fn toy(collectives: bool) -> Execution {
    let count = if collectives { 4 } else { 3 };
    let mut tasks = Vec::new();
    for rank in 0..2 {
        for slot in 0..count {
            let (dependencies, stages, ordered_collective) = if collectives {
                match slot {
                    0 => (vec![dependency(rank, count)], vec![vec![]], false),
                    1 => ((0..2).map(|r| dependency(r, 0)).collect(), vec![], true),
                    2 => (
                        vec![dependency(rank, 1)],
                        vec![vec![dependency(rank, 1)], vec![dependency(rank, 3)]],
                        false,
                    ),
                    3 => ((0..2).map(|r| dependency(r, 1)).collect(), vec![], true),
                    _ => unreachable!(),
                }
            } else if slot == 0 {
                (vec![dependency(1 - rank, count)], vec![], false)
            } else {
                (vec![], vec![vec![], vec![dependency(rank, 0)]], false)
            };
            tasks.push(Task {
                rank,
                slot,
                action: slot,
                operation: slot,
                coordinate: [0; 3],
                shared_memory_bytes: 0,
                dependencies,
                stages: stages
                    .into_iter()
                    .map(|dependencies| Stage { dependencies })
                    .collect(),
                ordered_collective,
            });
        }
    }
    Execution {
        tasks,
        tasks_per_rank: count,
        output_dependencies: (0..2)
            .map(|rank| vec![dependency(rank, count - 1)])
            .collect(),
        launches: vec![],
    }
}
#[test]
fn exhaustively_checks_two_workers_late_sources_full_credits_and_collective_order() {
    for collectives in [false, true] {
        let execution = toy(collectives);
        let model = Model {
            execution: &execution,
            world: 2,
            workers: 2,
            epoch: 2,
        };
        let initial = model.initial(true);
        let mut seen = HashSet::from([initial.clone()]);
        let mut queue = VecDeque::from([initial]);
        while let Some(state) = queue.pop_front() {
            for event in model.events(&state) {
                let next = model.step(&state, &event);
                if seen.insert(next.clone()) {
                    queue.push_back(next);
                }
            }
        }
        assert!(seen.len() > 100);
        eprintln!("{collectives}: {} complete interleaving states", seen.len());
    }
}
#[test]
fn occupied_credit_does_not_block_a_ready_producer_or_fully_ready_gemm() {
    let execution = toy(false);
    let model = Model {
        execution: &execution,
        world: 2,
        workers: 2,
        epoch: 1,
    };
    let s = model.initial(true);
    let s = model.step(&s, &Event::Claim(1, true));
    assert!(
        !model
            .events(&s)
            .iter()
            .any(|e| matches!(e, Event::Claim(2, _)))
    );
    let s = model.step(&s, &Event::Input(1));
    assert!(
        model
            .events(&s)
            .iter()
            .any(|e| matches!(e, Event::Claim(0, false)))
    );
    let s = model.step(&s, &Event::Claim(0, false));
    let s = model.step(&s, &Event::Complete(vec![0]));
    assert!(
        model
            .events(&s)
            .iter()
            .any(|e| matches!(e, Event::Claim(2, false)))
    );
}
#[test]
fn generated_plans_complete_under_varied_dispatch_and_repeated_epochs() {
    let mut plans = vec![
        support::gemm(256, 256, 192, 2),
        support::peer_chain("peer_push", "peer_pull", 2),
        support::peer_chain("peer_pull", "peer_push", 2),
        support::peer_chain("one_shot_push_nbi", "one_shot_push_nbi", 2),
    ];
    for backend in ["peer_push", "peer_pull", "one_shot_push_nbi"] {
        for axis in 0..2 {
            plans.push(support::lhs_gather(backend, axis, 2));
            plans.push(support::input_gather(backend, axis, 2, false));
            plans.push(support::input_gather(backend, axis, 2, true));
            plans.push(support::output_gather(backend, axis, 2));
        }
    }
    for plan in plans {
        let source = emit(&plan).unwrap();
        let execution = source.execution();
        for seed in 1..=24 {
            let mut random = seed;
            let mut previous = None;
            for epoch in 1..=3 {
                let model = Model {
                    execution,
                    world: 2,
                    workers: 2,
                    epoch,
                };
                let mut state = model.initial(true);
                if let Some(tokens) = previous.take() {
                    state.tokens = tokens;
                    assert!(
                        execution
                            .tasks
                            .iter()
                            .all(|t| !model.ready(&state, &[dependency(t.rank, t.slot)])),
                        "stale epoch accepted"
                    );
                }
                let mut steps = 0;
                loop {
                    let events = model.events(&state);
                    if events.is_empty() {
                        break;
                    }
                    random ^= random << 13;
                    random ^= random >> 7;
                    random ^= random << 17;
                    state = model.step(&state, &events[random as usize % events.len()]);
                    steps += 1;
                    assert!(steps < execution.tasks.len() * 32 + 32);
                }
                previous = Some(state.tokens);
            }
        }
    }
}

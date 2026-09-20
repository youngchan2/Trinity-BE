//! Fixtures built exclusively through the public physical-plan API.
pub mod loop_ir;
pub mod tensor;
use trinity_lowering::*;

const TARGET: TargetCapability = TargetCapability::Cuda(CudaTargetCapability::Hopper);

pub struct Builder {
    world: usize,
    shapes: Vec<[usize; 2]>,
    inputs: Vec<(String, usize)>,
    operations: Vec<(Vec<usize>, usize, OperationPayload)>,
}
impl Builder {
    pub fn new(world: usize) -> Self {
        Self {
            world,
            shapes: vec![],
            inputs: vec![],
            operations: vec![],
        }
    }
    pub fn input(&mut self, name: &str, shape: [usize; 2]) -> usize {
        let id = self.shapes.len();
        self.shapes.push(shape);
        self.inputs.push((name.to_owned(), id));
        id
    }
    pub fn gemm(&mut self, a: usize, b: usize, m: usize, n: usize, k: usize) -> usize {
        let out = self.shapes.len();
        self.shapes.push([m, n]);
        let implementation = gemm_implementations(TARGET)[0]
            .enumerate([DType::Bf16; 3], [&[m, k], &[k, n], &[m, n]])
            .pop()
            .unwrap();
        self.operations.push((
            vec![a, b],
            out,
            OperationPayload::Compute(ComputeOperation::new(implementation)),
        ));
        out
    }
    pub fn gather(
        &mut self,
        input: usize,
        shape: [usize; 2],
        axis: usize,
        world: usize,
        backend: &str,
    ) -> usize {
        let mut target = shape;
        target[axis] *= world;
        let out = self.shapes.len();
        self.shapes.push(target);
        let implementation = all_gather_implementations(TARGET)
            .iter()
            .find(|b| b.id().as_str().ends_with(backend))
            .unwrap()
            .enumerate(DType::Bf16, [&shape, &target], axis, world)
            .pop()
            .unwrap();
        self.operations.push((
            vec![input],
            out,
            OperationPayload::Communication(CommunicationOperation::new(
                CommunicationKind::AllGather,
                implementation,
            )),
        ));
        out
    }
    pub fn finish(self, output: usize) -> PhysicalPlan {
        let mut b = PhysicalPlanBuilder::new(TARGET, self.world);
        let values: Vec<_> = self
            .shapes
            .iter()
            .enumerate()
            .map(|(i, shape)| {
                let storage = if i == output || self.inputs.iter().any(|(_, v)| *v == i) {
                    Storage::External
                } else {
                    Storage::Global
                };
                b.add_value(DType::Bf16, *shape, storage)
            })
            .collect();
        for (name, id) in self.inputs {
            b.bind_input(name, values[id]);
        }
        for (inputs, out, payload) in self.operations {
            let op = b.add_operation(
                inputs.into_iter().map(|v| values[v]),
                [values[out]],
                payload,
            );
            b.add_statement(trinity_lowering::Statement::Operation(op));
        }
        b.finalize("result", values[output]).unwrap()
    }
}

pub fn gemm(m: usize, n: usize, k: usize, world: usize) -> PhysicalPlan {
    let mut b = Builder::new(world);
    let x = b.input("X with spaces\"", [m, k]);
    let w = b.input("W/weight", [k, n]);
    let y = b.gemm(x, w, m, n, k);
    b.finish(y)
}
pub fn gemm_chain() -> PhysicalPlan {
    gemm_chain_world(1)
}
pub fn gemm_chain_world(world: usize) -> PhysicalPlan {
    let mut b = Builder::new(world);
    let x = b.input("X", [256, 192]);
    let w = b.input("W", [192, 256]);
    let v = b.input("V", [256, 128]);
    let y = b.gemm(x, w, 256, 256, 192);
    let z = b.gemm(y, v, 256, 128, 256);
    b.finish(z)
}
pub fn gather(backend: &str, axis: usize, world: usize) -> PhysicalPlan {
    let mut b = Builder::new(world);
    let shape = [256, 256];
    let x = b.input("shard", shape);
    let y = b.gather(x, shape, axis, world, backend);
    b.finish(y)
}
pub fn input_gather(backend: &str, axis: usize, world: usize, producer: bool) -> PhysicalPlan {
    let mut b = Builder::new(world);
    let shape = [256, 256];
    let w = if producer {
        let a = b.input("P", [256, 192]);
        let c = b.input("Q", [192, 256]);
        b.gemm(a, c, 256, 256, 192)
    } else {
        b.input("W", shape)
    };
    let mut full = shape;
    full[axis] *= world;
    let x = b.input("X", [256, full[0]]);
    let gathered = b.gather(w, shape, axis, world, backend);
    let y = b.gemm(x, gathered, 256, full[1], full[0]);
    b.finish(y)
}
pub fn output_gather(backend: &str, axis: usize, world: usize) -> PhysicalPlan {
    let mut b = Builder::new(world);
    let x = b.input("X", [256, 192]);
    let w = b.input("W", [192, 256]);
    let y = b.gemm(x, w, 256, 256, 192);
    let z = b.gather(y, [256, 256], axis, world, backend);
    b.finish(z)
}
pub fn peer_chain(first: &str, second: &str, world: usize) -> PhysicalPlan {
    let mut b = Builder::new(world);
    let x = b.input("X", [256, 256]);
    let y = b.gather(x, [256, 256], 0, world, first);
    let z = b.gather(y, [256 * world, 256], 1, world, second);
    b.finish(z)
}

pub fn lhs_gather(backend: &str, axis: usize, world: usize) -> PhysicalPlan {
    let mut b = Builder::new(world);
    let shape = [256, 256];
    let x = b.input("X", shape);
    let mut full = shape;
    full[axis] *= world;
    let w = b.input("W", [full[1], 256]);
    let gathered = b.gather(x, shape, axis, world, backend);
    let y = b.gemm(gathered, w, full[0], 256, full[1]);
    b.finish(y)
}

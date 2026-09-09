#![allow(dead_code)]
use trinity_lowering::*;
pub const TARGET: TargetCapability = TargetCapability::Cuda(CudaTargetCapability::Hopper);

pub struct Builder {
    pub plan: PhysicalPlanBuilder,
    values: Vec<(DType, Vec<usize>)>,
}

impl Builder {
    pub fn new(world: usize) -> Self {
        Self {
            plan: PhysicalPlanBuilder::new(TARGET, world),
            values: vec![],
        }
    }
    pub fn value(&mut self, dtype: DType, shape: &[usize], storage: Storage) -> ValueInstanceId {
        self.values.push((dtype, shape.to_vec()));
        self.plan.add_value(dtype, shape.iter().copied(), storage)
    }
    pub fn input(&mut self, name: &str, dtype: DType, shape: &[usize]) -> ValueInstanceId {
        let v = self.value(dtype, shape, Storage::External);
        self.plan.bind_input(name, v);
        v
    }
    pub fn operation(
        &mut self,
        inputs: &[ValueInstanceId],
        output: ValueInstanceId,
        implementation: ImplementationInstance,
    ) {
        let op = self.plan.add_operation(
            inputs.iter().copied(),
            [output],
            OperationPayload::Compute(ComputeOperation::new(implementation)),
        );
        self.plan.add_action([op]);
    }
    pub fn pointwise(
        &mut self,
        name: &str,
        inputs: &[ValueInstanceId],
        dtype: DType,
        scalar: Option<f32>,
        storage: Storage,
    ) -> ValueInstanceId {
        let shape = self.values[inputs[0].index()].1.clone();
        let output = self.value(dtype, &shape, storage);
        let values: Vec<_> = inputs
            .iter()
            .chain([&output])
            .map(|v| &self.values[v.index()])
            .collect();
        let dtypes: Vec<_> = values.iter().map(|v| v.0).collect();
        let shapes: Vec<_> = values.iter().map(|v| v.1.as_slice()).collect();
        let implementation = pointwise_implementations(TARGET)
            .iter()
            .find(|d| d.id().as_str() == format!("cuda.{name}"))
            .unwrap()
            .enumerate(&dtypes, &shapes, scalar)
            .pop()
            .unwrap();
        self.operation(inputs, output, implementation);
        output
    }
    pub fn reduce(&mut self, input: ValueInstanceId, storage: Storage) -> ValueInstanceId {
        let (dtype, shape) = self.values[input.index()].clone();
        let output = self.value(DType::Fp32, &shape[..1], storage);
        let implementation = reduce_sum_implementations(TARGET)[0]
            .enumerate([dtype, DType::Fp32], [&shape, &shape[..1]], 1)
            .pop()
            .unwrap();
        self.operation(&[input], output, implementation);
        output
    }
    pub fn broadcast(
        &mut self,
        input: ValueInstanceId,
        columns: usize,
        storage: Storage,
    ) -> ValueInstanceId {
        let (dtype, shape) = self.values[input.index()].clone();
        let target = [shape[0], columns];
        let output = self.value(dtype, &target, storage);
        let implementation = broadcast_implementations(TARGET)[0]
            .enumerate(dtype, [&shape, &target], 1)
            .pop()
            .unwrap();
        self.operation(&[input], output, implementation);
        output
    }
    pub fn finish(self, output: ValueInstanceId) -> PhysicalPlan {
        self.plan.finalize("Y", output).unwrap()
    }
}

pub fn pointwise(
    name: &str,
    shape: &[usize],
    dtypes: &[DType],
    scalar: Option<f32>,
    world: usize,
) -> PhysicalPlan {
    let mut b = Builder::new(world);
    let inputs: Vec<_> = dtypes[..dtypes.len() - 1]
        .iter()
        .enumerate()
        .map(|(i, &dtype)| b.input(&format!("X{i}"), dtype, shape))
        .collect();
    let out = b.pointwise(
        name,
        &inputs,
        dtypes[dtypes.len() - 1],
        scalar,
        Storage::External,
    );
    b.finish(out)
}

pub fn normalization(m: usize, n: usize, world: usize) -> PhysicalPlan {
    let mut b = Builder::new(world);
    let x = b.input("X", DType::Bf16, &[m, n]);
    let square = b.pointwise("square", &[x], DType::Fp32, None, Storage::Global);
    let sum = b.reduce(square, Storage::Global);
    let mean = b.pointwise(
        "scalar_div",
        &[sum],
        DType::Fp32,
        Some(n as f32),
        Storage::Global,
    );
    let root = b.pointwise("sqrt", &[mean], DType::Fp32, None, Storage::Global);
    let denom = b.broadcast(root, n, Storage::Global);
    let output = b.pointwise("div", &[x, denom], DType::Bf16, None, Storage::External);
    b.finish(output)
}

pub fn silu(m: usize, n: usize, world: usize) -> PhysicalPlan {
    let mut b = Builder::new(world);
    let x = b.input("X", DType::Bf16, &[m, n]);
    let sig = b.pointwise("sigmoid", &[x], DType::Fp32, None, Storage::Global);
    let out = b.pointwise("mul", &[sig, x], DType::Bf16, None, Storage::External);
    b.finish(out)
}

pub fn gather_normalization(world: usize) -> PhysicalPlan {
    let mut b = Builder::new(world);
    let x = b.input("X", DType::Bf16, &[128, 128]);
    let gathered = b.value(DType::Bf16, &[128, 128 * world], Storage::Global);
    let imp = all_gather_implementations(TARGET)
        .iter()
        .find(|d| d.id().as_str().ends_with("peer_pull"))
        .unwrap()
        .enumerate(DType::Bf16, [&[128, 128], &[128, 128 * world]], 1, world)
        .pop()
        .unwrap();
    let op = b.plan.add_operation(
        [x],
        [gathered],
        OperationPayload::Communication(CommunicationOperation::new(
            CommunicationKind::AllGather,
            imp,
        )),
    );
    b.plan.add_action([op]);
    let square = b.pointwise("square", &[gathered], DType::Fp32, None, Storage::Global);
    let sum = b.reduce(square, Storage::External);
    b.finish(sum)
}

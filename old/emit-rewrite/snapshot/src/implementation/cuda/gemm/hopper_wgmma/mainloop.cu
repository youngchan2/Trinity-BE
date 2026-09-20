  int ${stage_base}=${stage_cursor};
  int ${stage_count}=(${STOP}-${START})/64;
  if (!runtime.await_stage(${stage_base})) return false;
  ${load}(${START},0);
  for(std::int64_t ${VAR}=${START},${iteration}=0;${VAR}<${STOP};${VAR}+=64,++${iteration}) {
    cp_async_wait<0>(); __syncthreads();
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
    bool ${prefetch}=runtime.prefetch_stage(${stage_base}+${iteration}+1,${stage_base}+${stage_count});
    if(${prefetch}) ${load}(${VAR}+64,(${iteration}+1)%2);
    auto ${sa}=make_tensor(make_smem_ptr(${shared}.a[${iteration}%2].begin()),${la});
    auto ${sb}=make_tensor(make_smem_ptr(${shared}.b[${iteration}%2].begin()),${lb});
    auto ${fa}=${tmma}.make_fragment_A(${tmma}.partition_A(${sa}));
    auto ${fb}=${tmma}.make_fragment_B(${tmma}.partition_B(${sb}));
    warpgroup_fence_operand(${accumulator}); warpgroup_arrive();
    cute::gemm(${mma},${fa},${fb},${accumulator});
    warpgroup_commit_batch(); warpgroup_wait<0>(); warpgroup_fence_operand(${accumulator});
    __syncthreads();
    if(${iteration}+1<${stage_count}&&!${prefetch}) {
      if(!runtime.await_stage(${stage_base}+${iteration}+1)) return false;
      ${load}(${VAR}+64,(${iteration}+1)%2);
    }
  }
  ${stage_cursor}+=${stage_count};

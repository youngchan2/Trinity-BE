{
  using namespace cute;

  using ${Element} = cutlass::bfloat16_t;

  auto ${la} = tile_to_shape(GMMA::Layout_K_SW128_Atom<${Element}>{}, make_shape(Int<128>{}, Int<64>{}));
  auto ${lb} = tile_to_shape(GMMA::Layout_MN_SW128_Atom<${Element}>{}, make_shape(Int<128>{}, Int<64>{}));

  struct ${Shared} {
    alignas(128) ArrayEngine<${Element}, cosize_v<decltype(${la})>> a[2];
    alignas(128) ArrayEngine<${Element}, cosize_v<decltype(${lb})>> b[2];
  };

  static_assert(sizeof(${Shared})==65536);

  auto& ${shared} = *static_cast<${Shared}*>(memory);
  auto ${mma} = make_tiled_mma(SM90_64x64x16_F32BF16BF16_SS<GMMA::Major::K,GMMA::Major::MN>{});
  auto ${tmma}=${mma}.get_slice(threadIdx.x);
  auto ${cc}=${tmma}.partition_C(make_identity_tensor(make_shape(Int<128>{},Int<128>{})));
  auto ${accumulator}=${tmma}.make_fragment_C(${cc});
  ${INITIALIZER}
  auto ${load}=[&](std::int64_t ${VAR},int ${slot}) {
    auto ${sa}=make_tensor(make_smem_ptr(${shared}.a[${slot}].begin()),${la});
    auto ${sb}=make_tensor(make_smem_ptr(${shared}.b[${slot}].begin()),${lb});
    for(int ${linear}=threadIdx.x*8;${linear}<128*64;${linear}+=blockDim.x*8) {
      int ${row}=${linear}/64, ${col}=${linear}%64;
      unsigned ${dst}=static_cast<unsigned>(__cvta_generic_to_shared(&${sa}(${row},${col})));
      int ${bytes}=${row}<${M}?16:0;
      auto ${src}=${A}+(${row}<${M}?(${AO}):0);
      ${COPY_A}
    }
    for(int ${linear}=threadIdx.x*8;${linear}<128*64;${linear}+=blockDim.x*8) {
      int ${row}=${linear}/128, ${col}=${linear}%128;
      unsigned ${dst}=static_cast<unsigned>(__cvta_generic_to_shared(&${sb}(${col},${row})));
      auto ${src}=${B}+(${BO});
      asm volatile("cp.async.ca.shared.global [%0], [%1], 16;" :: "r"(${dst}), "l"(${src}) : "memory");
    }
    cp_async_fence();
  };

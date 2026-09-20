  CUTE_UNROLL
  for(int ${i}=0;${i}<size(${accumulator});++${i}) {
    if(get<0>(${cc}(${i}))<${M}) { ${OUTPUT} }
  }
  __syncthreads();
}

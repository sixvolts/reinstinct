// Entry-point set for a matvec_batched_nr.h decoder; included after the
// decoder is defined. BNR_R<n> is the rows per wave of entry n, BNR_L<n>
// whether it stages activations in LDS; the defaults are the gfx906 tune
// (prefill.rs bench_small_batch_matmul) and a kernel file may override
// one before the include. The host sizes the grid from the same numbers
// (prefill.rs SmallBatchMatvec::entry) — keep the two in step.
#ifndef BNR_R1
#define BNR_R1 2
#endif
#ifndef BNR_L1
#define BNR_L1 false
#endif
#ifndef BNR_R2
#define BNR_R2 4
#endif
#ifndef BNR_L2
#define BNR_L2 true
#endif
#ifndef BNR_R3
#define BNR_R3 4
#endif
#ifndef BNR_L3
#define BNR_L3 true
#endif
#ifndef BNR_R4
#define BNR_R4 4
#endif
#ifndef BNR_L4
#define BNR_L4 true
#endif
#ifndef BNR_R8
#define BNR_R8 2
#endif
#ifndef BNR_L8
#define BNR_L8 true
#endif
#define BNR_ENTRIES(PREFIX, W)                                  \
    BNR_ENTRY(PREFIX, W, n1,  BNR_R1,  1,  BNR_L1,  true)       \
    BNR_ENTRY(PREFIX, W, n2,  BNR_R2,  2,  BNR_L2,  true)       \
    BNR_ENTRY(PREFIX, W, n3,  BNR_R3,  3,  BNR_L3,  true)       \
    BNR_ENTRY(PREFIX, W, n4,  BNR_R4,  4,  BNR_L4,  true)       \
    BNR_ENTRY(PREFIX, W, w8,  BNR_R8,  8,  BNR_L8,  false)

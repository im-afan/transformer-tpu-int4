/* infer_rq.h — the default requant table for infer.c.
 *
 * 16 {m0,n} words per layer, in the block order that kernel's enum declares.
 * The word is m0 in the low 12 bits and n above, and the op it feeds computes
 * `clip((acc*m0 + 2**(n-1)) >> n)` — REQUANT to [-8, 7], DYT to [-7, 7],
 * QUANT4 to [-8, 7] written 4 bits wide.
 *
 * THESE ARE NOT A CHECKPOINT'S SCALES. They are tuned for the SYNTHETIC
 * operands accel/tpulang/fw_vectors.py stages, so `make fw FWPROG=infer` is a
 * self-contained datapath regression with no model file involved. Too small and
 * every tensor pins at the clip, too large and the model collapses to zeros —
 * and a golden answer of all zeros passes against any datapath at all, which is
 * why fw_vectors.py's reference warns when the generated sequence is constant.
 *
 * WHY THIS IS NOT adder_rq.h. The two kernels shared one table while they were
 * the same model. They are both `adder_int4_wide` now — d=128, f=512 — but
 * adder.c runs the whole T=128 sequence and this one runs T=64, so RQ_A, whose
 * shift is set by the contraction over keys, differs. The rest of the row is
 * the same arithmetic and would agree; keeping a separate header is what stops
 * a later change to one kernel's shape from silently moving the other's.
 *
 * Each shift is one bit per doubling of the contraction that feeds it, which is
 * how these were carried over from the d=64 / T=32 table: Q/K/V/O/H gained one
 * (D 64 -> 128), S gained one (head_dim 16 -> 32), A gained one (T 32 -> 64)
 * and F gained one (DFF 256 -> 512).
 *
 * A real run overrides this file wholesale:
 *
 *   python accel/tpulang/infer_export.py --model-path model/saved/int4_d128_f512_l4.pt
 *
 * writes the same ADDER_RQ_INIT macro from the checkpoint's learned ActQuant
 * scales and Int4Linear weight scales, and builds against it with -DADDER_RQ_H.
 */
#ifndef INFER_RQ_H_DEFAULT
#define INFER_RQ_H_DEFAULT

#define RQW(m0, n) ((uint16_t)(((n) << 12) | (m0)))
#define RQ_ONE     RQW(1u, 0u)

/* The same row at every layer: the synthetic operands have the same shape at
 * every depth, so nothing here varies with the layer index.
 *
 *   Q  K  V   contract over D=128       S  contracts over head_dim=32
 *   A         contracts over T=64       O  contracts over D=128
 *   H         contracts over D=128      F  contracts over DFF=512
 *   X1 X2     add two int4 tensors, so the input is bounded by 16
 */
#define ADDER_RQ_ROW                                                          \
    { RQW(1u, 6u), RQW(1u, 6u), RQW(1u, 6u),   /* Q  K  V   */                \
      RQ_ONE,      RQ_ONE,                     /* KP VP     */                \
      RQW(1u, 4u),                             /* S         */                \
      RQ_ONE,      RQ_ONE,                     /* ID P      */                \
      RQW(1u, 6u),                             /* A         */                \
      RQW(1u, 6u), RQ_ONE,                     /* O  XO     */                \
      RQW(1u, 1u),                             /* X1        */                \
      RQW(1u, 6u), RQ_ONE,                     /* H  HR     */                \
      RQW(1u, 7u),                             /* F         */                \
      RQW(1u, 1u) }                            /* X2        */

#define ADDER_RQ_INIT { ADDER_RQ_ROW, ADDER_RQ_ROW, ADDER_RQ_ROW, ADDER_RQ_ROW }

#endif /* INFER_RQ_H_DEFAULT */

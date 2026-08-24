/* adder_rq.h — the default requant table for adder.c and infer.c.
 *
 * 16 {m0,n} words per layer, in the block order those kernels' enum declares.
 * The word is m0 in the low 12 bits and n above, and the op it feeds computes
 * `clip((acc*m0 + 2**(n-1)) >> n)` — REQUANT to [-8, 7], DYT to [-7, 7],
 * QUANT4 to [-8, 7] written 4 bits wide.
 *
 * THESE ARE NOT A CHECKPOINT'S SCALES. They are tuned for the SYNTHETIC
 * operands accel/tpulang/fw_vectors.py stages, so `make fw FWPROG=adder` is a
 * self-contained datapath regression with no model file involved. The shifts
 * were measured, not derived: too small and every tensor pins at the clip, too
 * large and the model collapses to zeros — and a golden answer of all zeros
 * passes against any datapath at all.
 *
 * A real run overrides this file wholesale:
 *
 *   python accel/tpulang/adder_export.py --model-path model/saved/int4_d64_f256_l4.pt
 *
 * writes the same ADDER_RQ_INIT macro from the checkpoint's learned ActQuant
 * scales and Int4Linear weight scales, and builds against it with -DADDER_RQ_H.
 */
#ifndef ADDER_RQ_H_DEFAULT
#define ADDER_RQ_H_DEFAULT

#define RQW(m0, n) ((uint16_t)(((n) << 12) | (m0)))
#define RQ_ONE     RQW(1u, 0u)

/* The same row at every layer: the synthetic operands have the same shape at
 * every depth, so nothing here varies with the layer index.
 *
 *   Q  K  V   contract over D=64        S  contracts over head_dim=16
 *   A         contracts over T=32       O  contracts over D=64
 *   H         contracts over D=64       F  contracts over DFF=256
 *   X1 X2     add two int4 tensors, so the input is bounded by 16
 */
#define ADDER_RQ_ROW                                                          \
    { RQW(1u, 5u), RQW(1u, 5u), RQW(1u, 5u),   /* Q  K  V   */                \
      RQ_ONE,      RQ_ONE,                     /* KP VP     */                \
      RQW(1u, 3u),                             /* S         */                \
      RQ_ONE,      RQ_ONE,                     /* ID P      */                \
      RQW(1u, 5u),                             /* A         */                \
      RQW(1u, 5u), RQ_ONE,                     /* O  XO     */                \
      RQW(1u, 1u),                             /* X1        */                \
      RQW(1u, 5u), RQ_ONE,                     /* H  HR     */                \
      RQW(1u, 6u),                             /* F         */                \
      RQW(1u, 1u) }                            /* X2        */

#define ADDER_RQ_INIT { ADDER_RQ_ROW, ADDER_RQ_ROW, ADDER_RQ_ROW, ADDER_RQ_ROW }

#endif /* ADDER_RQ_H_DEFAULT */

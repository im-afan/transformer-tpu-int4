/* infer_rq.h — the default requant table for infer.c. Tuned for the synthetic
 * operands fw_vectors.py stages, NOT a checkpoint's scales — a real run
 * overrides this wholesale via infer_export.py -DADDER_RQ_H. See docs/fw.md. */
#ifndef INFER_RQ_H_DEFAULT
#define INFER_RQ_H_DEFAULT

#define RQW(m0, n) ((uint16_t)(((n) << 12) | (m0)))
#define RQ_ONE     RQW(1u, 0u)

/* The same row at every layer: the synthetic operands have the same shape at
 * every depth, so nothing here varies with the layer index.
 *
 *   Q  K  V   contract over D=64        S  contracts over head_dim=16
 *   A         contracts over T=64        O  contracts over D=64
 *   H         contracts over D=64        F  contracts over DFF=256
 *   X1 X2     add two int4 tensors, so the input is bounded by 16
 */
#define ADDER_RQ_ROW                                                          \
    { RQW(1u, 6u), RQW(1u, 6u), RQW(1u, 6u),   /* Q  K  V   */                \
      RQ_ONE,      RQ_ONE,                     /* retired   */                \
      RQW(1u, 4u),                             /* S         */                \
      RQ_ONE,      RQ_ONE,                     /* ID P      */                \
      RQW(1u, 6u),                             /* A         */                \
      RQW(1u, 6u), RQ_ONE,                     /* O  XO     */                \
      RQW(1u, 1u),                             /* X1        */                \
      RQW(1u, 6u), RQ_ONE,                     /* H  HR     */                \
      RQW(1u, 7u),                             /* F         */                \
      RQW(1u, 1u) }                            /* X2        */

#define ADDER_RQ_INIT { ADDER_RQ_ROW, ADDER_RQ_ROW, ADDER_RQ_ROW, ADDER_RQ_ROW }

/* The output head, which is not per-layer and so is not in the row above. The
 * MXU requantizes on store, so the logits are int4 and this is what decides
 * whether an argmax over them can separate anything: too small a shift pins
 * every logit at the clip and the answer is token 0 every time. The
 * contraction is over D=64 int4 pairs; measured on the synthetic operands the
 * raw head accumulator spans about +-40, so a shift of 3 lands it across the
 * grid without reaching the clip. */
#ifndef INFER_RQ_LOGIT
#define INFER_RQ_LOGIT RQW(1u, 3u)
#endif

#endif /* INFER_RQ_H_DEFAULT */

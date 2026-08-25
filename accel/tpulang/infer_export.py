#!/usr/bin/env python3
"""infer_export.py — a real checkpoint *generating*, through fw/infer.c.

`adder_export.py` scores the model the way training does: hand the kernel a
finished sequence, read every position's logits, count how many argmaxes match
the token that was already there. That measures the arithmetic, and it is
teacher forcing — at position 20 the kernel is told what the model said at 19,
whatever it actually said.

This scores the same checkpoint the way it would be used. The device is handed
the prompt's token ids and nothing else; it prefills, then decodes its own
output token by token against a KV cache, and what comes back is a sequence it
chose. A single wrong digit derails everything after it, which is exactly the
property teacher forcing hides.

    python accel/tpulang/infer_export.py -n 16
    python accel/tpulang/infer_export.py -n 4 --show

Three things happen, in order — the same three as `adder_export.py`, and the
first two are literally its code:

1. **Derive.** `adder_export.derive` turns the checkpoint into int4 weights and
   the 16 {m0,n} words per layer. infer.c and adder.c are the same model and the
   same arithmetic in a different order, so they take the same table, unchanged
   — the two kernels differ only in sequence length, and nothing in the table
   depends on T.
2. **Build.** The table is a compile-time input (the requant word is a literal
   in the macro-op), so this writes a header and compiles `infer.c` natively
   against it with the host cc.
3. **Generate.** One co-execution run per problem: the ISS executes the kernel's
   commands as it emits them and answers the scratchpad reads its argmax makes,
   which is what a data-dependent kernel needs from a simulator. Weights, mask,
   head and embedding table are staged once and stay; a problem writes 32 int32
   token ids and reads 32 back.

WHAT THE HOST STILL DOES: **tokenize**. Not the embedding (the table is a DRAM
tensor and the gather is a DMA at a computed address) and not the argmax (the
logits land in the scratchpad and the CPU reads them). Compare adder.c, where
both were the host's — the ISA has not changed, cpu_subsys.sv's scratchpad
window was simply always there.

THE SCRATCHPAD IS NOT CLEARED BETWEEN PROBLEMS, here or on the board. The KV
cache from problem n-1 is still sitting there when problem n starts, and the
mask is what makes it harmless — so running the problems through one TPU
instance is not a shortcut, it is the test.
"""
from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
import tempfile

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch  # noqa: E402

import model.numbers_data as numbers_data  # noqa: E402
import adder_export as ax  # noqa: E402
from adder_export import (FW_DIR, Q4_MAX, Q4_MIN, act_scale, derive,  # noqa: E402
                          load_model, write_rq_header)
import fw_vectors as fv  # noqa: E402
from fw_vectors import coexecute  # noqa: E402
from iss import TPU  # noqa: E402

# Geometry and the DRAM map, from fw/infer.c. The six WEIGHT blocks per layer
# are adder.c's, at the same addresses — both kernels are `adder_int4_wide` —
# but everything else is this kernel's own and is COMPUTED off the shape, so it
# is taken from `fw_vectors` rather than restated here. That is the same chain
# fw/infer.c's DR_* macros walk; see the note above them for why it moved.
T, D, VOCAB, VPAD = fv.IN_T, fv.IN_D, fv.IN_VOCAB, fv.IN_VPAD
# 32 — the first ANSWER position, PINNED rather than read from numbers_data.
# numbers_data.EQUALS_POS is 64 (fw/adder.c's prompt); this kernel's is half
# that, so both it and the operand ceiling it implies are passed back into the
# generator explicitly. MAX_DIGITS is (PROMPT - 2) // 2: both operands and the
# '+' have to fit before the '=' at PROMPT - 1.
PROMPT = fv.IN_PROMPT
MAX_DIGITS = (PROMPT - 2) // 2
NGEN = T - PROMPT
DR_EMB, DR_TOK, DR_MASK = fv.IN_EMB, fv.IN_TOK, fv.IN_MASK
DR_LOG, DR_WFC = fv.IN_LOG, fv.IN_WFC
LOGIT_ROW = VPAD * 4                      # bytes per position in DR_LOG
ROWS = COLS = 8


def layout(batch: int = 1) -> dict:
    """fw/infer.c's DRAM map at this BATCH — the chain its DR_* macros walk.

    The token block and the logit block are per sequence, so everything below
    them (the mask, the output head, the caches) moves when BATCH does. The
    weights do not: they sit at adder.c's fixed addresses above 0x20000.
    """
    align = lambda a: (a + 63) & ~63
    emb = 0
    tok = align(emb + VOCAB * D)
    log = align(tok + batch * T * 4)
    mask = align(log + batch * T * VPAD * 4)
    wfc = align(mask + T * T)
    return {"emb": emb, "tok": tok, "log": log, "mask": mask, "wfc": wfc}


# The module-level DR_* are fw_vectors' single-sequence constants; this is the
# check that the chain above is the same one.
assert layout(1) == {"emb": DR_EMB, "tok": DR_TOK, "log": DR_LOG,
                     "mask": DR_MASK, "wfc": DR_WFC}


def token_addr(seq: int, pos: int, batch: int = 1) -> int:
    """Where sequence `seq`'s token id at `pos` lives in DRAM."""
    return layout(batch)["tok"] + (seq * T + pos) * 4


def logit_addr(seq: int, pos: int, batch: int = 1) -> int:
    """Where sequence `seq`'s VPAD logits for position `pos` live in DRAM."""
    return layout(batch)["log"] + (seq * T + pos) * LOGIT_ROW


PHASE_FLAGS = {
    "both": (),
    "prefill": ("-DINFER_PREFILL=1", "-DINFER_DECODE=0"),
    "decode": ("-DINFER_PREFILL=0", "-DINFER_DECODE=1"),
}


def build_kernel(rq_header: str, workdir: str, gen: int, phase: str = "both",
                 batch: int = 1, block=None) -> str:
    """Compile infer.c natively against `rq_header`; return the binary's path.

    Unlike adder_export's `build_trace`, this does not run it: the command
    stream depends on the tokens the kernel chooses, so there is no trace until
    something is answering its reads. `coexecute` runs it, once per problem.

    `phase`, `batch` and `block` are the same compile-time inputs the firmware
    Makefile takes as PHASE=, BATCH= and BLOCK=, so the native binary and the
    RISC-V image are the same kernel however they were built. `block` None
    leaves the kernel's own default (the array's dispatch limit).
    """
    exe = os.path.join(workdir, f"infer_{phase}_b{batch}_k{block or 0}.trace")
    # shlex, so HOSTCC can be a driver plus its first argument ("zig cc",
    # "ccache gcc") and not only a bare program name.
    cc = shlex.split(os.environ.get("HOSTCC", "cc"))
    cmd = [*cc, "-DTPU_TRACE", f'-DADDER_RQ_H="{os.path.abspath(rq_header)}"',
           f"-DINFER_GEN={gen}", f"-DBATCH={batch}", *PHASE_FLAGS[phase],
           *((f"-DBLOCK={block}",) if block else ()),
           "-I", FW_DIR, "-O1", "-o", exe,
           os.path.join(FW_DIR, "infer.c"),
           os.path.join(FW_DIR, "mock", "tpu_trace.c")]
    subprocess.run(cmd, check=True)
    return exe


def embedding_image(model) -> dict:
    """The embedding table as `{addr: byte}`, quantized onto the learned site.

    adder_export does this per problem, on the rows a problem happens to use
    (`embedded / s_x0`, rounded and clipped); the same numbers, computed once
    for all 13 rows, are the table the device gathers from. Nothing else about
    the front end changes — there is still no positional encoding to add.
    """
    s_x0 = act_scale(model.q_embed)
    with torch.no_grad():
        codes = (model.embedding.weight / s_x0).round().clamp(Q4_MIN, Q4_MAX)
    return {DR_EMB + v * D + d: int(codes[v][d]) & 0xFF
            for v in range(VOCAB) for d in range(D)}


def prompt_image(ids, seq: int = 0, batch: int = 1) -> dict:
    """One prompt's token ids as int32, little-endian, in sequence `seq`'s block."""
    base = token_addr(seq, 0, batch)
    return {base + i * 4 + b: (int(tok) >> (8 * b)) & 0xFF
            for i, tok in enumerate(ids) for b in range(4)}


def batch_prompt_image(rows, batch: int = 1) -> dict:
    """`batch` prompts, one per sequence, as one image."""
    img: dict = {}
    for seq, ids in enumerate(rows):
        img.update(prompt_image(ids, seq, batch))
    return img


def mask_image(batch: int = 1) -> dict:
    """The causal mask at THIS kernel's T and THIS kernel's address.

    adder.c's is [128][128]; read as [64][64] it would put row 2t where row t
    belongs, so the row stride is the reason this cannot be shared even though
    the values are the same rule.
    """
    base = layout(batch)["mask"]
    return {base + t * T + s: (0 if s <= t else -8) & 0xFF
            for t in range(T) for s in range(T)}


def head_image(weights: dict, batch: int = 1) -> dict:
    """The output head at THIS kernel's address.

    Same [D][VPAD] int4 block adder.c gets, moved: infer.c computes its whole
    map off the shape and the head lands wherever that chain puts it.
    """
    img: dict = {}
    fc = weights["fc"]
    ax.put_rowmajor_i4(img, layout(batch)["wfc"], D, VPAD,
                       lambda r, c: int(fc[r][c]))
    return img


def static_image(model, weights: dict, batch: int = 1) -> dict:
    """Everything the device is handed once: weights, mask, head, embedding.

    `adder_export.static_image` is the source for the six weight blocks per
    layer — the same checkpoint at the same addresses, which is the whole reason
    one staging pass feeds either kernel. Its mask and its head are at ADDER's
    addresses and this kernel's are computed, so both are dropped and rebuilt
    rather than left to land somewhere harmless: an upload is 16 KB smaller and
    a reader of this image sees exactly what the device will read.
    """
    img = ax.static_image(weights)
    for addr in range(ax.DR_MASK, ax.DR_MASK + ax.T * ax.T):
        img.pop(addr, None)
    for addr in range(ax.DR_WFC, ax.DR_WFC + ax.D * (ax.VPAD // 2)):
        img.pop(addr, None)

    img.update(mask_image(batch))
    img.update(head_image(weights, batch))
    img.update(embedding_image(model))
    return img


def decode_tokens(raw: bytes) -> list:
    """Generated ids out of the int32 block at `DR_TOK + PROMPT*4`.

    Takes raw bytes so the board's `R` reply and the ISS's DRAM decode through
    the same function.
    """
    return [int.from_bytes(raw[i:i + 4], "little") for i in range(0, len(raw), 4)]


def stage_static(tpu: TPU, model, weights: dict) -> None:
    """`static_image` into an ISS instance's DRAM."""
    for addr, byte in static_image(model, weights).items():
        tpu.dram[addr] = byte


def stage_prompt(tpu: TPU, ids) -> None:
    for addr, byte in prompt_image(ids).items():
        tpu.dram[addr] = byte


def read_tokens(tpu: TPU) -> list:
    """The generated ids, DR_TOK[PROMPT:T], from an ISS instance's DRAM."""
    return decode_tokens(bytes(tpu.dram[DR_TOK + PROMPT * 4:DR_TOK + T * 4]))


def torch_generate(model, prompt_ids: list, n_gen: int) -> list:
    """The same greedy decode in PyTorch, as the reference.

    The prefix only, not the padded sequence: attention is causal and has no
    normalization over the source axis, so position L-1's logits do not depend
    on anything past L — feeding the tail would change nothing and hide a real
    difference if it did.
    """
    toks = list(prompt_ids)
    out = []
    with torch.no_grad():
        for _ in range(n_gen):
            inp = torch.tensor([toks])
            logits = model(inp, None)[0, -1]
            nxt = int(torch.argmax(logits))
            out.append(nxt)
            toks.append(nxt)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-path", default="model/saved/int4_d128_f512_l4.pt")
    ap.add_argument("-n", "--problems", type=int, default=16,
                    help="addition problems to generate (each is one ISS run of "
                         f"the whole kernel — a prefill and {NGEN - 1} decode "
                         f"steps)")
    ap.add_argument("-g", "--gen", type=int, default=NGEN,
                    help=f"tokens to generate per problem (default {NGEN}, the "
                         f"whole answer field)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--show", action="store_true",
                    help="print every problem, not the first four")
    args = ap.parse_args()

    if not 1 <= args.gen <= NGEN:
        raise SystemExit(f"--gen must be in 1..{NGEN}")

    model = load_model(args.model_path)

    rq_table, weights = derive(model)

    workdir = tempfile.mkdtemp(prefix="infer_export_")
    header = os.path.join(workdir, "adder_rq_ckpt.h")
    write_rq_header(header, rq_table, f"from {args.model_path}")
    exe = build_kernel(header, workdir, args.gen)

    tpu = TPU(rows=ROWS, cols=COLS)
    stage_static(tpu, model, weights)   # weights, mask, head, embedding table

    torch.manual_seed(args.seed)
    import random
    random.seed(args.seed)
    exprs, tokens, _ = numbers_data.create_addition_batch(
        args.problems, T, max_digits=MAX_DIGITS, equals_pos=PROMPT)

    dev_seq = dev_tok = ref_seq = ref_tok = agree = 0
    n_tok = n_cmds = 0

    print(f"generating {args.gen} tokens per problem, {args.problems} problems, "
          f"{args.model_path}")
    for i in range(args.problems):
        ids = tokens[i][:PROMPT]
        target = tokens[i][PROMPT:PROMPT + args.gen]

        stage_prompt(tpu, ids)
        cmds, _ = coexecute(tpu, exe, quiet=True)
        n_cmds = len(cmds)
        got = read_tokens(tpu)[:args.gen]
        ref = torch_generate(model, ids, args.gen)

        dev_tok += sum(a == b for a, b in zip(got, target))
        ref_tok += sum(a == b for a, b in zip(ref, target))
        dev_seq += int(got == target)
        ref_seq += int(ref == target)
        agree += int(got == ref)
        n_tok += len(target)

        if args.show or i < 4:
            prompt = numbers_data.unreverse_expression(
                numbers_data.detokenize(ids))
            said = numbers_data.unreverse_expression(
                numbers_data.detokenize(got))
            want = numbers_data.unreverse_expression(
                numbers_data.detokenize(target))
            flag = "" if got == target else "   <-- wrong"
            print(f"  {prompt:>18s}  device says {said:<10s} want {want}{flag}")
        if (i + 1) % 8 == 0 or i + 1 == args.problems:
            print(f"  {i + 1:4d}/{args.problems}  device {100 * dev_seq / (i + 1):6.2f}% "
                  f"exact-seq  model {100 * ref_seq / (i + 1):6.2f}%  "
                  f"agree {agree}/{i + 1}", flush=True)

    n = args.problems
    print()
    print(f"kernel: {n_cmds} commands per problem (one prefill + "
          f"{args.gen - 1} decode steps)")
    print(f"{'':26s} {'exact-sequence':>16s} {'token':>10s}")
    print(f"{'QAT model, greedy':26s} {100 * ref_seq / n:15.2f}% "
          f"{100 * ref_tok / n_tok:9.2f}%")
    print(f"{'TPU kernel, generating':26s} {100 * dev_seq / n:15.2f}% "
          f"{100 * dev_tok / n_tok:9.2f}%")
    print()
    print(f"device and model generated the same sequence on {agree} of {n} "
          f"problems")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

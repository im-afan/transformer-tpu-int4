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
   the 16 {m0,n} words per layer. infer.c and adder.c are the same arithmetic in
   a different order, so they take the same table, unchanged.
2. **Build.** The table is a compile-time input (the requant word is a literal
   in the macro-op), so this writes a header and compiles `infer.c` natively
   against it with the host cc.
3. **Generate.** One co-execution run per problem: the ISS executes the kernel's
   commands as it emits them and answers the scratchpad reads its argmax makes,
   which is what a data-dependent kernel needs from a simulator. Weights, mask,
   head and embedding table are staged once and stay; a problem writes 15 int32
   token ids and reads 17 back.

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
import model.transformer as transformer  # noqa: E402
from adder_export import (FW_DIR, Q4_MAX, Q4_MIN, act_scale, derive,  # noqa: E402
                          stage_static, write_rq_header)
from fw_vectors import coexecute  # noqa: E402
from iss import TPU  # noqa: E402

# Geometry, from fw/infer.c. Everything but the two new blocks is adder.c's map.
T, D, VOCAB, VPAD = 32, 64, 13, 16
PROMPT = numbers_data.EQUALS_POS          # 15 — the first ANSWER position
NGEN = T - PROMPT
DR_EMB, DR_TOK, DR_LOG = 0x00000, 0x00400, 0x01800
ROWS = COLS = 8


def build_kernel(rq_header: str, workdir: str, gen: int) -> str:
    """Compile infer.c natively against `rq_header`; return the binary's path.

    Unlike adder_export's `build_trace`, this does not run it: the command
    stream depends on the tokens the kernel chooses, so there is no trace until
    something is answering its reads. `coexecute` runs it, once per problem.
    """
    exe = os.path.join(workdir, "infer.trace")
    # shlex, so HOSTCC can be a driver plus its first argument ("zig cc",
    # "ccache gcc") and not only a bare program name.
    cc = shlex.split(os.environ.get("HOSTCC", "cc"))
    cmd = [*cc, "-DTPU_TRACE", f'-DADDER_RQ_H="{os.path.abspath(rq_header)}"',
           f"-DINFER_GEN={gen}", "-I", FW_DIR, "-O1", "-o", exe,
           os.path.join(FW_DIR, "infer.c"),
           os.path.join(FW_DIR, "mock", "tpu_trace.c")]
    subprocess.run(cmd, check=True)
    return exe


def stage_embedding(tpu: TPU, model) -> None:
    """The embedding table, quantized onto the site the model learned.

    adder_export does this per problem, on the rows a problem happens to use
    (`embedded / s_x0`, rounded and clipped); the same numbers, computed once
    for all 13 rows, are the table the device gathers from. Nothing else about
    the front end changes — there is still no positional encoding to add.
    """
    s_x0 = act_scale(model.q_embed)
    with torch.no_grad():
        codes = (model.embedding.weight / s_x0).round().clamp(Q4_MIN, Q4_MAX)
    for v in range(VOCAB):
        for d in range(D):
            tpu.dram[DR_EMB + v * D + d] = int(codes[v][d]) & 0xFF


def stage_prompt(tpu: TPU, ids) -> None:
    """The prompt's token ids as int32, little-endian, at DR_TOK."""
    for i, tok in enumerate(ids):
        for b in range(4):
            tpu.dram[DR_TOK + i * 4 + b] = (int(tok) >> (8 * b)) & 0xFF


def read_tokens(tpu: TPU) -> list:
    """The generated ids, DR_TOK[PROMPT:T]."""
    out = []
    for p in range(PROMPT, T):
        v = sum(tpu.dram[DR_TOK + p * 4 + b] << (8 * b) for b in range(4))
        out.append(v)
    return out


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
    ap.add_argument("--model-path", default="model/saved/int4_d64_f256_l4.pt")
    ap.add_argument("-n", "--problems", type=int, default=16,
                    help="addition problems to generate (each is one ISS run of "
                         "the whole kernel — a prefill and 16 decode steps)")
    ap.add_argument("-g", "--gen", type=int, default=NGEN,
                    help=f"tokens to generate per problem (default {NGEN}, the "
                         f"whole answer field)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--show", action="store_true",
                    help="print every problem, not the first four")
    args = ap.parse_args()

    if not 1 <= args.gen <= NGEN:
        raise SystemExit(f"--gen must be in 1..{NGEN}")

    model = transformer.adder_int4_vanilla()
    path = (args.model_path if os.path.isabs(args.model_path)
            else os.path.join(REPO, args.model_path))
    model.load_state_dict(torch.load(path, map_location="cpu"))
    model.eval()

    rq_table, weights = derive(model)

    workdir = tempfile.mkdtemp(prefix="infer_export_")
    header = os.path.join(workdir, "adder_rq_ckpt.h")
    write_rq_header(header, rq_table, f"from {args.model_path}")
    exe = build_kernel(header, workdir, args.gen)

    tpu = TPU(rows=ROWS, cols=COLS)
    stage_static(tpu, weights)          # weights, mask, head — adder.c's map
    stage_embedding(tpu, model)

    torch.manual_seed(args.seed)
    import random
    random.seed(args.seed)
    exprs, tokens, _ = numbers_data.create_addition_batch(args.problems, T)

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

"""make_dummy_checkpoint.py — an UNTRAINED `adder_int4_wide` checkpoint.

    python -m model.make_dummy_checkpoint

There is no trained checkpoint at the wide shape (d=128, f=512, layers=4,
q_heads=kv_heads=4) yet, and every host-side path below `accel/` needs one
before it can run at all: `adder_export.py`, `infer_export.py` and
`accel/tpu/host/run_adder.py` all start by loading a `.pt`, deriving the 16
requant `{m0,n}` words per layer from its scales, and compiling a kernel against
them. This writes a checkpoint that satisfies every one of those steps and
computes addition no better than chance.

**It measures the plumbing, not the model.** A run against it exercises the
staging, the DRAM map, the kernel's command stream and the ISS/RTL agreement;
the accuracy number it produces is noise, and the sequence it generates is
whatever random int4 weights happen to argmax to.

WHY IT IS NOT JUST `torch.save(adder_int4_wide().state_dict())`. Every learned
`ActQuant` scale is seeded lazily, by the first tensor that flows through it
(`ActQuant.init_scale_from`), and a freshly constructed model has `initialized`
clear and `scale` at exactly 1.0 everywhere. Exporting that gives a requant
table derived from a scale nothing chose, and `quant.QATCalibration` would fall
back to a calibration set rather than reading the checkpoint. So this runs one
forward pass over real batches with the quantizers live, which is what seeds
them, and then checks that every site actually came up.

The sites the hardware *pins* to one another — `q_o`/`q_xo` to the residual
stream, `q_hr` to `q_h`, `q_p` to `q_s` — share an `ActQuant` **instance**, so
they are seeded together and `adder_export.derive`'s equality checks hold by
construction. `q_x1`, `q_f` and `q_x2` are pinned to `1/7` analytically and are
never seeded at all.

Nothing here depends on the sequence length: the model has no positional
encoding, so `--max-tokens` and `--equals-pos` only shape the batches the scales
are seeded from. The defaults are `numbers_data`'s, i.e. what `model/train.py`
would train against.
"""
import argparse
import os
import random

import torch

import model.numbers_data as numbers_data
import model.transformer as transformer

DEFAULT_OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "saved", "int4_d128_f512_l4.pt")


def learned_sites(model):
    """Every `ActQuant` whose scale is learned rather than pinned.

    A pinned site holds its scale in a buffer and is born initialized; only the
    `nn.Parameter` ones need a tensor to flow through them. Instances are
    deduplicated by identity, because the shared ones are the whole point.
    """
    seen, out = set(), []
    for name, mod in model.named_modules():
        if isinstance(mod, transformer.ActQuant) and id(mod) not in seen:
            seen.add(id(mod))
            if isinstance(mod.scale, torch.nn.Parameter):
                out.append((name, mod))
    return out


def seed_scales(model, batches: int, batch_size: int, max_tokens: int,
                equals_pos: int) -> None:
    """Run forward passes until every learned site has a scale.

    `eval()`, so the two dropouts are off: a scale seeded from a tensor 10% of
    which was zeroed is a scale seeded from a tensor the deployed model never
    produces. `no_grad`, because nothing here is being trained.
    """
    max_digits = (equals_pos - 2) // 2
    model.eval()
    with torch.no_grad():
        for _ in range(batches):
            _, tokens, masks = numbers_data.create_addition_batch(
                batch_size, max_tokens, max_digits=max_digits,
                equals_pos=equals_pos)
            model(torch.tensor(tokens), torch.stack(masks))


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-o", "--out", default=DEFAULT_OUT,
                    help="where to write the checkpoint (default: %(default)s)")
    ap.add_argument("--batches", type=int, default=4,
                    help="forward passes used to seed the activation scales")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--max-tokens", type=int, default=numbers_data.MAX_TOKENS)
    ap.add_argument("--equals-pos", type=int, default=numbers_data.EQUALS_POS)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--force", action="store_true",
                    help="overwrite an existing file. Without it this refuses, "
                         "because the default path is also where a REAL "
                         "checkpoint would live")
    args = ap.parse_args()

    if os.path.exists(args.out) and not args.force:
        raise SystemExit(f"{args.out} already exists — pass --force to replace "
                         f"it (if that is a trained checkpoint, do not)")

    torch.manual_seed(args.seed)
    random.seed(args.seed)

    net = transformer.adder_int4_wide()
    print(f"model     : d={net.d} f={net.f} layers={len(net.layers)} "
          f"q_heads={net.q_heads} kv_heads={net.kv_heads} "
          f"head_dim={net.head_dim} vocab={net.vocab_size}")
    print(f"parameters: {sum(p.numel() for p in net.parameters())}")

    sites = learned_sites(net)
    seed_scales(net, args.batches, args.batch_size, args.max_tokens,
                args.equals_pos)

    unseeded = [name for name, site in sites if not bool(site.initialized)]
    if unseeded:
        raise SystemExit(
            "these activation sites never saw a tensor and would export a "
            "scale of 1.0: " + ", ".join(unseeded) + ". A site the forward "
            "pass does not reach is a bug in the model, not in this script.")
    print(f"scales    : {len(sites)} learned sites seeded over "
          f"{args.batches} x {args.batch_size} sequences of {args.max_tokens} "
          f"(prompt {args.equals_pos})")
    for name, site in sites:
        print(f"            {name:32s} {float(site.scale.abs()):.6g}")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    torch.save(net.state_dict(), args.out)
    print(f"wrote     : {args.out} ({os.path.getsize(args.out)} bytes) — "
          f"UNTRAINED, so any accuracy it scores is chance")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

import random
from typing import List, Tuple
import torch

VOCAB = {str(i): i for i in range(10)}
PAD_TOKEN = 'N'
PAD_ID = 12
VOCAB.update({'+': 10, '=': 11, 'N': PAD_ID})  # after the digits, or detokenize KeyErrors
INV_VOCAB = {v: k for k, v in VOCAB.items()}
MAX_INTEGER = 99999
MIN_TOKEN_LENGTH = 5  # e.g. "0+0=0"
# First answer position; '=' sits at EQUALS_POS-1. Bounds operands at
# max_digits <= (EQUALS_POS - 2) // 2, see model/docs/notes.md.
EQUALS_POS = 64
MAX_TOKENS = 128

# Digits are written least-significant first: "123+45=168" -> "321+54=861".
# See model/docs/notes.md for why (carry direction + fixed place value).
REVERSE_DIGITS = True


def _digits(n: int) -> str:
    """Render an integer in the generator's digit order."""
    s = str(n)
    return s[::-1] if REVERSE_DIGITS else s

def tokenize(expression: str, max_tokens: int = None):
    """Convert an expression like '123+45=168' into token ids."""
    token_ids = [VOCAB[ch] for ch in expression]
    if max_tokens is None:
        return token_ids
    if len(token_ids) > max_tokens:
        raise ValueError(f"Expression {expression!r} has {len(token_ids)} tokens, exceeds max_tokens={max_tokens}")
    token_ids = token_ids + [PAD_ID] * (max_tokens - len(token_ids))

    mask = torch.tensor([-1e9 if token_ids[i] == PAD_ID else 0 for i in range(max_tokens)])
    mask = mask.repeat(max_tokens, 1)
    mask = mask + mask.T

    return token_ids, mask


def detokenize(token_ids: List[int]) -> str:
    """Inverse of tokenize; digits stay least-significant first. Use
    unreverse_expression for display order."""
    return ''.join(INV_VOCAB[token_id] for token_id in token_ids if token_id != PAD_ID)


def unreverse_expression(expr: str) -> str:
    """Restore human digit order for display: 321+54=861 -> 123+45=168.

    Tolerant of malformed input: anything not a run of digits passes through.
    """
    if not REVERSE_DIGITS:
        return expr

    out = []
    run = ''
    for ch in expr:
        if ch.isdigit():
            run += ch
        else:
            out.append(run[::-1])
            run = ''
            out.append(ch)
    out.append(run[::-1])
    return ''.join(out)


def _sample_number(max_digits: int) -> int:
    """Sample an integer where each digit-length 1..max_digits is equally likely."""
    n_digits = random.randint(1, max_digits)
    lo = 10 ** (n_digits - 1) if n_digits > 1 else 0
    hi = 10 ** n_digits - 1
    return random.randint(lo, hi)


def generate_addition_expression(max_digits: int = 31, max_length: int = MAX_TOKENS,
                                 equals_pos: int = EQUALS_POS) -> str:
    """Random addition expression, each operand length equally likely.

    equals_pos defaults to EQUALS_POS; accel/test/export.py pins it
    to the older 32-token kernel shape instead.
    """
    left = _sample_number(max_digits)
    right = _sample_number(max_digits)
    expr = f"{_digits(left)}+{_digits(right)}"
    if len(expr) > equals_pos - 1:
        raise ValueError(f"max_digits={max_digits} does not fit before "
                         f"equals_pos={equals_pos}")
    expr += PAD_TOKEN * (equals_pos - len(expr) - 1)
    expr += f"={_digits(left + right)}"
    expr += PAD_TOKEN * (max_length - len(expr))
    return expr


def create_addition_batch(batch_size: int, max_tokens: int, max_digits: int = 31,
                          equals_pos: int = EQUALS_POS) -> Tuple[List[str], List[List[int]]]:
    """Create a batch of addition expressions and corresponding token id sequences."""
    if max_tokens < MIN_TOKEN_LENGTH:
        raise ValueError(f"max_tokens must be at least {MIN_TOKEN_LENGTH}")

    expressions: List[str] = []
    token_batches: List[List[int]] = []
    attention_masks = []

    for _ in range(batch_size):
        while True:
            expr = generate_addition_expression(max_digits=max_digits,
                                                max_length=max_tokens,
                                                equals_pos=equals_pos)
            if len(expr) <= max_tokens:
                break
        expressions.append(expr)
        token_ids, mask = tokenize(expr, max_tokens=max_tokens)
        token_batches.append(token_ids)
        attention_masks.append(mask)

    return expressions, token_batches, attention_masks

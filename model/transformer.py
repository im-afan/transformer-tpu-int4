import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical

import model.numbers_data as numbers_data

device = torch.device("cpu")
if torch.cuda.is_available():
    device = torch.device("cuda")



# Symmetric int4 grid; every scale is absmax/qmax with qmax=7, -8 is
# representable but never a scale target.
INT4_QMIN, INT4_QMAX = -8, 7


class RoundClip(torch.autograd.Function):
    """round(x).clip(qmin, qmax), straight-through estimator backward."""

    @staticmethod
    def forward(input, qmin, qmax):
        return input.round().clip(qmin, qmax)

    @staticmethod
    def setup_context(ctx, inputs, output):
        input, qmin, qmax = inputs
        ctx.save_for_backward(input)
        ctx.qmin, ctx.qmax = qmin, qmax

    @staticmethod
    def backward(ctx, grad_output):
        (input,) = ctx.saved_tensors
        inside = (input >= ctx.qmin) & (input <= ctx.qmax)
        return grad_output * inside, None, None


class FakeQuant(torch.autograd.Function):
    """Symmetric per-tensor fake quantization: round(x/s).clip(qmin,qmax) * s.

    Returns the dequantized value. Backward is LSQ (arxiv 1902.08153).
    """

    @staticmethod
    def forward(input, scale, qmin, qmax, grad_scale):
        return (input / scale).round().clamp(qmin, qmax) * scale

    @staticmethod
    def setup_context(ctx, inputs, output):
        input, scale, qmin, qmax, grad_scale = inputs
        ctx.save_for_backward(input, scale)
        ctx.qmin, ctx.qmax, ctx.grad_scale = qmin, qmax, grad_scale

    @staticmethod
    def backward(ctx, grad_output):
        input, scale = ctx.saved_tensors
        q = input / scale
        inside = (q >= ctx.qmin) & (q <= ctx.qmax)
        grad_input = grad_output * inside

        q_clipped = q.clamp(ctx.qmin, ctx.qmax)
        local = torch.where(inside, q_clipped.round() - q, q_clipped)
        grad_scale = (grad_output * local).sum() * ctx.grad_scale
        return grad_input, grad_scale.reshape(scale.shape), None, None, None


def dynamic_fake_quant(x, qmin=-128, qmax=127):
    """Per-tensor int8 fake-quant with a scale taken from the current absmax."""
    scale = (x.detach().abs().max() / qmax).clamp_min(1e-8)
    return FakeQuant.apply(x, scale, qmin, qmax, 0.0)


class ActQuant(nn.Module):
    """One activation quantization site, one entry in the requant table.

    A site owns a per-tensor scale. Sites the hardware pins to another
    tensor's scale share the same ActQuant instance, or pin fixed_scale.
    """

    def __init__(self, qmin=INT4_QMIN, qmax=INT4_QMAX, fixed_scale=None,
                 init="absmax"):
        super().__init__()
        self.qmin = qmin
        self.qmax = qmax
        self.init = init
        self.enabled = True

        if fixed_scale is None:
            self.scale = nn.Parameter(torch.ones(()))
            self.register_buffer("initialized", torch.zeros((), dtype=torch.bool))
        else:
            self.register_buffer("scale", torch.tensor(float(fixed_scale)))
            self.register_buffer("initialized", torch.ones((), dtype=torch.bool))

    def init_scale_from(self, x):
        """Seed the scale from x's absmax, once. No-op if already seeded."""
        if bool(self.initialized):
            return
        with torch.no_grad():
            a = x.detach().abs()
            seed = a.mean() if self.init == "absmean" else a.max() / self.qmax
            self.scale.copy_(seed.clamp_min(1e-8))
            self.initialized.fill_(True)

    def forward(self, x):
        if not self.enabled:
            return x

        self.init_scale_from(x)

        scale = self.scale.abs().clamp_min(1e-8)
        grad_scale = 1.0 / math.sqrt(x.numel() * self.qmax)
        return FakeQuant.apply(x, scale, self.qmin, self.qmax, grad_scale)

    def extra_repr(self):
        kind = "learned" if isinstance(self.scale, nn.Parameter) else "pinned"
        return f"qmin={self.qmin}, qmax={self.qmax}, {kind}"


def set_quant_enabled(module, enabled):
    """Turn every quantization site on/off (float baseline vs. QAT)."""
    for m in module.modules():
        if isinstance(m, ActQuant):
            m.enabled = enabled
        elif isinstance(m, Model):
            m.quantize_head = enabled

class TernaryLinear(nn.Module):
    """Linear layer with 1.58-bit (ternary {-1, 0, 1}) weights, BitNet-style.

    No config builds one any more, and nothing in accel/ reads one.
    """

    def __init__(self, in_dim, out_dim, bias=True, eps=1e-5):
        super().__init__()
        self.w = nn.Parameter(torch.empty(in_dim, out_dim))
        self.bias = nn.Parameter(torch.zeros(out_dim)) if bias else None
        self.eps = eps

        # Per-tensor symmetric int8 activation scale. NaN => not calibrated.
        self.register_buffer("act_scale", torch.tensor(float("nan")))
        # Running absmax accumulated while calibrating (not persisted).
        self.register_buffer("_act_absmax", torch.zeros(()), persistent=False)
        self.calibrating = False

        nn.init.kaiming_uniform_(self.w, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.w)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        # x is quantized by the ActQuant site that produced it, not here.
        scale = self.w.abs().mean() + self.eps
        w_quant = RoundClip.apply(self.w / scale, -1, 1) * self.w.abs().mean()
        out = x @ w_quant
        if self.bias is not None:
            out = out + self.bias
        return out


class Int4Linear(nn.Module):
    """Linear layer with symmetric per-tensor int4 weights, scale = absmax/7.

    Weight is [in_dim, out_dim] with x @ w, the transpose of nn.Linear, so
    checkpoints are not interchangeable with a float config.
    """

    def __init__(self, in_dim, out_dim, bias=True, eps=1e-5):
        super().__init__()
        self.w = nn.Parameter(torch.empty(in_dim, out_dim))
        self.bias = nn.Parameter(torch.zeros(out_dim)) if bias else None
        self.eps = eps

        nn.init.kaiming_uniform_(self.w, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.w)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        # x is quantized by the ActQuant site that produced it, not here.
        scale = (self.w.detach().abs().max() / INT4_QMAX).clamp_min(self.eps)
        w_quant = RoundClip.apply(self.w / scale, INT4_QMIN, INT4_QMAX) * scale
        out = x @ w_quant
        if self.bias is not None:
            out = out + self.bias
        return out


def make_linear(in_dim, out_dim, use_int4=False, bias=True):
    if use_int4:
        return Int4Linear(in_dim, out_dim, bias=bias)
    return nn.Linear(in_dim, out_dim, bias=bias)


# https://arxiv.org/pdf/2503.10622 — no gamma/beta
class DyT(nn.Module):
    def __init__(self, C, init_a=0.5):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(1) * init_a)

    def forward(self, x):
        return F.hardtanh(x * self.alpha, min_val=-1, max_val=1)

class MultiHeadAttention(nn.Module):
    """Quantization sites, in ``adder_kernel.md`` §4 block order: RQ_Q/RQ_K/RQ_V,
    RQ_S, RQ_P/RQ_XO ({1,0} identities), RQ_A, RQ_O (pinned to s_x).

    Every tensor here is int4, K and V included; the old RQ_KT/RQ_VT trit
    narrow is gone now that the MXU can multiply int4 by int4.
    """

    def __init__(self, d, q_heads, kv_heads, head_dim, x_quant,
                 use_int4=False, use_bias=True):
        super().__init__()

        assert q_heads % kv_heads == 0

        self.q_heads = q_heads
        self.kv_heads = kv_heads
        self.d = d
        self.heads_per_q = q_heads // kv_heads
        self.head_dim = head_dim

        self.Wq = make_linear(d, q_heads * self.head_dim, use_int4, bias=use_bias)
        self.Wk = make_linear(d, kv_heads * self.head_dim, use_int4, bias=use_bias)
        self.Wv = make_linear(d, kv_heads * self.head_dim, use_int4, bias=use_bias)
        self.Wo = make_linear(q_heads * self.head_dim, d, use_int4, bias=use_bias)

        self.q_q = ActQuant()   # RQ_Q
        self.q_k = ActQuant()   # RQ_K
        self.q_v = ActQuant()   # RQ_V
        self.q_s = ActQuant()   # RQ_S: scaled on post-mask, post-ReLU scores
        self.q_p = self.q_s     # RQ_P  {1,0}: s_p = s_s
        self.q_a = ActQuant()   # RQ_A
        self.q_o = x_quant      # RQ_O  pinned to s_x
        self.q_xo = x_quant     # RQ_XO {1,0}: stays on s_x for the second add

    def forward(self, X, attn_mask, use_custom_attention=False):
        batch_size = X.shape[0]
        n_tokens = X.shape[1]

        Q = self.q_q(self.Wq(X))
        K = self.q_k(self.Wk(X))
        V = self.q_v(self.Wv(X))

        Q = torch.reshape(
            Q, [batch_size, n_tokens, self.kv_heads, self.heads_per_q, self.head_dim]
        )
        K = torch.reshape(K, [batch_size, n_tokens, self.kv_heads, self.head_dim])
        V = torch.reshape(V, [batch_size, n_tokens, self.kv_heads, self.head_dim])


        mask = torch.triu(torch.ones([n_tokens, n_tokens]) * -1e9, diagonal=1).reshape(
            [1, n_tokens, n_tokens, 1, 1]
        )
        mask = mask.to(X.device)

        attention_scores = torch.einsum("btkgh,bskh->btskg", Q, K) / math.sqrt(self.head_dim)
        self.q_s.init_scale_from(F.relu(attention_scores + mask))
        attention_scores = self.q_s(attention_scores)

        attention_scores = F.relu(attention_scores + mask)
        attention_scores = self.q_p(attention_scores)

        A = torch.einsum("btskg,bskh->btkgh", attention_scores, V).reshape(
            [batch_size, n_tokens, self.q_heads * self.head_dim]
        )
        A = self.q_a(A)

        O = self.q_o(self.Wo(A))
        return self.q_xo(O + X)

class Transformer(nn.Module):
    def __init__(
        self,
        d,
        f,
        q_heads,
        kv_heads,
        head_dim,
        x_quant,
        use_int4=False,
        use_bias=True,
    ):
        super().__init__()

        self.use_int4 = use_int4

        self.attention = MultiHeadAttention(
            d, q_heads, kv_heads, head_dim, x_quant,
            use_int4=use_int4, use_bias=use_bias,
        )
        self.norm1 = DyT(d)

        self.ff = nn.Sequential(
            make_linear(d, f, use_int4, bias=use_bias),
            nn.ReLU(),
            make_linear(f, d, use_int4, bias=use_bias),
        )

        self.norm2 = DyT(d)
        self.dropout = nn.Dropout(p=0.1)

        # RQ_X1/RQ_X2 are `dyt` ops: analytic scale 1/7, symmetric clip at -7.
        self.q_x1 = ActQuant(qmin=-INT4_QMAX, fixed_scale=1.0 / INT4_QMAX)
        self.q_h = ActQuant()                        # RQ_H
        self.q_hr = self.q_h                         # RQ_HR {1,0}: s_hr = s_h
        self.q_f = ActQuant(fixed_scale=1.0 / INT4_QMAX)   # RQ_F: requant, clips at -8
        self.q_x2 = ActQuant(qmin=-INT4_QMAX, fixed_scale=1.0 / INT4_QMAX)

    def forward(self, X, attn_mask, use_custom_attention=False):
        # Attention residual is 2X + O: attention() already returned O + X.
        X = self.q_x1(
            self.norm1(X + self.dropout(self.attention(X, attn_mask, use_custom_attention)))
        )

        H = self.q_h(self.ff[0](X))
        HR = self.q_hr(self.ff[1](H))
        Fout = self.q_f(self.ff[2](HR))
        return self.q_x2(self.norm2(X + self.dropout(Fout)))


class Model(nn.Module):
    def __init__(
        self,
        vocab_size,
        d=128,
        f=256,
        layers=8,
        q_heads=8,
        kv_heads=8,
        head_dim=None,
        use_int4=False,
        use_bias=True,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = d

        if head_dim == None:
            head_dim = d // q_heads

        self.d = d
        self.f = f
        self.q_heads = q_heads
        self.kv_heads = kv_heads
        self.head_dim = head_dim
        self.layers = layers

        self.embedding = nn.Embedding(vocab_size, d)

        # The s_x chain: layer 0's scale comes from the embedding, every
        # later layer's from the previous layer's q_x2 (1/7).
        self.q_embed = ActQuant()

        blocks = []
        x_quant = self.q_embed
        for i in range(layers):
            block = Transformer(
                self.d,
                self.f,
                self.q_heads,
                self.kv_heads,
                self.head_dim,
                x_quant,
                use_int4=use_int4,
                use_bias=use_bias,
            )
            blocks.append(block)
            x_quant = block.q_x2
        self.layers = nn.ModuleList(blocks)
        # The output head is int4 like every other projection.
        self.fc = make_linear(self.d, self.vocab_size, use_int4, bias=use_bias)
        self.quantize_head = True
        self.dropout = nn.Dropout(p=0.1)

    # No positional encoding: the only position signal is the causal mask.

    def forward(self, inputs, attn_mask, use_custom_attention=False):
        X = self.q_embed(self.dropout(self.embedding(inputs)))
        for layer in self.layers:
            X = layer(X, attn_mask, use_custom_attention=use_custom_attention)

        # Logits are never requantized: an argmax does not care about scale.
        if isinstance(self.fc, (TernaryLinear, Int4Linear)):
            return self.fc(X)
        w = self.fc.weight
        if self.quantize_head:
            w = dynamic_fake_quant(w)
        return F.linear(X, w, self.fc.bias)

    def sample_pred(self, logits):
        dist = Categorical(logits=logits)
        return dist.sample()

    def sample_pred_best(self, logits: torch.Tensor):
        return logits.argmax(dim=-1)


def adder_vanilla():
    model = Model(
        len(numbers_data.VOCAB),
        d=128,
        f=512,
        layers=4,
        q_heads=4,
        kv_heads=4,
    )
    return model


def adder_gqa():
    model = Model(
        len(numbers_data.VOCAB),
        d=128,
        f=512,
        layers=6,
        q_heads=8,
        kv_heads=2,
    )
    return model


def adder_int4_wide():
    # The live shape: head_dim = 32.
    model = Model(
        len(numbers_data.VOCAB),
        d=128,
        f=512,
        layers=4,
        q_heads=4,
        kv_heads=4,
        use_int4=True,
        use_bias=False,
    )
    return model


def adder_int4_vanilla():
    # head_dim = d // q_heads = 16. No biases: see model/docs/notes.md.
    model = Model(
        len(numbers_data.VOCAB),
        d=64,
        f=256,
        layers=4,
        q_heads=4,
        kv_heads=4,
        use_int4=True,
        use_bias=False,
    )
    return model

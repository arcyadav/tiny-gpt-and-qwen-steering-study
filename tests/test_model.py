import math

import pytest
import torch

from tinygpt.model import GPT, GPTConfig

# every switch, covered at least once (the project trains only the baseline)
VARIANTS = {
    "baseline": dict(),
    "pos_learned": dict(pos="learned"),
    "pos_sinusoidal": dict(pos="sinusoidal"),
    "pos_alibi": dict(pos="alibi"),
    "pos_none": dict(pos="none"),
    "norm_layernorm": dict(norm="layernorm"),
    "norm_post": dict(norm_place="post"),
    "ffn_gelu": dict(ffn="gelu"),
    "attn_gqa": dict(n_kv_head=2),
    "attn_mqa": dict(n_kv_head=1),
}

TINY = dict(vocab_size=64, block_size=32, n_layer=2, n_head=4, n_embd=32)
ALL_RUNS = list(VARIANTS)


def tiny(name, **extra):
    return GPT(GPTConfig(**{**VARIANTS[name], **TINY, **extra}))


# ---------------------------------------------------------------- the real 12M model

def test_baseline_param_count_matches_spec():
    model = GPT(GPTConfig())
    d, L, V = 384, 6, 4096
    weights = V * d + L * (4 * d * d + 3 * d * 1024)  # embedding + attention + SwiGLU
    norms = (2 * L + 1) * d
    assert model.num_params() == weights + norms == 12_194_688


def test_swiglu_and_gelu_have_equal_ffn_params():
    swiglu = GPT(GPTConfig()).num_params()
    gelu = GPT(GPTConfig(ffn="gelu")).num_params()
    assert swiglu == gelu


def test_gqa_mqa_shrink_kv_cache():
    mha, gqa, mqa = (GPT(GPTConfig(n_kv_head=k)) for k in (6, 2, 1))
    assert mha.kv_cache_bytes_per_token() == 3 * gqa.kv_cache_bytes_per_token() == 6 * mqa.kv_cache_bytes_per_token()
    assert mqa.num_params() < gqa.num_params() < mha.num_params()


# ---------------------------------------------------------------- every switch, on a tiny model

@pytest.mark.parametrize("name", ALL_RUNS)
def test_forward_shapes_and_initial_loss(name):
    torch.manual_seed(0)
    model = tiny(name)
    idx = torch.randint(0, 64, (2, 16))
    logits, loss = model(idx, targets=torch.randint(0, 64, (2, 16)))
    assert logits.shape == (2, 16, 64)
    # an untrained model should be close to uniform: loss ~ ln(vocab)
    assert abs(loss.item() - math.log(64)) < 0.5


@pytest.mark.parametrize("name", ALL_RUNS)
def test_causal_no_peeking_at_future_tokens(name):
    torch.manual_seed(0)
    model = tiny(name).eval()
    idx = torch.randint(0, 64, (1, 20))
    changed = idx.clone()
    changed[0, 12:] = (changed[0, 12:] + 1) % 64
    a, _ = model(idx)
    b, _ = model(changed)
    assert torch.allclose(a[0, :12], b[0, :12], atol=1e-5)
    assert not torch.allclose(a[0, 12:], b[0, 12:], atol=1e-5)


@pytest.mark.parametrize("name", ALL_RUNS)
def test_explicit_attention_path_matches_fast_path(name):
    torch.manual_seed(0)
    model = tiny(name).eval()
    idx = torch.randint(0, 64, (2, 16))
    fast, _ = model(idx)
    slow, _, attns = model(idx, return_attn=True)
    assert torch.allclose(fast, slow, atol=1e-4)
    assert len(attns) == 2 and attns[0].shape == (2, 4, 16, 16)
    assert torch.allclose(attns[0].sum(-1), torch.ones(2, 4, 16), atol=1e-5)
    assert attns[0][0, 0].triu(1).abs().max() == 0  # nothing attends to the future


@pytest.mark.parametrize("name", ["baseline", "pos_sinusoidal", "pos_alibi", "pos_none"])
def test_position_methods_run_past_training_length(name):
    model = tiny(name).eval()
    logits, _ = model(torch.randint(0, 64, (1, 4 * 32)))
    assert torch.isfinite(logits).all()


def test_learned_positions_refuse_longer_sequences():
    model = tiny("pos_learned").eval()
    with pytest.raises(ValueError):
        model(torch.randint(0, 64, (1, 33)))


def test_no_positions_model_is_permutation_blind_for_the_last_token():
    # with one layer and no positions, the last token sees its prefix as an unordered set, so
    # shuffling earlier tokens must not change its output. (From layer 2 on this no longer holds:
    # each earlier token's state depends on its own prefix, which is how order leaks in.)
    torch.manual_seed(0)
    model = tiny("pos_none", n_layer=1).eval()
    idx = torch.randint(0, 64, (1, 10))
    shuffled = torch.cat([idx[:, :9][:, torch.randperm(9)], idx[:, 9:]], dim=1)
    a, _ = model(idx)
    b, _ = model(shuffled)
    assert torch.allclose(a[0, -1], b[0, -1], atol=1e-5)


def test_alibi_slopes_for_six_heads():
    from tinygpt.model import alibi_slopes
    s = alibi_slopes(6)
    assert len(s) == 6 and all(0 < x < 1 for x in s) and len(set(s)) == 6


@pytest.mark.parametrize("name", ALL_RUNS)
def test_can_overfit_one_batch(name):
    torch.manual_seed(0)
    model = tiny(name)
    idx = torch.randint(0, 64, (4, 32))
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    first = None
    for _ in range(60):
        _, loss = model(idx[:, :-1], targets=idx[:, 1:])
        first = first if first is not None else loss.item()
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert loss.item() < 0.3 * first


def test_generate_extends_sequence():
    model = tiny("baseline").eval()
    out = model.generate(torch.zeros(1, 3, dtype=torch.long), max_new_tokens=50, top_k=10)
    assert out.shape == (1, 53)


def test_unknown_switch_is_rejected():
    with pytest.raises(AssertionError):
        GPTConfig(pos="spiral")

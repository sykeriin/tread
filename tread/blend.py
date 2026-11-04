"""Phase 0 core: chunk KV prefill, RoPE re-rotation, and selective recompute (blending).

Written against transformers==4.46.3 Qwen2 modules. KV tensors are [1, kv_heads, T, head_dim]
and keys are stored post-RoPE, exactly as HF's DynamicCache holds them.
"""
import torch
import torch.nn.functional as F
from transformers import DynamicCache
from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb, repeat_kv, rotate_half


def prefill_kv(model, ids):
    """Run a forward pass over `ids` (1D LongTensor) and return per-layer (K, V) lists.

    Calls the base model, skipping lm_head: full-vocab logits for every position cost
    ~400 MB per call and are never needed here.
    """
    pkv = model.model(ids[None], past_key_values=DynamicCache(), use_cache=True).past_key_values
    return list(pkv.key_cache), list(pkv.value_cache)


def rotate_keys(model, K, delta):
    """Shift post-RoPE keys by `delta` positions: R(p + delta) = R(delta) @ R(p).

    Exact for any RoPE whose angle is linear in position (default and Llama-3 scaled RoPE).
    HF folds `attention_scaling` (≠1 for YaRN-style RoPE) into cos/sin; the cached keys already
    carry it once, so it is divided out to get a pure rotation.

    K may have any leading dims (e.g. all layers stacked): the rotation is the same for every layer
    and head, so one cos/sin of shape [n, head_dim] broadcasts over all of them.
    """
    if delta == 0:
        return K
    n = K.shape[-2]
    pos = torch.full((1, n), delta, device=K.device, dtype=torch.long)
    cos, sin = model.model.rotary_emb(K, pos)
    scale = getattr(model.model.rotary_emb, "attention_scaling", 1.0)
    cos, sin = cos[0] / scale, sin[0] / scale
    return K * cos + rotate_half(K) * sin


def chunk_kv(model, prefix_ids, doc_ids, mode):
    """KV for one document, computed independently of other documents.

    mode="shared": prefill [prefix][doc] and keep the doc slice (doc sees the prefix / sink tokens).
    mode="alone":  prefill [doc] at position 0 with no context at all.
    Returns (K list, V list, base_position).
    """
    if mode == "shared":
        K, V = prefill_kv(model, torch.cat([prefix_ids, doc_ids]))
        P = len(prefix_ids)
        return [k[:, :, P:] for k in K], [v[:, :, P:] for v in V], P
    K, V = prefill_kv(model, doc_ids)
    return K, V, 0


def assemble(model, prefix_ids, docs_ids, mode):
    """Build the reused context cache: exact prefix KV + independently computed, re-rotated doc KV."""
    Kp, Vp = prefill_kv(model, prefix_ids)
    Ks, Vs = [[k] for k in Kp], [[v] for v in Vp]
    offset = len(prefix_ids)
    for d in docs_ids:
        Kd, Vd, base = chunk_kv(model, prefix_ids, d, mode)
        for l in range(len(Ks)):
            Ks[l].append(rotate_keys(model, Kd[l], offset - base))
            Vs[l].append(Vd[l])
        offset += len(d)
    return [torch.cat(x, dim=2) for x in Ks], [torch.cat(x, dim=2) for x in Vs]


def _subset_ctx(model, S, T, ref):
    """Per-request constants for recomputing positions S against a T-token cache.

    RoPE cos/sin for S and the causal mask are identical for every layer, so they are built once
    per blend instead of once per layer.
    """
    rope = model.model.rotary_emb(ref, S[None])
    mask = (torch.arange(T, device=S.device)[None, :] <= S[:, None])[None, None]
    return rope, mask


def _qkv(layer, model, x, pos, rope=None):
    attn = layer.self_attn
    n = x.shape[1]
    q = attn.q_proj(x).view(1, n, attn.num_heads, attn.head_dim).transpose(1, 2)
    k = attn.k_proj(x).view(1, n, attn.num_key_value_heads, attn.head_dim).transpose(1, 2)
    v = attn.v_proj(x).view(1, n, attn.num_key_value_heads, attn.head_dim).transpose(1, 2)
    cos, sin = rope if rope is not None else model.model.rotary_emb(v, pos[None])
    q, k = apply_rotary_pos_emb(q, k, cos, sin)
    return q, k, v


def _layer_subset(layer, model, h, S, K, V, inplace=False, ctx=None):
    """Recompute one decoder layer for tokens S (positions == indices) against the full cache.

    Fresh K/V for S are spliced into the layer cache before attention, so S attends to the
    mix of reused (non-S) and fresh (S) states, as in CacheBlend. `inplace` skips the copy when
    the caller owns K/V (the engine's freshly assembled cache); Phase 0 reuses K/V, so it copies.
    `ctx` is `_subset_ctx(...)`, built once per blend.
    """
    attn = layer.self_attn
    rope, mask = ctx if ctx is not None else _subset_ctx(model, S, K.shape[2], h)
    q, k, v = _qkv(layer, model, layer.input_layernorm(h), S, rope)
    if not inplace:
        K, V = K.clone(), V.clone()
    K[:, :, S], V[:, :, S] = k, v
    # Not enable_gqa=True: in torch 2.5, a boolean mask + GQA falls back to the math kernel, which
    # materialises the full attention matrix (3.4 GB at 8k tokens) and is ~3x slower than repeat_kv.
    groups = attn.num_heads // attn.num_key_value_heads
    o = F.scaled_dot_product_attention(q, repeat_kv(K, groups), repeat_kv(V, groups), attn_mask=mask)
    o = attn.o_proj(o.transpose(1, 2).reshape(1, len(S), -1))
    h = h + o
    h = h + layer.mlp(layer.post_attention_layernorm(h))
    return h, K, V


def layer1_deviation(model, ctx_ids, K, V, all_c, inplace=False):
    """Per-token drift of reused layer-1 KV from a fresh recompute, for candidate positions `all_c`.

    Layer 0 KV depends only on token + position, so it is already exact after rotation.
    Recompute layer 0 for every candidate, then measure how far layer-1 KV drifts.
    Returns (dev [len(all_c)], layer-0 hidden for all_c, fresh layer-0 K, fresh layer-0 V).
    """
    layers = model.model.layers
    h0 = model.model.embed_tokens(ctx_ids[all_c])[None]
    ctx = _subset_ctx(model, all_c, K[0].shape[2], h0)
    h, K0, V0 = _layer_subset(layers[0], model, h0, all_c, K[0], V[0], inplace=inplace, ctx=ctx)
    _, k1, v1 = _qkv(layers[1], model, layers[1].input_layernorm(h), all_c, ctx[0])
    dev = (k1 - K[1][:, :, all_c]).norm(dim=-1).sum(dim=1) + (v1 - V[1][:, :, all_c]).norm(dim=-1).sum(dim=1)
    return dev[0], h, K0, V0


def blend(model, ctx_ids, K, V, candidates, ratio, policy, gen=None, inplace=False, runner=None):
    """Selectively recompute a `ratio` fraction of `candidates` (doc token positions).

    policy: "boundary" (first tokens of each doc, see `candidates` as list of per-doc ranges),
            "deviation" (CacheBlend-style: rank by layer-1 KV deviation), "random".
    Returns new (K, V) lists, the selected positions, and the effective recompute ratio in
    token-layers: deviation also pays for recomputing layer 0 over every candidate.
    """
    K, V = list(K), list(V)
    if not candidates:  # request with no documents: nothing to recompute
        return K, V, torch.empty(0, dtype=torch.long, device=K[0].device), 0.0
    all_c = torch.cat(candidates)
    budget = int(round(ratio * len(all_c)))
    if budget == 0:
        return K, V, all_c[:0], 0.0
    layers = model.model.layers
    embed = model.model.embed_tokens
    start = 0

    if policy == "boundary":
        # Water-fill: take the first w tokens of every doc, short docs simply run out first.
        rank = torch.cat([torch.arange(len(c), device=c.device) for c in candidates])
        S = all_c[rank.argsort(stable=True)[:budget]]
    elif policy == "random":
        S = all_c[torch.randperm(len(all_c), generator=gen, device="cpu").to(all_c.device)[:budget]]
    elif policy == "deviation":
        dev, h, K[0], V[0] = layer1_deviation(model, ctx_ids, K, V, all_c, inplace=inplace)
        top = dev.topk(budget).indices
        S, h, start = all_c[top], h[:, top], 1
    else:
        raise ValueError(policy)

    S, order = S.sort()
    if start == 0:
        h = embed(ctx_ids[S])[None]
    else:
        h = h[:, order]
    if runner is not None and runner.worth_it(K[0].shape[2]):
        # CUDA-graph replay of the whole layer loop (tread/graphs.py).
        h, K[start:], V[start:] = runner.run(h, S, K[start:], V[start:], start)
    else:
        ctx = _subset_ctx(model, S, K[0].shape[2], h)
        for l in range(start, len(layers)):
            h, K[l], V[l] = _layer_subset(layers[l], model, h, S, K[l], V[l], inplace=inplace, ctx=ctx)
    L, n = len(layers), len(all_c)
    eff = (len(S) * (L - start) + start * n) / (n * L)
    return K, V, S, eff


def to_cache(K, V):
    c = DynamicCache()
    for l, (k, v) in enumerate(zip(K, V)):
        c.update(k, v, l)
    return c

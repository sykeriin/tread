"""Reuse engine: build a request's KV cache from content-addressed chunk blocks.

Request = [prefix][doc_1]...[doc_N][query], given as separate segments (the API contract; it avoids
tokenization-boundary ambiguity). The prefix is reused as an exact prefix. Each doc block is computed
once *after the prefix* (attention-sink handling, see phase0/RESULTS.md), cached under a key
that includes the prefix, and re-rotated to wherever it lands in later requests. Optional
selective recompute (`blend`) repairs the missing cross-doc attention.
"""
import time
from dataclasses import dataclass, field

import torch

from tread.blend import blend, prefill_kv, rotate_keys, to_cache
from tread.graphs import SubsetRunner
from tread.store import BlockStore, block_key


@dataclass
class RequestStats:
    hits: int = 0
    misses: int = 0
    tokens_reused: int = 0
    tokens_prefilled: int = 0
    recompute_eff: float = 0.0
    t: dict = field(default_factory=dict)  # stage -> seconds


class Engine:
    def __init__(self, model, store: BlockStore, graphs=False):
        self.model = model
        self.store = store
        # CUDA-graph the recompute layer loop (tread/graphs.py); off by default since each new
        # size bucket pays a one-off capture.
        self.runner = SubsetRunner(model) if graphs else None
        cfg = model.config
        self.sig = (cfg._name_or_path, str(model.dtype), getattr(cfg, "rope_theta", None),
                    getattr(cfg, "rope_scaling", None))

    def _sync(self):
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    @torch.inference_mode()
    def prefix_block(self, prefix_ids, st, namespace=""):
        # Doc keys derive from this key, so a namespace isolates a tenant's whole block tree.
        key = block_key(*self.sig, namespace, "prefix", prefix_ids)
        b = self.store.get(key)
        if b is None:
            st.misses += 1
            st.tokens_prefilled += len(prefix_ids)
            K, V = (torch.stack(x) for x in prefill_kv(self.model, prefix_ids))
            b = self.store.put(key, K, V, 0) or _Tmp(K, V, 0)
        else:
            st.hits += 1
            st.tokens_reused += len(prefix_ids)
        return key, b

    @torch.inference_mode()
    def doc_blocks(self, prefix_key, pb, docs, st):
        """Look up every doc's block; prefill all misses together in one forward.

        Missed docs are packed into one sequence on top of the cached prefix KV, with a block-diagonal
        mask (each doc sees the prefix and itself, never another doc) and positions restarting at P
        for each doc. That is exactly the per-doc prefill, in one launch sequence instead of one per doc.
        """
        keys = [block_key(*self.sig, prefix_key, d) for d in docs]
        blocks = [self.store.get(k) for k in keys]
        miss = [i for i, b in enumerate(blocks) if b is None]
        st.hits += len(docs) - len(miss)
        st.tokens_reused += sum(len(d) for d, b in zip(docs, blocks) if b is not None)
        if not miss:
            return blocks
        st.misses += len(miss)
        P = pb.n_tokens
        ids = torch.cat([docs[i] for i in miss])
        n, dev = len(ids), ids.device
        seg = torch.cat([torch.full((len(docs[i]),), j, device=dev) for j, i in enumerate(miss)])
        local = torch.cat([torch.arange(len(docs[i]), device=dev) for i in miss])
        st.tokens_prefilled += n
        # Additive 4D mask over [prefix | packed docs]: HF 4.46 passes a 4D mask through unchanged.
        allowed = torch.ones(n, P + n, dtype=torch.bool, device=dev)
        allowed[:, P:] = (seg[:, None] == seg[None, :]) & (local[:, None] >= local[None, :])
        mask = torch.zeros(1, 1, n, P + n, dtype=self.model.dtype, device=dev)
        mask.masked_fill_(~allowed, torch.finfo(self.model.dtype).min)
        pkv = self.model.model(ids[None], past_key_values=to_cache(pb.K.unbind(0), pb.V.unbind(0)),
                               attention_mask=mask, position_ids=(P + local)[None],
                               cache_position=torch.arange(P, P + n, device=dev), use_cache=True).past_key_values
        Kall, Vall = torch.stack(pkv.key_cache)[..., P:, :], torch.stack(pkv.value_cache)[..., P:, :]
        off = 0
        for i in miss:
            L = len(docs[i])
            K, V = Kall[..., off:off + L, :].contiguous(), Vall[..., off:off + L, :].contiguous()
            blocks[i] = self.store.put(keys[i], K, V, P) or _Tmp(K, V, P)
            off += L
        return blocks

    @torch.inference_mode()
    def prefill(self, prefix_ids, docs, query_ids, ratio=0.0, policy="deviation", namespace=""):
        """Build the context cache from blocks, optionally blend, run the query.

        Returns (logits of the last query position, DynamicCache for decoding, RequestStats).
        """
        st = RequestStats()
        self._sync()
        t0 = time.perf_counter()

        pkey, pb = self.prefix_block(prefix_ids, st, namespace)
        Ks, Vs = [pb.K], [pb.V]
        offset = pb.n_tokens
        cands = []
        for d, b in zip(docs, self.doc_blocks(pkey, pb, docs, st)):
            Ks.append(rotate_keys(self.model, b.K, offset - b.base))  # all layers in one op
            Vs.append(b.V)
            cands.append(torch.arange(offset, offset + len(d), device=d.device))
            offset += len(d)
        # One concat per K/V; per-layer views of a tensor this request owns, so blend can
        # write into it in place.
        K = list(torch.cat(Ks, dim=-2).unbind(0))
        V = list(torch.cat(Vs, dim=-2).unbind(0))
        self._sync()
        t1 = time.perf_counter()

        if ratio > 0:
            ctx = torch.cat([prefix_ids] + list(docs))
            K, V, _, st.recompute_eff = blend(self.model, ctx, K, V, cands, ratio, policy, inplace=True,
                                              runner=self.runner)
        self._sync()
        t2 = time.perf_counter()

        cache = to_cache(K, V)
        logits = self.model(query_ids[None], past_key_values=cache, use_cache=True, num_logits_to_keep=1).logits[0, -1]
        self._sync()
        t3 = time.perf_counter()
        st.t = dict(assemble=t1 - t0, blend=t2 - t1, query=t3 - t2, ttft=t3 - t0)
        return logits, cache, st

    @torch.inference_mode()
    def prefill_prefix_only(self, prefix_ids, docs, query_ids):
        """Baseline: exact-prefix caching only (what vLLM/SGLang prefix caching gives when doc
        order changes). The prefix comes from the store; every doc is prefilled again."""
        st = RequestStats()
        self._sync()
        t0 = time.perf_counter()
        _, pb = self.prefix_block(prefix_ids, st)
        rest = torch.cat(list(docs) + [query_ids])
        st.tokens_prefilled += len(rest) - len(query_ids)
        out = self.model(rest[None], past_key_values=to_cache(pb.K.unbind(0), pb.V.unbind(0)), use_cache=True,
                         num_logits_to_keep=1)
        self._sync()
        st.t = dict(ttft=time.perf_counter() - t0)
        return out.logits[0, -1], out.past_key_values, st


@torch.inference_mode()
def prefill_full(model, prefix_ids, docs, query_ids):
    """Baseline: no caching at all."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    ids = torch.cat([prefix_ids] + list(docs) + [query_ids])
    out = model(ids[None], use_cache=True, num_logits_to_keep=1)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    st = RequestStats(tokens_prefilled=len(ids) - len(query_ids), t=dict(ttft=time.perf_counter() - t0))
    return out.logits[0, -1], out.past_key_values, st


class _Tmp:
    """Uncached block (store budget too small for it): same fields as store.Block."""

    def __init__(self, K, V, base):
        self.K, self.V, self.base, self.n_tokens = K, V, base, K.shape[-2]

"""CUDA-graph replay of the selective-recompute layer loop.

Eagerly, recomputing a subset S through L layers launches ~a dozen small kernels per layer from
Python, so blend costs ~100 ms even when S is tiny (bench/RESULTS.md). Capturing the whole loop
as one CUDA graph replaces those launches with a single replay.

Graphs need static shapes, but |S| and the cache length T vary per request. Each request is padded
into a bucket (nb >= |S|, Tb >= T + nb - |S|):
  - padded query slots are placed at positions T, T+1, ... in the padding region of the cache.
    Real queries sit at positions < T, so the causal mask already hides padding keys from them.
    Padded slots' outputs and the K/V they write land in the padding region and are discarded;
  - the cache is copied into a static [L, 1, kv_heads, Tb, head_dim] buffer, and results are
    returned as views of it, valid until the next replay of the same bucket.
"""
from collections import OrderedDict

import torch

from tread.blend import _layer_subset, _subset_ctx


def _bucket(n, T, n_min=64, n_step=256, t_step=1024):
    # Powers of 2 up to n_step, then multiples of n_step: pow-2 rounding wasted up to ~2x compute
    # on large subsets (1,560 -> 2,048), where recompute is FLOP-bound rather than launch-bound.
    nb = max(n_min, 1 << (n - 1).bit_length()) if n <= n_step else -(-n // n_step) * n_step
    Tb = -(-(T + nb - n) // t_step) * t_step
    return nb, Tb


class _Graph:
    def __init__(self, model, start, nb, Tb, like_h, like_k):
        L = len(model.model.layers)
        kv_shape = (L - start, *like_k.shape[:2], Tb, like_k.shape[-1])
        self.K = torch.zeros(kv_shape, dtype=like_k.dtype, device=like_k.device)
        self.V = torch.zeros_like(self.K)
        self.h = torch.zeros((1, nb, like_h.shape[-1]), dtype=like_h.dtype, device=like_h.device)
        self.S = torch.arange(nb, device=like_k.device)
        self.nbytes = 2 * self.K.numel() * self.K.element_size()

        def body():
            ctx = _subset_ctx(model, self.S, Tb, self.h)
            h = self.h
            for i, layer in enumerate(model.model.layers[start:]):
                h, _, _ = _layer_subset(layer, model, h, self.S, self.K[i], self.V[i], inplace=True, ctx=ctx)
            return h

        # Warm up on a side stream (required before capture), then capture.
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(2):
                body()
        torch.cuda.current_stream().wait_stream(s)
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self.out = body()


class SubsetRunner:
    """Caches one graph per (start layer, bucket); keeps at most `max_graphs` (LRU)."""

    def __init__(self, model, max_graphs=4, max_ctx=2048):
        self.model = model
        self.max_graphs = max_graphs
        # Above ~2k tokens recompute is FLOP-bound, and copying the cache into and out of the static
        # buffers costs more than the launches saved (bench/RESULTS.md): fall back to eager there.
        self.max_ctx = max_ctx
        self.graphs: "OrderedDict[tuple, _Graph]" = OrderedDict()
        self.captures = 0

    def worth_it(self, T):
        return T <= self.max_ctx

    @torch.inference_mode()
    def run(self, h, S, K, V, start):
        """Recompute layers start..L-1 for sorted positions S. K/V: per-layer lists for those layers.

        Returns (h, K list, V list); K/V are views into the graph's static buffers.
        """
        n, T = len(S), K[0].shape[2]
        nb, Tb = _bucket(n, T)
        key = (start, nb, Tb)
        g = self.graphs.get(key)
        if g is None:
            while len(self.graphs) >= self.max_graphs:
                self.graphs.popitem(last=False)
            g = _Graph(self.model, start, nb, Tb, h, K[0])
            self.graphs[key] = g
            self.captures += 1
        self.graphs.move_to_end(key)

        g.K[..., :T, :].copy_(torch.stack(K))
        g.V[..., :T, :].copy_(torch.stack(V))
        g.S[:n].copy_(S)
        g.S[n:].copy_(torch.arange(T, T + nb - n, device=S.device))
        g.h[:, :n].copy_(h)
        g.h[:, n:].zero_()
        g.graph.replay()
        return g.out[:, :n], list(g.K[..., :T, :].unbind(0)), list(g.V[..., :T, :].unbind(0))

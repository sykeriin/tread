"""Content-addressed KV block store with a byte budget and LRU eviction.

A block is the K/V of one chunk for all layers, stacked as [layers, 1, kv_heads, tokens, head_dim],
plus the position it was computed at (`base`). Stacking lets a block be rotated and concatenated
in one op instead of one per layer.
Keys are content hashes, so the same chunk hits regardless of where it appears in a prompt.
Single-threaded for now: no refcounts, an evicted block is only dropped from the index and
any request already holding its tensors keeps them alive.
"""
import hashlib
from collections import OrderedDict
from dataclasses import dataclass, field

import torch


def block_key(*parts) -> str:
    """Hash of everything the KV depends on: model, dtype, RoPE config, context key, token ids."""
    h = hashlib.sha256()
    for p in parts:
        if isinstance(p, torch.Tensor):
            h.update(p.detach().to("cpu", torch.int64).numpy().tobytes())
        else:
            h.update(str(p).encode())
        h.update(b"\x00")
    return h.hexdigest()


@dataclass
class Block:
    K: torch.Tensor
    V: torch.Tensor
    base: int
    n_tokens: int
    nbytes: int
    hits: int = 0


@dataclass
class StoreStats:
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    bytes_used: int = 0
    extra: dict = field(default_factory=dict)


class BlockStore:
    def __init__(self, budget_bytes: int):
        self.budget = budget_bytes
        self.blocks: "OrderedDict[str, Block]" = OrderedDict()
        self.stats = StoreStats()

    def get(self, key):
        b = self.blocks.get(key)
        if b is None:
            self.stats.misses += 1
            return None
        self.blocks.move_to_end(key)
        b.hits += 1
        self.stats.hits += 1
        return b

    def put(self, key, K, V, base):
        nbytes = sum(t.numel() * t.element_size() for t in (K, V))
        if nbytes > self.budget:
            return None  # never fits; serve this request without caching
        if key in self.blocks:
            self._drop(key)
        while self.stats.bytes_used + nbytes > self.budget:
            self._drop(next(iter(self.blocks)))
            self.stats.evictions += 1
        b = Block(K=K, V=V, base=base, n_tokens=K.shape[-2], nbytes=nbytes)
        self.blocks[key] = b
        self.stats.bytes_used += nbytes
        return b

    def _drop(self, key):
        self.stats.bytes_used -= self.blocks.pop(key).nbytes

    def __len__(self):
        return len(self.blocks)

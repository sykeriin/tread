"""Content-addressed KV block store: GPU tier + optional pinned-CPU tier, byte budgets, LRU.

A block is the K/V of one chunk for all layers, stacked as [layers, 1, kv_heads, tokens, head_dim],
plus the position it was computed at (`base`). Stacking lets a block be rotated and concatenated
in one op instead of one per layer.
Keys are content hashes, so the same chunk hits regardless of where it appears in a prompt.

When the GPU tier is full, its least-recently-used block moves to pinned CPU memory (if
`cpu_budget_bytes` > 0) instead of being dropped; a later hit copies it back to the GPU. That
costs a PCIe transfer, which is far cheaper than recomputing the chunk. The CPU tier evicts LRU
blocks for good. Single-threaded for now: no refcounts, an evicted block is only dropped from the
index and any request already holding its tensors keeps them alive.
"""
import hashlib
from collections import OrderedDict
from dataclasses import dataclass

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
    device: torch.device
    hits: int = 0


@dataclass
class StoreStats:
    hits: int = 0  # GPU + CPU-tier hits
    misses: int = 0
    evictions: int = 0  # blocks leaving the GPU tier (offloaded or dropped)
    bytes_used: int = 0  # GPU tier
    cpu_hits: int = 0
    offloads: int = 0
    cpu_evictions: int = 0  # blocks dropped for good from the CPU tier
    cpu_bytes_used: int = 0


class BlockStore:
    def __init__(self, budget_bytes: int, cpu_budget_bytes: int = 0):
        self.budget = budget_bytes
        self.cpu_budget = cpu_budget_bytes
        self.blocks: "OrderedDict[str, Block]" = OrderedDict()  # GPU tier
        self.cpu_blocks: "OrderedDict[str, Block]" = OrderedDict()
        self.stats = StoreStats()

    def get(self, key):
        b = self.blocks.get(key)
        if b is not None:
            self.blocks.move_to_end(key)
        else:
            b = self.cpu_blocks.pop(key, None)
            if b is None:
                self.stats.misses += 1
                return None
            self.stats.cpu_bytes_used -= b.nbytes
            self.stats.cpu_hits += 1
            b.K = b.K.to(b.device, non_blocking=True)
            b.V = b.V.to(b.device, non_blocking=True)
            self._insert(key, b)
        b.hits += 1
        self.stats.hits += 1
        return b

    def put(self, key, K, V, base):
        nbytes = sum(t.numel() * t.element_size() for t in (K, V))
        if nbytes > self.budget:
            return None  # never fits; serve this request without caching
        for tier in (self.blocks, self.cpu_blocks):
            if key in tier:
                self._remove(tier, key)
        b = Block(K=K, V=V, base=base, n_tokens=K.shape[-2], nbytes=nbytes, device=K.device)
        self._insert(key, b)
        return b

    def _insert(self, key, b):
        while self.stats.bytes_used + b.nbytes > self.budget:
            self._evict_gpu()
        self.blocks[key] = b
        self.stats.bytes_used += b.nbytes

    def _evict_gpu(self):
        key, b = self.blocks.popitem(last=False)
        self.stats.bytes_used -= b.nbytes
        self.stats.evictions += 1
        if b.nbytes > self.cpu_budget:
            return
        while self.stats.cpu_bytes_used + b.nbytes > self.cpu_budget:
            _, old = self.cpu_blocks.popitem(last=False)
            self.stats.cpu_bytes_used -= old.nbytes
            self.stats.cpu_evictions += 1
        # A new Block, never a mutated one: the request that just looked this block up may still be
        # assembling from it, and must keep seeing GPU tensors.
        b = Block(K=b.K.to("cpu").pin_memory(), V=b.V.to("cpu").pin_memory(), base=b.base,
                  n_tokens=b.n_tokens, nbytes=b.nbytes, device=b.device, hits=b.hits)
        self.cpu_blocks[key] = b
        self.stats.cpu_bytes_used += b.nbytes
        self.stats.offloads += 1

    def _remove(self, tier, key):
        b = tier.pop(key)
        if tier is self.blocks:
            self.stats.bytes_used -= b.nbytes
        else:
            self.stats.cpu_bytes_used -= b.nbytes

    def __len__(self):
        return len(self.blocks) + len(self.cpu_blocks)

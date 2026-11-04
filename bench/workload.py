"""Reordered-RAG workload shared by every benchmark (tread, vLLM, llama.cpp), so all systems see
identical token sequences.

Each request: [system][docs in order 1][question], then the same docs in order 2. A question's
gold paragraphs are padded to a target context length with distractors from other HotpotQA
questions (validation 1000-1399, unused elsewhere). The random draws happen in a fixed order
(pool permutation, then order 1, then order 2) so runs are reproducible from the seed.
"""
import torch
from datasets import load_dataset

SYSTEM = ("<|im_start|>system\nAnswer the question using the documents. "
          "Reply with a short phrase only.<|im_end|>\n<|im_start|>user\n")


class Workload:
    def __init__(self, tok, device="cpu", seed=0, pool_split="validation[1000:1400]"):
        self.tok, self.device = tok, device
        self.g = torch.Generator().manual_seed(seed)
        pool = load_dataset("hotpotqa/hotpot_qa", "distractor", split=pool_split)
        self.pool = [self.enc(f"Document: {t}\n{''.join(s)}\n\n")
                     for ex in pool for t, s in zip(ex["context"]["title"], ex["context"]["sentences"])]
        self.prefix = self.enc(SYSTEM)

    def enc(self, s):
        return torch.tensor(self.tok(s, add_special_tokens=False).input_ids, device=self.device)

    @staticmethod
    def questions(start=400, n=15):
        return load_dataset("hotpotqa/hotpot_qa", "distractor", split=f"validation[{start}:{start + n}]")

    def request(self, ex, target_len):
        """Returns (docs, query, order1, order2, n_tokens). Call in a fixed sequence for reproducibility."""
        gold = set(ex["supporting_facts"]["title"])
        docs = [self.enc(f"Document: {t}\n{''.join(s)}\n\n")
                for t, s in zip(ex["context"]["title"], ex["context"]["sentences"]) if t in gold]
        n_tok = len(self.prefix) + sum(len(d) for d in docs)
        for j in torch.randperm(len(self.pool), generator=self.g).tolist():
            if n_tok >= target_len:
                break
            docs.append(self.pool[j])
            n_tok += len(self.pool[j])
        query = self.enc(f"Question: {ex['question']}<|im_end|>\n<|im_start|>assistant\n")
        o1 = [docs[j] for j in torch.randperm(len(docs), generator=self.g)]
        o2 = [docs[j] for j in torch.randperm(len(docs), generator=self.g)]
        return docs, query, o1, o2, n_tok

    def flat(self, order, query):
        """Whole prompt as one token list, for engines that take a single sequence."""
        return torch.cat([self.prefix] + list(order) + [query]).tolist()

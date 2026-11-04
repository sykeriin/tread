# I tried to make LLM servers reuse KV cache when documents come back in a different order

*Draft for you to edit and publish; first person, written as if by the project author.*

If you ask an LLM server the same question twice with the same documents in the prompt, the second request is
fast: the server remembers the work it did on the shared beginning of the prompt. Reorder the documents and that
memory is useless, because it only matches an exact prefix. RAG systems do this constantly. The vLLM maintainers
closed the request for generalized reuse as not planned, and the open-source attempts I found had open bugs.

So I built a small serving layer to find out how far you can get, and what it costs. Here is what I learned,
including the parts that did not work.

## The idea, in two parts

**Positions are the easy part.** Models like Llama and Qwen encode a token's position by rotating its key vector.
Rotations compose, so a key cached at position 100 moves to position 4,000 by rotating it by the difference. I
tested that this matches computing the key in place, to floating-point noise.

**The hard part is attention you never did.** A document cached on its own never looked at the documents before
it. Its deeper-layer values differ from what a full pass would give. You can patch that by recomputing a
selected fraction of tokens against the assembled cache. How much that costs, and how well it works, is the
whole project.

## What worked

- Caching each document *after* the system prompt instead of alone roughly halved the error. Documents seem to
  need the model's attention-sink tokens in view.
- Recomputing the tokens whose cached values drifted most beat random picks, and drove error from KL 0.263 to
  0.099 / 0.070 / 0.029 at 20 / 30 / 50% recompute (1.5B model, 300 held-out questions).
- At 4k-8k tokens, 20% recompute was 1.4-1.6× faster than a full pass, and plain reuse up to 5.8× faster, on a
  0.5B model.

## What did not work

- **Adaptive recompute.** I found a signal that looked like it could decide per request how much to recompute
  (how much the question attends to later documents). On 100 examples it cut error 20-30%. On 300 fresh
  examples it did nothing. I had cross-validated, and it still overfit. The error is heavy-tailed: a few hard
  requests dominate, so small samples mislead.
- **A lot of my first speedups were fake.** My first 1.5B benchmark said 14-26× faster. It was a 4 GB GPU with no
  free memory, so Windows was quietly spilling to system RAM and only the big baseline passes slowed down. I
  only caught it because the baseline was 5× slower than it measured by hand an hour earlier.
- **Partial recompute can make things worse.** On a quarter of requests at 8 documents, recomputing some tokens
  was worse than recomputing none.
- **Short contexts do not benefit.** At 1k tokens recompute is slower than a full pass. It is dominated by
  per-layer launch overhead, and CUDA graphs fixed that at 1k but lost at 4k+.

## What it honestly adds up to

A reuse layer that is exact where it can be, measurable where it cannot, and faster than full prefill from
roughly 4k tokens, at a quality cost I can quantify (top-1 agreement with full prefill of 69-92% at 20%
recompute on the first token). It is not a free lunch, and I would not claim end-to-end accuracy until I have
run larger models and tasks at n in the hundreds.

## What is missing

Everything here is Qwen2.5 0.5B/1.5B on HotpotQA on a laptop GPU. The comparisons against vLLM and llama.cpp,
and 7B results, need a rented GPU. The scripts are written but not run. I would rather say that than guess.

Code, tests, raw results and the failed experiments: github.com/sykeriin/tread

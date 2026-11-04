# Running the benchmark on a rented GPU

The dev laptop (RTX 3050, 4 GB) can only time Qwen2.5-0.5B reliably: with 1.5B loaded there is no VRAM
headroom, Windows spills to system RAM and every timing is wrong (see `bench/RESULTS.md`). vLLM also has no
Windows build. Run the headline numbers on a Linux box with a 24-80 GB GPU.

**Cost estimate**: ~2-4 GPU-hours for 1.5B and 7B at 1k-16k tokens, n=30 → roughly $3-10 on RunPod / Lambda / Vast.

1. Rent a Linux GPU instance (CUDA 12.x image), clone the repo.
2. Create the three venvs per `requirements-cloud.txt`.
3. For llama.cpp, convert/download an fp16 GGUF of the same model.
4. `MODEL=Qwen/Qwen2.5-7B-Instruct GGUF=... bash bench/cloud/run_all.sh`
5. Check `gpu.csv` (steady SM clock, no thermal throttling) before trusting any number.

**Status: untested.** `bench/baselines.py` was written without being run. Check that vLLM and llama.cpp
actually reuse their prefix cache on the second pass (e.g. vLLM log lines `prefix cache hit rate`); a baseline
that silently has caching off would flatter tread.

**Comparability notes**
- All systems receive identical token ids (`bench/workload.py`, seed 0).
- tread's TTFT excludes the first decode step; the baselines' 1-token generation includes it. This makes the
  baselines look slightly slower than they are. State this when quoting speedups.
- The exact-prefix baseline inside tread's own codebase is pessimistic at long contexts (HF takes a slower
  attention path with a cache). vLLM/llama.cpp are the fair reference for absolute latency.
- Quality (KL / top-1 vs full prefill) is measured by `bench_scaling.py` only; vLLM and llama.cpp reuse only
  exact prefixes, so they are lossless by construction.

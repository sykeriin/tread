#!/usr/bin/env bash
# One-command benchmark on a rented Linux GPU. Assumes three venvs (see requirements-cloud.txt):
#   .venv-tread, .venv-vllm, .venv-llamacpp   and a GGUF of the same model in $GGUF.
# Usage: MODEL=Qwen/Qwen2.5-7B-Instruct GGUF=models/qwen2.5-7b-instruct-fp16.gguf bash bench/cloud/run_all.sh
set -euo pipefail
MODEL=${MODEL:-Qwen/Qwen2.5-1.5B-Instruct}
LENGTHS=${LENGTHS:-"1000 2000 4000 8000 16000"}
N=${N:-30}
OUT=${OUT:-bench/results/cloud}
mkdir -p "$OUT"

# Log GPU clocks so thermal/clock problems are visible afterwards.
nvidia-smi --query-gpu=timestamp,utilization.gpu,clocks.sm,power.draw,temperature.gpu,memory.used \
  --format=csv -l 1 > "$OUT/gpu.csv" &
SMI=$!; trap 'kill $SMI 2>/dev/null || true' EXIT

echo "== tread (full prefill, prefix cache, tread at 0/20/50% recompute) =="
.venv-tread/bin/python bench/bench_scaling.py --model "$MODEL" --lengths $LENGTHS --n "$N" --store-mb 4096 --out "$OUT"

echo "== vLLM, prefix caching on =="
.venv-vllm/bin/python bench/baselines.py --engine vllm --model "$MODEL" --lengths $LENGTHS --n "$N" --out "$OUT"

if [ -n "${GGUF:-}" ]; then
  echo "== llama.cpp, prompt cache =="
  .venv-llamacpp/bin/python bench/baselines.py --engine llamacpp --model "$MODEL" --gguf "$GGUF" --lengths $LENGTHS --n "$N" --out "$OUT"
fi
echo "done: $OUT"

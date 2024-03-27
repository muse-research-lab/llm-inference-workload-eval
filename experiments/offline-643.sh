#/bin/bash

set -e

NUM_REQUESTS=1000
RESULTS_DIR="results/offline-643"

MAX_NUM_BATCHED_TOKENS=2560
MAX_NUM_SEQS=2048
MAX_MODEL_LEN=2048

HF_TOKEN=""

MODEL="facebook/opt-13b"
GPU_MEM_UTIL=0.95

# Alpaca Dataset
python vllm/benchmarks/benchmark_throughput_custom.py \
    --model $MODEL \
    --seed 0 --gpu-memory-utilization $GPU_MEM_UTIL \
    --swap-space 0 --max-num-batched-tokens $MAX_NUM_BATCHED_TOKENS \
    --max-num-sequences $MAX_NUM_SEQS --n1 1.0 \
    --dataset alpaca --num-requests $NUM_REQUESTS --token $HF_TOKEN \
    --block-size 16 --collect-stats --max-model-len $MAX_MODEL_LEN \
    --results-dir $RESULTS_DIR

# CNN DailyMail Dataset
python vllm/benchmarks/benchmark_throughput_custom.py \
    --model $MODEL \
    --seed 0 --gpu-memory-utilization $GPU_MEM_UTIL \
    --swap-space 0 --max-num-batched-tokens $MAX_NUM_BATCHED_TOKENS \
    --max-num-sequences $MAX_NUM_SEQS --n1 1.0 \
    --dataset cnn_dailymail --num-requests $NUM_REQUESTS --token $HF_TOKEN \
    --block-size 16 --collect-stats --max-model-len $MAX_MODEL_LEN \
    --results-dir $RESULTS_DIR

# Dolly Dataset
python vllm/benchmarks/benchmark_throughput_custom.py \
    --model $MODEL \
    --seed 0 --gpu-memory-utilization $GPU_MEM_UTIL \
    --swap-space 0 --max-num-batched-tokens $MAX_NUM_BATCHED_TOKENS \
    --max-num-sequences $MAX_NUM_SEQS --n1 1.0 \
    --dataset dolly --num-requests $NUM_REQUESTS --token $HF_TOKEN \
    --block-size 16 --collect-stats --max-model-len $MAX_MODEL_LEN \
    --results-dir $RESULTS_DIR

# ShareGPT Dataset
python vllm/benchmarks/benchmark_throughput_custom.py \
    --model $MODEL \
    --seed 0 --gpu-memory-utilization $GPU_MEM_UTIL \
    --swap-space 0 --max-num-batched-tokens $MAX_NUM_BATCHED_TOKENS \
    --max-num-sequences $MAX_NUM_SEQS --n1 1.0 \
    --dataset sharegpt --num-requests $NUM_REQUESTS --token $HF_TOKEN \
    --block-size 16 --collect-stats --max-model-len $MAX_MODEL_LEN \
    --results-dir $RESULTS_DIR

MODEL="meta-llama/Llama-2-13b-hf"
GPU_MEM_UTIL=0.9083

# Alpaca Dataset
python vllm/benchmarks/benchmark_throughput_custom.py \
    --model $MODEL \
    --seed 0 --gpu-memory-utilization $GPU_MEM_UTIL \
    --swap-space 0 --max-num-batched-tokens $MAX_NUM_BATCHED_TOKENS \
    --max-num-sequences $MAX_NUM_SEQS --n1 1.0 \
    --dataset alpaca --num-requests $NUM_REQUESTS --token $HF_TOKEN \
    --block-size 16 --collect-stats --max-model-len $MAX_MODEL_LEN \
    --results-dir $RESULTS_DIR

# CNN DailyMail Dataset
python vllm/benchmarks/benchmark_throughput_custom.py \
    --model $MODEL \
    --seed 0 --gpu-memory-utilization $GPU_MEM_UTIL \
    --swap-space 0 --max-num-batched-tokens $MAX_NUM_BATCHED_TOKENS \
    --max-num-sequences $MAX_NUM_SEQS --n1 1.0 \
    --dataset cnn_dailymail --num-requests $NUM_REQUESTS --token $HF_TOKEN \
    --block-size 16 --collect-stats --max-model-len $MAX_MODEL_LEN \
    --results-dir $RESULTS_DIR

# Dolly Dataset
python vllm/benchmarks/benchmark_throughput_custom.py \
    --model $MODEL \
    --seed 0 --gpu-memory-utilization $GPU_MEM_UTIL \
    --swap-space 0 --max-num-batched-tokens $MAX_NUM_BATCHED_TOKENS \
    --max-num-sequences $MAX_NUM_SEQS --n1 1.0 \
    --dataset dolly --num-requests $NUM_REQUESTS --token $HF_TOKEN \
    --block-size 16 --collect-stats --max-model-len $MAX_MODEL_LEN \
    --results-dir $RESULTS_DIR

# ShareGPT Dataset
python vllm/benchmarks/benchmark_throughput_custom.py \
    --model $MODEL \
    --seed 0 --gpu-memory-utilization $GPU_MEM_UTIL \
    --swap-space 0 --max-num-batched-tokens $MAX_NUM_BATCHED_TOKENS \
    --max-num-sequences $MAX_NUM_SEQS --n1 1.0 \
    --dataset sharegpt --num-requests $NUM_REQUESTS --token $HF_TOKEN \
    --block-size 16 --collect-stats --max-model-len $MAX_MODEL_LEN \
    --results-dir $RESULTS_DIR

#/bin/bash

set -e

DURATION=600
RESULTS_DIR="results/online-643"

MAX_NUM_BATCHED_TOKENS=2560
MAX_NUM_SEQS=2048
MAX_MODEL_LEN=2048

HF_TOKEN=""

MODEL="facebook/opt-13b"
GPU_MEM_UTIL=0.95

# Alpaca Dataset
for request_rate in 27.0 27.5 28.0 28.5 29.0 29.5 30.0 30.5 31.0; do
    python vllm/benchmarks/benchmark_text_completion.py \
        --model $MODEL \
        --seed 0 --gpu-memory-utilization $GPU_MEM_UTIL \
        --swap-space 0 --max-num-batched-tokens $MAX_NUM_BATCHED_TOKENS \
        --max-num-sequences $MAX_NUM_SEQS --duration $DURATION --n1 1.0 \
        --dataset alpaca --request-rate $request_rate --token $HF_TOKEN \
        --block-size 16 --collect-stats --max-model-len $MAX_MODEL_LEN \
        --results-dir $RESULTS_DIR
done

# CNN DailyMail Dataset
for request_rate in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 1.1 1.2 1.3 1.4 1.5 1.6 1.7 1.8 1.9 2.0; do
    python vllm/benchmarks/benchmark_text_completion.py \
        --model $MODEL \
        --seed 0 --gpu-memory-utilization $GPU_MEM_UTIL \
        --swap-space 0 --max-num-batched-tokens $MAX_NUM_BATCHED_TOKENS \
        --max-num-sequences $MAX_NUM_SEQS --duration $DURATION --n1 1.0 \
        --dataset cnn_dailymail --request-rate $request_rate --token $HF_TOKEN \
        --block-size 16 --collect-stats --max-model-len $MAX_MODEL_LEN \
        --results-dir $RESULTS_DIR
done

# Dolly Dataset
for request_rate in 16.0 16.5 17.0 17.5 18.0 18.5 19.0; do
    python vllm/benchmarks/benchmark_text_completion.py \
        --model $MODEL \
        --seed 0 --gpu-memory-utilization $GPU_MEM_UTIL \
        --swap-space 0 --max-num-batched-tokens $MAX_NUM_BATCHED_TOKENS \
        --max-num-sequences $MAX_NUM_SEQS --duration $DURATION --n1 1.0 \
        --dataset dolly --request-rate $request_rate --token $HF_TOKEN \
        --block-size 16 --collect-stats --max-model-len $MAX_MODEL_LEN \
        --results-dir $RESULTS_DIR
done

# ShareGPT Dataset
for request_rate in 3.0 3.1 3.2 3.3 3.4 3.5 3.6 3.7 3.8 3.9 4.0; do
    python vllm/benchmarks/benchmark_text_completion.py \
        --model $MODEL \
        --seed 0 --gpu-memory-utilization $GPU_MEM_UTIL \
        --swap-space 0 --max-num-batched-tokens $MAX_NUM_BATCHED_TOKENS \
        --max-num-sequences $MAX_NUM_SEQS --duration $DURATION --n1 1.0 \
        --dataset sharegpt --request-rate $request_rate --token $HF_TOKEN \
        --block-size 16 --collect-stats --max-model-len $MAX_MODEL_LEN \
        --results-dir $RESULTS_DIR
done

MODEL="meta-llama/Llama-2-13b-hf"
GPU_MEM_UTIL=0.9083

# Alpaca Dataset
for request_rate in 24.0 24.5 25.0 25.5 26.0; do
    python vllm/benchmarks/benchmark_text_completion.py \
        --model $MODEL \
        --seed 0 --gpu-memory-utilization $GPU_MEM_UTIL \
        --swap-space 0 --max-num-batched-tokens $MAX_NUM_BATCHED_TOKENS \
        --max-num-sequences $MAX_NUM_SEQS --duration $DURATION --n1 1.0 \
        --dataset alpaca --request-rate $request_rate --token $HF_TOKEN \
        --block-size 16 --collect-stats --max-model-len $MAX_MODEL_LEN \
        --results-dir $RESULTS_DIR
done

# CNN DailyMail Dataset
for request_rate in 0.01 0.02 0.03 0.04 0.05 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0; do
    python vllm/benchmarks/benchmark_text_completion.py \
        --model $MODEL \
        --seed 0 --gpu-memory-utilization $GPU_MEM_UTIL \
        --swap-space 0 --max-num-batched-tokens $MAX_NUM_BATCHED_TOKENS \
        --max-num-sequences $MAX_NUM_SEQS --duration $DURATION --n1 1.0 \
        --dataset cnn_dailymail --request-rate $request_rate --token $HF_TOKEN \
        --block-size 16 --collect-stats --max-model-len $MAX_MODEL_LEN \
        --results-dir $RESULTS_DIR
done

# Dolly Dataset
for request_rate in 10.0 10.5 11.0 11.5 12.0 12.5 13.0 13.5 14.0; do
    python vllm/benchmarks/benchmark_text_completion.py \
        --model $MODEL \
        --seed 0 --gpu-memory-utilization $GPU_MEM_UTIL \
        --swap-space 0 --max-num-batched-tokens $MAX_NUM_BATCHED_TOKENS \
        --max-num-sequences $MAX_NUM_SEQS --duration $DURATION --n1 1.0 \
        --dataset dolly --request-rate $request_rate --token $HF_TOKEN \
        --block-size 16 --collect-stats --max-model-len $MAX_MODEL_LEN \
        --results-dir $RESULTS_DIR
done

# ShareGPT Dataset
for request_rate in 3.0 3.1 3.2 3.3 3.4 3.5 3.6 3.7 3.8 3.9 4.0; do
    python vllm/benchmarks/benchmark_text_completion.py \
        --model $MODEL \
        --seed 0 --gpu-memory-utilization $GPU_MEM_UTIL \
        --swap-space 0 --max-num-batched-tokens $MAX_NUM_BATCHED_TOKENS \
        --max-num-sequences $MAX_NUM_SEQS --duration $DURATION --n1 1.0 \
        --dataset sharegpt --request-rate $request_rate --token $HF_TOKEN \
        --block-size 16 --collect-stats --max-model-len $MAX_MODEL_LEN \
        --results-dir $RESULTS_DIR
done


from src.tokenizer import ChessTokenizer
from datasets import load_dataset
import os

print("Warming up... Downloading dataset and building vocab.")

# 1. Download dataset (caches to ~/.cache/huggingface)
print("Downloading dataset...")
dataset = load_dataset("dlouapre/lichess_2025-01_1M", split="train")
print(f"Dataset downloaded: {len(dataset)} examples")

# 2. Build and save tokenizer (optional but good for cache)
print("Building tokenizer...")
tokenizer = ChessTokenizer.build_vocab_from_dataset(
    dataset_name="dlouapre/lichess_2025-01_1M",
    max_samples=100000
)
print(f"Tokenizer built with vocab size: {tokenizer.vocab_size}")

print("\nSUCCESS! Data is cached.")

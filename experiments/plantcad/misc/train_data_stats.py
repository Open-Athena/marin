#!/usr/bin/env python3
"""Count tokens and examples in PlantCAD dataset, and analyze token frequencies."""

import os
import numpy as np
from collections import Counter
from huggingface_hub import snapshot_download
from levanter.store.tree_store import TreeStore
import tensorstore as ts


def main():
    # Download dataset
    print("Loading PlantCAD dataset...")
    cached_path = snapshot_download(
        repo_id="plantcad/_dev_marin_plantcad1_v1_tokenized",
        repo_type="dataset",
        revision="main",
    )
    
    # Open with TreeStore
    exemplar = {
        "input_ids": np.array([0], dtype=np.int32),
        "token_type_ids": np.array([0], dtype=np.int32)
    }
    
    train_path = os.path.join(cached_path, "train")
    store = TreeStore.open(exemplar, train_path, mode="r", cache_metadata=True)
    
    # Count examples
    num_examples = len(store)
    print(f"Number of examples: {num_examples:,}")
    
    # Count tokens (assuming all examples have same length)
    first_example = store[0]
    tokens_per_example = len(first_example['input_ids'])
    total_tokens = num_examples * tokens_per_example
    
    print(f"Tokens per example: {tokens_per_example:,}")
    print(f"Total tokens: {total_tokens:,}")
    print(f"Total tokens (billions): {total_tokens / 1e9:.2f}B")
    
    # Count token frequencies efficiently by reading directly from tensorstore
    print("\nAnalyzing token frequencies from ENTIRE dataset...")
    
    # Access the underlying JaggedArrayStore for input_ids directly
    input_ids_store = store.tree['input_ids']
    
    print(f"Reading all {total_tokens:,} tokens directly from tensorstore...")
    slice_obj: ts.TensorStore = input_ids_store.data[:total_tokens]
    read_future: ts.Future = slice_obj.read()
    raw_data: np.ndarray = read_future.result()
    print(f"Shape: {raw_data.shape}, dtype: {raw_data.dtype}")
    
    # Count frequencies using numpy - much faster!
    print("Counting token frequencies with np.unique...")
    unique_tokens, counts = np.unique(raw_data, return_counts=True)
    
    # Sort by count (descending)
    sorted_indices = np.argsort(counts)[::-1]
    sorted_tokens = unique_tokens[sorted_indices]
    sorted_counts = counts[sorted_indices]
    
    # Show most common tokens
    print(f"\nToken frequency analysis (FULL dataset - {total_tokens:,} tokens):")
    print("Most common tokens:")
    for i in range(min(20, len(sorted_tokens))):
        token_id = sorted_tokens[i]
        count = sorted_counts[i]
        frequency = count / total_tokens * 100
        print(f"  Token {token_id}: {count:,} occurrences ({frequency:.2f}%)")
    
    print(f"\nTotal unique tokens: {len(unique_tokens)}")
    print(f"Token ID range: {unique_tokens.min()} - {unique_tokens.max()}")


if __name__ == "__main__":
    main()

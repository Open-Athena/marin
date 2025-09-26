#!/usr/bin/env python3
"""
Model sizing utility for finding LlamaConfig models that hit specific parameter targets.

Uses exact parameter counting (not theoretical) to find configurations that come close
to 100M, 50M, 30M, and 15M parameter targets for genomic models.
"""

import jax.random as jrandom
from haliax import Axis
from levanter.models.llama import LlamaConfig, LlamaLMHeadModel
from levanter.utils.jax_utils import parameter_count


# PlantCaduceus tokenizer vocab size
GENOMIC_VOCAB_SIZE = 7


def count_exact_parameters(config: LlamaConfig, vocab_size: int = GENOMIC_VOCAB_SIZE) -> int:
    """Get exact parameter count by instantiating the model."""
    Vocab = Axis("vocab", vocab_size)
    key = jrandom.PRNGKey(0)
    model = LlamaLMHeadModel.init(Vocab, config, key=key)
    return parameter_count(model)


def find_config_for_target(target_params_millions: int, start_hidden_dim: int, start_layers: int, tolerance_millions: int = 3) -> tuple[LlamaConfig, int]:
    """
    Find a configuration close to the target parameter count.
    
    Args:
        target_params_millions: Target parameter count in millions
        start_hidden_dim: Starting hidden dimension to search around
        start_layers: Starting number of layers to search around
        tolerance_millions: Acceptable deviation in millions of parameters
    
    Returns:
        Tuple of (best_config, actual_param_count)
    """
    target_params = target_params_millions * 1_000_000
    tolerance = tolerance_millions * 1_000_000
    
    best_config = None
    best_params = float('inf')
    best_diff = float('inf')
    
    # Search around the starting point with increments of 128 for hidden_dim
    # Reduce search range for very large models to avoid memory issues
    search_range = 128 if target_params_millions >= 300 else 256
    for hidden_dim in range(start_hidden_dim + search_range, max(128, start_hidden_dim - search_range), -128):
        # Calculate intermediate_dim as multiples of 128
        for int_multiplier in [3.0, 3.5, 4.0]:
            intermediate_dim = int(hidden_dim * int_multiplier)
            # Round to nearest 128
            intermediate_dim = ((intermediate_dim + 63) // 128) * 128
            
            # Ensure num_heads is even and divisible into hidden_dim
            if hidden_dim >= 512:
                num_heads = hidden_dim // 64  # 64 dims per head
            elif hidden_dim >= 256:
                num_heads = hidden_dim // 32  # 32 dims per head
            else:
                num_heads = max(2, hidden_dim // 16)  # Minimum 2 heads
                
            # Ensure num_heads is even
            if num_heads % 2 != 0:
                num_heads += 1
                
            # KV heads - ensure even and divisible
            for ratio in [1, 2, 4]:
                if num_heads >= ratio and num_heads % ratio == 0:
                    num_kv_heads = num_heads // ratio
                    if num_kv_heads >= 2 and num_kv_heads % 2 == 0:
                        break
            else:
                num_kv_heads = 2  # Even fallback
            
            # Try layer counts around the starting point (only even)
            for layer_offset in [0, -2, 2, -4, 4]:
                num_layers = start_layers + layer_offset
                if num_layers < 2 or num_layers % 2 != 0:
                    continue
                    
                config = LlamaConfig(
                    seq_len=512,
                    hidden_dim=hidden_dim,
                    intermediate_dim=intermediate_dim,
                    num_heads=num_heads,
                    num_kv_heads=num_kv_heads,
                    num_layers=num_layers,
                )
                
                try:
                    actual_params = count_exact_parameters(config)
                    diff = abs(actual_params - target_params)
                    
                    if diff < best_diff and diff <= tolerance:
                        best_config = config
                        best_params = actual_params
                        best_diff = diff
                        
                except Exception:
                    continue
    
    return best_config, best_params


def print_config_summary(config: LlamaConfig, actual_params: int, target_millions: int):
    """Print a summary of the configuration."""
    actual_millions = actual_params / 1_000_000
    print(f"\nTarget: {target_millions}M parameters")
    print(f"Actual: {actual_params:,} ({actual_millions:.1f}M) parameters")
    print(f"Difference: {abs(actual_millions - target_millions):.1f}M")
    print(f"Config:")
    print(f"  hidden_dim: {config.hidden_dim}")
    print(f"  intermediate_dim: {config.intermediate_dim}")
    print(f"  num_layers: {config.num_layers}")
    print(f"  num_heads: {config.num_heads}")
    print(f"  num_kv_heads: {config.num_kv_heads}")
    print(f"  seq_len: {config.seq_len}")


if __name__ == "__main__":
    print(f"Using genomic vocab size: {GENOMIC_VOCAB_SIZE}")
    print("=" * 60)
    
    # Define targets with smart starting points based on previous results
    targets = [
        (300, 896, 18),   # 300M: start around 896 hidden_dim, 18 layers (more conservative)
        (100, 768, 12),   # 100M: start around 768 hidden_dim, 12 layers
        (30, 512, 8),     # 30M: start around 512 hidden_dim, 8 layers  
        (10, 384, 6),     # 10M: start around 384 hidden_dim, 6 layers
    ]
    
    configs = {}
    
    for target_millions, start_hidden, start_layers in targets:
        print(f"Finding configuration for {target_millions}M parameters...")
        print(f"Starting search from hidden_dim={start_hidden}, layers={start_layers}")
        
        # Use higher tolerance for larger models
        tolerance = 20 if target_millions >= 300 else 3
        config, params = find_config_for_target(target_millions, start_hidden, start_layers, tolerance)
        
        if config is not None:
            configs[f"{target_millions}m"] = config
            print_config_summary(config, params, target_millions)
        else:
            print(f"Could not find configuration within tolerance for {target_millions}M parameters")
        print("-" * 40)
    
    # Generate the configurations as code
    print("\n" + "=" * 60)
    print("GENERATED CONFIGURATIONS FOR GENOMIC MODELS (EVEN NUMBERS)")
    print("=" * 60)
    
    for name, config in configs.items():
        actual_params = count_exact_parameters(config)
        print(f"\nllama_genomic_{name} = LlamaConfig(")
        print(f"    seq_len={config.seq_len},")
        print(f"    hidden_dim={config.hidden_dim},")
        print(f"    intermediate_dim={config.intermediate_dim},")
        print(f"    num_heads={config.num_heads},")
        print(f"    num_kv_heads={config.num_kv_heads},")
        print(f"    num_layers={config.num_layers},")
        print(f")")
        print(f"# Actual params: {actual_params:,} ({actual_params/1e6:.1f}M)")


# Using genomic vocab size: 7
# ============================================================
# Finding configuration for 300M parameters...
# Starting search from hidden_dim=896, layers=18

# Target: 300M parameters
# Actual: 299,953,152 (300.0M) parameters
# Difference: 0.0M
# Config:
#   hidden_dim: 1024
#   intermediate_dim: 3072
#   num_layers: 22
#   num_heads: 16
#   num_kv_heads: 16
#   seq_len: 512
# ----------------------------------------
# Finding configuration for 100M parameters...
# Starting search from hidden_dim=768, layers=12

# Target: 100M parameters
# Actual: 102,659,328 (102.7M) parameters
# Difference: 2.7M
# Config:
#   hidden_dim: 768
#   intermediate_dim: 2688
#   num_layers: 12
#   num_heads: 12
#   num_kv_heads: 12
#   seq_len: 512
# ----------------------------------------
# Finding configuration for 30M parameters...
# Starting search from hidden_dim=512, layers=8

# Target: 30M parameters
# Actual: 30,424,576 (30.4M) parameters
# Difference: 0.4M
# Config:
#   hidden_dim: 512
#   intermediate_dim: 1792
#   num_layers: 8
#   num_heads: 8
#   num_kv_heads: 8
#   seq_len: 512
# ----------------------------------------
# Finding configuration for 10M parameters...
# Starting search from hidden_dim=384, layers=6

# Target: 10M parameters
# Actual: 9,511,680 (9.5M) parameters
# Difference: 0.5M
# Config:
#   hidden_dim: 256
#   intermediate_dim: 896
#   num_layers: 10
#   num_heads: 8
#   num_kv_heads: 8
#   seq_len: 512
# ----------------------------------------

# ============================================================
# GENERATED CONFIGURATIONS FOR GENOMIC MODELS (EVEN NUMBERS)
# ============================================================

# llama_genomic_300m = LlamaConfig(
#     seq_len=512,
#     hidden_dim=1024,
#     intermediate_dim=3072,
#     num_heads=16,
#     num_kv_heads=16,
#     num_layers=22,
# )
# # Actual params: 299,953,152 (300.0M)

# llama_genomic_100m = LlamaConfig(
#     seq_len=512,
#     hidden_dim=768,
#     intermediate_dim=2688,
#     num_heads=12,
#     num_kv_heads=12,
#     num_layers=12,
# )
# # Actual params: 102,659,328 (102.7M)

# llama_genomic_30m = LlamaConfig(
#     seq_len=512,
#     hidden_dim=512,
#     intermediate_dim=1792,
#     num_heads=8,
#     num_kv_heads=8,
#     num_layers=8,
# )
# # Actual params: 30,424,576 (30.4M)

# llama_genomic_10m = LlamaConfig(
#     seq_len=512,
#     hidden_dim=256,
#     intermediate_dim=896,
#     num_heads=8,
#     num_kv_heads=8,
#     num_layers=10,
# )
# # Actual params: 9,511,680 (9.5M)
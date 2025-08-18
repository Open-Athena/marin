#!/usr/bin/env python3
"""
Parameter counting utility for LlamaConfig models.

This script creates an actual LlamaLMHeadModel instance and counts its parameters,
which gives you the exact parameter count for a given configuration.
"""

import jax.random as jrandom
from haliax import Axis
from haliax.partitioning import round_axis_for_partitioning
from levanter.models.llama import LlamaConfig, LlamaLMHeadModel
from levanter.utils.jax_utils import parameter_count


def count_parameters_exact(config: LlamaConfig, vocab_size: int = 128256) -> dict:
    """
    Count the exact number of parameters by instantiating a LlamaLMHeadModel.
    
    Args:
        config: LlamaConfig instance
        vocab_size: Vocabulary size (default: 128256 for Llama3 tokenizer)
    
    Returns:
        Dictionary with parameter count and config info
    """
    # Create vocab axis (no partitioning needed for parameter counting)
    Vocab = Axis("vocab", vocab_size)
    
    # Generate a random key for model initialization
    key = jrandom.PRNGKey(0)
    
    # Initialize the actual model
    model = LlamaLMHeadModel.init(Vocab, config, key=key)
    
    # Count the actual parameters using JAX
    actual_params = parameter_count(model)
    
    # Also get the theoretical count for comparison
    theoretical_params = config.total_trainable_params(vocab_size)
    
    return {
        "actual_params": actual_params,
        "theoretical_params": theoretical_params,
        "match": actual_params == theoretical_params,
        "actual_params_millions": actual_params / 1e6,
        "actual_params_billions": actual_params / 1e9,
        "vocab_size": vocab_size,
        "config": {
            "hidden_dim": config.hidden_dim,
            "intermediate_dim": config.intermediate_dim,
            "num_layers": config.num_layers,
            "num_heads": config.num_heads,
            "num_kv_heads": config.num_kv_heads,
            "seq_len": config.seq_len,
            "tie_word_embeddings": config.tie_word_embeddings,
            "hybrid_norm": config.hybrid_norm,
            "input_embedding_norm": config.input_embedding_norm,
        }
    }


def count_parameters_theoretical(config: LlamaConfig, vocab_size: int = 128256) -> dict:
    """
    Count parameters using the theoretical calculation (for comparison/verification).
    
    Args:
        config: LlamaConfig instance
        vocab_size: Vocabulary size (default: 128256 for Llama3 tokenizer)
    
    Returns:
        Dictionary with parameter breakdown and totals
    """
    # Use the built-in method from LlamaConfig
    total_params = config.total_trainable_params(vocab_size)
    
    # Calculate breakdown for better understanding
    head_size = config.hidden_dim // config.num_heads
    
    # Attention parameters per layer
    q_proj = config.hidden_dim * head_size * config.num_heads
    kv_proj = 2 * config.hidden_dim * head_size * config.num_kv_heads
    o_proj = head_size * config.num_heads * config.hidden_dim
    attn_per_layer = q_proj + kv_proj + o_proj
    
    # MLP parameters per layer (3 layers for SwiGLU: gate, up, down)
    mlp_per_layer = 3 * config.hidden_dim * config.intermediate_dim
    
    # Layer norm parameters per layer (2 RMSNorm layers)
    layernorm_per_layer = 2 * config.hidden_dim
    
    # Hybrid norm (additional layer norms after attention and MLP)
    hybrid_norm_per_layer = 2 * config.hidden_dim if config.hybrid_norm else 0
    
    # Total per transformer layer
    transformer_layer_params = attn_per_layer + mlp_per_layer + layernorm_per_layer + hybrid_norm_per_layer
    
    # All transformer layers
    all_transformer_layers = config.num_layers * transformer_layer_params
    
    # Final layer norm
    final_layernorm = config.hidden_dim
    
    # Input embedding layer norm (if enabled)
    input_embedding_norm = config.hidden_dim if config.input_embedding_norm else 0
    
    # Embedding parameters (token embedding + lm_head)
    if config.tie_word_embeddings:
        embedding_params = vocab_size * config.hidden_dim  # only input embedding
    else:
        embedding_params = vocab_size * config.hidden_dim * 2  # input embedding + output lm_head
    
    # Total non-embedding parameters
    non_embedding_params = all_transformer_layers + final_layernorm + input_embedding_norm
    
    return {
        "total_params": total_params,
        "total_params_millions": total_params / 1e6,
        "total_params_billions": total_params / 1e9,
        "breakdown": {
            "embedding_params": embedding_params,
            "non_embedding_params": non_embedding_params,
            "transformer_layers": {
                "num_layers": config.num_layers,
                "params_per_layer": transformer_layer_params,
                "total_layers_params": all_transformer_layers,
                "attention_per_layer": attn_per_layer,
                "mlp_per_layer": mlp_per_layer,
                "layernorm_per_layer": layernorm_per_layer,
                "hybrid_norm_per_layer": hybrid_norm_per_layer,
            },
            "final_layernorm": final_layernorm,
            "input_embedding_norm": input_embedding_norm,
        }
    }


def print_parameter_summary(config: LlamaConfig, vocab_size: int = 128256, use_exact: bool = True):
    """Print a nice summary of parameters for a LlamaConfig."""
    if use_exact:
        result = count_parameters_exact(config, vocab_size)
        print(f"Model Configuration:")
        print(f"  Hidden dim: {result['config']['hidden_dim']}")
        print(f"  Intermediate dim: {result['config']['intermediate_dim']}")
        print(f"  Num layers: {result['config']['num_layers']}")
        print(f"  Num heads: {result['config']['num_heads']}")
        print(f"  Num KV heads: {result['config']['num_kv_heads']}")
        print(f"  Seq length: {result['config']['seq_len']}")
        print(f"  Vocab size: {result['vocab_size']}")
        print(f"  Tie embeddings: {result['config']['tie_word_embeddings']}")
        print()
        
        print(f"Parameter Count:")
        print(f"  Actual: {result['actual_params']:,} ({result['actual_params_millions']:.1f}M)")
        if result['actual_params_billions'] >= 1.0:
            print(f"          {result['actual_params_billions']:.2f}B parameters")
        print(f"  Theoretical: {result['theoretical_params']:,}")
        print(f"  Match: {result['match']}")
    else:
        result = count_parameters_theoretical(config, vocab_size)
        
        print(f"Model Configuration:")
        print(f"  Hidden dim: {config.hidden_dim}")
        print(f"  Intermediate dim: {config.intermediate_dim}")
        print(f"  Num layers: {config.num_layers}")
        print(f"  Num heads: {config.num_heads}")
        print(f"  Num KV heads: {config.num_kv_heads}")
        print(f"  Seq length: {config.seq_len}")
        print(f"  Vocab size: {vocab_size}")
        print(f"  Tie embeddings: {config.tie_word_embeddings}")
        print()
        
        print(f"Parameter Count:")
        print(f"  Total: {result['total_params']:,} ({result['total_params_millions']:.1f}M)")
        if result['total_params_billions'] >= 1.0:
            print(f"         {result['total_params_billions']:.2f}B parameters")
        print()
        
        print(f"  Embedding params: {result['breakdown']['embedding_params']:,}")
        print(f"  Non-embedding params: {result['breakdown']['non_embedding_params']:,}")
        print()
        
        layers = result['breakdown']['transformer_layers']
        print(f"  Per layer ({layers['num_layers']} layers):")
        print(f"    Attention: {layers['attention_per_layer']:,}")
        print(f"    MLP: {layers['mlp_per_layer']:,}")
        print(f"    Layer norm: {layers['layernorm_per_layer']:,}")
        if layers['hybrid_norm_per_layer'] > 0:
            print(f"    Hybrid norm: {layers['hybrid_norm_per_layer']:,}")
        print(f"    Total per layer: {layers['params_per_layer']:,}")


if __name__ == "__main__":
    # Example usage with different model sizes

    genomic_vocab_size = 8
    
    # Your example config
    example_config = LlamaConfig(
        seq_len=512,
        hidden_dim=512,
        intermediate_dim=1792,
        num_heads=8,
        num_kv_heads=8,
        num_layers=6,
    )
    
    print("=" * 60)
    print("EXAMPLE CONFIG (from your query)")
    print("=" * 60)
    print_parameter_summary(example_config, vocab_size=genomic_vocab_size, use_exact=True)
    
    # Compare with PlantCAD configs
    from experiments.plantcad.utils import get_plantcad_config
    
    print("\n" + "=" * 60)
    print("PLANTCAD NANO CONFIG")
    print("=" * 60)
    nano_config = get_plantcad_config("nano")
    print_parameter_summary(nano_config, vocab_size=genomic_vocab_size, use_exact=True)
    
    print("\n" + "=" * 60)
    print("PLANTCAD 10M CONFIG")
    print("=" * 60)
    config_10m = get_plantcad_config("10m")
    print_parameter_summary(config_10m, vocab_size=genomic_vocab_size, use_exact=True)
    
    print("\n" + "=" * 60)
    print("PLANTCAD 30M CONFIG (DEFAULT)")
    print("=" * 60)
    config_30m = get_plantcad_config("30m")
    print_parameter_summary(config_30m, vocab_size=genomic_vocab_size, use_exact=True)
    
    print("\n" + "=" * 60)
    print("PLANTCAD 100M CONFIG")
    print("=" * 60)
    config_100m = get_plantcad_config("100m")
    print_parameter_summary(config_100m, vocab_size=genomic_vocab_size, use_exact=True)
    
    # Show PlantCAD configs with genomic vocab size
    print("\nWith genomic vocab size (7):")
    print(f"\nNano config with genomic vocab: {count_parameters_exact(nano_config, genomic_vocab_size)['actual_params_millions']:.1f}M")
    print(f"10M config with genomic vocab: {count_parameters_exact(config_10m, genomic_vocab_size)['actual_params_millions']:.1f}M")
    print(f"30M config with genomic vocab: {count_parameters_exact(config_30m, genomic_vocab_size)['actual_params_millions']:.1f}M")
    print(f"100M config with genomic vocab: {count_parameters_exact(config_100m, genomic_vocab_size)['actual_params_millions']:.1f}M")


# python -m experiments.plantcad.misc.param_counter
# ============================================================
# EXAMPLE CONFIG (from your query)
# ============================================================
# Model Configuration:
#   Hidden dim: 512
#   Intermediate dim: 1792
#   Num layers: 6
#   Num heads: 8
#   Num KV heads: 8
#   Seq length: 512
#   Vocab size: 8
#   Tie embeddings: False

# Parameter Count:
#   Actual: 22,821,376 (22.8M)
#   Theoretical: 22,821,376
#   Match: True

# ============================================================
# PLANTCAD NANO CONFIG
# ============================================================
# Model Configuration:
#   Hidden dim: 32
#   Intermediate dim: 128
#   Num layers: 2
#   Num heads: 2
#   Num KV heads: 2
#   Seq length: 512
#   Vocab size: 8
#   Tie embeddings: False

# Parameter Count:
#   Actual: 33,440 (0.0M)
#   Theoretical: 33,440
#   Match: True

# ============================================================
# PLANTCAD 10M CONFIG
# ============================================================
# Model Configuration:
#   Hidden dim: 256
#   Intermediate dim: 896
#   Num layers: 10
#   Num heads: 8
#   Num KV heads: 8
#   Seq length: 512
#   Vocab size: 8
#   Tie embeddings: False

# Parameter Count:
#   Actual: 9,512,192 (9.5M)
#   Theoretical: 9,512,192
#   Match: True

# ============================================================
# PLANTCAD 30M CONFIG (DEFAULT)
# ============================================================
# Model Configuration:
#   Hidden dim: 512
#   Intermediate dim: 1792
#   Num layers: 8
#   Num heads: 8
#   Num KV heads: 8
#   Seq length: 512
#   Vocab size: 8
#   Tie embeddings: False

# Parameter Count:
#   Actual: 30,425,600 (30.4M)
#   Theoretical: 30,425,600
#   Match: True

# ============================================================
# PLANTCAD 100M CONFIG
# ============================================================
# Model Configuration:
#   Hidden dim: 768
#   Intermediate dim: 2688
#   Num layers: 12
#   Num heads: 12
#   Num KV heads: 12
#   Seq length: 512
#   Vocab size: 8
#   Tie embeddings: False

# Parameter Count:
#   Actual: 102,660,864 (102.7M)
#   Theoretical: 102,660,864
#   Match: True

# With genomic vocab size (7):

# Nano config with genomic vocab: 0.0M
# 10M config with genomic vocab: 9.5M
# 30M config with genomic vocab: 30.4M
# 100M config with genomic vocab: 102.7M
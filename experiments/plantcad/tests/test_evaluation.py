# Copyright 2025 The Marin Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pytest
import jax
import jax.numpy as jnp
import haliax as hax
from experiments.plantcad.evaluation import (
    create_alternate_sequences,
    compute_sequence_logprob,
    compute_masked_conservation,
    compute_causal_conservation,
)


def test_create_alternate_sequences():
    """Test create_alternate_sequences with simple sequences."""

    # Test setup: 2 batches, 5 positions, 4 nucleotide tokens
    batch_size = 2
    seq_length = 5

    # Nucleotide token IDs: A=0, C=1, T=2, G=3
    nucleotide_token_ids = [0, 1, 2, 3]

    # Create axes
    Batch = hax.Axis("batch", batch_size)
    Position = hax.Axis("position", seq_length)

    # Create test tokens:
    tokens_data = jnp.array(
        [
            [0, 1, 2, 3, 0],  # A, C, T, G, A
            [1, 2, 3, 0, 1],  # C, T, G, A, C
        ]
    )
    tokens = hax.named(tokens_data, (Batch, Position))

    # Target positions for substitution:
    # Batch 0: position 2 (currently T=2)
    # Batch 1: position 1 (currently T=2)
    positions_data = jnp.array([2, 1])
    nucleotide_positions = hax.named(positions_data, (Batch,))

    # Run the function
    alt, ref = create_alternate_sequences(tokens, nucleotide_positions, nucleotide_token_ids)

    # Check output shapes
    assert alt.ndim == 3
    Batch_out, Position_out, Variant = alt.axes
    assert Batch_out.size == batch_size
    assert Position_out.size == seq_length
    assert Variant.size == 4

    assert ref.ndim == 1
    assert ref.axes[0].size == batch_size

    # Check that substitutions happened correctly
    alt_array = alt.array

    # For batch 0, position 2 should be substituted with [A, C, T, G] = [0, 1, 2, 3]
    # Original: [A, C, T, G, A] = [0, 1, 2, 3, 0]
    # Expected variants at position 2:
    # Variant 0: [0, 1, 0, 3, 0] (A substituted)
    # Variant 1: [0, 1, 1, 3, 0] (C substituted)
    # Variant 2: [0, 1, 2, 3, 0] (T substituted - same as original)
    # Variant 3: [0, 1, 3, 3, 0] (G substituted)

    expected_batch0 = [
        [0, 1, 0, 3, 0],  # A substituted at position 2
        [0, 1, 1, 3, 0],  # C substituted at position 2
        [0, 1, 2, 3, 0],  # T substituted at position 2 (original)
        [0, 1, 3, 3, 0],  # G substituted at position 2
    ]

    _assert_batch_variants(alt_array, 0, expected_batch0, seq_length, "Batch 0")

    # For batch 1, position 1 should be substituted with [A, C, T, G] = [0, 1, 2, 3]
    # Original: [C, T, G, A, C] = [1, 2, 3, 0, 1]
    # Expected variants at position 1:
    # Variant 0: [1, 0, 3, 0, 1] (A substituted)
    # Variant 1: [1, 1, 3, 0, 1] (C substituted)
    # Variant 2: [1, 2, 3, 0, 1] (T substituted - same as original)
    # Variant 3: [1, 3, 3, 0, 1] (G substituted)

    expected_batch1 = [
        [1, 0, 3, 0, 1],  # A substituted at position 1
        [1, 1, 3, 0, 1],  # C substituted at position 1
        [1, 2, 3, 0, 1],  # T substituted at position 1 (original)
        [1, 3, 3, 0, 1],  # G substituted at position 1
    ]

    _assert_batch_variants(alt_array, 1, expected_batch1, seq_length, "Batch 1")

    # Check reference indices - should point to variants matching original tokens
    # Batch 0: original token at position 2 is T=2, so ref should be 2 (nucleotide_token_ids[2] = 2)
    # Batch 1: original token at position 1 is T=2, so ref should be 2 (nucleotide_token_ids[2] = 2)
    assert ref.array[0] == 2, f"Batch 0 ref should be 2 (T), got {ref.array[0]}"
    assert ref.array[1] == 2, f"Batch 1 ref should be 2 (T), got {ref.array[1]}"


def test_create_alternate_sequences_invalid_nucleotide():
    """Test that create_alternate_sequences raises ValueError for OOV nucleotides at target positions."""

    # Create simple test case with invalid nucleotide
    Batch = hax.Axis("batch", 1)
    Position = hax.Axis("position", 3)

    # Valid nucleotide token IDs: A=0, C=1, T=2, G=3
    nucleotide_token_ids = [0, 1, 2, 3]

    # Token sequence with invalid nucleotide (5) at position 1
    tokens_data = jnp.array([[0, 5, 2]])  # A, INVALID, T
    tokens = hax.named(tokens_data, (Batch, Position))

    # Target position 1 (the invalid nucleotide)
    nucleotide_positions = hax.named(jnp.array([1]), (Batch,))

    # Should raise ValueError
    with pytest.raises(ValueError, match="Found invalid sequences in batch with OOV nucleotides"):
        create_alternate_sequences(tokens, nucleotide_positions, nucleotide_token_ids)


def test_compute_sequence_logprob():
    """Test compute_sequence_logprob with manually specified sequences and expected ordering."""

    # Test with 3 sequences, 4 positions, 4 vocab tokens
    batch_size = 3
    seq_length = 4
    vocab_size = 4

    # Create axes
    Batch = hax.Axis("batch", batch_size)
    Position = hax.Axis("position", seq_length)
    Vocab = hax.Axis("vocab", vocab_size)

    # Create three test sequences: [start_token, tok1, tok2, tok3]
    # Sequence 0: [0, 1, 2, 3] - will get HIGH log prob
    # Sequence 1: [0, 3, 2, 1] - will get MEDIUM log prob
    # Sequence 2: [0, 0, 0, 0] - will get LOW log prob
    tokens_data = jnp.array(
        [
            [0, 1, 2, 3],  # High prob sequence
            [0, 3, 2, 1],  # Medium prob sequence
            [0, 0, 0, 0],  # Low prob sequence
        ]
    )
    tokens = hax.named(tokens_data, (Batch, Position))

    # Create logits that favor the next tokens for sequence 0, somewhat for sequence 1, not for sequence 2
    # Logits predict: pos0->tok1, pos1->tok2, pos2->tok3 (shifted by 1)
    logits_data = jnp.array(
        [
            # Sequence 0: logits strongly favor the actual next tokens [1, 2, 3]
            [
                [0.0, 3.0, 0.0, 0.0],  # pos 0: strongly predict token 1 (next is 1) ✓
                [0.0, 0.0, 3.0, 0.0],  # pos 1: strongly predict token 2 (next is 2) ✓
                [0.0, 0.0, 0.0, 3.0],  # pos 2: strongly predict token 3 (next is 3) ✓
                [0.0, 0.0, 0.0, 0.0],  # pos 3: uniform (no next token)
            ],
            # Sequence 1: logits moderately favor the actual next tokens [3, 2, 1]
            [
                [0.0, 0.0, 0.0, 1.5],  # pos 0: moderately predict token 3 (next is 3) ✓
                [0.0, 0.0, 1.5, 0.0],  # pos 1: moderately predict token 2 (next is 2) ✓
                [0.0, 1.5, 0.0, 0.0],  # pos 2: moderately predict token 1 (next is 1) ✓
                [0.0, 0.0, 0.0, 0.0],  # pos 3: uniform (no next token)
            ],
            # Sequence 2: logits oppose the actual next tokens [0, 0, 0]
            [
                [0.0, 3.0, 3.0, 3.0],  # pos 0: predict anything BUT token 0 (next is 0) ✗
                [0.0, 3.0, 3.0, 3.0],  # pos 1: predict anything BUT token 0 (next is 0) ✗
                [0.0, 3.0, 3.0, 3.0],  # pos 2: predict anything BUT token 0 (next is 0) ✗
                [0.0, 0.0, 0.0, 0.0],  # pos 3: uniform (no next token)
            ],
        ]
    )
    logits = hax.named(logits_data, (Batch, Position, Vocab))

    # Run the function
    result = compute_sequence_logprob(logits, tokens)

    # Check output shape
    assert result.axes == (Batch,)
    assert result.array.shape == (batch_size,)

    # Check that result values are finite
    assert jnp.all(jnp.isfinite(result.array))

    # Check expected ordering: sequence 0 > sequence 1 > sequence 2
    # (higher log prob = less negative = better prediction)
    seq0_logprob = result.array[0]  # Should be highest (least negative)
    seq1_logprob = result.array[1]  # Should be medium
    seq2_logprob = result.array[2]  # Should be lowest (most negative)

    assert seq0_logprob > seq1_logprob, f"Seq0 should have higher log prob than Seq1: {seq0_logprob} > {seq1_logprob}"
    assert seq1_logprob > seq2_logprob, f"Seq1 should have higher log prob than Seq2: {seq1_logprob} > {seq2_logprob}"
    assert seq0_logprob > seq2_logprob, f"Seq0 should have higher log prob than Seq2: {seq0_logprob} > {seq2_logprob}"


def test_compute_masked_conservation():
    """Test compute_masked_conservation with valid and invalid nucleotides."""
    setup = _conservation_test_setup(batch_axis_name="batch")
    mask_token_id = 99

    # Test 1: Valid nucleotides - should work
    valid_tokens_data = jnp.array([[0, 1, 2]])  # A, C, T
    valid_tokens = hax.named(valid_tokens_data, (setup["Batch"], setup["Position"]))

    result = compute_masked_conservation(
        tokens=valid_tokens,
        logit_function=setup["mock_logit_function"],
        nucleotide_positions=setup["nucleotide_positions"],
        nucleotide_token_ids=setup["nucleotide_token_ids"],
        mask_token_id=mask_token_id,
    )

    # Check basic properties
    assert result.axes == (setup["Batch"],)
    assert result.array.shape == (setup["batch_size"],)
    assert jnp.all(jnp.isfinite(result.array))

    # Test 2: Invalid nucleotides - should raise error
    invalid_tokens_data = jnp.array([[0, 5, 2]])  # A, INVALID, T
    invalid_tokens = hax.named(invalid_tokens_data, (setup["Batch"], setup["Position"]))

    with pytest.raises(ValueError, match="Found invalid sequences in batch with OOV nucleotides"):
        compute_masked_conservation(
            tokens=invalid_tokens,
            logit_function=setup["mock_logit_function"],
            nucleotide_positions=setup["nucleotide_positions"],
            nucleotide_token_ids=setup["nucleotide_token_ids"],
            mask_token_id=mask_token_id,
        )


def test_compute_causal_conservation():
    """Test compute_causal_conservation with valid and invalid nucleotides."""
    setup = _conservation_test_setup(batch_axis_name="batch_variant")

    # Test 1: Valid nucleotides - should work
    valid_tokens_data = jnp.array([[0, 1, 2]])  # A, C, T
    valid_tokens = hax.named(valid_tokens_data, (setup["Batch"], setup["Position"]))

    result = compute_causal_conservation(
        tokens=valid_tokens,
        logit_function=setup["mock_logit_function"],
        nucleotide_positions=setup["nucleotide_positions"],
        nucleotide_token_ids=setup["nucleotide_token_ids"],
    )

    # Check basic properties
    assert result.axes == (setup["Batch"],)
    assert result.array.shape == (setup["batch_size"],)
    assert jnp.all(jnp.isfinite(result.array))

    # Test 2: Invalid nucleotides - should raise error
    invalid_tokens_data = jnp.array([[0, 5, 2]])  # A, INVALID, T
    invalid_tokens = hax.named(invalid_tokens_data, (setup["Batch"], setup["Position"]))

    with pytest.raises(ValueError, match="Found invalid sequences in batch with OOV nucleotides"):
        compute_causal_conservation(
            tokens=invalid_tokens,
            logit_function=setup["mock_logit_function"],
            nucleotide_positions=setup["nucleotide_positions"],
            nucleotide_token_ids=setup["nucleotide_token_ids"],
        )


def _assert_batch_variants(alt_array, batch_idx, expected_variants, seq_length, batch_name):
    """Helper to assert variant sequences match expected values for a batch."""
    for variant_idx in range(4):
        for pos_idx in range(seq_length):
            actual = alt_array[batch_idx, pos_idx, variant_idx]
            expected = expected_variants[variant_idx][pos_idx]
            assert (
                actual == expected
            ), f"{batch_name}, position {pos_idx}, variant {variant_idx}: expected {expected}, got {actual}"


def _conservation_test_setup(batch_axis_name: str = "batch"):
    """Shared setup for conservation function tests.

    Args:
        batch_axis_name: Name of the expected batch axis in input tokens.
    """
    batch_size = 1
    seq_length = 3
    vocab_size = 4

    # Create axes
    Batch = hax.Axis("batch", batch_size)
    Position = hax.Axis("position", seq_length)
    Vocab = hax.Axis("vocab", vocab_size)

    # Nucleotide token IDs
    nucleotide_token_ids = [0, 1, 2, 3]

    # Target position 1
    nucleotide_positions = hax.named(jnp.array([1]), (Batch,))

    # Mock function that returns random logits with fixed seed
    def mock_logit_function(input_tokens):
        batch_size, seq_len = input_tokens.array.shape
        BatchAxis = input_tokens.resolve_axis(batch_axis_name)
        assert input_tokens.axes == (BatchAxis, Position)
        key = jax.random.PRNGKey(42)
        logits_array = jax.random.normal(key, (batch_size, seq_len, vocab_size))
        return hax.named(logits_array, (BatchAxis, Position, Vocab))

    return {
        "batch_size": batch_size,
        "seq_length": seq_length,
        "vocab_size": vocab_size,
        "Batch": Batch,
        "Position": Position,
        "Vocab": Vocab,
        "nucleotide_token_ids": nucleotide_token_ids,
        "nucleotide_positions": nucleotide_positions,
        "mock_logit_function": mock_logit_function,
    }

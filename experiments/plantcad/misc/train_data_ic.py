#!/usr/bin/env python3
"""
Script to download, decompress, analyze, and recompress a file from HuggingFace.
Uses Docker for bz2 compression since it's not available locally.

References: https://deepmind.google/research/publications/39768/
"""

import os
import subprocess
import sys
import json
import argparse
import multiprocessing as mp
import numpy as np
from pathlib import Path
import zstandard as zstd
import pandas as pd
import tiktoken
from huggingface_hub import hf_hub_download
from tqdm import tqdm

def verify_download_integrity(file_path: Path, expected_size: int = None) -> bool:
    """Verify that a downloaded file is complete and not corrupted."""
    if not file_path.exists():
        return False
    
    actual_size = file_path.stat().st_size
    print(f"File size verification: {actual_size:,} bytes")
    
    if expected_size and actual_size != expected_size:
        print(f"WARNING: File size mismatch! Expected: {expected_size:,}, Got: {actual_size:,}")
        return False
    
    # Test if we can read the beginning and end of the file
    try:
        with open(file_path, 'rb') as f:
            # Read first 1KB
            f.read(1024)
            # Seek to end-1KB and read
            if actual_size > 1024:
                f.seek(-1024, 2)
                f.read(1024)
        print("✅ File integrity check passed - file is readable")
        return True
    except Exception as e:
        print(f"❌ File integrity check failed: {e}")
        return False

def download_file_hf(repo_id: str, filename: str, output_path: Path) -> None:
    """Download a file from HuggingFace Hub using the official API."""
    print(f"Downloading {filename} from {repo_id} to {output_path}")
    
    try:
        # HuggingFace Hub download with built-in resume capability
        downloaded_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            repo_type="dataset",
            local_dir=output_path.parent
        )
        
        # If the downloaded file has a different name, rename it
        downloaded_file = Path(downloaded_path)
        if downloaded_file != output_path:
            downloaded_file.rename(output_path)
        
        print(f"✅ Downloaded: {output_path} ({get_file_size_str(output_path)})")
        
    except Exception as e:
        print(f"❌ Download error: {e}")
        raise

def extract_sequences_from_zst(input_path: Path, output_path: Path) -> int:
    """Stream through compressed file and extract just the 'seq' values."""
    print(f"Extracting sequences from {input_path} to {output_path}")
    print("Streaming through compressed file...")
    
    sequence_count = 0
    
    with open(input_path, 'rb') as compressed_file:
        dctx = zstd.ZstdDecompressor()
        
        with open(output_path, 'w', encoding='utf-8') as seq_file:
            # Stream decompress and process line by line
            text_stream = dctx.stream_reader(compressed_file)
            
            buffer = ""
            chunk_size = 8192
            
            # Create progress bar based on compressed file size
            file_size = input_path.stat().st_size
            with tqdm(total=file_size, desc="Processing compressed data", unit="B", unit_scale=True) as pbar:
                bytes_processed = 0
                
                while True:
                    chunk = text_stream.read(chunk_size)
                    if not chunk:
                        break
                    
                    bytes_processed += len(chunk)
                    pbar.update(len(chunk))
                    pbar.set_postfix(sequences=sequence_count)
                    
                    # Decode chunk and add to buffer
                    buffer += chunk.decode('utf-8', errors='ignore')
                    
                    # Process complete lines
                    while '\n' in buffer:
                        line, buffer = buffer.split('\n', 1)
                        line = line.strip()
                        
                        if line:
                            try:
                                record = json.loads(line)
                                if 'seq' in record:
                                    seq_file.write(record['seq'] + '\n')
                                    sequence_count += 1
                                        
                            except json.JSONDecodeError:
                                # Skip invalid JSON lines
                                continue
                
                # Process any remaining data in buffer
                if buffer.strip():
                    try:
                        record = json.loads(buffer.strip())
                        if 'seq' in record:
                            seq_file.write(record['seq'] + '\n')
                            sequence_count += 1
                    except json.JSONDecodeError:
                        pass
    
    print(f"✅ Extracted {sequence_count:,} sequences to {output_path} ({get_file_size_str(output_path)})")
    return sequence_count

def extract_text_from_parquet(input_path: Path, output_path: Path) -> int:
    """Extract 'text' field from parquet file."""
    print(f"Extracting text from {input_path} to {output_path}")
    print("Reading parquet file...")
    
    # Read the parquet file
    df = pd.read_parquet(input_path)
    
    if 'text' not in df.columns:
        raise ValueError(f"No 'text' column found in parquet file. Available columns: {list(df.columns)}")
    
    text_count = 0
    
    with open(output_path, 'w', encoding='utf-8') as text_file:
        with tqdm(total=len(df), desc="Extracting texts", unit="texts") as pbar:
            for idx, text in enumerate(df['text']):
                if pd.notna(text) and text.strip():  # Skip empty or NaN texts
                    text_file.write(str(text) + '\n')
                    text_count += 1
                
                pbar.update(1)
                pbar.set_postfix(extracted=text_count)
    
    print(f"✅ Extracted {text_count:,} texts to {output_path} ({get_file_size_str(output_path)})")
    return text_count

def extract_text_from_multiple_parquets(parquet_files: list, output_path: Path) -> int:
    """Extract 'text' field from multiple parquet files into one combined file."""
    total_text_count = 0
    
    with open(output_path, 'w', encoding='utf-8') as combined_file:
        for i, parquet_file in enumerate(parquet_files):
            print(f"Processing file {i+1}/{len(parquet_files)}: {parquet_file.name}")
            
            # Read the parquet file
            df = pd.read_parquet(parquet_file)
            
            if 'text' not in df.columns:
                raise ValueError(f"No 'text' column found in {parquet_file}. Available columns: {list(df.columns)}")
            
            file_text_count = 0
            
            with tqdm(total=len(df), desc=f"Extracting from {parquet_file.name}", unit="texts") as pbar:
                for idx, text in enumerate(df['text']):
                    if pd.notna(text) and text.strip():  # Skip empty or NaN texts
                        combined_file.write(str(text) + '\n')
                        file_text_count += 1
                        total_text_count += 1
                    
                    pbar.update(1)
                    pbar.set_postfix(extracted=file_text_count)
            
            print(f"✅ Extracted {file_text_count:,} texts from {parquet_file.name}")
    
    print(f"✅ Total extracted: {total_text_count:,} texts to {output_path} ({get_file_size_str(output_path)})")
    return total_text_count

def decompress_zst(input_path: Path, output_path: Path) -> None:
    """Decompress a .zst file to plain text."""
    print(f"Decompressing {input_path} to {output_path}")
    
    with open(input_path, 'rb') as compressed_file:
        dctx = zstd.ZstdDecompressor()
        with open(output_path, 'wb') as output_file:
            dctx.copy_stream(compressed_file, output_file)
    
    print(f"Decompressed: {output_path} ({get_file_size_str(output_path)})")

def count_lines(file_path: Path) -> int:
    """Count the number of lines in a file."""
    print(f"Counting lines in {file_path}")
    
    line_count = 0
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line_count += 1
    
    print(f"Line count: {line_count:,}")
    return line_count

def compress_with_docker(input_path: Path, output_path: Path) -> None:
    """Use Docker to compress a file with bz2."""
    print(f"Compressing {input_path} to {output_path} using Docker")
    print("This may take several minutes for large files...")
    
    # Get absolute paths
    input_abs = input_path.absolute()
    output_abs = output_path.absolute()
    
    # Docker command to install bzip2 and compress
    docker_cmd = [
        'docker', 'run', '--rm',
        '-v', f'{input_abs.parent}:/data',
        'ubuntu:latest',
        'bash', '-c',
        f'apt-get update -qq && apt-get install -y -qq bzip2 && echo "Starting compression..." && bzip2 -v -c /data/{input_abs.name} > /data/{output_abs.name} && echo "Compression completed" && ls -lh /data/{output_abs.name}'
    ]
    
    try:
        print("Running Docker container...")
        result = subprocess.run(docker_cmd, check=True, text=True)
        print(f"Compressed: {output_path} ({get_file_size_str(output_path)})")
    except subprocess.CalledProcessError as e:
        print(f"Docker compression failed: {e}")
        print(f"Return code: {e.returncode}")
        if hasattr(e, 'stdout') and e.stdout:
            print(f"stdout: {e.stdout}")
        if hasattr(e, 'stderr') and e.stderr:
            print(f"stderr: {e.stderr}")
        raise

def count_dna_tokens_worker(sequence_chunk):
    """Worker function to count DNA tokens for a chunk of sequences."""
    total_tokens = 0
    
    for sequence in sequence_chunk:
        if sequence and sequence.strip():  # Skip empty lines
            total_tokens += len(sequence.strip())
    
    return total_tokens

def count_dna_tokens(file_path: Path) -> int:
    """Count DNA tokens (1 token per character in DNA sequences) with multiprocessing."""
    print(f"Counting DNA tokens in {file_path} using 5 parallel processes with 25 chunks...")
    
    # Read all lines into memory
    print("Loading file into memory...")
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    total_lines = len(lines)
    print(f"Processing {total_lines:,} sequences in 25 chunks with 5 parallel workers...")
    
    # Split lines into 25 chunks for better parallelization
    chunk_size = max(1, total_lines // 25)
    chunks = [lines[i:i + chunk_size] for i in range(0, total_lines, chunk_size)]
    
    total_tokens = 0
    
    # Use multiprocessing with progress bar
    with mp.Pool(processes=5) as pool:
        with tqdm(total=len(chunks), desc="Processing chunks", unit="chunks") as pbar:
            # Submit all chunks for processing
            results = []
            for chunk in chunks:
                result = pool.apply_async(count_dna_tokens_worker, (chunk,))
                results.append(result)
            
            # Collect results as they complete
            for result in results:
                chunk_tokens = result.get()
                total_tokens += chunk_tokens
                pbar.update(1)
                pbar.set_postfix(tokens=f"{total_tokens:,}")
    
    print(f"✅ Total DNA tokens: {total_tokens:,}")
    return total_tokens

def count_tokens_worker(text_chunk):
    """Worker function to count tokens for a chunk of text."""
    encoding = tiktoken.get_encoding("cl100k_base")
    total_tokens = 0
    
    for text in text_chunk:
        if text and text.strip():  # Skip empty lines
            tokens = encoding.encode(text.strip())
            total_tokens += len(tokens)
    
    return total_tokens

def count_text_tokens(file_path: Path) -> int:
    """Count text tokens using tiktoken (GPT-4 encoding) with multiprocessing."""
    print(f"Counting text tokens in {file_path} using 5 parallel processes with 25 chunks...")
    
    # Read all lines into memory
    print("Loading file into memory...")
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    total_lines = len(lines)
    print(f"Processing {total_lines:,} lines in 25 chunks with 5 parallel workers...")
    
    # Split lines into 25 chunks for better parallelization
    chunk_size = max(1, total_lines // 25)
    chunks = [lines[i:i + chunk_size] for i in range(0, total_lines, chunk_size)]
    
    total_tokens = 0
    
    # Use multiprocessing with progress bar
    with mp.Pool(processes=5) as pool:
        with tqdm(total=len(chunks), desc="Processing chunks", unit="chunks") as pbar:
            # Submit all chunks for processing
            results = []
            for chunk in chunks:
                result = pool.apply_async(count_tokens_worker, (chunk,))
                results.append(result)
            
            # Collect results as they complete
            for result in results:
                chunk_tokens = result.get()
                total_tokens += chunk_tokens
                pbar.update(1)
                pbar.set_postfix(tokens=f"{total_tokens:,}")
    
    print(f"✅ Total text tokens: {total_tokens:,}")
    return total_tokens

def get_file_size_str(file_path: Path) -> str:
    """Get human-readable file size."""
    size = file_path.stat().st_size
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size < 1024.0:
            return f"{size:.1f} {unit}"
        size /= 1024.0
    return f"{size:.1f} PB"

def format_number(num: float, precision: int = 2) -> str:
    """Format large numbers with appropriate units."""
    if num >= 1_000_000_000:
        return f"{num/1_000_000_000:.{precision}f}B"
    elif num >= 1_000_000:
        return f"{num/1_000_000:.{precision}f}M"
    elif num >= 1_000:
        return f"{num/1_000:.{precision}f}K"
    else:
        return f"{num:.{precision}f}"

def process_plantcad_dataset():
    """Process the plant genomics dataset."""
    print("🧬 Processing PlantCAD genomics dataset...")
    
    # File paths
    base_dir = Path.cwd()
    zst_file = base_dir / "train.jsonl.zst"
    sequences_file = base_dir / "train.sequences"
    sequences_bz2_file = base_dir / "train.sequences.bz2"
    
    # HuggingFace repository and file info
    repo_id = "kuleshov-group/Angiosperm_16_genomes"
    filename = "data/train/train.jsonl.zst"
    
    # Step 1: Download the compressed file using HuggingFace Hub
    if not zst_file.exists():
        download_file_hf(repo_id, filename, zst_file)
    else:
        print(f"File already exists: {zst_file} ({get_file_size_str(zst_file)})")
        
    # Verify download integrity
    if not verify_download_integrity(zst_file):
        print("❌ Download verification failed. Removing and re-downloading...")
        zst_file.unlink()
        download_file_hf(repo_id, filename, zst_file)
    
    # Step 2: Extract sequences directly from compressed file
    if not sequences_file.exists():
        item_count = extract_sequences_from_zst(zst_file, sequences_file)
    else:
        print(f"Sequences file already exists: {sequences_file} ({get_file_size_str(sequences_file)})")
        # Still count for the final report
        item_count = sum(1 for _ in open(sequences_file, 'r'))
        print(f"Found {item_count:,} sequences in existing file")
    
    # Step 3: Compress sequences file with Docker (bz2)
    if not sequences_bz2_file.exists():
        compress_with_docker(sequences_file, sequences_bz2_file)
    else:
        print(f"BZ2 sequences file already exists: {sequences_bz2_file} ({get_file_size_str(sequences_bz2_file)})")
    
    # Step 4: Count DNA tokens
    token_count = count_dna_tokens(sequences_file)
    
    return sequences_file, sequences_bz2_file, item_count, token_count, "sequences"

def process_wikipedia_dataset():
    """Process the Wikipedia dataset using 6 randomly sampled files."""
    print("📚 Processing Wikipedia dataset with 6 randomly sampled files...")
    
    # Fixed seed for reproducible results using numpy's default_rng
    rng = np.random.default_rng(42)
    
    # Randomly choose 6 file indices between 0 and 40 (inclusive)
    selected_indices = rng.choice(41, size=6, replace=False)
    selected_indices = sorted(selected_indices.tolist())  # Sort for cleaner output
    print(f"Selected file indices: {selected_indices}")
    
    # File paths
    base_dir = Path.cwd()
    texts_file = base_dir / "wikipedia.texts"
    texts_bz2_file = base_dir / "wikipedia.texts.bz2"
    
    # HuggingFace repository info
    repo_id = "wikimedia/wikipedia"
    
    total_item_count = 0
    
    # Step 1: Download and process all selected parquet files
    all_parquet_files = []
    for idx in selected_indices:
        parquet_file = base_dir / f"train-{idx:05d}-of-00041.parquet"
        filename = f"20231101.en/train-{idx:05d}-of-00041.parquet"
        all_parquet_files.append(parquet_file)
        
        print(f"\nProcessing file {idx:05d}...")
        
        # Download if needed
        if not parquet_file.exists():
            download_file_hf(repo_id, filename, parquet_file)
        else:
            print(f"File already exists: {parquet_file} ({get_file_size_str(parquet_file)})")
            
        # Verify download integrity
        if not verify_download_integrity(parquet_file):
            print("❌ Download verification failed. Removing and re-downloading...")
            parquet_file.unlink()
            download_file_hf(repo_id, filename, parquet_file)
    
    # Step 2: Extract text from all parquet files into one combined file
    if not texts_file.exists():
        print(f"\nExtracting texts from {len(selected_indices)} files to {texts_file}")
        total_item_count = extract_text_from_multiple_parquets(all_parquet_files, texts_file)
    else:
        print(f"Combined texts file already exists: {texts_file} ({get_file_size_str(texts_file)})")
        # Still count for the final report
        total_item_count = sum(1 for _ in open(texts_file, 'r'))
        print(f"Found {total_item_count:,} texts in existing file")
    
    # Step 3: Compress texts file with Docker (bz2)
    if not texts_bz2_file.exists():
        compress_with_docker(texts_file, texts_bz2_file)
    else:
        print(f"BZ2 texts file already exists: {texts_bz2_file} ({get_file_size_str(texts_bz2_file)})")
    
    # Step 4: Count text tokens using tiktoken
    token_count = count_text_tokens(texts_file)
    
    return texts_file, texts_bz2_file, total_item_count, token_count, "texts"

def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description='Download and analyze datasets from HuggingFace')
    parser.add_argument('dataset', choices=['plantcad', 'wikipedia'], 
                      help='Dataset to analyze: plantcad (genomics) or wikipedia (text)')
    
    args = parser.parse_args()
    
    try:
        if args.dataset == 'plantcad':
            data_file, compressed_file, item_count, token_count, data_type = process_plantcad_dataset()
        elif args.dataset == 'wikipedia':
            data_file, compressed_file, item_count, token_count, data_type = process_wikipedia_dataset()
        
        # Calculate file sizes
        uncompressed_size = data_file.stat().st_size
        compressed_size = compressed_file.stat().st_size
        
        # Step 4: Show final results
        print("\n" + "="*60)
        print(f"{args.dataset.upper()} ANALYSIS RESULTS:")
        print("="*60)
        print(f"Total {data_type} extracted: {item_count:,}")
        print(f"Total tokens: {token_count:,}")
        print(f"Data file (uncompressed): {get_file_size_str(data_file)}")
        print(f"Data file (bz2 compressed): {get_file_size_str(compressed_file)}")
        
        print(f"\nCompression analysis:")
        print(f"BZ2 vs uncompressed: {(compressed_size / uncompressed_size) * 100:.1f}%")
        
        # Calculate space saved
        space_saved = uncompressed_size - compressed_size
        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
            if space_saved < 1024.0:
                print(f"Space saved: {space_saved:.1f} {unit}")
                break
            space_saved /= 1024.0
        
        # Token analysis
        print(f"\nToken analysis:")
        print(f"Bytes per token (uncompressed): {uncompressed_size / token_count:.2f}")
        print(f"Bytes per token (compressed): {compressed_size / token_count:.2f}")
        
        # Store results for summary
        results = {
            'dataset': args.dataset,
            'token_count': token_count,
            'uncompressed_size': uncompressed_size,
            'compressed_size': compressed_size,
            'data_type': data_type
        }
        
        # Show dataset summary
        print("\n" + "="*80)
        print("DATASET SUMMARY:")
        print("="*80)
        print(f"Dataset: {args.dataset.capitalize()}")
        print(f"Token Count: {format_number(token_count)} ({token_count:,})")
        print(f"Compressed Size: {get_file_size_str(compressed_file)} ({compressed_size:,} bytes)")
        print(f"Uncompressed Size: {get_file_size_str(data_file)} ({uncompressed_size:,} bytes)")
        print(f"Compression Ratio: {(compressed_size/uncompressed_size)*100:.1f}%")
        print(f"Compressed Bytes per Token: {compressed_size/token_count:.3f}")
        print(f"Uncompressed Bytes per Token: {uncompressed_size/token_count:.3f}")
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    # Needed for multiprocessing on Windows and macOS
    mp.set_start_method('spawn', force=True)
    main()


# 📚 Processing Wikipedia dataset with 6 randomly sampled files...
# Selected file indices: [3, 17, 24, 28, 35, 39]
# ...
# Extracting texts from 6 files to /Users/eczech/repos/dev/wikipedia.texts
# Processing file 1/6: train-00003-of-00041.parquet
# Extracting from train-00003-of-00041.parquet: 100%|█| 156289/156289 [00:05<00:00, 28552.72texts/s, extra
# ✅ Extracted 156,289 texts from train-00003-of-00041.parquet
# Processing file 2/6: train-00017-of-00041.parquet
# Extracting from train-00017-of-00041.parquet: 100%|█| 156288/156288 [00:05<00:00, 28694.79texts/s, extra
# ✅ Extracted 156,288 texts from train-00017-of-00041.parquet
# Processing file 3/6: train-00024-of-00041.parquet
# Extracting from train-00024-of-00041.parquet: 100%|█| 156288/156288 [00:05<00:00, 29179.22texts/s, extra
# ✅ Extracted 156,288 texts from train-00024-of-00041.parquet
# Processing file 4/6: train-00028-of-00041.parquet
# Extracting from train-00028-of-00041.parquet: 100%|█| 156288/156288 [00:05<00:00, 30351.26texts/s, extra
# ✅ Extracted 156,288 texts from train-00028-of-00041.parquet
# Processing file 5/6: train-00035-of-00041.parquet
# Extracting from train-00035-of-00041.parquet: 100%|█| 156288/156288 [00:05<00:00, 29295.33texts/s, extra
# ✅ Extracted 156,288 texts from train-00035-of-00041.parquet
# Processing file 6/6: train-00039-of-00041.parquet
# Extracting from train-00039-of-00041.parquet: 100%|█| 156288/156288 [00:05<00:00, 27663.54texts/s, extra
# ✅ Extracted 156,288 texts from train-00039-of-00041.parquet
# ✅ Total extracted: 937,729 texts to /Users/eczech/repos/dev/wikipedia.texts (2.6 GB)
# Compressing /Users/eczech/repos/dev/wikipedia.texts to /Users/eczech/repos/dev/wikipedia.texts.bz2 using Docker
# This may take several minutes for large files...
# Running Docker container...
# debconf: delaying package configuration, since apt-utils is not installed
# Selecting previously unselected package bzip2.
# (Reading database ... 4376 files and directories currently installed.)
# Preparing to unpack .../bzip2_1.0.8-5.1build0.1_arm64.deb ...
# Unpacking bzip2 (1.0.8-5.1build0.1) ...
# Setting up bzip2 (1.0.8-5.1build0.1) ...
# Starting compression...
#   /data/wikipedia.texts:  3.372:1,  2.373 bits/byte, 70.34% saved, 2770736803 in, 821807103 out.
# Compression completed
# -rw-r--r-- 1 root root 784M Sep 19 10:55 /data/wikipedia.texts.bz2
# Compressed: /Users/eczech/repos/dev/wikipedia.texts.bz2 (783.7 MB)
# Counting text tokens in /Users/eczech/repos/dev/wikipedia.texts using 5 parallel processes with 25 chunks...
# Loading file into memory...
# Processing 34,811,401 lines in 25 chunks with 5 parallel workers...
# Processing chunks: 100%|████████████████████████| 26/26 [00:46<00:00,  1.80s/chunks, tokens=633,552,609]
# ✅ Total text tokens: 633,552,609

# ============================================================
# WIKIPEDIA ANALYSIS RESULTS:
# ============================================================
# Total texts extracted: 937,729
# Total tokens: 633,552,609
# Data file (uncompressed): 2.6 GB
# Data file (bz2 compressed): 783.7 MB

# Compression analysis:
# BZ2 vs uncompressed: 29.7%
# Space saved: 1.8 GB

# Token analysis:
# Bytes per token (uncompressed): 4.37
# Bytes per token (compressed): 1.30

# ================================================================================
# DATASET SUMMARY:
# ================================================================================
# Dataset: Wikipedia
# Token Count: 633.55M (633,552,609)
# Compressed Size: 783.7 MB (821,807,103 bytes)
# Uncompressed Size: 2.6 GB (2,770,736,803 bytes)
# Compression Ratio: 29.7%
# Compressed Bytes per Token: 1.297
# Uncompressed Bytes per Token: 4.373


# ============================================================
# PLANTCAD ANALYSIS RESULTS:
# ============================================================
# Total sequences extracted: 5,485,282
# Total tokens: 2,808,464,384
# Data file (uncompressed): 2.6 GB
# Data file (bz2 compressed): 750.8 MB

# Compression analysis:
# BZ2 vs uncompressed: 28.0%
# Space saved: 1.9 GB

# Token analysis:
# Bytes per token (uncompressed): 1.00
# Bytes per token (compressed): 0.28

# ================================================================================
# DATASET SUMMARY:
# ================================================================================
# Dataset: Plantcad
# Token Count: 2.81B (2,808,464,384)
# Compressed Size: 750.8 MB (787,297,639 bytes)
# Uncompressed Size: 2.6 GB (2,813,949,666 bytes)
# Compression Ratio: 28.0%
# Compressed Bytes per Token: 0.280
# Uncompressed Bytes per Token: 1.002
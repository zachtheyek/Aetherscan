"""
Verify train/test data exist, can be accessed, and have sensible statistics
Uses memory mapping to avoid loading entire files into RAM
"""

import gc
import os
from typing import Any

import numpy as np
import psutil


def get_memory_usage():
    """Get current memory usage in GB"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024 / 1024


def verify_data_file(filepath: str) -> dict[str, Any]:
    """
    Verify a single data file using memory mapping

    Args:
        filepath: Path to the numpy file

    Returns:
        Dictionary with file information
    """
    filename = os.path.basename(filepath)
    print(f"\nFile: {filename}")

    # Get file size
    file_size_gb = os.path.getsize(filepath) / 1e9
    print(f"File size: {file_size_gb:.2f} GB")

    # Use memory mapping to avoid loading entire file
    data = np.load(filepath, mmap_mode="r")

    info = {
        "filename": filename,
        "shape": data.shape,
        "dtype": str(data.dtype),
        "file_size_gb": file_size_gb,
    }

    print(f"Shape: {data.shape}")
    print(f"Data type: {data.dtype}")

    # Sample data statistics (don't process entire array)
    # Sample from different parts of the file
    n_samples = min(50, data.shape[0])
    sample_indices = np.linspace(0, data.shape[0] - 1, n_samples, dtype=int)

    sample_mins = []
    sample_maxs = []
    sample_means = []

    print(f"Sampling {n_samples} snippets for statistics...")

    for idx in sample_indices:
        # Load just one snippet at a time
        snippet = data[idx]
        sample_mins.append(np.min(snippet))
        sample_maxs.append(np.max(snippet))
        sample_means.append(np.mean(snippet))

        # Clear snippet from memory
        del snippet
        gc.collect()

    info["sample_min"] = np.min(sample_mins)
    info["sample_max"] = np.max(sample_maxs)
    info["sample_mean"] = np.mean(sample_means)

    print(f"Sample min value: {info['sample_min']:.2e}")
    print(f"Sample max value: {info['sample_max']:.2e}")
    print(f"Sample mean value: {info['sample_mean']:.2e}")

    # Calculate expected memory requirement if loaded fully
    expected_memory_gb = np.prod(data.shape) * np.dtype(data.dtype).itemsize / 1e9
    print(f"Memory required if fully loaded: {expected_memory_gb:.2f} GB")

    info["expected_memory_gb"] = expected_memory_gb

    # Clean up
    del data
    gc.collect()

    return info


def verify_data_format(data_path: str):
    """
    Verify data format for all files in a directory

    Args:
        data_path: Directory containing .npy files
    """
    files = sorted([f for f in os.listdir(data_path) if f.endswith(".npy")])

    print(f"Found {len(files)} .npy files")
    print(f"Initial memory usage: {get_memory_usage():.2f} GB")

    all_info = []

    for i, file in enumerate(files):
        print(f"\n--- Processing file {i + 1}/{len(files)} ---")
        filepath = os.path.join(data_path, file)

        try:
            info = verify_data_file(filepath)
            all_info.append(info)
        except Exception as e:
            print(f"Error processing {file}: {e}")
            continue

        print(f"Current memory usage: {get_memory_usage():.2f} GB")

        # Force garbage collection
        gc.collect()

    return all_info


def print_summary(info_list: list):
    """Print summary statistics"""
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    total_size = sum(info["file_size_gb"] for info in info_list)
    total_memory = sum(info["expected_memory_gb"] for info in info_list)

    print(f"Total files processed: {len(info_list)}")
    print(f"Total disk space: {total_size:.2f} GB")
    print(f"Total memory if all loaded: {total_memory:.2f} GB")

    # Check shape consistency
    shapes = [info["shape"] for info in info_list]
    unique_shapes = list({str(s[1:]) for s in shapes})  # Ignore first dimension

    if len(unique_shapes) == 1:
        print(f"Consistent data shape (excluding first dim): {unique_shapes[0]}")
    else:
        print(f"WARNING: Inconsistent shapes found: {unique_shapes}")

    # Data type consistency
    dtypes = list({info["dtype"] for info in info_list})
    if len(dtypes) == 1:
        print(f"Consistent data type: {dtypes[0]}")
    else:
        print(f"WARNING: Multiple data types: {dtypes}")


def main():
    """Main verification function"""
    print("=== Memory-Efficient Data Verification ===")

    # Check available memory
    total_memory = psutil.virtual_memory().total / 1e9
    available_memory = psutil.virtual_memory().available / 1e9

    print(f"System memory: {total_memory:.1f} GB total, {available_memory:.1f} GB available")

    # Verify training data
    print("\n=== Training Data ===")
    training_info = verify_data_format("/datax/scratch/zachy/data/aetherscan/training")

    # Verify test data
    print("\n\n=== Test Data ===")
    test_info = verify_data_format("/datax/scratch/zachy/data/aetherscan/testing")

    # Print summaries
    all_info = training_info + test_info
    print_summary(all_info)

    # Recommendations
    print("\n" + "=" * 60)
    print("RECOMMENDATIONS")
    print("=" * 60)

    if all_info:
        sample_info = all_info[0]
        # Heuristic check: if 1 file alone > 80% of available memory, warn user of dangerous potential memory pressure
        if sample_info["expected_memory_gb"] > available_memory * 0.8:
            print("WARNING: Data files are too large to load entirely into memory!")
            print("Recommendations:")
            print("1. Use memory mapping (mmap_mode='r') when loading files")
            print("2. Process data in smaller chunks")
            print("3. Consider downsampling or using a subset for initial testing")
            print("4. Upgrade to a machine with more RAM if full loading is required")
        else:
            print("None")

    print("\nVerification complete!")


if __name__ == "__main__":
    main()

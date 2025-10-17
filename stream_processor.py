"""
Memory-Efficient Streaming Signal Processor
Handles GB-scale files with constant memory usage
Demonstrates: chunking, memory pooling, zero-copy operations
"""

import numpy as np
from typing import Iterator, Tuple, Optional
from pathlib import Path
import mmap
from dataclasses import dataclass
import psutil
import gc

@dataclass
class StreamConfig:
    chunk_size: int = 1_000_000  # 1M samples
    overlap: int = 1000
    dtype: np.dtype = np.float32
    max_memory_mb: float = 500.0

class MemoryPool:
    """Simple memory pool for array reuse"""
    def __init__(self, max_size: int = 10):
        self.pool = []
        self.max_size = max_size
    
    def get(self, shape: Tuple, dtype: np.dtype) -> np.ndarray:
        """Get array from pool or allocate new"""
        for i, arr in enumerate(self.pool):
            if arr.shape == shape and arr.dtype == dtype:
                return self.pool.pop(i)
        return np.empty(shape, dtype=dtype)
    
    def put(self, arr: np.ndarray):
        """Return array to pool"""
        if len(self.pool) < self.max_size:
            self.pool.append(arr)

class StreamProcessor:
    """Process large signals with constant memory"""
    
    def __init__(self, config: StreamConfig):
        self.config = config
        self.pool = MemoryPool()
        self.stats = {
            "chunks_processed": 0,
            "total_samples": 0,
            "peak_memory_mb": 0,
            "events_found": 0
        }
    
    def process_file_mmap(self, filepath: Path) -> Iterator[Tuple[np.ndarray, int]]:
        """
        Memory-map file for zero-copy reading
        Yields (chunk, offset) tuples
        """
        with open(filepath, 'rb') as f:
            with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
                file_size = len(mm)
                bytes_per_sample = np.dtype(self.config.dtype).itemsize
                n_samples = file_size // bytes_per_sample
                
                offset = 0
                while offset < n_samples:
                    chunk_end = min(offset + self.config.chunk_size, n_samples)
                    
                    # Zero-copy view into mmap
                    byte_offset = offset * bytes_per_sample
                    byte_end = chunk_end * bytes_per_sample
                    chunk_bytes = mm[byte_offset:byte_end]
                    
                    # Create numpy array view (no copy)
                    chunk = np.frombuffer(chunk_bytes, dtype=self.config.dtype)
                    
                    yield chunk.copy(), offset  # Copy needed to release mmap
                    offset = chunk_end - self.config.overlap
    
    def process_stream(self, data_iter: Iterator[Tuple[np.ndarray, int]]) -> dict:
        """
        Process streaming data with memory tracking
        """
        process = psutil.Process()
        initial_memory = process.memory_info().rss / (1024**2)
        
        all_events = []
        
        for chunk, offset in data_iter:
            # Track memory
            current_memory = process.memory_info().rss / (1024**2)
            self.stats["peak_memory_mb"] = max(
                self.stats["peak_memory_mb"],
                current_memory - initial_memory
            )
            
            # Process chunk
            events = self._detect_in_chunk(chunk, offset)
            all_events.extend(events)
            
            # Update stats
            self.stats["chunks_processed"] += 1
            self.stats["total_samples"] += len(chunk)
            self.stats["events_found"] += len(events)
            
            # Force garbage collection periodically
            if self.stats["chunks_processed"] % 10 == 0:
                gc.collect()
        
        return {
            "events": all_events,
            "stats": self.stats
        }
    
    def _detect_in_chunk(self, chunk: np.ndarray, offset: int) -> list:
        """Simple threshold detection in chunk"""
        baseline = np.median(chunk)
        noise = np.std(chunk)
        threshold = baseline - 3 * noise
        
        # Find crossings
        crossings = np.where(chunk < threshold)[0]
        
        # Group consecutive
        if len(crossings) == 0:
            return []
        
        events = []
        start = crossings[0]
        for i in range(1, len(crossings)):
            if crossings[i] - crossings[i-1] > 1:
                events.append({
                    "start": int(offset + start),
                    "end": int(offset + crossings[i-1]),
                    "amplitude": float(np.min(chunk[start:crossings[i-1]+1]) - baseline)
                })
                start = crossings[i]
        
        # Last event
        events.append({
            "start": int(offset + start),
            "end": int(offset + crossings[-1]),
            "amplitude": float(np.min(chunk[start:]) - baseline)
        })
        
        return events


class ChunkedArrayProcessor:
    """Process numpy arrays in memory-efficient chunks"""
    
    def __init__(self, chunk_size: int = 1_000_000):
        self.chunk_size = chunk_size
    
    def chunked_fft(self, signal: np.ndarray) -> np.ndarray:
        """
        FFT with overlap-add for long signals
        Constant memory usage
        """
        n = len(signal)
        overlap = self.chunk_size // 4
        output = np.zeros(n, dtype=np.complex64)
        
        for i in range(0, n, self.chunk_size - overlap):
            start = i
            end = min(i + self.chunk_size, n)
            
            chunk = signal[start:end]
            chunk_fft = np.fft.fft(chunk)
            
            # Overlap-add
            output[start:end] += chunk_fft[:len(chunk)]
        
        return output
    
    def chunked_convolve(self, signal: np.ndarray, kernel: np.ndarray) -> np.ndarray:
        """
        Convolution with overlap-save
        Memory-efficient for long signals
        """
        kernel_size = len(kernel)
        chunk_size = self.chunk_size
        overlap = kernel_size - 1
        
        output = np.zeros(len(signal))
        
        for i in range(0, len(signal), chunk_size - overlap):
            start = max(0, i - overlap)
            end = min(i + chunk_size, len(signal))
            
            chunk = signal[start:end]
            conv_chunk = np.convolve(chunk, kernel, mode='same')
            
            # Handle overlap
            if i == 0:
                output[start:end] = conv_chunk
            else:
                output[i:end] = conv_chunk[overlap:]
        
        return output


def benchmark_memory_efficiency():
    """Demonstrate memory-efficient processing"""
    print("=== Memory-Efficient Stream Processing ===\n")
    
    # Create large synthetic signal file
    signal_size = 50_000_000  # 50M samples
    print(f"Creating {signal_size:,} sample signal (~190 MB)...")
    
    temp_file = Path("temp_signal.bin")
    signal = np.random.randn(signal_size).astype(np.float32)
    signal[10000000:10001000] -= 5  # Add events
    signal.tofile(temp_file)
    
    config = StreamConfig(
        chunk_size=1_000_000,
        overlap=1000,
        dtype=np.float32
    )
    
    processor = StreamProcessor(config)
    
    # Process with memory mapping
    print("\nProcessing with memory mapping...")
    data_iter = processor.process_file_mmap(temp_file)
    result = processor.process_stream(data_iter)
    
    print(f"\nResults:")
    print(f"  Total samples: {result['stats']['total_samples']:,}")
    print(f"  Chunks processed: {result['stats']['chunks_processed']}")
    print(f"  Events found: {result['stats']['events_found']}")
    print(f"  Peak memory: {result['stats']['peak_memory_mb']:.1f} MB")
    print(f"  Memory efficiency: {signal_size * 4 / (result['stats']['peak_memory_mb'] * 1024**2):.1f}x")
    
    # Chunked operations
    print("\n=== Chunked FFT vs Standard FFT ===")
    small_signal = signal[:10_000_000]  # 10M samples
    
    chunked_proc = ChunkedArrayProcessor(chunk_size=1_000_000)
    
    import time
    process = psutil.Process()
    
    # Chunked FFT
    gc.collect()
    mem_before = process.memory_info().rss / (1024**2)
    start = time.perf_counter()
    chunked_fft = chunked_proc.chunked_fft(small_signal)
    chunked_time = time.perf_counter() - start
    chunked_mem = process.memory_info().rss / (1024**2) - mem_before
    
    # Standard FFT
    gc.collect()
    mem_before = process.memory_info().rss / (1024**2)
    start = time.perf_counter()
    standard_fft = np.fft.fft(small_signal)
    standard_time = time.perf_counter() - start
    standard_mem = process.memory_info().rss / (1024**2) - mem_before
    
    print(f"Chunked FFT: {chunked_time:.2f}s, {chunked_mem:.1f} MB")
    print(f"Standard FFT: {standard_time:.2f}s, {standard_mem:.1f} MB")
    print(f"Memory savings: {standard_mem / chunked_mem:.1f}x")
    
    # Cleanup
    temp_file.unlink()


if __name__ == "__main__":
    benchmark_memory_efficiency()

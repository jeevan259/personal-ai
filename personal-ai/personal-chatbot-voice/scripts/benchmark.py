#!/usr/bin/env python3
"""
Benchmark performance of voice assistant components
"""

import argparse
import asyncio
import time
import statistics
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from config.settings import get_settings
from utils.logging_config import setup_logging
from speech.stt.whisper_stt import WhisperSTT
from speech.tts.pyttsx3_local import PyTTSX3TTS
from llm.providers import MockLLMProvider
from audio.processing.vad import VoiceActivityDetector
import numpy as np
import logging

logger = logging.getLogger(__name__)

class Benchmark:
    """Performance benchmark suite"""
    
    def __init__(self):
        self.settings = get_settings()
        setup_logging("WARNING", self.settings.logs_dir)  # Less verbose for benchmarks
        
    async def benchmark_stt(self, iterations: int = 10):
        """Benchmark Speech-to-Text"""
        print(f"\n{'='*50}")
        print("Benchmarking Speech-to-Text")
        print(f"{'='*50}")
        
        stt = WhisperSTT(model_size="tiny")
        
        # Create test audio
        audio_data = np.random.randn(16000).astype(np.float32)
        
        times = []
        
        for i in range(iterations):
            start = time.time()
            await stt.transcribe(audio_data)
            end = time.time()
            
            duration = end - start
            times.append(duration)
            
            print(f"  Iteration {i+1}: {duration:.3f}s")
            
        # Calculate statistics
        stats = self._calculate_stats(times)
        self._print_stats(stats, "STT")
        
        return stats
        
    async def benchmark_llm(self, iterations: int = 20):
        """Benchmark LLM response"""
        print(f"\n{'='*50}")
        print("Benchmarking LLM Response")
        print(f"{'='*50}")
        
        llm = MockLLMProvider()
        
        times = []
        
        for i in range(iterations):
            start = time.time()
            await llm.generate(f"Test query {i}")
            end = time.time()
            
            duration = end - start
            times.append(duration)
            
            if i < 5:  # Only print first few
                print(f"  Iteration {i+1}: {duration:.3f}s")
                
        # Calculate statistics
        stats = self._calculate_stats(times)
        self._print_stats(stats, "LLM")
        
        return stats
        
    async def benchmark_tts(self, iterations: int = 10):
        """Benchmark Text-to-Speech"""
        print(f"\n{'='*50}")
        print("Benchmarking Text-to-Speech")
        print(f"{'='*50}")
        
        tts = PyTTSX3TTS()
        
        test_text = "This is a test sentence for benchmarking text to speech performance."
        
        times = []
        
        for i in range(iterations):
            start = time.time()
            await tts.speak(test_text)
            end = time.time()
            
            duration = end - start
            times.append(duration)
            
            print(f"  Iteration {i+1}: {duration:.3f}s")
            
        # Calculate statistics
        stats = self._calculate_stats(times)
        self._print_stats(stats, "TTS")
        
        return stats
        
    def benchmark_vad(self, iterations: int = 1000):
        """Benchmark Voice Activity Detection"""
        print(f"\n{'='*50}")
        print("Benchmarking Voice Activity Detection")
        print(f"{'='*50}")
        
        vad = VoiceActivityDetector()
        
        # Create test audio frames
        frame_size = vad.get_frame_size()
        
        times = []
        
        for i in range(iterations):
            audio_frame = np.random.randint(-1000, 1000, frame_size, dtype=np.int16)
            
            start = time.perf_counter()
            vad.is_speech(audio_frame)
            end = time.perf_counter()
            
            duration_ms = (end - start) * 1000  # Convert to milliseconds
            times.append(duration_ms)
            
            if i < 5:  # Only print first few
                print(f"  Iteration {i+1}: {duration_ms:.3f}ms")
                
        # Calculate statistics
        stats = self._calculate_stats(times)
        self._print_stats(stats, "VAD", unit="ms")
        
        return stats
        
    def _calculate_stats(self, times):
        """Calculate statistics from timing data"""
        if not times:
            return {}
            
        return {
            "iterations": len(times),
            "min": min(times),
            "max": max(times),
            "mean": statistics.mean(times),
            "median": statistics.median(times),
            "stdev": statistics.stdev(times) if len(times) > 1 else 0,
            "total": sum(times)
        }
        
    def _print_stats(self, stats, component, unit="s"):
        """Print statistics"""
        if not stats:
            print(f"  No data for {component}")
            return
            
        print(f"\n{component} Statistics:")
        print(f"  Iterations: {stats['iterations']}")
        print(f"  Min time: {stats['min']:.3f}{unit}")
        print(f"  Max time: {stats['max']:.3f}{unit}")
        print(f"  Mean time: {stats['mean']:.3f}{unit}")
        print(f"  Median time: {stats['median']:.3f}{unit}")
        print(f"  Std Dev: {stats['stdev']:.3f}{unit}")
        print(f"  Total time: {stats['total']:.3f}{unit}")
        
    async def run_all(self, iterations: dict = None):
        """Run all benchmarks"""
        if iterations is None:
            iterations = {
                "stt": 5,
                "llm": 20,
                "tts": 5,
                "vad": 1000
            }
            
        results = {}
        
        print("="*60)
        print("Personal Voice Assistant - Performance Benchmark")
        print("="*60)
        
        # STT benchmark
        results["stt"] = await self.benchmark_stt(iterations["stt"])
        
        # LLM benchmark
        results["llm"] = await self.benchmark_llm(iterations["llm"])
        
        # TTS benchmark
        results["tts"] = await self.benchmark_tts(iterations["tts"])
        
        # VAD benchmark
        results["vad"] = self.benchmark_vad(iterations["vad"])
        
        # Summary
        print(f"\n{'='*50}")
        print("Benchmark Summary")
        print(f"{'='*50}")
        
        total_iterations = sum(r["iterations"] for r in results.values())
        total_time = sum(r["total"] for r in results.values())
        
        print(f"Total iterations: {total_iterations}")
        print(f"Total benchmark time: {total_time:.2f}s")
        
        # Check for performance issues
        thresholds = {
            "stt": 2.0,  # seconds
            "llm": 0.1,  # seconds
            "tts": 3.0,  # seconds
            "vad": 10.0  # milliseconds
        }
        
        issues = []
        for component, stats in results.items():
            if component in thresholds and stats["mean"] > thresholds[component]:
                issues.append(f"{component}: {stats['mean']:.2f} > {thresholds[component]}")
                
        if issues:
            print(f"\n⚠️  Performance issues detected:")
            for issue in issues:
                print(f"  - {issue}")
        else:
            print(f"\n✅ All benchmarks within acceptable limits")
            
        return results

async def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Benchmark voice assistant performance")
    parser.add_argument(
        "--iterations",
        type=int,
        nargs=4,
        default=[5, 20, 5, 1000],
        help="Iterations for [STT, LLM, TTS, VAD] (default: 5 20 5 1000)"
    )
    parser.add_argument(
        "--component",
        choices=["all", "stt", "llm", "tts", "vad"],
        default="all",
        help="Component to benchmark (default: all)"
    )
    
    args = parser.parse_args()
    
    benchmark = Benchmark()
    
    if args.component == "all":
        iterations_dict = {
            "stt": args.iterations[0],
            "llm": args.iterations[1],
            "tts": args.iterations[2],
            "vad": args.iterations[3]
        }
        await benchmark.run_all(iterations_dict)
    else:
        # Run single component
        if args.component == "stt":
            await benchmark.benchmark_stt(args.iterations[0])
        elif args.component == "llm":
            await benchmark.benchmark_llm(args.iterations[1])
        elif args.component == "tts":
            await benchmark.benchmark_tts(args.iterations[2])
        elif args.component == "vad":
            benchmark.benchmark_vad(args.iterations[3])
            
    return 0

if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
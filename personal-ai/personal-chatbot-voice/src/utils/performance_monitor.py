"""
Performance monitoring for timing and metrics
"""

import time
from typing import Dict, Any, Optional
import logging
from datetime import datetime
import json
from pathlib import Path

logger = logging.getLogger(__name__)

class PerformanceMonitor:
    """Monitor performance metrics and timing"""
    
    def __init__(self, log_dir: Optional[Path] = None):
        self.log_dir = log_dir
        self.metrics: Dict[str, list] = {}
        self.timers: Dict[str, float] = {}
        self.counters: Dict[str, int] = {}
        
        if log_dir:
            log_dir.mkdir(parents=True, exist_ok=True)
            
        logger.info("Performance monitor initialized")
        
    def start_timer(self, name: str):
        """Start a timer"""
        self.timers[name] = time.time()
        
    def stop_timer(self, name: str) -> Optional[float]:
        """Stop a timer and record the duration"""
        if name not in self.timers:
            logger.warning(f"Timer '{name}' not started")
            return None
            
        duration = time.time() - self.timers[name]
        
        # Record metric
        if name not in self.metrics:
            self.metrics[name] = []
        self.metrics[name].append(duration)
        
        # Increment counter
        self.counters[name] = self.counters.get(name, 0) + 1
        
        del self.timers[name]
        
        logger.debug(f"Timer '{name}': {duration:.3f}s")
        return duration
        
    def increment_counter(self, name: str, amount: int = 1):
        """Increment a counter"""
        self.counters[name] = self.counters.get(name, 0) + amount
        
    def record_metric(self, name: str, value: float):
        """Record a metric value"""
        if name not in self.metrics:
            self.metrics[name] = []
        self.metrics[name].append(value)
        
    def get_stats(self, name: str) -> Dict[str, float]:
        """Get statistics for a metric"""
        if name not in self.metrics or not self.metrics[name]:
            return {}
            
        values = self.metrics[name]
        
        return {
            "count": len(values),
            "min": min(values),
            "max": max(values),
            "mean": sum(values) / len(values),
            "latest": values[-1],
        }
        
    def get_all_stats(self) -> Dict[str, Dict[str, float]]:
        """Get statistics for all metrics"""
        return {name: self.get_stats(name) for name in self.metrics}
        
    def save_report(self, filename: Optional[str] = None):
        """Save performance report to file"""
        if not self.log_dir:
            return
            
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"performance_{timestamp}.json"
            
        filepath = self.log_dir / filename
        
        report = {
            "timestamp": datetime.now().isoformat(),
            "metrics": self.get_all_stats(),
            "counters": self.counters,
            "summary": self.get_summary(),
        }
        
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
                
            logger.info(f"Performance report saved to: {filepath}")
            
        except Exception as e:
            logger.error(f"Failed to save performance report: {e}")
            
    def get_summary(self) -> Dict[str, Any]:
        """Get summary of performance"""
        summary = {}
        
        # Calculate overall statistics
        total_calls = sum(len(values) for values in self.metrics.values())
        total_time = sum(sum(values) for values in self.metrics.values())
        
        summary["total_metric_calls"] = total_calls
        summary["total_monitored_time"] = total_time
        
        # Find slowest operation
        if self.metrics:
            avg_times = {
                name: sum(values) / len(values)
                for name, values in self.metrics.items()
            }
            if avg_times:
                slowest = max(avg_times.items(), key=lambda x: x[1])
                summary["slowest_operation"] = {
                    "name": slowest[0],
                    "avg_time": slowest[1]
                }
                
        return summary
        
    def reset(self):
        """Reset all metrics"""
        self.metrics.clear()
        self.timers.clear()
        self.counters.clear()
        logger.info("Performance monitor reset")
        
    def __del__(self):
        """Destructor - save report on exit"""
        if self.metrics:
            self.save_report(f"performance_exit_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
"""
Baseline Comparison for Impact Analysis
"""

import random
import numpy as np

class BaselineComparison:
    def __init__(self):
        self.smart_times = []
        self.random_times = []
        self.worst_times = []
        
    def add_scenario(self, lane_waiting_times):
        """Compare smart vs random vs worst selection"""
        if not lane_waiting_times:
            return
        
        times = list(lane_waiting_times.values())
        if not times:
            return
        
        # Smart: Best lane
        self.smart_times.append(min(times))
        
        # Random: Any lane
        self.random_times.append(random.choice(times))
        
        # Worst: Worst lane
        self.worst_times.append(max(times))
    
    def get_comparison(self):
        """Get comparison statistics"""
        if not self.smart_times:
            return {}
        
        avg_smart = np.mean(self.smart_times)
        avg_random = np.mean(self.random_times)
        avg_worst = np.mean(self.worst_times)
        
        improvement_vs_random = ((avg_random - avg_smart) / avg_random * 100) if avg_random > 0 else 0
        improvement_vs_worst = ((avg_worst - avg_smart) / avg_worst * 100) if avg_worst > 0 else 0
        
        return {
            'avg_smart_wait': round(avg_smart, 2),
            'avg_random_wait': round(avg_random, 2),
            'avg_worst_wait': round(avg_worst, 2),
            'improvement_vs_random_percent': round(improvement_vs_random, 2),
            'improvement_vs_worst_percent': round(improvement_vs_worst, 2),
            'time_saved_vs_random': round(avg_random - avg_smart, 2),
            'time_saved_vs_worst': round(avg_worst - avg_smart, 2)
        }
    
    def generate_comparison_report(self):
        """Generate comparison report"""
        comp = self.get_comparison()
        
        return f"""
╔══════════════════════════════════════════════════════╗
║          BASELINE COMPARISON REPORT                  ║
╠══════════════════════════════════════════════════════╣
║ Smart System: {comp.get('avg_smart_wait', 0):>38}s ║
║ Random Selection: {comp.get('avg_random_wait', 0):>34}s ║
║ Worst Case: {comp.get('avg_worst_wait', 0):>40}s ║
╠══════════════════════════════════════════════════════╣
║ Improvement vs Random: {comp.get('improvement_vs_random_percent', 0):>27}% ║
║ Improvement vs Worst: {comp.get('improvement_vs_worst_percent', 0):>28}% ║
╚══════════════════════════════════════════════════════╝
"""
"""
Performance Metrics Calculator
"""

import numpy as np
from collections import defaultdict

class MetricsCalculator:
    def __init__(self):
        self.detections_history = []
        self.processing_times = []
        
    def add_detection_result(self, frame_num, detections, processing_time):
        """Record detection results"""
        self.detections_history.append({
            'frame': frame_num,
            'count': len(detections),
            'classes': [d.get('class', 'unknown') for d in detections],
            'confidences': [d.get('confidence', 0) for d in detections]
        })
        self.processing_times.append(processing_time)
    
    def calculate_metrics(self):
        """Calculate comprehensive metrics"""
        if not self.detections_history:
            return {}
        
        total_detections = sum(d['count'] for d in self.detections_history)
        avg_detections_per_frame = total_detections / len(self.detections_history)
        
        all_confidences = []
        for d in self.detections_history:
            all_confidences.extend(d['confidences'])
        
        avg_confidence = np.mean(all_confidences) if all_confidences else 0
        
        avg_processing_time = np.mean(self.processing_times) if self.processing_times else 0
        fps = 1.0 / avg_processing_time if avg_processing_time > 0 else 0
        
        vehicle_counts = defaultdict(int)
        for d in self.detections_history:
            for v_class in d['classes']:
                vehicle_counts[v_class] += 1
        
        return {
            'total_frames_processed': len(self.detections_history),
            'total_vehicles_detected': total_detections,
            'avg_vehicles_per_frame': round(avg_detections_per_frame, 2),
            'avg_detection_confidence': round(avg_confidence * 100, 2),
            'avg_processing_time_ms': round(avg_processing_time * 1000, 2),
            'processing_fps': round(fps, 2),
            'vehicle_type_distribution': dict(vehicle_counts)
        }
    
    def generate_report(self):
        """Generate formatted report"""
        metrics = self.calculate_metrics()
        
        report = f"""
╔══════════════════════════════════════════════╗
║       PERFORMANCE METRICS REPORT            ║
╠══════════════════════════════════════════════╣
║ Total Frames Processed: {metrics.get('total_frames_processed', 0):>17} ║
║ Total Vehicles Detected: {metrics.get('total_vehicles_detected', 0):>16} ║
║ Avg Vehicles/Frame: {metrics.get('avg_vehicles_per_frame', 0):>21} ║
║ Detection Confidence: {metrics.get('avg_detection_confidence', 0):>18}% ║
║ Processing Speed: {metrics.get('processing_fps', 0):>24} FPS ║
╚══════════════════════════════════════════════╝
"""
        return report
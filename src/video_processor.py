"""
Video Processing with Vehicle Detection and Lane Analysis
"""

import cv2
import numpy as np
from src.vehicle_detector import VehicleDetector
from src.lane_analyzer import LaneAnalyzer
from src.config import LANE_COLORS, VEHICLE_COLORS, FRAME_SKIP

class VideoProcessor:
    def __init__(self, lane_polygons):
        """Initialize processor with lane configuration"""
        self.detector = VehicleDetector()
        self.analyzer = LaneAnalyzer(lane_polygons)
        self.lane_polygons = self._format_polygons(lane_polygons)
    
    def _format_polygons(self, polygons):
        """Ensure all polygons are properly formatted numpy arrays"""
        formatted = []
        for poly in polygons:
            if isinstance(poly, list):
                poly = np.array(poly, dtype=np.int32)
            else:
                poly = np.array(poly, dtype=np.int32)
            
            if len(poly.shape) == 2 and poly.shape[1] == 2:
                formatted.append(poly)
            else:
                print(f"⚠️ Warning: Skipping invalid polygon shape {poly.shape}")
        
        return formatted
        
    def draw_lanes(self, frame):
        """Draw lane polygons on frame"""
        overlay = frame.copy()
        
        for i, polygon in enumerate(self.lane_polygons):
            color = LANE_COLORS[i % len(LANE_COLORS)]
            polygon = polygon.astype(np.int32)
            
            # Draw outline
            cv2.polylines(overlay, [polygon], True, color, 3)
            
            # Fill with transparency
            cv2.fillPoly(overlay, [polygon], color)
            
            # Add lane number at center
            M = cv2.moments(polygon)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                cv2.putText(overlay, f'Lane {i+1}', (cx-30, cy),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Blend overlay with original
        frame = cv2.addWeighted(overlay, 0.3, frame, 0.7, 0)
        return frame
    
    def draw_detections(self, frame, detections):
        """Draw bounding boxes around detected vehicles"""
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            color = VEHICLE_COLORS.get(det['class'], (255, 255, 255))
            
            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Draw label
            label = f"{det['class']} {det['confidence']:.2f}"
            cv2.putText(frame, label, (x1, y1-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        return frame
    
    def draw_statistics(self, frame, lane_data, best_lane):
        """Draw statistics panel at bottom of frame"""
        panel_height = 120
        panel = np.zeros((panel_height, frame.shape[1], 3), dtype=np.uint8)
        
        for lane_id in range(len(self.lane_polygons)):
            data = lane_data.get(lane_id, {
                'count': 0,
                'total_weight': 0,
                'waiting_time': 0
            })
            
            x = 20 + lane_id * 250
            color = (0, 255, 0) if lane_id == best_lane else (255, 255, 255)
            
            # Lane title
            cv2.putText(panel, f"LANE {lane_id+1}", (x, 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            # Vehicle count
            cv2.putText(panel, f"Vehicles: {data['count']}", (x, 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            
            # Load weight
            cv2.putText(panel, f"Load: {data['total_weight']:.1f}", (x, 75),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            
            # Waiting time
            cv2.putText(panel, f"Wait: {data['waiting_time']:.0f}s", (x, 100),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Best lane indicator
            if lane_id == best_lane:
                cv2.putText(panel, "✓ BEST", (x+150, 25),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        return np.vstack([frame, panel])
    
    def process_video(self, video_path, output_path, progress_callback=None):
        """
        Process entire video
        
        Args:
            video_path: Input video file path
            output_path: Output video file path
            progress_callback: Optional function(percent) to report progress
        """
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"❌ Could not open video: {video_path}")
        
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"📹 Video info: {width}x{height} @ {fps}fps, {total_frames} frames")
        
        # Try H264 codec first, fallback to mp4v
        fourcc = cv2.VideoWriter_fourcc(*'avc1')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height + 120))
        
        if not out.isOpened():
            print("⚠️ H264 codec failed, trying mp4v...")
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height + 120))
        
        frame_num = 0
        processed_count = 0
        total_detections = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_num += 1
            
            # Process every Nth frame
            if frame_num % FRAME_SKIP == 0:
                try:
                    # Detect vehicles
                    detections = self.detector.detect(frame)
                    total_detections += len(detections)
                    
                    # Analyze lanes
                    lane_data = self.analyzer.analyze_frame(detections)
                    best_lane = self.analyzer.get_best_lane(lane_data)
                    
                    # Draw visualizations
                    frame = self.draw_lanes(frame)
                    frame = self.draw_detections(frame, detections)
                    frame = self.draw_statistics(frame, lane_data, best_lane)
                    
                    processed_count += 1
                    
                except Exception as e:
                    print(f"⚠️ Error processing frame {frame_num}: {str(e)}")
            
            out.write(frame)
            
            # Update progress
            if progress_callback and frame_num % 10 == 0:
                progress = int((frame_num / total_frames) * 100)
                progress_callback(progress)
        
        cap.release()
        out.release()
        
        print(f"\n✅ Processing complete!")
        print(f"📊 Statistics:")
        print(f"   - Total frames: {frame_num}")
        print(f"   - Processed frames: {processed_count}")
        print(f"   - Total detections: {total_detections}")
        if processed_count > 0:
            print(f"   - Avg detections/frame: {total_detections/processed_count:.1f}")
        
        return True
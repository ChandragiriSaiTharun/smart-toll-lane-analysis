"""
Test lane detection logic
"""

import cv2
import numpy as np
from src.lane_analyzer import LaneAnalyzer
from src.vehicle_detector import VehicleDetector

def test_lane_detection(video_path, lanes_json='outputs/lanes.json'):
    import json
    
    # Load lanes
    with open(lanes_json, 'r') as f:
        lanes = json.load(f)
    
    print(f"✅ Loaded {len(lanes)} lanes")
    
    # Initialize
    analyzer = LaneAnalyzer(lanes)
    detector = VehicleDetector()
    
    # Load first frame
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        print("❌ Could not read video")
        return
    
    # Detect
    detections = detector.detect(frame)
    print(f"\n📍 Detected {len(detections)} vehicles:")
    
    for i, det in enumerate(detections):
        center = det['center']
        lane_id = analyzer.assign_lane(center)
        
        print(f"   {i+1}. {det['class']} at {center} → Lane {lane_id + 1 if lane_id is not None else 'NONE'}")
    
    # Analyze
    lane_data = analyzer.analyze_frame(detections)
    
    print(f"\n📊 Lane Statistics:")
    for lane_id, data in lane_data.items():
        print(f"   Lane {lane_id + 1}: {data['count']} vehicles, weight={data['total_weight']:.1f}, wait={data['waiting_time']:.1f}s")
    
    # Draw and save
    for i, poly in enumerate(analyzer.lane_polygons):
        cv2.polylines(frame, [poly], True, (0, 255, 0), 2)
    
    for det in detections:
        x1, y1, x2, y2 = det['bbox']
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
        
        center = det['center']
        cv2.circle(frame, (int(center[0]), int(center[1])), 5, (0, 0, 255), -1)
    
    cv2.imwrite('outputs/lane_test.jpg', frame)
    print(f"\n✅ Saved test image to: outputs/lane_test.jpg")

if __name__ == "__main__":
    import sys
    video_path = sys.argv[1] if len(sys.argv) > 1 else 'data/test_toll.mp4'
    test_lane_detection(video_path)
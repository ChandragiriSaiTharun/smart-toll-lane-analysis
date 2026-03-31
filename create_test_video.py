"""
Create synthetic test video for demonstration
Run: python create_test_video.py
"""

import cv2
import numpy as np
import os

def create_test_video(output_path='data/test_toll.mp4', duration=30):
    """
    Create a simple test video with moving rectangles (simulating vehicles)
    """
    # Video settings
    width, height = 768, 432
    fps = 30
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    print(f"🎬 Creating test video: {output_path}")
    print(f"   Resolution: {width}x{height}")
    print(f"   Duration: {duration}s")
    print(f"   FPS: {fps}")
    
    # Define 3 lanes
    lane_width = width // 3
    
    # Create vehicles (rectangles moving upward)
    vehicles = [
        # Lane 1 vehicles
        {'x': 50, 'y': height, 'lane': 0, 'speed': 3, 'color': (0, 255, 0), 'width': 60, 'height': 80, 'type': 'car'},
        {'x': 80, 'y': height + 150, 'lane': 0, 'speed': 2.5, 'color': (0, 255, 0), 'width': 60, 'height': 80, 'type': 'car'},
        
        # Lane 2 vehicles
        {'x': lane_width + 60, 'y': height, 'lane': 1, 'speed': 1.5, 'color': (0, 0, 255), 'width': 80, 'height': 120, 'type': 'truck'},
        {'x': lane_width + 50, 'y': height + 200, 'lane': 1, 'speed': 2, 'color': (0, 255, 0), 'width': 60, 'height': 80, 'type': 'car'},
        
        # Lane 3 vehicles
        {'x': 2*lane_width + 70, 'y': height, 'lane': 2, 'speed': 2.8, 'color': (0, 255, 0), 'width': 60, 'height': 80, 'type': 'car'},
        {'x': 2*lane_width + 50, 'y': height + 100, 'lane': 2, 'speed': 3, 'color': (0, 255, 0), 'width': 60, 'height': 80, 'type': 'car'},
        {'x': 2*lane_width + 60, 'y': height + 250, 'lane': 2, 'speed': 2, 'color': (255, 165, 0), 'width': 40, 'height': 60, 'type': 'motorcycle'},
    ]
    
    total_frames = duration * fps
    
    for frame_num in range(total_frames):
        # Create blank frame (gray background)
        frame = np.ones((height, width, 3), dtype=np.uint8) * 50
        
        # Draw lane dividers
        cv2.line(frame, (lane_width, 0), (lane_width, height), (150, 150, 150), 3)
        cv2.line(frame, (2*lane_width, 0), (2*lane_width, height), (150, 150, 150), 3)
        
        # Draw lane numbers at top
        for i in range(3):
            lane_center = i * lane_width + lane_width // 2
            cv2.putText(frame, f'LANE {i+1}', (lane_center - 50, 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Draw and move vehicles
        for vehicle in vehicles:
            vehicle['y'] -= vehicle['speed']
            
            # Reset if vehicle exits top
            if vehicle['y'] < -vehicle['height']:
                vehicle['y'] = height + 100
            
            # Draw rectangle (vehicle)
            x1 = int(vehicle['x'])
            y1 = int(vehicle['y'])
            x2 = x1 + vehicle['width']
            y2 = y1 + vehicle['height']
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), vehicle['color'], -1)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 2)
            
            # Add vehicle type label
            cv2.putText(frame, vehicle['type'], (x1, y1-5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, vehicle['color'], 1)
        
        # Add frame number and timestamp
        cv2.putText(frame, f"Frame: {frame_num} | Time: {frame_num/fps:.1f}s", (10, height-20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        out.write(frame)
        
        # Progress indicator
        if frame_num % (fps * 5) == 0:
            print(f"   Progress: {frame_num}/{total_frames} frames ({100*frame_num/total_frames:.0f}%)")
    
    out.release()
    print(f"\n✅ Test video created successfully!")
    print(f"   Saved to: {output_path}")
    print(f"   Size: {os.path.getsize(output_path) / 1024 / 1024:.2f} MB")
    print(f"\n💡 You can now test with: python test_detection.py {output_path}")

if __name__ == "__main__":
    create_test_video()
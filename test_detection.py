"""
Test YOLO detection on first frame
Run: python test_detection.py data/your_video.mp4
"""

import cv2
import sys
import os
from src.vehicle_detector import VehicleDetector
from src.config import VEHICLE_COLORS

def test_detection(video_path):
    """Test detection on first frame"""
    
    if not os.path.exists(video_path):
        print(f"❌ Video not found: {video_path}")
        return
    
    print(f"📹 Testing detection on: {video_path}")
    
    # Read first frame
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    
    if not ret:
        print("❌ Could not read video")
        cap.release()
        return
    
    height, width = frame.shape[:2]
    print(f"✅ Frame size: {width}x{height}")
    cap.release()
    
    # Detect
    print("\n🔍 Running YOLO detection...")
    detector = VehicleDetector()
    detections = detector.detect(frame)
    
    print(f"\n📊 Detection Results:")
    print(f"{'='*60}")
    print(f"   Total detections: {len(detections)}")
    
    if detections:
        print(f"\n   Detected vehicles:")
        for i, det in enumerate(detections, 1):
            print(f"   {i}. {det['class'].upper()}")
            print(f"      - Confidence: {det['confidence']:.2%}")
            print(f"      - Position: {det['bbox']}")
        
        # Draw on frame
        result_frame = frame.copy()
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            color = VEHICLE_COLORS.get(det['class'], (255, 255, 255))
            
            cv2.rectangle(result_frame, (x1, y1), (x2, y2), color, 3)
            
            label = f"{det['class']} {det['confidence']:.2f}"
            (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(result_frame, (x1, y1-label_h-10), (x1+label_w, y1), color, -1)
            cv2.putText(result_frame, label, (x1, y1-5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Save result
        os.makedirs('outputs', exist_ok=True)
        output_path = 'outputs/test_detection.jpg'
        cv2.imwrite(output_path, result_frame)
        
        print(f"\n✅ SUCCESS! Result saved to: {output_path}")
        print(f"\n💡 Open the image to see detected vehicles:")
        print(f"   xdg-open {output_path}")
        
    else:
        print("\n❌ No vehicles detected!")
        print("\n🔍 Troubleshooting:")
        print("   1. Check if video has visible vehicles")
        print("   2. Try lowering confidence threshold in src/config.py")
        print("   3. Video might be too blurry or dark")
        
        os.makedirs('outputs', exist_ok=True)
        cv2.imwrite('outputs/original_frame.jpg', frame)
        print(f"\n💾 Saved original frame to: outputs/original_frame.jpg")
    
    print(f"\n{'='*60}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("❌ Usage: python test_detection.py <video_path>")
        print("📝 Example: python test_detection.py data/toll_video.mp4")
        sys.exit(1)
    
    video_path = sys.argv[1]
    test_detection(video_path)
"""
Vehicle Detection using YOLOv8
"""

from ultralytics import YOLO
import cv2
from src.config import MODEL_NAME, CONFIDENCE_THRESHOLD

class VehicleDetector:
    def __init__(self):
        """Initialize YOLO model"""
        print("📦 Loading YOLO model...")
        self.model = YOLO(MODEL_NAME)
        
        # Vehicle classes from COCO dataset
        self.vehicle_classes = ['car', 'motorcycle', 'bus', 'truck', 'bicycle']
        print("✅ Model loaded successfully!")
        
    def detect(self, frame):
        """
        Detect vehicles in a frame
        
        Args:
            frame: Input image/frame
            
        Returns:
            List of detections: [{'class': str, 'confidence': float, 'bbox': tuple, 'center': tuple}]
        """
        # Run YOLO detection with lower confidence for better detection
        results = self.model(frame, conf=CONFIDENCE_THRESHOLD, verbose=False)[0]
        detections = []
        
        for box in results.boxes:
            class_id = int(box.cls[0])
            class_name = results.names[class_id]
            confidence = float(box.conf[0])
            
            # Filter only vehicle classes
            if class_name in self.vehicle_classes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                bbox = (int(x1), int(y1), int(x2), int(y2))
                center = (int((x1 + x2) / 2), int((y1 + y2) / 2))
                
                detections.append({
                    'class': class_name,
                    'confidence': confidence,
                    'bbox': bbox,
                    'center': center
                })
        
        return detections
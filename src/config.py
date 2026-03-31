"""
Configuration settings for Smart Toll Lane Analysis System
"""

# Vehicle processing time weights (multiplier for base toll time)
VEHICLE_WEIGHTS = {
    'car': 1.0,         # Standard car: 8 seconds
    'motorcycle': 0.5,  # Faster: 4 seconds
    'bus': 2.5,         # Slower: 20 seconds
    'truck': 3.0,       # Slowest: 24 seconds
    'bicycle': 0.5,
}

# YOLO Model Settings
MODEL_NAME = 'yolov8s.pt'  # Small model (better accuracy, still fast)
CONFIDENCE_THRESHOLD = 0.20  # Lower = more detections

# Processing Settings
FRAME_SKIP = 3  # Process every 3rd frame
AVERAGE_TOLL_TIME = 8  # Base processing time in seconds per weight unit

# Display Colors
LANE_COLORS = [
    (255, 0, 0),      # Red
    (0, 255, 0),      # Green
    (0, 0, 255),      # Blue
    (255, 255, 0),    # Yellow (BGR)
    (255, 0, 255),    # Magenta
    (0, 255, 255),    # Cyan (BGR)
]

VEHICLE_COLORS = {
    'car': (0, 255, 0),
    'motorcycle': (255, 165, 0),
    'bus': (0, 165, 255),
    'truck': (0, 0, 255),
    'bicycle': (255, 255, 0),
}
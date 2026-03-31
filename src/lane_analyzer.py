"""
Lane Analysis with Correct Assignment Logic
"""

import cv2
import numpy as np
from collections import defaultdict
from src.config import VEHICLE_WEIGHTS, AVERAGE_TOLL_TIME


class LaneAnalyzer:
    def __init__(self, lane_polygons):
        """Initialize with lane polygons"""
        self.lane_polygons = []
        for poly in lane_polygons:
            p = np.array(poly, dtype=np.int32)
            # Ensure polygon is in correct shape
            if len(p.shape) == 2 and p.shape[1] == 2:
                self.lane_polygons.append(p)
            else:
                print(f"⚠️ Invalid polygon shape: {p.shape}")
        
        self.num_lanes = len(self.lane_polygons)
        print(f"✅ LaneAnalyzer initialized with {self.num_lanes} lanes")

    def assign_lane(self, point):
        """
        Determine which lane a point belongs to
        
        Args:
            point: (x, y) tuple - center of vehicle
            
        Returns:
            lane_id (int) or None
        """
        if point is None:
            return None
        
        # Ensure point is tuple of integers
        try:
            x, y = int(point[0]), int(point[1])
            point = (x, y)
        except (TypeError, IndexError):
            return None
        
        for lane_id, polygon in enumerate(self.lane_polygons):
            # Check if point is inside polygon
            result = cv2.pointPolygonTest(polygon, point, False)
            if result >= 0:  # Inside or on boundary
                return lane_id
        
        return None

    def analyze_frame(self, detections):
        """
        Analyze detections in current frame
        
        Args:
            detections: List of detection dicts
            
        Returns:
            dict with lane statistics
        """
        lane_data = {}
        
        # Initialize all lanes
        for lane_id in range(self.num_lanes):
            lane_data[lane_id] = {
                'vehicles': [],
                'count': 0,
                'total_weight': 0.0,
                'waiting_time': 0.0,
                'vehicle_types': {}
            }
        
        # Assign each detection to a lane
        for det in detections:
            center = det.get('center')
            if center is None:
                continue
            
            lane_id = self.assign_lane(center)
            
            if lane_id is not None and lane_id in lane_data:
                vehicle_class = det.get('class', 'car')
                weight = VEHICLE_WEIGHTS.get(vehicle_class, 1.0)
                
                lane_data[lane_id]['vehicles'].append(det)
                lane_data[lane_id]['count'] += 1
                lane_data[lane_id]['total_weight'] += weight
                
                # Count vehicle types
                if vehicle_class not in lane_data[lane_id]['vehicle_types']:
                    lane_data[lane_id]['vehicle_types'][vehicle_class] = 0
                lane_data[lane_id]['vehicle_types'][vehicle_class] += 1
        
        # Calculate waiting time for each lane
        for lane_id in range(self.num_lanes):
            total_weight = lane_data[lane_id]['total_weight']
            lane_data[lane_id]['waiting_time'] = round(total_weight * AVERAGE_TOLL_TIME, 1)
        
        return lane_data

    def get_best_lane(self, lane_data):
        """Find lane with minimum waiting time"""
        if not lane_data:
            return 0
        
        best_lane = 0
        min_time = float('inf')
        
        for lane_id in range(self.num_lanes):
            wait_time = lane_data.get(lane_id, {}).get('waiting_time', float('inf'))
            if wait_time < min_time:
                min_time = wait_time
                best_lane = lane_id
        
        return best_lane
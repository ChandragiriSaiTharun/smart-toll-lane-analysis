"""
Simple and Accurate Vehicle Tracker
"""

import numpy as np
from collections import OrderedDict
from scipy.spatial import distance as dist
from scipy.optimize import linear_sum_assignment


class VehicleTracker:
    def __init__(self, max_disappeared=30, max_distance=80):
        """
        Initialize tracker
        
        Args:
            max_disappeared: Frames before removing vehicle
            max_distance: Max pixel distance for matching
        """
        self.next_id = 0
        self.objects = OrderedDict()      # {id: centroid}
        self.disappeared = OrderedDict()  # {id: frames_missing}
        self.metadata = OrderedDict()     # {id: {class, lane, etc.}}
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance
        
        # Statistics
        self.total_tracked = 0
        self.lane_entry_count = {}
        self.lane_exit_count = {}
        
        print(f"✅ VehicleTracker initialized (max_dist={max_distance}, max_disappear={max_disappeared})")

    def register(self, centroid, vehicle_class, lane_id, frame_num):
        """Register new vehicle"""
        self.objects[self.next_id] = centroid
        self.disappeared[self.next_id] = 0
        self.metadata[self.next_id] = {
            'class': vehicle_class,
            'lane': lane_id,
            'entry_frame': frame_num,
            'tracked': True
        }
        
        # Update entry count
        if lane_id is not None:
            if lane_id not in self.lane_entry_count:
                self.lane_entry_count[lane_id] = 0
            self.lane_entry_count[lane_id] += 1
        
        self.total_tracked += 1
        self.next_id += 1

    def deregister(self, object_id):
        """Remove vehicle from tracking"""
        if object_id in self.metadata:
            lane_id = self.metadata[object_id].get('lane')
            
            if lane_id is not None:
                if lane_id not in self.lane_exit_count:
                    self.lane_exit_count[lane_id] = 0
                self.lane_exit_count[lane_id] += 1
            
            self.metadata[object_id]['tracked'] = False
        
        if object_id in self.objects:
            del self.objects[object_id]
        if object_id in self.disappeared:
            del self.disappeared[object_id]

    def update(self, detections, lane_assignments, frame_num):
        """
        Update tracker with new detections
        
        Args:
            detections: List of detection dicts
            lane_assignments: dict {detection_idx: lane_id}
            frame_num: Current frame number
            
        Returns:
            dict: Currently tracked objects metadata
        """
        # No detections - increment disappeared counter
        if len(detections) == 0:
            for obj_id in list(self.disappeared.keys()):
                self.disappeared[obj_id] += 1
                
                if self.disappeared[obj_id] > self.max_disappeared:
                    self.deregister(obj_id)
            
            return self._get_active_objects()
        
        # Extract centroids from detections
        input_centroids = []
        for det in detections:
            center = det.get('center')
            if center is not None:
                input_centroids.append(center)
        
        if len(input_centroids) == 0:
            return self._get_active_objects()
        
        input_centroids = np.array(input_centroids)
        
        # If no existing objects, register all
        if len(self.objects) == 0:
            for i, centroid in enumerate(input_centroids):
                lane_id = lane_assignments.get(i)
                vehicle_class = detections[i].get('class', 'car')
                self.register(centroid, vehicle_class, lane_id, frame_num)
        else:
            # Match existing objects to new detections
            object_ids = list(self.objects.keys())
            object_centroids = list(self.objects.values())
            
            # Calculate distance matrix
            D = dist.cdist(np.array(object_centroids), input_centroids)
            
            # Optimal assignment (Hungarian algorithm) — conflict-free
            row_indices, col_indices = linear_sum_assignment(D)
            
            used_rows = set()
            used_cols = set()
            
            for row, col in zip(row_indices, col_indices):
                # Check if distance is within threshold
                if D[row, col] > self.max_distance:
                    continue
                
                # Update existing object
                object_id = object_ids[row]
                self.objects[object_id] = input_centroids[col]
                self.disappeared[object_id] = 0
                
                # Update lane if changed
                new_lane = lane_assignments.get(col)
                if new_lane is not None:
                    self.metadata[object_id]['lane'] = new_lane
                
                used_rows.add(row)
                used_cols.add(col)
            
            # Handle unmatched existing objects (disappeared)
            for row in range(len(object_ids)):
                if row not in used_rows:
                    object_id = object_ids[row]
                    self.disappeared[object_id] += 1
                    
                    if self.disappeared[object_id] > self.max_disappeared:
                        self.deregister(object_id)
            
            # Handle unmatched new detections (new vehicles)
            for col in range(len(input_centroids)):
                if col not in used_cols:
                    lane_id = lane_assignments.get(col)
                    vehicle_class = detections[col].get('class', 'car')
                    self.register(input_centroids[col], vehicle_class, lane_id, frame_num)
        
        return self._get_active_objects()

    def _get_active_objects(self):
        """Get currently active tracked objects"""
        active = {}
        for obj_id, centroid in self.objects.items():
            if obj_id in self.metadata:
                active[obj_id] = {
                    **self.metadata[obj_id],
                    'centroid': centroid,
                    'tracked': True
                }
        return active

    def get_vehicle_type_distribution(self):
        """Count each tracked ID once → true unique vehicle type breakdown"""
        counts = {}
        for obj_id, meta in self.metadata.items():
            v_class = meta.get('class', 'unknown')
            counts[v_class] = counts.get(v_class, 0) + 1
        return counts

    def get_statistics(self):
        """Get tracking statistics"""
        return {
            'total_tracked': self.total_tracked,
            'currently_active': len(self.objects),
            'lane_entries': self.lane_entry_count.copy(),
            'lane_exits': self.lane_exit_count.copy()
        }
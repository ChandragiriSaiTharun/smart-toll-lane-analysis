"""
Real-time Processor — clean version
No dimension-changing overlays. Stats shown in Streamlit only.
"""

import cv2
import numpy as np
import time
from src.vehicle_detector import VehicleDetector
from src.lane_analyzer import LaneAnalyzer
from src.vehicle_tracker import VehicleTracker
from src.metrics_calculator import MetricsCalculator
from src.baseline_comparison import BaselineComparison
from src.config import LANE_COLORS, VEHICLE_COLORS, FRAME_SKIP, VEHICLE_WEIGHTS, AVERAGE_TOLL_TIME


class RealtimeProcessor:
    def __init__(self, lane_polygons, frame_skip=None):
        print("🚀 Initializing Real-time Processor...")

        # Core components
        self.detector = VehicleDetector()
        self.analyzer = LaneAnalyzer(lane_polygons)

        # Metrics
        self.metrics_calc = MetricsCalculator()
        self.baseline_comp = BaselineComparison()

        # Lane polygons
        self.lane_polygons = [np.array(poly, dtype=np.int32) for poly in lane_polygons]
        self.num_lanes = len(self.lane_polygons)

        # Frame skip
        self.frame_skip = frame_skip if frame_skip else FRAME_SKIP

        # Tracker created lazily on first frame
        self.tracker = None

        # Statistics
        self.all_seen_ids = set()
        self.frame_count = 0
        self.session_history = []

        # Cache for skipped frames
        self.last_annotated = None
        self.last_stats = None

        print(f"✅ Processor Ready with {self.num_lanes} lanes, skip={self.frame_skip}")
        for i, poly in enumerate(self.lane_polygons):
            area = cv2.contourArea(poly)
            x, y, w, h = cv2.boundingRect(poly)
            print(f"   Lane {i+1}: {len(poly)} pts, bbox=({x},{y},{w},{h}), area={area:.0f}px²")

    # ===========================================================
    # MAIN PROCESSING
    # ===========================================================
    def process_frame(self, frame, frame_num):
        """Process single frame"""
        start_time = time.time()
        self.frame_count = frame_num

        # Skip frames — return cached result
        if frame_num % self.frame_skip != 0:
            if self.last_annotated is not None:
                return self.last_annotated, self.last_stats
            # Before first real processing, just draw lanes
            return self.draw_lanes(frame.copy()), None

        # --- STEP 1: DETECT ---
        detections = self.detector.detect(frame)

        # Create tracker on first frame, scaled to resolution
        if self.tracker is None:
            h, w = frame.shape[:2]
            diag = (h**2 + w**2) ** 0.5
            max_dist = max(int(diag * 0.03 * self.frame_skip), 60)
            self.tracker = VehicleTracker(max_disappeared=3, max_distance=max_dist)
            print(f"📏 Tracker: max_dist={max_dist}px, max_disappeared=3")

        # --- STEP 2: FILTER TO IN-LANE ONLY ---
        lane_assignments = {}
        valid_detections = []

        for det in detections:
            center = det.get('center')
            if center is None:
                continue
            lane_id = self.analyzer.assign_lane(center)
            if lane_id is not None:
                lane_assignments[len(valid_detections)] = lane_id
                det['lane_id'] = lane_id
                valid_detections.append(det)

        # --- STEP 3: UPDATE TRACKER ---
        tracked_objects = self.tracker.update(valid_detections, lane_assignments, frame_num)
        self.all_seen_ids.update(tracked_objects.keys())

        # --- STEP 4: LANE STATS (re-verify centroid is still inside lane) ---
        lane_data = {}
        for lane_id in range(self.num_lanes):
            lane_data[lane_id] = {
                'count': 0, 'total_weight': 0.0, 'waiting_time': 0.0,
                'vehicle_types': {}, 'vehicle_ids': []
            }

        verified = {}
        for obj_id, obj in tracked_objects.items():
            centroid = obj.get('centroid')
            if centroid is None:
                continue
            # Re-verify centroid is CURRENTLY inside a lane
            lane = self.analyzer.assign_lane((int(centroid[0]), int(centroid[1])))
            if lane is None:
                continue  # centroid left the lane → don't count

            obj['lane'] = lane
            verified[obj_id] = obj

            v_class = obj.get('class', 'car')
            weight = VEHICLE_WEIGHTS.get(v_class, 1.0)
            lane_data[lane]['count'] += 1
            lane_data[lane]['total_weight'] += weight
            lane_data[lane]['vehicle_ids'].append(obj_id)
            lane_data[lane]['vehicle_types'][v_class] = lane_data[lane]['vehicle_types'].get(v_class, 0) + 1

        for lid in range(self.num_lanes):
            lane_data[lid]['waiting_time'] = round(lane_data[lid]['total_weight'] * AVERAGE_TOLL_TIME, 1)

        # --- STEP 5: BEST LANE ---
        best_lane = min(lane_data, key=lambda lid: lane_data[lid]['waiting_time'])

        # --- STEP 6: METRICS ---
        processing_time = time.time() - start_time
        self.metrics_calc.add_detection_result(frame_num, valid_detections, processing_time)
        self.baseline_comp.add_scenario({lid: d['waiting_time'] for lid, d in lane_data.items()})

        # --- STEP 7: DRAW (lanes + verified vehicles only, NO dimension changes) ---
        annotated = self.draw_lanes(frame.copy())
        annotated = self.draw_vehicles(annotated, verified)

        # --- STEP 8: STATS ---
        stats = {
            'frame_num': frame_num,
            'detections_this_frame': len(valid_detections),
            'total_unique_vehicles': len(self.all_seen_ids),
            'lane_data': lane_data,
            'best_lane': best_lane,
            'detections': valid_detections,
            'tracker_stats': self.tracker.get_statistics(),
            'metrics': self.metrics_calc.calculate_metrics(),
            'comparison': self.baseline_comp.get_comparison(),
            'tracked_objects': tracked_objects,
            'processing_time_ms': round(processing_time * 1000, 2)
        }

        row = {
            'frame': frame_num, 'best_lane': best_lane + 1,
            'total_unique_vehicles': len(self.all_seen_ids),
            'processing_ms': round(processing_time * 1000, 2),
        }
        for lid, ld in lane_data.items():
            row[f'lane_{lid+1}_count'] = ld['count']
            row[f'lane_{lid+1}_weight'] = round(ld['total_weight'], 2)
            row[f'lane_{lid+1}_wait_s'] = ld['waiting_time']
        self.session_history.append(row)

        self.last_annotated = annotated
        self.last_stats = stats
        return annotated, stats

    # ===========================================================
    # DRAWING — NO DIMENSION CHANGES
    # ===========================================================
    def draw_lanes(self, frame):
        """Draw lane polygons: semi-transparent fill + solid outlines"""
        overlay = frame.copy()
        for i, poly in enumerate(self.lane_polygons):
            color = LANE_COLORS[i % len(LANE_COLORS)]
            cv2.fillPoly(overlay, [poly], color)

        frame = cv2.addWeighted(overlay, 0.25, frame, 0.75, 0)

        for i, poly in enumerate(self.lane_polygons):
            color = LANE_COLORS[i % len(LANE_COLORS)]
            cv2.polylines(frame, [poly], True, color, 3)
            M = cv2.moments(poly)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                cv2.putText(frame, f'L{i+1}', (cx - 20, cy),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 4)
                cv2.putText(frame, f'L{i+1}', (cx - 20, cy),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 2)
        return frame

    def draw_vehicles(self, frame, verified_objects):
        """Draw markers only for verified in-lane vehicles"""
        for obj_id, obj in verified_objects.items():
            centroid = obj.get('centroid')
            lane_id = obj.get('lane')
            if centroid is None or lane_id is None:
                continue

            v_class = obj.get('class', 'car')
            color = VEHICLE_COLORS.get(v_class, (255, 255, 255))
            cx, cy = int(centroid[0]), int(centroid[1])

            # Filled circle + white border
            cv2.circle(frame, (cx, cy), 8, color, -1)
            cv2.circle(frame, (cx, cy), 10, (255, 255, 255), 2)

            # Label
            label = f"ID:{obj_id} {v_class} [L{lane_id+1}]"
            (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            cv2.rectangle(frame, (cx - lw//2, cy - lh - 14),
                          (cx + lw//2 + 4, cy - 6), color, -1)
            cv2.putText(frame, label, (cx - lw//2 + 2, cy - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

        return frame

    # ===========================================================
    # VIDEO STREAM
    # ===========================================================
    def process_video_stream(self, video_path):
        """Generator for video processing"""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        print(f"📹 Processing: {total_frames} frames @ {fps} FPS")

        frame_num = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_num += 1
            annotated, stats = self.process_frame(frame, frame_num)
            progress = int((frame_num / total_frames) * 100) if total_frames > 0 else 0
            yield {
                'frame': annotated,
                'frame_num': frame_num,
                'total_frames': total_frames,
                'progress': progress,
                'stats': stats,
                'fps': fps
            }

        cap.release()
        print(f"✅ Processing complete: {frame_num} frames")

    def get_final_metrics(self):
        """Get final metrics"""
        return {
            'metrics': self.metrics_calc.calculate_metrics(),
            'comparison': self.baseline_comp.get_comparison(),
            'tracker_stats': self.tracker.get_statistics() if self.tracker else {},
            'total_unique_vehicles': len(self.all_seen_ids),
            'unique_vehicle_types': self.tracker.get_vehicle_type_distribution() if self.tracker else {},
            'session_history': self.session_history,
        }
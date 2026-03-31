"""
Interactive Lane Drawing Module
"""

import cv2
import numpy as np
import json
import os


class LaneDrawer:
    def __init__(self):
        """Initialize lane drawer"""
        self.lanes = []
        self.current_polygon = []
        self.temp_frame = None
        self.original_frame = None
        self.orig_width = 0
        self.orig_height = 0
        self.display_width = 0
        self.display_height = 0
        self.scale = 1.0

    def mouse_callback(self, event, x, y, flags, param):
        """
        Mouse callback.
        We show a RESIZED image via imshow, so (x, y) are in
        display-pixel space. Map back to original frame coords.
        """
        orig_x = int(x / self.scale)
        orig_y = int(y / self.scale)

        # Clamp to frame bounds
        orig_x = max(0, min(orig_x, self.orig_width - 1))
        orig_y = max(0, min(orig_y, self.orig_height - 1))

        if event == cv2.EVENT_LBUTTONDOWN:
            self.current_polygon.append((orig_x, orig_y))
            print(f"✓ Point added: ({orig_x}, {orig_y})")

            cv2.circle(self.temp_frame, (orig_x, orig_y), 5, (0, 255, 0), -1)

            if len(self.current_polygon) > 1:
                cv2.line(self.temp_frame,
                         self.current_polygon[-2],
                         self.current_polygon[-1],
                         (0, 255, 0), 2)

        elif event == cv2.EVENT_RBUTTONDOWN:
            if len(self.current_polygon) >= 3:
                cv2.line(self.temp_frame,
                         self.current_polygon[-1],
                         self.current_polygon[0],
                         (0, 255, 0), 2)
                self.lanes.append(self.current_polygon.copy())
                print(f"✅ Lane {len(self.lanes)} completed with {len(self.current_polygon)} points")
                self.current_polygon = []
            else:
                print("❌ Need at least 3 points for a polygon")

    def draw_interactive(self, video_path):
        """
        Interactive polygon drawing.

        We manually cv2.resize() the frame to a small display size,
        then show it with WINDOW_AUTOSIZE so the window is exactly
        the display size. Mouse coords come back in display-pixel space
        and we divide by `scale` to get original-frame coords.

        Controls:
          Left click  → add point
          Right click → finish polygon (need ≥ 3 points)
          s → save & exit
          c → clear current polygon
          r → reset all
          q → quit without saving
        """
        cap = cv2.VideoCapture(video_path)
        ret, frame = cap.read()
        cap.release()

        if not ret:
            print("❌ Error: Could not read video")
            return None

        self.original_frame = frame.copy()
        self.orig_height, self.orig_width = frame.shape[:2]

        # Calculate display scale to fit on screen (max 1200x800)
        max_w, max_h = 1200, 800
        self.scale = min(max_w / self.orig_width, max_h / self.orig_height, 1.0)
        self.display_width = int(self.orig_width * self.scale)
        self.display_height = int(self.orig_height * self.scale)

        window_name = 'Draw Lane Polygons'
        cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(window_name, self.mouse_callback)

        self.temp_frame = frame.copy()

        print("\n" + "=" * 60)
        print("🎨 INTERACTIVE LANE DRAWING MODE")
        print("=" * 60)
        print(f"Frame size: {self.orig_width}x{self.orig_height}")
        print(f"Display size: {self.display_width}x{self.display_height}")
        print(f"Scale factor: {self.scale:.4f}")
        print("\nControls:")
        print("  LEFT CLICK  → add point")
        print("  RIGHT CLICK → finish polygon")
        print("  s → save & exit | c → clear | r → reset | q → quit")
        print("=" * 60 + "\n")

        while True:
            # Build display on original-res canvas
            display = self.temp_frame.copy()

            # Draw completed lanes
            for i, lane in enumerate(self.lanes):
                pts = np.array(lane, dtype=np.int32)
                color = self.get_lane_color(i)
                cv2.polylines(display, [pts], True, color, 3)

                M = cv2.moments(pts)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    cv2.putText(display, f'Lane {i+1}', (cx - 40, cy),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 3)

            # HUD
            info = f"Lanes: {len(self.lanes)} | Points: {len(self.current_polygon)}"
            cv2.rectangle(display, (5, 5), (600, 80), (0, 0, 0), -1)
            cv2.putText(display, info, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(display, "s=Save | c=Clear | r=Reset | q=Quit", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

            # RESIZE to display size, then show
            display_resized = cv2.resize(display,
                                         (self.display_width, self.display_height))
            cv2.imshow(window_name, display_resized)

            key = cv2.waitKey(1) & 0xFF

            if key == ord('s'):
                if self.lanes:
                    cv2.destroyAllWindows()
                    print(f"\n✅ Saved {len(self.lanes)} lanes!")
                    return self.lanes
                else:
                    print("❌ No lanes to save!")

            elif key == ord('c'):
                self.current_polygon = []
                self.temp_frame = self.original_frame.copy()
                print("🔄 Current polygon cleared")

            elif key == ord('r'):
                self.lanes = []
                self.current_polygon = []
                self.temp_frame = self.original_frame.copy()
                print("🔄 All lanes reset")

            elif key == ord('q'):
                cv2.destroyAllWindows()
                print("❌ Quit without saving")
                return None

        cv2.destroyAllWindows()
        return self.lanes

    def get_lane_color(self, lane_id):
        """Get color for lane"""
        colors = [
            (255, 0, 0),    # Red
            (0, 255, 0),    # Green
            (0, 0, 255),    # Blue
            (255, 255, 0),  # Yellow (BGR)
            (255, 0, 255),  # Magenta
            (0, 255, 255),  # Cyan (BGR)
        ]
        return colors[lane_id % len(colors)]

    def save_lanes(self, filepath):
        """Save lanes to JSON file"""
        lanes_list = []
        for lane in self.lanes:
            if isinstance(lane, np.ndarray):
                lanes_list.append(lane.tolist())
            else:
                lanes_list.append(lane)
        with open(filepath, 'w') as f:
            json.dump(lanes_list, f, indent=2)
        print(f"💾 Lanes saved to {filepath}")

    def load_lanes(self, filepath):
        """Load lanes from JSON file"""
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                self.lanes = json.load(f)
            print(f"📂 Loaded {len(self.lanes)} lanes from {filepath}")
            return self.lanes
        else:
            print(f"❌ File not found: {filepath}")
            return None


def draw_lanes_cli(video_path, output_json='lanes.json'):
    """Command-line interface for drawing lanes"""
    drawer = LaneDrawer()
    lanes = drawer.draw_interactive(video_path)
    if lanes:
        drawer.save_lanes(output_json)
        return lanes
    return None


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python -m src.lane_drawer <video_path> [output_json]")
    else:
        video_path = sys.argv[1]
        output_json = sys.argv[2] if len(sys.argv) > 2 else 'outputs/lanes.json'
        draw_lanes_cli(video_path, output_json)
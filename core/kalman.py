import numpy as np
from filterpy.kalman import KalmanFilter
from scipy.stats import chi2

class KalmanBallTracker:
    def __init__(self):
        self.miss_count = 0
        self.MAX_MISS = 6

        self.kf = KalmanFilter(dim_x=4, dim_z=2)
        dt = 1.0

        self.kf.F = np.array([
            [1,0,dt,0],
            [0,1,0,dt],
            [0,0,1,0 ],
            [0,0,0,1 ]
        ])
        self.kf.H = np.array([
            [1,0,0,0],
            [0,1,0,0]
        ])

        self.kf.P *= 500.
        self.kf.R = np.eye(2) * 10
        self.kf.Q = np.eye(4) * 0.1

        self.ball_xy = None
        self.miss = 0

        self.ball_state = "VISIBLE"


    def is_valid_measurement(self, kf, z, thresh=40):  # 95%, 2D
        y = z - kf.H @ kf.x
        S = kf.H @ kf.P @ kf.H.T + kf.R
        d = y.T @ np.linalg.inv(S) @ y
        return d < thresh

    def select_ball(self, dets, prev_xy, max_dist):
        if len(dets) == 0:
            return None

        if prev_xy is None:
            # 第一帧：选置信度最高
            return max(dets, key=lambda d: d["conf"])

        px, py = prev_xy
        best = None

        best_dist = max_dist
        for d in dets:
            cx, cy = d["cx"], d["cy"]
            dist = np.hypot(cx - px, cy - py)
            if dist < best_dist:
                best_dist = dist
                best = d
        return best
    
    def select_ball(self, dets, prev_xy, max_dist, is_initialized):
        if len(dets) == 0:
            return None

        # Fix: If not initialized, don't check distance, just take the best detection
        if not is_initialized:
            return max(dets, key=lambda d: d["conf"])

        px, py = prev_xy
        best = None
        best_dist = max_dist
        
        for d in dets:
            cx, cy = d["cx"], d["cy"]
            dist = np.hypot(cx - px, cy - py)
            if dist < best_dist:
                best_dist = dist
                best = d
        return best
    
    def ball_detection(self, detections):
        # 1. Properly format YOLO detections from the Boxes object
        formatted_dets = []
        for box in detections:
            # Extract center x, center y, and confidence
            xywh = box.xywh.cpu().numpy()[0] 
            conf = box.conf.cpu().numpy()[0]
            formatted_dets.append({"cx": xywh[0], "cy": xywh[1], "conf": conf})

        self.kf.predict()
        
        # Check if we have a valid previous position
        is_initialized = self.ball_xy is not None
        pred_xy = (self.kf.x[0,0], self.kf.x[1,0]) if is_initialized else None

        # 2. Select the ball (Ignoring distance if it's the first time we see it)
        sel = self.select_ball(
            formatted_dets,
            prev_xy=pred_xy,
            max_dist=80 if self.ball_state == "VISIBLE" else 200,
            is_initialized=is_initialized
        )

        if sel is not None:
            z = np.array([[sel["cx"]], [sel["cy"]]])
            
            # 3. If first detection, jump the Kalman Filter to the ball's location
            if not is_initialized:
                self.kf.x[0,0] = sel["cx"]
                self.kf.x[1,0] = sel["cy"]
                self.ball_xy = (int(sel["cx"]), int(sel["cy"]))
                self.ball_state = "VISIBLE"
                self.miss = 0
                return # Skip update for the first frame to stabilize

            # Validation gate (only if already tracking)
            if self.is_valid_measurement(self.kf, z):
                self.kf.update(z)
                self.ball_state = "VISIBLE"
                self.miss = 0
            else:
                self.miss += 1
        else:
            self.miss += 1

        if self.miss >= 3:
            self.ball_state = "OCCLUDED"
            print("Ball occluded.")

        # Update the visual coordinates
        self.ball_xy = (int(self.kf.x[0,0]), int(self.kf.x[1,0]))
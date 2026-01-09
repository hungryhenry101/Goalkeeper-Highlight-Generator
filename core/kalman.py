import numpy as np
from filterpy.kalman import KalmanFilter

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
        self.kf.Q = np.eye(4) * 0.01

        self.ball_xy = None
        self.miss = 0

    def select_ball(self, dets, prev_xy, max_dist=80):
        if len(dets) == 0:
            return None

        if prev_xy is None:
            # 第一帧：选置信度最高
            return max(dets, key=lambda d: d["conf"])

        px, py = prev_xy
        best = None
        best_dist = 1e9
        for d in dets:
            cx, cy = d["cx"], d["cy"]
            dist = np.hypot(cx - px, cy - py)
            if dist < best_dist and dist < max_dist:
                best_dist = dist
                best = d
        return best
    
    def ball_detection(self, detections):
        ball_dets = []
        #for box in ball_detect[0].boxes:
        for box in detections:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            ball_dets.append({"cx": cx, "cy": cy, "conf": float(box.conf[0])})
        sel = self.select_ball(ball_dets, self.ball_xy)
        if sel is not None:
            z = np.array([sel["cx"], sel["cy"]])
            if self.ball_xy is None:
                self.kf.x[:2] = z.reshape(2,1)
            self.kf.update(z)
            self.miss = 0
        else:
            self.kf.predict()
            self.miss += 1

        self.ball_xy = (int(self.kf.x[0]), int(self.kf.x[1]))
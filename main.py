import cv2
from ultralytics import YOLO
from tqdm import tqdm
import csv
import random
import numpy as np
from collections import defaultdict, deque
import core.ball_tracker as ball_tracker
import core.cmc as cmc

MODEL_PATH = "./models/best.pt"  # YOUR MODEL PATH HERE
VIDEO_PATH = "./input_vids/test1.mp4" # YOUR VIDEO PATH HERE
OUTPUT_VIDEO = "./output/out.mp4"
CONF_THRES = 0.15  # 降低阈值以提高检测率

player_model = YOLO(MODEL_PATH)
ball_model = YOLO(MODEL_PATH)
names = player_model.names    
name2id = {v: k for k, v in player_model.names.items()}
ball_cls = name2id["ball"]

csv_file = open("./output/track_log.csv", "w", newline="")
csv_writer = csv.writer(csv_file)
csv_writer.writerow(["frame","id","raw_x","raw_y","comp_x","comp_y","dx","dy","dist"])

# VIDEO PROCESSING
cap = cv2.VideoCapture(VIDEO_PATH)
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
writer = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, fps, (width, height))

# Ball Tracker
football_tracker = ball_tracker.BallTracker()

# CMC
compensator = cmc.CMC(width, height)

track_colors = {}

# raw and compensated trajectories (players)
raw_traj = defaultdict(lambda: deque(maxlen=50))
comp_traj = defaultdict(lambda: deque(maxlen=50))

# 累积变换矩阵：从参考帧（第一帧）到当前帧的变换

def get_color(track_id):
    if track_id not in track_colors:
        track_colors[track_id] = (
            random.randint(50, 255),
            random.randint(50, 255),
            random.randint(50, 255)
        )
    return track_colors[track_id]

def draw_traj(frame, traj, color):
    pts = list(traj)
    if len(pts) < 2:
        return
    for i in range(1, len(pts)):
        x1, y1 = pts[i-1]
        x2, y2 = pts[i]
        cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)

cv2.namedWindow("preview", cv2.WINDOW_NORMAL)
for frame_idx in tqdm(range(total_frames)):
    ret, frame = cap.read()
    if not ret:
        break
    
    raw_frame = frame.copy() # For CMC

    ball_detect = ball_model(
        frame,
        conf=0.05,
        classes=[ball_cls],
        verbose=False
    )

    # Ball tracking
    football_tracker.ball_detection(ball_detect[0].boxes)
    cv2.circle(frame, football_tracker.ball_xy, 6, (0,255,255), -1)
    cv2.putText(
        frame,
        f"Ball: {football_tracker.ball_state}",
        (10,30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0,255,255),
        2
    )

    compensator.gray = cv2.cvtColor(raw_frame, cv2.COLOR_BGR2GRAY)
    compensator.grays.append(compensator.gray.copy())

    # YOLO tracker
    results = player_model.track(
        frame,
        persist=True,
        conf=CONF_THRES,
        tracker="bytetrack.yaml",
        verbose=False
    )

    if not results or results[0].boxes is None or results[0].boxes.id is None:
        print("no detections")
        writer.write(frame)
        if compensator.ref_gray is None:
            print("first frame set as reference")
            compensator.ref_gray = compensator.grays[0].copy()
        else:
            compensator.ref_gray = compensator.grays[frame_idx - 9].copy()
        compensator.prev_gray = compensator.gray.copy()
        continue

    boxes = results[0].boxes
    ids = boxes.id.cpu().numpy()
    xys = boxes.xyxy.cpu().numpy()

    # Camera Motion Compensation
    compensator.compensating(football_tracker.ball_xy, xys,frame)

    # Update trajectories and draw per-track
    for tid, (x1, y1, x2, y2) in zip(ids, xys):
        tid = int(tid)
        x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
        color = get_color(tid)

        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)

        raw_traj[tid].append((cx, cy))

        if compensator.M_to_ref is not None:
            # M_to_ref是从当前帧到参考帧的2x3仿射变换矩阵
            # 变换公式: x' = a*x + b*y + tx
            #           y' = c*x + d*y + ty
            pt = np.array([cx, cy], dtype=np.float32)
            # M_to_ref[:, :2] 是旋转/缩放部分，M_to_ref[:, 2] 是平移部分
            pt_c = compensator.M_to_ref[:, :2] @ pt + compensator.M_to_ref[:, 2]
            cx_c, cy_c = int(pt_c[0]), int(pt_c[1])
            comp_traj[tid].append((cx_c, cy_c))
            dx = cx - cx_c; dy = cy - cy_c; dist = np.hypot(dx, dy)
            csv_writer.writerow([frame_idx, tid, cx, cy, cx_c, cy_c, dx, dy, f"{dist:.2f}"])
        else:
            comp_traj[tid].append((cx, cy))

        # Draw per-track trajectories and label
        draw_traj(frame, raw_traj[tid], (0, 0, 255))   # raw - red
        draw_traj(frame, comp_traj[tid], (0, 255, 0))  # compensated - green
        cv2.putText(
            frame,
            f"ID {tid}",
            (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2
        )

    compensator.prev_gray = compensator.gray.copy()
    delay = max(1, int(1000 / fps))
    cv2.imshow("preview", frame)
    if cv2.waitKey(delay) & 0xFF == ord('q'):
        break
    writer.write(frame)
cv2.destroyAllWindows()
cap.release()
writer.release()

print(f"Tracking video saved to {OUTPUT_VIDEO}")
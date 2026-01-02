import cv2
from ultralytics import YOLO
from tqdm import tqdm
import csv
import random
import numpy as np
from collections import defaultdict, deque

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

# CMC
track_colors = {}
prev_gray = None  # previous grayscale frame
ref_gray = None   # reference frame
grays = deque(maxlen=10)

# raw and compensated trajectories (players)
raw_traj = defaultdict(lambda: deque(maxlen=50))
comp_traj = defaultdict(lambda: deque(maxlen=50))

# 累积变换矩阵：从参考帧（第一帧）到当前帧的变换
# 使用3x3齐次矩阵便于累积
cumulative_M = np.eye(3, dtype=np.float32)  # 从参考帧到当前帧的累积变换

transform_history = deque(maxlen=5)

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
cv2.namedWindow("CMC Mask", cv2.WINDOW_NORMAL)
cv2.namedWindow("CMC Optical Flow", cv2.WINDOW_NORMAL)
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
    for box in ball_detect[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        cv2.circle(frame, (cx, cy), 6, (0, 255, 255), -1)

    gray = cv2.cvtColor(raw_frame, cv2.COLOR_BGR2GRAY)
    grays.append(gray.copy())

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
        if ref_gray is None:
            print("first frame set as reference")
            ref_gray = gray.copy()
        else:
            ref_gray = grays[frame_idx - 9].copy()
        prev_gray = gray.copy()
        continue

    boxes = results[0].boxes
    ids = boxes.id.cpu().numpy()
    xys = boxes.xyxy.cpu().numpy()

    # CMC BELOW HERE
    if ref_gray is None:
        print("first frame set as reference")
        ref_gray = gray.copy()
        cumulative_M = np.eye(3, dtype=np.float32)
    else:
        ref_gray = grays[0].copy()

    M_to_ref = None  # 从当前帧到参考帧的变换（用于补偿）
    
    if prev_gray is not None:
        mask = np.ones_like(gray, dtype=np.uint8) * 255
        for x1, y1, x2, y2 in xys.astype(int):
            margin = 10
            mask[max(0, y1-margin):min(height, y2+margin), 
                 max(0, x1-margin):min(width, x2+margin)] = 0
        for box in ball_detect[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            mask[y1-15:y2+15, x1-15:x2+15] = 0
        
        mask_vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        mask_vis[mask == 0] = (0, 0, 255)
        cv2.imshow("CMC Mask", mask_vis)

        pts_prev = cv2.goodFeaturesToTrack(
            prev_gray,
            maxCorners=1000,  # 特征点数量
            qualityLevel=0.02,  # 质量阈值
            minDistance=10,  # 最小距离
            mask=mask,
            blockSize=7,  # 块大小
            useHarrisDetector=False,
            k=0.04
        )

        if pts_prev is not None and len(pts_prev) >= 6:  # 更多点以提高精度
            pts_curr, status, err = cv2.calcOpticalFlowPyrLK(
                prev_gray,
                gray,
                pts_prev,
                None,
                winSize=(31, 31), # 搜索窗口
                maxLevel=4, #金字塔层数
                criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
            )
            good_mask = (status.flatten() == 1) & (err.flatten() < 10.0)
            good_prev = pts_prev[good_mask]
            good_curr = pts_curr[good_mask]
            flow_vis = frame.copy()

            if len(good_prev) >= 6:  # 至少需要6个点来估计仿射变换
                M, inliers = cv2.estimateAffine2D(
                    good_prev,
                    good_curr,
                    method=cv2.RANSAC,
                    ransacReprojThreshold=2.0,  # 重投影误差阈值
                    maxIters=3000,  # 迭代次数
                    confidence=0.995,
                    refineIters=10  # 细化迭代
                )

                # 可视化光流
                for i, (p_prev, p_curr) in enumerate(zip(good_prev, good_curr)):
                    x0, y0 = p_prev.ravel().astype(int)
                    x1, y1 = p_curr.ravel().astype(int)

                    if inliers[i]:
                        color = (0, 255, 0)   # 内点：绿
                    else:
                        color = (0, 0, 255)   # 外点：红

                    cv2.circle(flow_vis, (x1, y1), 2, color, -1)
                    cv2.line(flow_vis, (x0, y0), (x1, y1), color, 1)
                cv2.imshow("CMC Optical Flow", flow_vis)

                if M is not None and inliers is not None:
                    # 内点比例
                    inlier_ratio = np.sum(inliers) / len(inliers) if len(inliers) > 0 else 0
                    if inlier_ratio > 0.5:  # 至少一半是内点
                        # 计算变换的尺度因子，如果变化太大可能是错误的估计
                        scale_x = np.sqrt(M[0,0]**2 + M[0,1]**2)
                        scale_y = np.sqrt(M[1,0]**2 + M[1,1]**2)
                        if 0.5 < scale_x < 2.0 and 0.5 < scale_y < 2.0:
                            # 检查旋转角度是否合理（避免极端旋转）
                            det = M[0,0] * M[1,1] - M[0,1] * M[1,0]
                            if abs(det) > 0.1:  # 确保矩阵可逆
                                # M是从prev到curr的2x3仿射变换矩阵
                                # 转换为3x3齐次矩阵以便累积
                                M_3x3 = np.vstack([M, [0, 0, 1]])
                                # 对变换进行平滑处理（减少抖动）
                                if len(transform_history) > 0:
                                    # 计算与上一帧变换的差异
                                    last_M = transform_history[-1]
                                    diff = np.linalg.norm(M_3x3 - last_M)
                                    # 如果变化太大，可能是异常值，使用加权平均
                                    if diff < 0.5:  # 变化在合理范围内
                                        # 加权平均：当前帧权重0.7，历史平均权重0.3
                                        alpha = 0.7
                                        M_3x3 = alpha * M_3x3 + (1 - alpha) * last_M
                                
                                transform_history.append(M_3x3.copy())
                                # 累积变换：从参考帧到当前帧 = 从参考帧到上一帧 * 从上一帧到当前帧
                                cumulative_M = cumulative_M @ M_3x3
                                
                                # 计算从当前帧到参考帧的逆变换（用于补偿）
                                # cumulative_M是3x3矩阵，先求逆，然后提取2x3部分
                                cumulative_M_inv = np.linalg.inv(cumulative_M)
                                M_to_ref = cumulative_M_inv[:2, :]  # 提取前两行得到2x3矩阵
                            else:
                                print("det too small")
                                M_to_ref = None
                        else:
                            print("scale out of bounds")
                            M_to_ref = None
                    else:
                        print("low inlier ratio")
                        M_to_ref = None
                else:
                    print("M is None or inliers is None")
                    M_to_ref = None
            else:
                print("good prev < 6")
                M_to_ref = None
        else:
            print("Not enough good points")
            M_to_ref = None
    else:
        print("prev_gray is None")
        M_to_ref = None

    # Update trajectories and draw per-track
    for tid, (x1, y1, x2, y2) in zip(ids, xys):
        tid = int(tid)
        x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
        color = get_color(tid)

        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)

        raw_traj[tid].append((cx, cy))

        if M_to_ref is not None:
            # M_to_ref是从当前帧到参考帧的2x3仿射变换矩阵
            # 变换公式: x' = a*x + b*y + tx
            #           y' = c*x + d*y + ty
            pt = np.array([cx, cy], dtype=np.float32)
            # M_to_ref[:, :2] 是旋转/缩放部分，M_to_ref[:, 2] 是平移部分
            pt_c = M_to_ref[:, :2] @ pt + M_to_ref[:, 2]
            cx_c, cy_c = int(pt_c[0]), int(pt_c[1])
            comp_traj[tid].append((cx_c, cy_c))
            dx = cx - cx_c; dy = cy - cy_c; dist = np.hypot(dx, dy)
            csv_writer.writerow([frame_idx, tid, cx, cy, cx_c, cy_c, dx, dy, f"{dist:.2f}"])
        else:
            comp_traj[tid].append((cx, cy))

        # Draw per-track trajectories and label
        draw_traj(frame, raw_traj[tid], (0, 0, 255))   # raw
        draw_traj(frame, comp_traj[tid], (0, 255, 0))  # compensated
        cv2.putText(
            frame,
            f"ID {tid}",
            (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2
        )

    prev_gray = gray.copy()
    delay = max(1, int(1000 / fps))
    cv2.imshow("preview", frame)
    if cv2.waitKey(delay) & 0xFF == ord('q'):
        break
    writer.write(frame)
cv2.destroyAllWindows()
cap.release()
writer.release()

print(f"Tracking video saved to {OUTPUT_VIDEO}")
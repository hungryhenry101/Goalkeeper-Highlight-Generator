import cv2
from ultralytics import YOLO
from tqdm import tqdm
import csv
import random
import numpy as np
from collections import defaultdict, deque

model = YOLO("./models/yolo11m.pt") # YOUR MODEL PATH HERE

VIDEO_PATH = "./input_vids/test1.mp4" # YOUR VIDEO PATH HERE
OUTPUT_VIDEO = "./output/out.mp4"
CONF_THRES = 0.15  # 降低阈值以提高检测率，特别是对守门员和边裁

csv_file = open("./output/track_log.csv", "w", newline="")
csv_writer = csv.writer(csv_file)
csv_writer.writerow(["frame","id","raw_x","raw_y","comp_x","comp_y","dx","dy","dist"])

cap = cv2.VideoCapture(VIDEO_PATH)
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
writer = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, fps, (width, height))

track_colors = {}
prev_gray = None  # previous grayscale frame for motion compensation
ref_gray = None   # reference frame (first frame) for motion compensation

# store per-track raw and compensated trajectories
# raw_traj and comp_traj are dicts: track_id -> deque of (x,y)
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
for frame_idx in tqdm(range(total_frames)):
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Run YOLO tracker to obtain persistent track IDs (use built-in tracker)
    results = model.track(
        frame,
        persist=True,
        conf=CONF_THRES,
        classes=[0],  # only persons
        tracker="bytetrack.yaml",
        verbose=False
    )

    if not results or results[0].boxes is None or results[0].boxes.id is None:
        writer.write(frame)
        if ref_gray is None:
            ref_gray = gray.copy()
        prev_gray = gray
        continue

    boxes = results[0].boxes
    ids = boxes.id.cpu().numpy()
    xys = boxes.xyxy.cpu().numpy()

    # 初始化参考帧
    if ref_gray is None:
        ref_gray = gray.copy()
        cumulative_M = np.eye(3, dtype=np.float32)

    # Motion compensation: 计算从上一帧到当前帧的变换，并累积到参考帧
    M_to_ref = None  # 从当前帧到参考帧的变换（用于补偿）
    
    if prev_gray is not None:
        mask = np.ones_like(gray, dtype=np.uint8) * 255
        for x1, y1, x2, y2 in xys.astype(int):
            margin = 10
            mask[max(0, y1-margin):min(height, y2+margin), 
                 max(0, x1-margin):min(width, x2+margin)] = 0

        pts_prev = cv2.goodFeaturesToTrack(
            prev_gray,
            maxCorners=1000,  # 增加特征点数量
            qualityLevel=0.02,  # 提高质量阈值，选择更好的特征点
            minDistance=10,  # 稍微增加最小距离
            mask=mask,
            blockSize=7,  # 使用更大的块大小
            useHarrisDetector=False,
            k=0.04
        )

        if pts_prev is not None and len(pts_prev) >= 6:  # 需要更多点以提高精度
            # 使用更大的搜索窗口和更多金字塔层数以提高光流精度
            pts_curr, status, err = cv2.calcOpticalFlowPyrLK(
                prev_gray,
                gray,
                pts_prev,
                None,
                winSize=(31, 31),
                maxLevel=4,
                criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
            )
            good_mask = (status.flatten() == 1) & (err.flatten() < 10.0)
            good_prev = pts_prev[good_mask]
            good_curr = pts_curr[good_mask]

            if len(good_prev) >= 6:  # 至少需要6个点来估计仿射变换
                # 对于旋转镜头，使用完整的仿射变换（estimateAffine2D）可能更准确
                # 因为它可以处理任意方向的缩放，而不仅仅是相似变换
                M, inliers = cv2.estimateAffine2D(
                    good_prev,
                    good_curr,
                    method=cv2.RANSAC,
                    ransacReprojThreshold=2.0,  # 降低重投影误差阈值，更严格
                    maxIters=3000,  # 增加迭代次数
                    confidence=0.995,  # 提高置信度
                    refineIters=10  # 添加细化迭代
                )
                if M is not None and inliers is not None:
                    # 检查内点比例，确保估计可靠
                    inlier_ratio = np.sum(inliers) / len(inliers) if len(inliers) > 0 else 0
                    if inlier_ratio > 0.5:  # 至少50%的点是内点
                        # 检查变换矩阵是否有效（避免退化情况）
                        # 计算变换的尺度因子，如果变化太大可能是错误的估计
                        scale_x = np.sqrt(M[0,0]**2 + M[0,1]**2)
                        scale_y = np.sqrt(M[1,0]**2 + M[1,1]**2)
                        # 允许一定的尺度变化范围（0.5到2.0），更严格的范围
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
                                M_to_ref = None
                        else:
                            M_to_ref = None
                    else:
                        M_to_ref = None
                else:
                    M_to_ref = None
            else:
                M_to_ref = None
        else:
            M_to_ref = None
    else:
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

    prev_gray = gray
    delay = max(1, int(1000 / fps))
    cv2.imshow("preview", frame)
    if cv2.waitKey(delay) & 0xFF == ord('q'):
        break
    writer.write(frame)
cv2.destroyAllWindows()
cap.release()
writer.release()

print(f"Tracking video saved to {OUTPUT_VIDEO}")
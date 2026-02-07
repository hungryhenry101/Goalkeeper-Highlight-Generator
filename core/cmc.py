from collections import deque
import numpy as np
import cv2

class CMC:
    def __init__(self, width, height):
        self.optical_flow_visible = False
        self.mask_visible = True

        self.prev_gray = None  # previous grayscale frame
        self.ref_gray = None   # reference frame
        self.gray = None        # current grayscale frame
        self.grays = deque(maxlen=10)
        self.width = width
        self.height = height
        self.M_to_ref = None  # transformation matrix to reference frame
        self.transform_history = deque(maxlen=5)  # store recent transforms for smoothing
        self.cumulative_M = np.eye(3, dtype=np.float32)  # 从参考帧到当前帧的累积变换

        self.flow_magnitude_acc = np.zeros((height, width), dtype=np.float32)
        self.flow_count = np.zeros((height, width), dtype=np.uint16)
        self.ui_mask = np.ones((height, width), dtype=np.uint8) * 255

        if self.optical_flow_visible:
            cv2.namedWindow("CMC Optical Flow", cv2.WINDOW_NORMAL)
        if self.mask_visible:
            cv2.namedWindow("CMC Mask", cv2.WINDOW_NORMAL)

    def compensating(self, ball_xy ,xys,frame):
        # 光流向量小 => UI， 进行排除
        avg_flow = np.zeros_like(self.flow_magnitude_acc)
        valid = self.flow_count > 20
        
        detection_mask = np.ones((self.height, self.width), dtype=np.uint8) * 255
                
        if self.ref_gray is None:
            print("first frame set as reference")
            self.ref_gray = self.grays[0].copy()
            self.cumulative_M = np.eye(3, dtype=np.float32)
        else:
            self.ref_gray = self.grays[0].copy()
            
        if self.M_to_ref is not None:
            camera_speed = np.linalg.norm(self.M_to_ref[:, 2]) 
            if camera_speed > 0.5: 
                avg_flow[valid] = self.flow_magnitude_acc[valid] / self.flow_count[valid]
                self.ui_mask[valid & (avg_flow < 0.1)] = 0
                
                # print the count of ui_mask == 0
                print(f"masked pixels: {(self.ui_mask == 0).sum()}; valid count: {valid.sum()}")
            else:
                print("Camera speed too low, skipping UI mask update")
        else:
            print("M_to_ref is None, skipping UI mask update")
            
        if self.prev_gray is not None:
            for x1, y1, x2, y2 in xys.astype(int):
                margin = 10
                detection_mask[max(0, y1-margin):min(self.height, y2+margin), 
                    max(0, x1-margin):min(self.width, x2+margin)] = 0
            if ball_xy is not None:
                cx, cy = ball_xy
                detection_mask[max(0, cy-15):min(self.height, cy+15), 
                    max(0, cx-15):min(self.width, cx+15)] = 0
            
            # Visualize mask if enabled
            if self.mask_visible:
                if frame.ndim == 2 or (frame.ndim == 3 and frame.shape[2] == 1):
                    mask_vis = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                else:
                    mask_vis = frame.copy()
                # avoid modifying original frame
                #mask_vis[detection_mask == 0] = (0, 0, 255) # red
                #mask_vis[self.ui_mask == 0] = (255, 0, 0) # blue
                #把ui_mask周围10px的区域也标记
                kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 21))
                ui_mask_dilated = cv2.dilate(self.ui_mask == 0, kernel)
                mask_vis[ui_mask_dilated] = (255, 255, 0) 
                print(f"UI Masked pixels (dilated): {ui_mask_dilated.sum()}")
                
                cv2.imshow("CMC Mask", mask_vis)

            pts_prev = cv2.goodFeaturesToTrack(
                self.prev_gray,
                maxCorners=1000,  # 特征点数量
                qualityLevel=0.02,  # 质量阈值
                minDistance=10,  # 最小距离
                mask=detection_mask,
                blockSize=7,  # 块大小
                useHarrisDetector=False,
                k=0.04
            )

            if pts_prev is not None and len(pts_prev) >= 6:
                pts_curr, status, err = cv2.calcOpticalFlowPyrLK(
                    self.prev_gray,
                    self.gray,
                    pts_prev,
                    None,
                    winSize=(31, 31), # 搜索窗口
                    maxLevel=4,
                    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
                )
                
                # 计算光流矢量及其幅值用于 UI 排除
                flow = pts_curr - pts_prev
                flow_mag = np.linalg.norm(flow, axis=2).flatten()

                for (x, y), mag, ok in zip(
                        pts_prev.reshape(-1, 2),
                        flow_mag,
                        status.flatten()):
                    if not ok:
                        continue
                    x, y = int(x), int(y)
                    self.flow_magnitude_acc[y, x] += mag
                    self.flow_count[y, x] += 1
            
                # 仅使用状态为1且误差较小的点进行变换估计
                good_mask = (status.flatten() == 1) & (err.flatten() < 10.0)
                good_prev = pts_prev[good_mask]
                good_curr = pts_curr[good_mask]
                flow_vis = frame

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

                    # 可视化光流(if enabled)
                    if self.optical_flow_visible:
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
                                    if len(self.transform_history) > 0:
                                        # 计算与上一帧变换的差异
                                        last_M = self.transform_history[-1]
                                        diff = np.linalg.norm(M_3x3 - last_M)
                                        # 如果变化太大，可能是异常值，使用加权平均
                                        if diff < 0.5:  # 变化在合理范围内
                                            # 加权平均：当前帧权重0.7，历史平均权重0.3
                                            alpha = 0.7
                                            M_3x3 = alpha * M_3x3 + (1 - alpha) * last_M
                                    
                                    self.transform_history.append(M_3x3.copy())
                                    # 累积变换：从参考帧到当前帧 = 从参考帧到上一帧 * 从上一帧到当前帧
                                    self.cumulative_M = self.cumulative_M @ M_3x3
                                    
                                    # 计算从当前帧到参考帧的逆变换（用于补偿）
                                    # cumulative_M是3x3矩阵，先求逆，然后提取2x3部分
                                    cumulative_M_inv = np.linalg.inv(self.cumulative_M)
                                    self.M_to_ref = cumulative_M_inv[:2, :]  # 提取前两行得到2x3矩阵
                                else:
                                    print("det too small")
                                    self.M_to_ref = None
                            else:
                                print("scale out of bounds")
                                self.M_to_ref = None
                        else:
                            print("low inlier ratio")
                            self.M_to_ref = None
                    else:
                        print("M is None or inliers is None")
                        self.M_to_ref = None
                else:
                    print("good prev < 6")
                    self.M_to_ref = None
            else:
                print("Not enough good points")
                self.M_to_ref = None
        else:
            print("prev_gray is None")
            self.M_to_ref = None        
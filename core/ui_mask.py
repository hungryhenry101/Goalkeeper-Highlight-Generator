import numpy as np
import cv2
from collections import deque


class UIMaskDetector:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        
        # 光流累积
        self.flow_magnitude_acc = np.zeros((height, width), dtype=np.float32)
        self.flow_count = np.zeros((height, width), dtype=np.uint16)
        
        # UI 置信度图 - 连续多帧低光流会提高置信度
        self.ui_confidence = np.zeros((height, width), dtype=np.float32)
        
        # 最终 UI 掩码
        self.ui_mask = np.ones((height, width), dtype=np.uint8) * 255
        
        # 位置先验权重 - UI 更可能出现在边缘区域
        self._build_position_prior()

        # 用于检测 UI 的时间稳定性
        self.low_flow_history = deque(maxlen=3)
        
        # PARAMETERS
        self.min_flow_threshold = 1.0  # 低于此值认为是静止点（像素/帧）
        self.min_confidence = 0.5  # 置信度阈值
        self.min_frames = 3  # 最小累积帧数
        self.decay_factor = 0.9  # 置信度衰减因子
        self.motion_threshold = 400.0  # 相机运动阈值（根据实际运动强度调整）

    def _build_position_prior(self):
        """
        构建位置先验图
        UI 元素（比分牌、台标、字幕）通常出现在画面的顶部、底部或角落
        给这些区域更高的权重
        """
        self.position_prior = np.zeros((self.height, self.width), dtype=np.float32)

        y, x = np.ogrid[:self.height, :self.width]

        # 顶部区域（比分牌、台标）- 权重最高
        top_weight = np.exp(-(y.astype(np.float32) / (self.height * 0.15)) ** 2)
        self.position_prior += top_weight * 0.4

        # 底部区域（字幕、信息栏）- 权重次高
        bottom_weight = np.exp(-((self.height - y).astype(np.float32) / (self.height * 0.15)) ** 2)
        self.position_prior += bottom_weight * 0.3

        # 左侧区域（台标）- 权重较低
        left_weight = np.exp(-(x.astype(np.float32) / (self.width * 0.2)) ** 2)
        self.position_prior += left_weight * 0.2

        # 右侧区域 - 权重较低
        right_weight = np.exp(-((self.width - x).astype(np.float32) / (self.width * 0.2)) ** 2)
        self.position_prior += right_weight * 0.1

        # 归一化到 [0, 1]
        max_val = self.position_prior.max()
        if max_val > 0:
            self.position_prior /= max_val

    def check_camera_motion(self, M):
        """
        检测相机是否有显著运动

        参数:
            M: 2x3 仿射变换矩阵 (从 prev 到 curr)

        返回:
            (has_motion, motion_strength): 是否有运动及运动强度
        """
        if M is None:
            return False, 0.0

        # 提取平移分量
        tx, ty = M[0, 2], M[1, 2]
        translation = np.sqrt(tx ** 2 + ty ** 2)

        # 提取旋转分量（通过计算变换矩阵的旋转角度）
        # 仿射矩阵 [[a, b, tx], [c, d, ty]] 中，旋转角度可以通过 atan2(b, a) 估计
        rotation = np.abs(np.arctan2(M[0, 1], M[0, 0]))

        # 提取尺度分量
        scale_x = np.sqrt(M[0, 0] ** 2 + M[0, 1] ** 2)
        scale_y = np.sqrt(M[1, 0] ** 2 + M[1, 1] ** 2)
        scale_change = np.abs(scale_x - scale_y) + np.abs(1 - (scale_x + scale_y) / 2)

        # 综合运动评分
        motion_score = translation * 2.0 + rotation * 5.0 + scale_change * 3.0

        # 阈值判断（根据实际视频调整）
        has_motion = motion_score > self.motion_threshold

        return has_motion, motion_score

    def update(self, flow_mags, flow_positions, M, frame_gray=None):
        """
        更新 UI 检测结果

        参数:
            flow_mags: 光流幅值列表 [(x, y, mag), ...]
            flow_positions: 光流位置列表 [(x, y), ...]
            M: 2x3 仿射变换矩阵
            frame_gray: 当前帧灰度图（可选，用于调试）

        返回:
            ui_mask: 255 表示非 UI 区域，0 表示 UI 区域
        """
        # 1. 检查相机运动
        has_motion, motion_strength = self.check_camera_motion(M)

        if not has_motion:
            # 相机静止时，光流信息不可靠，只衰减置信度
            self.ui_confidence *= self.decay_factor
            self.ui_mask = (self.ui_confidence < self.min_confidence).astype(np.uint8) * 255
            return self.ui_mask

        # 2. 累积光流信息
        for x, y, mag in flow_mags:
            x, y = int(x), int(y)
            if 0 <= x < self.width and 0 <= y < self.height:
                self.flow_magnitude_acc[y, x] += mag
                self.flow_count[y, x] += 1

        # 3. 计算平均光流并更新置信度
        valid = self.flow_count >= self.min_frames
        avg_flow = np.zeros_like(self.flow_magnitude_acc)
        avg_flow[valid] = self.flow_magnitude_acc[valid] / self.flow_count[valid]

        # 使用相对阈值：低于平均光流 30% 的点认为是低光流
        global_avg_flow = avg_flow[valid].mean() if valid.any() else 1.0
        relative_threshold = max(global_avg_flow * 0.3, self.min_flow_threshold)

        # 低光流区域增加置信度，高光流区域降低置信度
        is_low_flow = avg_flow < relative_threshold

        # 只在边缘区域（位置先验>0.3）才可能是 UI
        is_edge_region = self.position_prior > 0.3

        # 记录当前帧的低光流区域
        self.low_flow_history.append(is_low_flow.copy())

        # 时间一致性检查：连续多帧都是低光流的区域更可能是 UI
        if len(self.low_flow_history) >= 2:
            consistent_low_flow = np.all(list(self.low_flow_history), axis=0)
        else:
            consistent_low_flow = is_low_flow

        # 更新置信度
        # 只有 边缘区域 + 一致低光流 才会增加置信度
        ui_candidate = consistent_low_flow & is_edge_region
        confidence_increase = ui_candidate.astype(np.float32) * (0.1 + self.position_prior * 0.05)
        confidence_decrease = (~ui_candidate).astype(np.float32) * 0.05

        self.ui_confidence = np.clip(
            self.ui_confidence * self.decay_factor + confidence_increase - confidence_decrease,
            0, 1
        )

        # 4. 生成 UI 掩码
        is_ui = self.ui_confidence >= self.min_confidence

        # 形态学处理：去除噪声，填充空洞
        # is_ui = is_ui.astype(np.uint8)
        # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        # is_ui = cv2.morphologyEx(is_ui, cv2.MORPH_CLOSE, kernel)  # 先闭运算填充空洞
        # is_ui = cv2.morphologyEx(is_ui, cv2.MORPH_OPEN, kernel)   # 再开运算去除噪声

        # 5. 输出掩码（255=非 UI，0=UI）
        self.ui_mask = (is_ui == 0).astype(np.uint8) * 255

        return self.ui_mask

    def reset(self):
        """重置所有状态"""
        self.flow_magnitude_acc.fill(0)
        self.flow_count.fill(0)
        self.ui_confidence.fill(0)
        self.ui_mask.fill(255)
        self.low_flow_history.clear()

    def soft_reset(self):
        """软重置 - 只衰减置信度，保留部分历史信息"""
        self.flow_magnitude_acc.fill(0)
        self.flow_count.fill(0)
        self.ui_confidence *= 0.5
        self.low_flow_history.clear()
    
    def get_debug_info(self):
        """获取调试信息"""
        return {
            'ui_pixels': (self.ui_mask == 0).sum(),
            'confidence_max': self.ui_confidence.max(),
            'confidence_mean': self.ui_confidence.mean(),
            'valid_points': (self.flow_count >= self.min_frames).sum(),
        }

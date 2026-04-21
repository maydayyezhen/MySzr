import cv2
import mediapipe as mp
import numpy as np

from landmark_schema import extract_upper_body_array


class PoseExtractor:
    """
    上半身关键点提取器
    使用 MediaPipe Pose 做实时人体姿态检测
    """

    def __init__(
        self,
        model_complexity: int = 1,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ):
        self.mp_pose = mp.solutions.pose

        # 使用视频流模式的经典 Pose API
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=model_complexity,
            smooth_landmarks=True,
            enable_segmentation=False,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    def extract(self, frame_bgr: np.ndarray) -> np.ndarray | None:
        """
        输入一帧 BGR 图像
        返回:
            - 成功检测到人体时: shape=[点数,4] 的 ndarray
            - 未检测到时: None
        """
        if frame_bgr is None:
            return None

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        results = self.pose.process(frame_rgb)

        if not results.pose_landmarks:
            return None

        return extract_upper_body_array(results.pose_landmarks)

    def close(self):
        """释放 MediaPipe 资源"""
        self.pose.close()
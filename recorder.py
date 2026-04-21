import math
import time
from typing import Optional

import cv2
import numpy as np

from landmark_schema import empty_frame_data
from pose_extractor import PoseExtractor
from save_utils import save_landmarks_npy
from ui_overlay import draw_button, draw_countdown, draw_landmarks, draw_status_panel


class PoseRecorderApp:
    """
    上半身点位录制应用
    功能：
    1. 打开摄像头实时预览
    2. 点击左上角 START 按钮
    3. 先倒计时 3 秒
    4. 再正式录制指定时长
    5. 录制结束后不自动保存，而是等待用户确认
       - 按 S 保存
       - 按 D 丢弃
       - 也可以再次点 START 重新录制
    """

    WINDOW_NAME = "Upper Body Pose Recorder"

    STATE_READY = "READY"
    STATE_COUNTDOWN = "COUNTDOWN"
    STATE_RECORDING = "RECORDING"
    STATE_REVIEW = "REVIEW"
    STATE_SAVED = "SAVED"
    STATE_DISCARDED = "DISCARDED"

    def __init__(
        self,
        duration_sec: float,
        output_path: str,
        camera_index: int = 0,
        countdown_sec: int = 3,
    ):
        self.duration_sec = float(duration_sec)
        self.output_path = output_path
        self.camera_index = int(camera_index)
        self.countdown_sec = int(countdown_sec)

        self.state = self.STATE_READY

        self.countdown_start_time: Optional[float] = None
        self.record_start_time: Optional[float] = None

        self.frames_data: list[np.ndarray] = []
        self.last_valid_points = empty_frame_data()

        self.saved_path = ""
        self.saved_shape = None

        # 左上角开始按钮
        self.button_rect = (20, 20, 220, 70)

        self.cap = cv2.VideoCapture(self.camera_index)
        if not self.cap.isOpened():
            raise RuntimeError(f"无法打开摄像头: {self.camera_index}")

        self.extractor = PoseExtractor()

        cv2.namedWindow(self.WINDOW_NAME)
        cv2.setMouseCallback(self.WINDOW_NAME, self._on_mouse)

    def _point_in_rect(self, x: int, y: int, rect: tuple[int, int, int, int]) -> bool:
        x1, y1, x2, y2 = rect
        return x1 <= x <= x2 and y1 <= y <= y2

    def _reset_record_buffer(self):
        """重置录制缓存"""
        self.frames_data = []
        self.saved_path = ""
        self.saved_shape = None
        self.last_valid_points = empty_frame_data()

    def _start_countdown(self):
        """进入倒计时"""
        self.state = self.STATE_COUNTDOWN
        self.countdown_start_time = time.time()
        self.record_start_time = None
        self._reset_record_buffer()

    def _start_recording(self):
        """正式开始录制"""
        self.state = self.STATE_RECORDING
        self.record_start_time = time.time()
        self.frames_data = []

    def _finish_recording(self):
        """录制结束，进入待确认状态，不自动保存"""
        self.state = self.STATE_REVIEW

    def _save_recording(self):
        """保存录制结果"""
        self.saved_path, self.saved_shape = save_landmarks_npy(
            self.output_path,
            self.frames_data,
        )
        self.state = self.STATE_SAVED

    def _discard_recording(self):
        """丢弃录制结果"""
        self._reset_record_buffer()
        self.state = self.STATE_DISCARDED

    def _on_mouse(self, event, x, y, flags, param):
        """
        鼠标点击事件：
        READY / REVIEW / SAVED / DISCARDED 状态下，都可以重新点 START 录制
        """
        if event != cv2.EVENT_LBUTTONDOWN:
            return

        if not self._point_in_rect(x, y, self.button_rect):
            return

        if self.state in [
            self.STATE_READY,
            self.STATE_REVIEW,
            self.STATE_SAVED,
            self.STATE_DISCARDED,
        ]:
            self._start_countdown()

    def _get_state_text(self) -> str:
        if self.state == self.STATE_READY:
            return "READY"
        if self.state == self.STATE_COUNTDOWN:
            return "COUNTDOWN"
        if self.state == self.STATE_RECORDING:
            return "RECORDING"
        if self.state == self.STATE_REVIEW:
            return "REVIEW (Press S to save / D to discard)"
        if self.state == self.STATE_SAVED:
            if self.saved_shape is not None:
                return f"SAVED {self.saved_shape}"
            return "SAVED"
        if self.state == self.STATE_DISCARDED:
            return "DISCARDED"
        return "UNKNOWN"

    def _mirror_points_for_display(self, points: Optional[np.ndarray]) -> Optional[np.ndarray]:
        """
        仅用于显示的点位镜像：
        将 x 从 x 改成 1 - x
        注意：不会影响实际保存的数据
        """
        if points is None:
            return None

        mirrored = points.copy()
        mirrored[:, 0] = 1.0 - mirrored[:, 0]
        return mirrored

    def run(self):
        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("读取摄像头画面失败，程序结束。")
                break

            # 原始帧用于检测和保存
            raw_frame = frame.copy()

            # 镜像帧仅用于显示
            display_frame = cv2.flip(raw_frame, 1)

            # 始终基于原始帧做检测
            detected_points = self.extractor.extract(raw_frame)

            now = time.time()

            # ------------------------------
            # 状态机
            # ------------------------------
            if self.state == self.STATE_COUNTDOWN:
                elapsed = now - self.countdown_start_time
                if elapsed >= self.countdown_sec:
                    self._start_recording()

            elif self.state == self.STATE_RECORDING:
                if detected_points is None:
                    current_points = self.last_valid_points.copy()
                else:
                    current_points = detected_points.copy()
                    self.last_valid_points = current_points.copy()

                self.frames_data.append(current_points)

                record_elapsed = now - self.record_start_time
                if record_elapsed >= self.duration_sec:
                    self._finish_recording()

            # ------------------------------
            # 预览点位
            # REVIEW / SAVED 状态下，优先显示录制结束时的最后一帧
            # 其他状态下显示当前实时点位
            # ------------------------------
            if self.state in [self.STATE_REVIEW, self.STATE_SAVED]:
                preview_points = self.last_valid_points if len(self.frames_data) > 0 else detected_points
            else:
                preview_points = detected_points
                if preview_points is None and len(self.frames_data) > 0:
                    preview_points = self.last_valid_points

            preview_points = self._mirror_points_for_display(preview_points)

            # 绘制关键点
            draw_landmarks(display_frame, preview_points)

            # 按钮
            button_enabled = self.state in [
                self.STATE_READY,
                self.STATE_REVIEW,
                self.STATE_SAVED,
                self.STATE_DISCARDED,
            ]
            draw_button(display_frame, self.button_rect, "START", enabled=button_enabled)

            # 状态面板
            draw_status_panel(
                frame=display_frame,
                state_text=self._get_state_text(),
                duration_sec=self.duration_sec,
                recorded_frames=len(self.frames_data),
                output_path=self.output_path,
            )

            # 倒计时显示
            if self.state == self.STATE_COUNTDOWN:
                remaining = self.countdown_sec - (now - self.countdown_start_time)
                remaining = max(0, math.ceil(remaining))
                draw_countdown(display_frame, remaining)

            # 录制剩余时间
            if self.state == self.STATE_RECORDING:
                record_elapsed = now - self.record_start_time
                remaining_record = max(0.0, self.duration_sec - record_elapsed)

                cv2.putText(
                    display_frame,
                    f"Remaining: {remaining_record:.2f}s",
                    (20, 260),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.75,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )

            # REVIEW 状态下给出操作提示
            if self.state == self.STATE_REVIEW:
                cv2.putText(
                    display_frame,
                    "Recording finished. Press S to save, D to discard.",
                    (20, 300),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.72,
                    (0, 255, 255),
                    2,
                    cv2.LINE_AA,
                )

            # SAVED 状态提示
            if self.state == self.STATE_SAVED and self.saved_path:
                cv2.putText(
                    display_frame,
                    "Save completed!",
                    (20, 300),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.75,
                    (0, 255, 255),
                    2,
                    cv2.LINE_AA,
                )

            # DISCARDED 状态提示
            if self.state == self.STATE_DISCARDED:
                cv2.putText(
                    display_frame,
                    "Recording discarded.",
                    (20, 300),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.75,
                    (0, 200, 255),
                    2,
                    cv2.LINE_AA,
                )

            cv2.imshow(self.WINDOW_NAME, display_frame)

            key = cv2.waitKey(1) & 0xFF

            # 退出
            if key in [27, ord("q")]:
                break

            # REVIEW 状态下按 S 保存
            if self.state == self.STATE_REVIEW and key == ord("s"):
                self._save_recording()

            # REVIEW 状态下按 D 丢弃
            if self.state == self.STATE_REVIEW and key == ord("d"):
                self._discard_recording()

        self.close()

    def close(self):
        try:
            self.extractor.close()
        except Exception:
            pass

        try:
            if self.cap is not None:
                self.cap.release()
        except Exception:
            pass

        cv2.destroyAllWindows()
import cv2
import numpy as np

from landmark_schema import NAME_TO_INDEX, UPPER_BODY_CONNECTIONS


def draw_button(
    frame: np.ndarray,
    rect: tuple[int, int, int, int],
    text: str,
    enabled: bool = True,
):
    """
    绘制开始按钮
    rect: (x1, y1, x2, y2)
    """
    x1, y1, x2, y2 = rect

    if enabled:
        color = (60, 180, 75)  # 绿色
    else:
        color = (120, 120, 120)  # 灰色

    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness=-1)
    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), thickness=2)

    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
    text_x = x1 + (x2 - x1 - text_size[0]) // 2
    text_y = y1 + (y2 - y1 + text_size[1]) // 2

    cv2.putText(
        frame,
        text,
        (text_x, text_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )


def draw_landmarks(frame: np.ndarray, points: np.ndarray | None):
    """
    在画面上绘制上半身关键点与连接线
    points shape = [点数, 4]
    """
    if points is None:
        return

    h, w = frame.shape[:2]

    # 先画线
    for name_a, name_b in UPPER_BODY_CONNECTIONS:
        idx_a = NAME_TO_INDEX[name_a]
        idx_b = NAME_TO_INDEX[name_b]

        xa, ya, _, va = points[idx_a]
        xb, yb, _, vb = points[idx_b]

        # 可见度太低就不画
        if va < 0.1 or vb < 0.1:
            continue

        pa = (int(xa * w), int(ya * h))
        pb = (int(xb * w), int(yb * h))

        cv2.line(frame, pa, pb, (255, 200, 0), 2, cv2.LINE_AA)

    # 再画点
    for i in range(points.shape[0]):
        x, y, _, v = points[i]

        if v < 0.1:
            continue

        px = int(x * w)
        py = int(y * h)

        cv2.circle(frame, (px, py), 4, (0, 0, 255), -1, cv2.LINE_AA)


def draw_status_panel(
    frame: np.ndarray,
    state_text: str,
    duration_sec: float,
    recorded_frames: int,
    output_path: str,
):
    """
    绘制状态面板
    """
    lines = [
        f"State: {state_text}",
        f"Duration: {duration_sec:.2f}s",
        f"Recorded Frames: {recorded_frames}",
        f"Output: {output_path}",
        "Press Q / ESC to quit",
    ]

    x = 20
    y = 100

    for line in lines:
        cv2.putText(
            frame,
            line,
            (x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        y += 30


def draw_countdown(frame: np.ndarray, remaining_sec: int):
    """
    居中绘制倒计时数字
    """
    text = str(remaining_sec)
    h, w = frame.shape[:2]

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 3.0
    thickness = 6

    text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
    text_x = (w - text_size[0]) // 2
    text_y = (h + text_size[1]) // 2

    cv2.putText(
        frame,
        text,
        (text_x, text_y),
        font,
        font_scale,
        (0, 255, 255),
        thickness,
        cv2.LINE_AA,
    )
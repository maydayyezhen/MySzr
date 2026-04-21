import mediapipe as mp
import numpy as np

# =========================
# 上半身关键点定义
# 最终保存顺序就按这里的顺序来
# 每个点保存 [x, y, z, visibility]
# =========================

MP_POSE = mp.solutions.pose

UPPER_BODY_LANDMARKS = [
    ("nose", MP_POSE.PoseLandmark.NOSE),
    ("left_eye", MP_POSE.PoseLandmark.LEFT_EYE),
    ("right_eye", MP_POSE.PoseLandmark.RIGHT_EYE),
    ("left_ear", MP_POSE.PoseLandmark.LEFT_EAR),
    ("right_ear", MP_POSE.PoseLandmark.RIGHT_EAR),

    ("left_shoulder", MP_POSE.PoseLandmark.LEFT_SHOULDER),
    ("right_shoulder", MP_POSE.PoseLandmark.RIGHT_SHOULDER),

    ("left_elbow", MP_POSE.PoseLandmark.LEFT_ELBOW),
    ("right_elbow", MP_POSE.PoseLandmark.RIGHT_ELBOW),

    ("left_wrist", MP_POSE.PoseLandmark.LEFT_WRIST),
    ("right_wrist", MP_POSE.PoseLandmark.RIGHT_WRIST),

    ("left_thumb", MP_POSE.PoseLandmark.LEFT_THUMB),
    ("right_thumb", MP_POSE.PoseLandmark.RIGHT_THUMB),

    ("left_index", MP_POSE.PoseLandmark.LEFT_INDEX),
    ("right_index", MP_POSE.PoseLandmark.RIGHT_INDEX),

    ("left_pinky", MP_POSE.PoseLandmark.LEFT_PINKY),
    ("right_pinky", MP_POSE.PoseLandmark.RIGHT_PINKY),

    ("left_hip", MP_POSE.PoseLandmark.LEFT_HIP),
    ("right_hip", MP_POSE.PoseLandmark.RIGHT_HIP),
]

LANDMARK_NAMES = [name for name, _ in UPPER_BODY_LANDMARKS]
NAME_TO_INDEX = {name: idx for idx, name in enumerate(LANDMARK_NAMES)}
NUM_POINTS = len(UPPER_BODY_LANDMARKS)

# =========================
# 用于预览绘制的连接关系
# 这里只画头部、肩部、胳膊、手腕与粗略手部点
# =========================
UPPER_BODY_CONNECTIONS = [
    ("left_eye", "nose"),
    ("right_eye", "nose"),
    ("left_eye", "left_ear"),
    ("right_eye", "right_ear"),

    ("left_shoulder", "right_shoulder"),
    ("left_shoulder", "left_elbow"),
    ("left_elbow", "left_wrist"),

    ("right_shoulder", "right_elbow"),
    ("right_elbow", "right_wrist"),

    ("left_wrist", "left_thumb"),
    ("left_wrist", "left_index"),
    ("left_wrist", "left_pinky"),

    ("right_wrist", "right_thumb"),
    ("right_wrist", "right_index"),
    ("right_wrist", "right_pinky"),

    ("left_shoulder", "left_hip"),
    ("right_shoulder", "right_hip"),
    ("left_hip", "right_hip"),
]


def empty_frame_data() -> np.ndarray:
    """
    创建一个空帧数据。
    shape = [点数, 4]
    最后一维是 [x, y, z, visibility]
    """
    return np.zeros((NUM_POINTS, 4), dtype=np.float32)


def extract_upper_body_array(pose_landmarks) -> np.ndarray:
    """
    从 MediaPipe 的 pose_landmarks 中提取上半身点位，
    返回 shape = [NUM_POINTS, 4] 的 float32 数组。
    """
    data = empty_frame_data()

    if pose_landmarks is None:
        return data

    for idx, (_, landmark_enum) in enumerate(UPPER_BODY_LANDMARKS):
        lm = pose_landmarks.landmark[landmark_enum.value]
        data[idx, 0] = lm.x
        data[idx, 1] = lm.y
        data[idx, 2] = lm.z
        data[idx, 3] = getattr(lm, "visibility", 0.0)

    return data
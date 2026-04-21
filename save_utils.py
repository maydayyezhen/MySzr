from pathlib import Path
import numpy as np


def normalize_output_path(output_path: str) -> Path:
    """
    规范化输出路径
    如果没有 .npy 后缀，则自动补上
    """
    path = Path(output_path).expanduser()

    if path.suffix.lower() != ".npy":
        path = path.with_suffix(".npy")

    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def save_landmarks_npy(output_path: str, frames: list[np.ndarray]) -> tuple[str, tuple]:
    """
    保存关键点数组到 .npy 文件
    返回:
        - 保存后的完整路径
        - 数组 shape
    """
    path = normalize_output_path(output_path)

    array = np.asarray(frames, dtype=np.float32)
    np.save(str(path), array)

    return str(path), array.shape
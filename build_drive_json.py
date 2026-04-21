import json
from pathlib import Path
from typing import Optional

import numpy as np

LANDMARK_INDEX = {
    "nose": 0,
    "left_eye": 1,
    "right_eye": 2,
    "left_ear": 3,
    "right_ear": 4,
    "left_shoulder": 5,
    "right_shoulder": 6,
    "left_elbow": 7,
    "right_elbow": 8,
    "left_wrist": 9,
    "right_wrist": 10,
    "left_thumb": 11,
    "right_thumb": 12,
    "left_index": 13,
    "right_index": 14,
    "left_pinky": 15,
    "right_pinky": 16,
    "left_hip": 17,
    "right_hip": 18,
}


def ask_str(prompt: str, default: str) -> str:
    raw = input(f"{prompt} [默认 {default}]: ").strip()
    return raw if raw else default


def ask_float(prompt: str, default: float) -> float:
    while True:
        raw = input(f"{prompt} [默认 {default}]: ").strip()
        if raw == "":
            return float(default)
        try:
            return float(raw)
        except ValueError:
            print("请输入合法数字。")


def normalize(v: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    n = np.linalg.norm(v)
    if n < eps:
        return np.zeros_like(v, dtype=np.float32)
    return (v / n).astype(np.float32)


def get_xyz(frame_points: np.ndarray, name: str) -> np.ndarray:
    return frame_points[LANDMARK_INDEX[name], :3].astype(np.float32)


def get_visibility(frame_points: np.ndarray, name: str) -> float:
    return float(frame_points[LANDMARK_INDEX[name], 3])


def to_body_coord(p: np.ndarray) -> np.ndarray:
    return np.array([p[0], -p[1], -p[2]], dtype=np.float32)


def compute_body_axes(frame_points: np.ndarray) -> Optional[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    ls = to_body_coord(get_xyz(frame_points, "left_shoulder"))
    rs = to_body_coord(get_xyz(frame_points, "right_shoulder"))
    lh = to_body_coord(get_xyz(frame_points, "left_hip"))
    rh = to_body_coord(get_xyz(frame_points, "right_hip"))

    shoulder_mid = (ls + rs) * 0.5
    hip_mid = (lh + rh) * 0.5

    x_axis = normalize(rs - ls)
    y_axis = normalize(shoulder_mid - hip_mid)
    if np.linalg.norm(x_axis) < 1e-6 or np.linalg.norm(y_axis) < 1e-6:
        return None

    z_axis = normalize(np.cross(x_axis, y_axis))
    if np.linalg.norm(z_axis) < 1e-6:
        return None

    y_axis = normalize(np.cross(z_axis, x_axis))
    if np.linalg.norm(y_axis) < 1e-6:
        return None

    return x_axis, y_axis, z_axis


def world_to_body_local(point_world_like: np.ndarray, origin_world_like: np.ndarray, axes, scale: float) -> np.ndarray:
    x_axis, y_axis, z_axis = axes
    diff = point_world_like - origin_world_like
    local = np.array([
        float(np.dot(diff, x_axis)),
        float(np.dot(diff, y_axis)),
        float(np.dot(diff, z_axis)),
    ], dtype=np.float32)
    if scale > 1e-8:
        local = local / scale
    return local.astype(np.float32)


def vec3_to_dict(v: np.ndarray) -> dict:
    return {"x": float(v[0]), "y": float(v[1]), "z": float(v[2])}


def dict_vec_to_np(d: dict) -> np.ndarray:
    return np.array([d["x"], d["y"], d["z"]], dtype=np.float32)


def build_frame_data(frame_points: np.ndarray, visibility_threshold: float) -> Optional[dict]:
    required = [
        "left_shoulder", "right_shoulder",
        "left_elbow", "right_elbow",
        "left_wrist", "right_wrist",
        "left_hip", "right_hip",
    ]
    for name in required:
        if get_visibility(frame_points, name) < visibility_threshold:
            return None

    axes = compute_body_axes(frame_points)
    if axes is None:
        return None

    ls = to_body_coord(get_xyz(frame_points, "left_shoulder"))
    rs = to_body_coord(get_xyz(frame_points, "right_shoulder"))
    le = to_body_coord(get_xyz(frame_points, "left_elbow"))
    re = to_body_coord(get_xyz(frame_points, "right_elbow"))
    lw = to_body_coord(get_xyz(frame_points, "left_wrist"))
    rw = to_body_coord(get_xyz(frame_points, "right_wrist"))

    shoulder_mid = (ls + rs) * 0.5
    shoulder_width = float(np.linalg.norm(rs - ls))
    if shoulder_width < 1e-6:
        return None

    return {
        "leftShoulder": vec3_to_dict(world_to_body_local(ls, shoulder_mid, axes, shoulder_width)),
        "leftElbow": vec3_to_dict(world_to_body_local(le, shoulder_mid, axes, shoulder_width)),
        "leftWrist": vec3_to_dict(world_to_body_local(lw, shoulder_mid, axes, shoulder_width)),
        "rightShoulder": vec3_to_dict(world_to_body_local(rs, shoulder_mid, axes, shoulder_width)),
        "rightElbow": vec3_to_dict(world_to_body_local(re, shoulder_mid, axes, shoulder_width)),
        "rightWrist": vec3_to_dict(world_to_body_local(rw, shoulder_mid, axes, shoulder_width)),
    }


def smooth_position_frames(frames: list[dict], alpha: float = 0.25) -> list[dict]:
    if not frames:
        return frames

    keys = [
        "leftShoulder", "leftElbow", "leftWrist",
        "rightShoulder", "rightElbow", "rightWrist",
    ]

    smoothed = []
    prev = None
    for frame in frames:
        if prev is None:
            prev = frame
            smoothed.append(frame)
            continue

        new_frame = {}
        for key in keys:
            pv = dict_vec_to_np(prev[key])
            cv = dict_vec_to_np(frame[key])
            mixed = (1.0 - alpha) * pv + alpha * cv
            new_frame[key] = vec3_to_dict(mixed.astype(np.float32))

        prev = new_frame
        smoothed.append(new_frame)

    return smoothed


def apply_deadzone_frames(frames: list[dict], pos_deadzone: float = 0.01) -> list[dict]:
    if not frames:
        return frames

    keys = [
        "leftShoulder", "leftElbow", "leftWrist",
        "rightShoulder", "rightElbow", "rightWrist",
    ]

    output = [frames[0]]
    for frame in frames[1:]:
        prev = output[-1]
        new_frame = {}
        for key in keys:
            pv = dict_vec_to_np(prev[key])
            cv = dict_vec_to_np(frame[key])
            if np.linalg.norm(cv - pv) < pos_deadzone:
                new_frame[key] = prev[key]
            else:
                new_frame[key] = frame[key]
        output.append(new_frame)
    return output


def main():
    print("=" * 60)
    print("将 upper_body_pose.npy 转成稳定版 Unity 上肢 IK 用 drive.json")
    print("当前版本只导出 shoulder / elbow / wrist 位置，不导出 hand 朝向。")
    print("=" * 60)

    npy_path = ask_str("请输入 .npy 文件路径", "outputs/upper_body_pose.npy")
    fps = ask_float("请输入录制 FPS（用于 Unity 播放）", 30.0)
    output_json_path = ask_str("请输入输出 JSON 路径", "outputs/drive.json")
    visibility_threshold = ask_float("请输入 visibility 阈值", 0.5)

    npy_file = Path(npy_path).expanduser()
    if not npy_file.exists():
        print(f"文件不存在：{npy_file}")
        return

    data = np.load(str(npy_file))
    if data.ndim != 3 or data.shape[1] != 19 or data.shape[2] != 4:
        print(f"数据 shape 不符合预期，当前 shape = {data.shape}")
        print("预期应该是 [num_frames, 19, 4]")
        return

    raw_frames = []
    last_valid = None
    for i in range(data.shape[0]):
        frame = build_frame_data(data[i], visibility_threshold)
        if frame is None:
            if last_valid is not None:
                raw_frames.append(last_valid)
            else:
                continue
        else:
            raw_frames.append(frame)
            last_valid = frame

    if not raw_frames:
        print("没有构建出有效帧，请检查录制动作或降低 visibility 阈值。")
        return

    smoothed = smooth_position_frames(raw_frames, alpha=0.25)
    smoothed = apply_deadzone_frames(smoothed, pos_deadzone=0.01)

    output = {
        "fps": float(fps),
        "frameCount": len(smoothed),
        "coordinateSystem": "body_local_normalized_by_shoulder_width",
        "origin": "shoulder_mid",
        "frames": smoothed,
    }

    output_file = Path(output_json_path).expanduser()
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"转换完成，已保存到：{output_file}")
    print(f"有效帧数：{len(smoothed)}")


if __name__ == "__main__":
    main()

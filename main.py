from recorder import PoseRecorderApp


def ask_float(prompt: str, default: float) -> float:
    """
    从终端读取浮点数
    直接回车则使用默认值
    """
    while True:
        raw = input(f"{prompt} [默认 {default}]: ").strip()
        if raw == "":
            return float(default)

        try:
            value = float(raw)
            if value <= 0:
                print("请输入大于 0 的数字。")
                continue
            return value
        except ValueError:
            print("输入无效，请重新输入数字。")


def ask_int(prompt: str, default: int) -> int:
    """
    从终端读取整数
    直接回车则使用默认值
    """
    while True:
        raw = input(f"{prompt} [默认 {default}]: ").strip()
        if raw == "":
            return int(default)

        try:
            value = int(raw)
            if value < 0:
                print("请输入大于等于 0 的整数。")
                continue
            return value
        except ValueError:
            print("输入无效，请重新输入整数。")


def ask_str(prompt: str, default: str) -> str:
    """
    从终端读取字符串
    直接回车则使用默认值
    """
    raw = input(f"{prompt} [默认 {default}]: ").strip()
    return raw if raw else default


def main():
    print("=" * 50)
    print("上半身点位录制工具")
    print("说明：点击窗口左上角 START，3 秒后开始采集")
    print("退出：按 Q 或 ESC")
    print("=" * 50)

    duration_sec = ask_float("请输入采集时长（秒）", 8.0)
    camera_index = ask_int("请输入摄像头编号", 0)
    output_path = ask_str("请输入输出文件路径", "outputs/upper_body_pose.npy")

    app = PoseRecorderApp(
        duration_sec=duration_sec,
        output_path=output_path,
        camera_index=camera_index,
        countdown_sec=3,
    )
    app.run()


if __name__ == "__main__":
    main()
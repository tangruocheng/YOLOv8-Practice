# YOLOv8で動体検知の練習  Practice motion detection with YOLOv8
## このリポジトリの目的     Purpose of this repository
# 新潟工業大学の研究でYOLOv8を使うための練習  Practice for using YOLOv8 in research at Niigata Institute of Technology
# 将来的に高齢者見守りシステム開発を目指す  Aim to develop a system to watch over the elderly in the future

## できること
# 1. リアルタイム人カウント
# 2. 転倒検知（開発中）



"""""
学習ログ (Learning Log)
#2025-07-05: カメラ検知の基本実装
#2025-07-06: 転倒検知アルゴリズム調査 → [調査ノート](./docs/fall_detect_research.md)
#2025-07-09：FPS展示追加
#2025-07-14:人検査に集中し、検査目標可信度向上
"""""


import cv2
from ultralytics import YOLO
import time

def main():
    # 1. 初始化模型（加载YOLOv8预训练权重）
    model = YOLO('yolov8n.pt')  # 自动下载（如果本地没有）

    # 2. 打开摄像头
    cap = cv2.VideoCapture(0)  # 0=默认摄像头
    if not cap.isOpened():
        print("错误：摄像头未找到！")
        return

    prev_time = 0  # 用于计算FPS

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 3. 执行检测（关键修改点！）
        results = model.predict(
            frame,
            augment=True,   # 启用轻量数据增强（提升鲁棒性）
            conf=0.5,       # 调高置信度阈值（减少误检）
            classes=[0]     # 只检测"人"（COCO类别0）
        )

        # 4. 计算FPS
        current_time = time.time()
        fps = 1 / (current_time - prev_time)
        prev_time = current_time

        # 5. 绘制检测框和FPS
        annotated_frame = results[0].plot()  # 自动渲染检测结果
        cv2.putText(annotated_frame, f"FPS: {int(fps)}", (10, 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # 6. 显示画面
        cv2.imshow('YOLOv8检测', annotated_frame)
        if cv2.waitKey(1) == ord('q'):  # 按Q退出
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

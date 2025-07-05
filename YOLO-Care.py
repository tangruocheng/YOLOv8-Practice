# YOLOv8で動体検知の練習  Practice motion detection with YOLOv8
## このリポジトリの目的     Purpose of this repository
# 新潟工業大学の研究でYOLOv8を使うための練習  Practice for using YOLOv8 in research at Niigata Institute of Technology
# 将来的に高齢者見守りシステム開発を目指す  Aim to develop a system to watch over the elderly in the future

## できること
# 1. リアルタイム人カウント
# 2. 転倒検知（開発中）




## 学習ログ (Learning Log)
- 2024-07-05: カメラ検知の基本実装  
- 2024-07-06: 転倒検知アルゴリズム調査 → [調査ノート](./docs/fall_detect_research.md)




import cv2
from ultralytics import YOLO

def main():
    # 1. 初始化模型（自动下载预训练权重）
    model = YOLO('yolov8n.pt')  # 确保联网

    # 2. 检测摄像头（改成视频路径也可）
    cap = cv2.VideoCapture(0)  # 0=默认摄像头
    if not cap.isOpened():
        print("错误：摄像头未找到！尝试改用视频文件路径")
        return
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # 3. 执行检测
        results = model.predict(frame, conf=0.25)  # 简化版
        
        # 4. 显示结果
        annotated_frame = results[0].plot()  # 自动画框
        cv2.imshow('YOLOv8检测', annotated_frame)
        
        if cv2.waitKey(1) == ord('q'):  # 按Q退出
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
from ultralytics import YOLO

if __name__ == '__main__':
    # 모델 로드
    model = YOLO('yolov8l.pt')

    # 학습 설정
    model.train(
        data='./data/data.yaml',
        epochs=100,
        imgsz=1920,
        batch=4,
        device=0,
        lr0=0.001,
        lrf=0.1,
        augment=True,
    )

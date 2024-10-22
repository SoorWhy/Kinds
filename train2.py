from ultralytics import YOLO
import wandb


def train():
    # W&B 설정 초기화
    wandb.init(project="Pedestrian_Detection")

    # 모델 로드
    #train 12, 27
    model = YOLO('./runs/detect/train27/weights/best.pt')

    # 모델 학습
    results = model.train(
        data='./data/data.yaml',
        epochs=100,  # W&B에서 설정한 epochs 값 사용
        imgsz=1080,  # 이미지 크기
        batch=4,  # 배치 크기
        device=0,  # GPU 사용
        lr0=0.0983869322001272,  # 초기 learning rate
        lrf=0.17581083062500835,  # learning rate final factor
        momentum=0.9,  # 모멘텀
        weight_decay=0.001,  # 가중치 감쇠
        patience=5
    )

    # 학습 결과 로그
    wandb.log({
        "train_loss": results['train/box_loss'],
        "val_loss": results['val/box_loss'],
        "mAP_50": results['metrics/mAP50'],
        "mAP_50_95": results['metrics/mAP50-95']
    })

    wandb.finish()

if __name__ == '__main__':
    # W&B API 키 로그인
    wandb.login(key='419b5e3b17000e880f1572c359068e78642b58e2')
    train()

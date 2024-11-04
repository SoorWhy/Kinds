import os
from ultralytics import YOLO
import cv2
import numpy as np
import torch
import random
from sklearn.metrics import precision_score, recall_score, accuracy_score
from sklearn.preprocessing import MultiLabelBinarizer

# 시드 고정 함수
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# 시드 고정
set_seed(42)

# YOLO 모델 로드
model = YOLO('./강성민_Kinds.pt')

# 클래스 통합 규칙 정의
def map_class(class_id):
    if class_id in [0, 3]:
        return 0  # 무단횡단 클래스
    elif class_id in [1, 4]:
        return 1  # 횡단보도 클래스
    elif class_id in [2, 5]:
        return 2  # 인도 클래스
    else:
        return None

# 정답 라벨 파일에서 좌표와 클래스 정보 읽기
def read_label(label_path):
    with open(label_path, 'r') as f:
        labels = []
        for line in f.readlines():
            parts = line.strip().split()
            class_id = int(parts[0])
            new_class_id = map_class(class_id)
            if new_class_id is not None:
                x_min, y_min, x_max, y_max = map(float, parts[1:])
                labels.append((new_class_id, x_min, y_min, x_max, y_max))
        return labels

# 예측 결과를 (x_min, y_min, x_max, y_max)로 변환
def convert_predictions_to_bbox(results):
    pred_labels = []
    for result in results:
        boxes = result.boxes
        for box in boxes:
            class_id = int(box.cls[0])
            new_class_id = map_class(class_id)
            if new_class_id is not None:
                x_min, y_min, x_max, y_max = map(float, box.xyxy[0])  # YOLO의 예측 결과를 변환
                pred_labels.append((new_class_id, x_min, y_min, x_max, y_max))
    return pred_labels

# 테스트할 이미지들이 있는 폴더 경로
input_folder = './data/test/images/'
label_folder = './data/test/labels/'
output_label_folder = './predicted_labels/'  # 예측된 라벨 저장 폴더
output_image_folder = './val_output/images/'  # 예측된 이미지 저장 폴더

# 출력 폴더가 없으면 생성
os.makedirs(output_label_folder, exist_ok=True)
os.makedirs(output_image_folder, exist_ok=True)

# 이미지 파일 목록 가져오기 (JPG 파일만)
image_files = [f for f in os.listdir(input_folder) if f.endswith('.jpg')]

# 평가 지표를 계산할 리스트
all_true_labels = []
all_pred_labels = []

# 여러 이미지에 대해 처리
for image_file in image_files:
    image_path = os.path.join(input_folder, image_file)
    label_file = image_file.replace('.jpg', '.txt')
    
    # 이미지와 라벨 로드
    image = cv2.imread(image_path)
    label_path = os.path.join(label_folder, label_file)
    true_labels = read_label(label_path)

    # 모델로 예측
    results = model(image)
    pred_labels = convert_predictions_to_bbox(results)

    # 예측된 결과를 라벨 파일로 저장
    output_label_file = os.path.join(output_label_folder, label_file)
    with open(output_label_file, 'w') as f:
        for label in pred_labels:
            class_id, x_min, y_min, x_max, y_max = label
            f.write(f"{class_id} {x_min} {y_min} {x_max} {y_max}\n")
    
    # 바운딩 박스를 이미지에 그려 저장
    for label in pred_labels:
        class_id, x_min, y_min, x_max, y_max = label
        color = (0, 255, 0) if class_id == 0 else (255, 0, 0) if class_id == 1 else (0, 0, 255)
        cv2.rectangle(image, (int(x_min), int(y_min)), (int(x_max), int(y_max)), color, 2)
        cv2.putText(image, f'ID: {class_id}', (int(x_min), int(y_min) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    output_image_path = os.path.join(output_image_folder, image_file)
    cv2.imwrite(output_image_path, image)

    # 스코어 계산을 위한 레이블 준비
    all_true_labels.append([label[0] for label in true_labels])
    all_pred_labels.append([label[0] for label in pred_labels])

# MultiLabelBinarizer로 레이블 이진화
mlb = MultiLabelBinarizer(classes=[0, 1, 2])
true_binary = mlb.fit_transform(all_true_labels)
pred_binary = mlb.transform(all_pred_labels)

# 스코어 계산 (Precision, Recall, Accuracy)
if true_binary.size > 0 and pred_binary.size > 0:
    precision = precision_score(true_binary, pred_binary, average='macro')
    recall = recall_score(true_binary, pred_binary, average='macro')
    accuracy = accuracy_score(true_binary, pred_binary)

    # 결과 파일로 저장
    result_file = './val_output/score_results.txt'
    with open(result_file, 'w') as f:
        f.write(f"Precision: {precision:.4f}\n")
        f.write(f"Recall: {recall:.4f}\n")
        f.write(f"Accuracy: {accuracy:.4f}\n")

    print(f"Scores saved to {result_file}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"Accuracy: {accuracy:.4f}")
else:
    print("No valid predictions for scoring.")

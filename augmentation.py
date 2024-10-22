import os
import cv2
import albumentations as A

# 멀리서 찍힌 이미지에 대한 증강 파이프라인
far_augmentation_pipeline = A.Compose([
    A.RandomScale(scale_limit=(0, 0.5), p=0.5),  # 객체 크기 확대
    A.RandomCrop(height=1080, width=1440, p=0.5),  # 부드러운 크롭 적용
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.3),  # 밝기 및 대비 조정 (조명 변화 대응)
    A.GaussNoise(var_limit=(10.0, 25.0), p=0.2),  # 적당히 낮은 강도의 가우시안 노이즈 (어두운 상황 대비)
    A.PadIfNeeded(min_height=1080, min_width=1920, border_mode=cv2.BORDER_CONSTANT, value=[0, 0, 0], p=1.0)  # 패딩 적용
], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels'], min_area=0.0001, min_visibility=0.1))


# 가까이서 찍힌 이미지에 대한 증강 파이프라인
close_augmentation_pipeline = A.Compose([
    A.HorizontalFlip(p=0.5),  # 수평 뒤집기
    A.Rotate(limit=10, p=0.3),  # 작은 각도 회전
    A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.0, p=0.2),  # 밝기만 미세하게 조정
    A.HueSaturationValue(hue_shift_limit=2, sat_shift_limit=10, val_shift_limit=5, p=0.2),  # 색상과 채도 변동을 더 낮게 설정
    A.Sharpen(alpha=(0.1, 0.3), lightness=(0.5, 0.9), p=0.2),  # 더 약하게 선명도 조정
], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels'], min_area=0.0001, min_visibility=0.1))


def load_yolo_labels(label_path):
    """YOLO 형식의 라벨 파일을 로드하여 bbox와 class_labels를 반환"""
    bboxes = []
    class_labels = []
    with open(label_path, 'r') as f:
        for line in f.readlines():
            class_id, x_center, y_center, width, height = map(float, line.strip().split())
            bboxes.append([x_center, y_center, width, height])
            class_labels.append(int(class_id))
    return bboxes, class_labels

def save_yolo_labels(label_path, bboxes, class_labels):
    """증강 후 YOLO 라벨 데이터를 저장"""
    with open(label_path, 'w') as f:
        for bbox, class_id in zip(bboxes, class_labels):
            x_center, y_center, width, height = bbox
            f.write(f"{class_id} {x_center} {y_center} {width} {height}\n")

def augment_image_and_labels(image_path, label_path, output_image_path, output_label_path, augmentation_pipeline):
    """이미지와 라벨을 동시에 증강하고 저장"""
    # 이미지 로드
    image = cv2.imread(image_path)
    
    # 라벨 로드
    bboxes, class_labels = load_yolo_labels(label_path)
    
    # 증강 적용
    augmented = augmentation_pipeline(image=image, bboxes=bboxes, class_labels=class_labels)
    augmented_image = augmented['image']
    augmented_bboxes = augmented['bboxes']
    augmented_class_labels = augmented['class_labels']
    
    # 증강된 이미지와 라벨 저장
    cv2.imwrite(output_image_path, augmented_image)
    save_yolo_labels(output_label_path, augmented_bboxes, augmented_class_labels)

def process_and_augment_images(image_folders, label_folder, output_image_folder, output_label_folder, augmentation_pipeline, num_augments):
    """이미지 폴더들에서 이미지를 가져오고, 라벨은 라벨 폴더에서 가져와 증강한 후, 결과를 저장"""
    if not os.path.exists(output_image_folder):
        os.makedirs(output_image_folder)
    if not os.path.exists(output_label_folder):
        os.makedirs(output_label_folder)

    # 각 이미지 폴더 처리
    for image_folder in image_folders:
        for filename in os.listdir(image_folder):
            if filename.endswith(('.png', '.jpg', '.jpeg')):
                image_path = os.path.join(image_folder, filename)
                label_path = os.path.join(label_folder, filename.replace('.jpg', '.txt').replace('.png', '.txt'))
                
                # num_augments만큼 반복하여 증강
                for i in range(num_augments):
                    output_image_path = os.path.join(output_image_folder, f"aug_{i}_{filename}")
                    output_label_path = os.path.join(output_label_folder, f"aug_{i}_{filename.replace('.jpg', '.txt').replace('.png', '.txt')}")
                    
                    # 이미지와 라벨 증강 후 저장
                    augment_image_and_labels(image_path, label_path, output_image_path, output_label_path, augmentation_pipeline)


# 입력 이미지 폴더들 (far_output, close_output), 그리고 라벨 폴더
far_image_folder = "./preprocess/far_output"
close_image_folder = "./preprocess/close_output"
label_folder = "./data/train/labels"

# 출력 이미지 및 라벨 폴더
output_image_folder = "./data_aug/train/images"
output_label_folder = "./data_aug/train/labels"

# 증강 프로세스 실행 (멀리서 찍힌 이미지에 대해)
process_and_augment_images([far_image_folder], label_folder, output_image_folder, output_label_folder, far_augmentation_pipeline, num_augments=2)

# 증강 프로세스 실행 (가까이서 찍힌 이미지에 대해)
process_and_augment_images([close_image_folder], label_folder, output_image_folder, output_label_folder, close_augmentation_pipeline, num_augments=2)

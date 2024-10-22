import cv2
import os

# CLAHE (Contrast Limited Adaptive Histogram Equalization) 적용 함수
def apply_clahe(image, clip_limit=2.0, tile_grid_size=(8, 8)):
    """
    CLAHE를 사용하여 이미지의 대비를 개선.
    Args:
        image: 입력 이미지
        clip_limit: 대비 제한 클립 값 (기본값 2.0)
        tile_grid_size: 타일 그리드 크기 (기본값 (8, 8))
    Returns:
        CLAHE가 적용된 이미지
    """
    yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)  # BGR -> YUV 변환
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)  # CLAHE 객체 생성
    yuv[:, :, 0] = clahe.apply(yuv[:, :, 0])  # Y 채널에만 CLAHE 적용 (밝기 정보)
    return cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)  # YUV -> BGR로 다시 변환

# Super-Resolution 적용 함수
def apply_super_resolution(image, upscale_factor=2):
    """
    OpenCV의 DNN 기반 Super-Resolution 모델을 사용하여 이미지 해상도 증가.
    Args:
        image: 입력 이미지
        upscale_factor: 확대 배율 (기본값 2배)
    Returns:
        초해상도(Super-Resolution)가 적용된 이미지
    """
    sr = cv2.dnn_superres.DnnSuperResImpl_create()  # Super-Resolution 객체 생성
    model_path = f"./ESPCN_x{upscale_factor}.pb"  # ESPCN 모델 경로 설정
    sr.readModel(model_path)  # 모델 로드
    sr.setModel("espcn", upscale_factor)  # 모델 및 배율 설정
    result = sr.upsample(image)  # Super-Resolution 적용하여 이미지 확대
    return result

# 전처리 함수: CLAHE와 Super-Resolution 결합
def preprocess_image(image_path, upscale_factor=2, output_size=(1920, 1080)):
    """
    입력 이미지에 CLAHE와 Super-Resolution을 적용한 후 크기를 조정하는 전처리 과정.
    Args:
        image_path: 입력 이미지 파일 경로
        upscale_factor: ESPCN에서 확대할 배율 (x2, x3, x4 선택)
        output_size: 결과 이미지 크기 (기본값 1920x1080)
    Returns:
        전처리된 이미지
    """
    # 이미지 로드
    image = cv2.imread(image_path)
    
    # CLAHE 적용
    image_clahe = apply_clahe(image)
    
    # Super-Resolution 적용
    image_sr = apply_super_resolution(image_clahe, upscale_factor=upscale_factor)
    
    # 이미지 크기 조정 (원본 해상도에 맞게 리사이즈)
    image_resized = cv2.resize(image_sr, output_size)
    
    return image_resized

# 폴더 내 모든 이미지를 처리하는 함수
def process_images_in_folder(input_folder, output_folder, upscale_factor):
    """
    입력 폴더 내의 모든 이미지를 전처리(CLAHE + Super-Resolution)하여 출력 폴더에 저장하는 함수.
    Args:
        input_folder: 입력 이미지들이 있는 폴더 경로
        output_folder: 전처리된 이미지를 저장할 폴더 경로
        upscale_factor: Super-Resolution의 확대 배율 (x2, x3, x4)
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 폴더 내 모든 이미지 파일 처리
    for filename in os.listdir(input_folder):
        if filename.endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(input_folder, filename)
            output_image_path = os.path.join(output_folder, filename)
            
            # 전처리 수행
            preprocessed_image = preprocess_image(image_path, upscale_factor=upscale_factor)
            
            # 전처리된 이미지 저장
            cv2.imwrite(output_image_path, preprocessed_image)

# 멀리서 찍힌 이미지 (far_image)와 가까이서 찍힌 이미지 (close_image)를 각각 처리
def process_all_images(far_image_folder, close_image_folder, far_output_folder, close_output_folder):
    """
    멀리서 찍힌 이미지와 가까이서 찍힌 이미지를 구분하여 전처리하는 함수.
    Args:
        far_image_folder: 먼 이미지들이 저장된 폴더 경로
        close_image_folder: 가까운 이미지들이 저장된 폴더 경로
        far_output_folder: 먼 이미지를 전처리하여 저장할 폴더 경로
        close_output_folder: 가까운 이미지를 전처리하여 저장할 폴더 경로
    """
    # 먼 이미지 처리 (확대 배율 x3)
    process_images_in_folder(far_image_folder, far_output_folder, upscale_factor=3)

    # 가까운 이미지 처리 (확대 배율 x2)
    process_images_in_folder(close_image_folder, close_output_folder, upscale_factor=2)

# 실행 예시
far_image_folder = "./far_image"  # 멀리서 찍힌 이미지가 저장된 폴더 경로
close_image_folder = "./close_image"  # 가까이서 찍힌 이미지가 저장된 폴더 경로
far_output_folder = "./far_output"  # 처리된 먼 이미지를 저장할 폴더 경로
close_output_folder = "./close_output"  # 처리된 가까운 이미지를 저장할 폴더 경로

# 모든 이미지 전처리 수행
process_all_images(far_image_folder, close_image_folder, far_output_folder, close_output_folder)

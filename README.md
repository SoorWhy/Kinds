# Kinds | 세종 자율주행 2회 경진대회
![자율주행](https://github.com/user-attachments/assets/8f9f1b44-1ba2-4277-8608-214c4a01ac8b)  
[세종 2회 경진대회]: https://www.sejong.ai/kor/contest_02.do
## 프로젝트 설명
### | 팀원
강성민, 김낙현
### | 프로젝트명
자율주행 차량의 안전 운행을 위한 CCTV 연계 보행자 탐지
### | 목표 성과
* 정확도(Accuracy) 90% 이상의 보행자 탐지 모델 구현
* 무단횡단 보행자, 횡단보도 보행자에 대한 정확한 분류 탐지
### | 모델
* yolo를 사용한 객체 탐지
![yolo](https://github.com/user-attachments/assets/e4f17194-845a-4761-9d60-b7045d48dbe2)

## 프로젝트에 사용된 기술 및 버전
### | Language
* Python `3.10.12`
* numpy `2.1.2`
* torch `2.4.1+cu121`
* ultralytics `8.3.13`
* wandb `0.18.3`
* opencv-python-headless `4.10.0.84`
* scikit-learn `1.5.2`
### | Tool
* Git Hub
* WandB
### | System
* Linux | Elice Cloud
* Driver Version: 531.14
* CUDA Version: 12.1

## 디렉터리 정보
### | data
* 이미지 학습을 위한 원본 데이터셋
```
data
 ┣ train
 ┃ ┣ images
 ┃ ┗ labels
 ┣ validation
 ┃ ┣ images
 ┃ ┗ labels
 ┗ data.yaml
```
### | preprocess
* 이미지 전처리 분류 데이터셋
* close_image : 다소 가까운 거리의 CCTV 이미지
* far_image : 먼 거리의 CCTV 이미지
 ```
preprocess
 ┣ close_image
 ┣ close_output
 ┣ far_image
 ┣ far_output
 ┗ preprocess_image.py
```
* _output : 각각 전처리 후 데이터
* preprocess_image.py : 각각 환경에 맞게 전처리 하는 코드
### | data_pp
* 이미지 전처리 후 데이터셋
```
data_pp
 ┣ train
 ┃ ┣ images
 ┃ ┗ labels
 ┗ data.yaml
```
### | augmentation.py
* 데이터 증강을 위한 python 코드
* 증강 데이터는 용량이 커서 업로드 하지 못함
### | code.ipynb
* 라이브러리 설치 및 모델 학습을 위한 notebook 코드
### | evaluate.py
* 모델 성능 평가 및 추론을 위한 python 코드
### | requirements.txt
* 종속성 설치를 위한 txt 파일
### | sweep.yaml
* 하이퍼파라미터 스윕을 위한 yaml 파일
### | train.py
* 딥러닝 모델 학습을 위한 python 코드
### | train2.py
* 딥러닝 모델 학습과 WandB 스윕을 위한 python 코드 

## 전처리 및 증강 기법
### | 전처리 과정
원본 이미지에 대해 기본적인 대비 향상과 해상도 개선

1. CLAHE (Contrast Limited Adaptive Histogram Equalization)
* 목적: 이미지의 대비를 향상시켜 밝기 차이가 큰 부분에서 보행자를 더 명확히 탐지
* 적용 방식: YUV 색상 공간에서 Y 채널(밝기 정보)에만 CLAHE를 적용, 어두운 부분을 좀 더 밝게 조정하고 이미지의 전체적인 선명도를 개선
2. Super-Resolution (ESPCN)
* 목적: 해상도를 높여 멀리서 찍힌 이미지에서 작은 객체를 더 잘 탐지
* 적용 방식: OpenCV의 ESPCN 모델을 사용해 2배, 3배까지 해상도를 개선
  
    - 멀리서 찍힌 이미지: 3배 확대
    - 가까이서 찍힌 이미지: 2배 확대

3. 이미지 리사이즈
* 목적: YOLO 학습에 적합한 해상도로 이미지 크기를 1920x1080으로 조정
* 적용 방식: CLAHE와 Super-Resolution을 적용한 후 이미지 크기를 고정된 해상도로 리사이즈

![preprocess](https://github.com/user-attachments/assets/0b59e5d0-5bdb-4973-a3d9-4509ea60b0f3)

### | 증강 기법
각각의 거리에 맞는 증강 기법을 적용하여 다양한 상황에서 보행자를 탐지할 수 있도록 데이터를 증강

#### 1. 멀리서 찍힌 이미지: 작은 객체를 제대로 학습하지 못하는 문제를 해결하기 위한 기법 사용
* RandomScale

  - 목적: 작은 객체 크기를 확대하여 학습
  - 설정: scale_limit=(0, 0.5)로 객체 크기를 최대 50% 확대

* RandomCrop

  - 목적: 작은 객체를 더 잘 학습할 수 있도록 다양한 위치에서 이미지 일부를 학습
  - 설정: height=1080, width=1440 크롭 후 패딩으로 원본 크기 복원

* GaussNoise

  - 목적: 어두운 환경이나 저화질 CCTV에서 발생하는 노이즈에 대응
  - 설정: var_limit=(10.0, 25.0), 20% 확률로 가우시안 노이즈 적용

* PadIfNeeded

  - 목적: 크롭 후 원본 해상도로 이미지 복구, 작은 객체가 잘리지 않도록 보완
  - 설정: min_height=1080, min_width=1920

#### 2. 가까이서 찍힌 이미지: 방향성이나 선명도에 중점을 둔 증강 기법 적용
* HorizontalFlip

  - 목적: 좌우 대칭적인 상황을 학습, 방향성에 유연하게 대응
  - 설정: 50% 확률로 좌우 뒤집기 적용

* Rotate

  - 목적: 다양한 각도에서 보행자를 탐지하도록 회전 적용
  - 설정: limit=10, 최대 ±10도의 회전 적용, 30% 확률

* RandomBrightnessContrast

  - 목적: 미세한 밝기 조정을 통해 낮과 밤의 차이를 학습
  - 설정: brightness_limit=0.1, 대비 조정은 제외 (contrast_limit=0.0)

* HueSaturationValue

  - 목적: 조명에 따른 색상 변화에 대응, 낮은 값으로 조정
  - 설정: hue_shift_limit=2, sat_shift_limit=10, val_shift_limit=5

* Sharpen

  - 목적: 가까이서 찍힌 이미지에서 보행자를 더 명확히 탐지하기 위해 선명도 보정
  - 설정: alpha=(0.1, 0.3), 20% 확률로 적용
   
## 결과
![0001-20240628-123941](https://github.com/user-attachments/assets/8ffb5b36-1792-41ae-b1b8-871d6822d185)
* 라벨: {0:무단횡단보행자, 1:횡단보도보행자, 2:인도보행자}
### | 평가 데이터 추론 결과
* Precision : 0.9278
* Recall : 0.8603
* Accuracy : 0.7220
### | 성과
- **우수상 수상**

## 프로젝트 회고
### | 어려웠던 점
* CCTV 영상과 작은 객체 인식 학습의 어려움
  * CCTV 영상 특징 상 화질 문제, 같은 장소, 낮과 밤의 변화 등 학습에 영향을 주는 요인이 많았음
* 원본 영상 학습 성능과 전처리, 증강 데이터 학습 성능의 미미한 차이
  * 전처리와 증강을 통한 학습을 수행하였지만 <b>전처리 전 vs 후</b>의 결과는 전이 더 좋았고 <b>전처리 전+증강 vs 전처리 후+증강</b>은 전처리 후+증강의 결과가 더 좋았어서 학습 데이터셋 구성 과정이 어려웠음
  * 성능 자체도 미미하게 좋아지거나 나빠지는 차이로 좀 더 개선적인 방안이 필요
 ### | 배운 점
 * CCTV 영상의 객체가 멀리 있는 영상, 다소 가깝게 있는 영상 등 같은 CCTV 영상이라도 다양한 환경에 맞게 전처리 하는 과정을 시도해 볼 수 있는 경험을 함
 * 보행자라는 객체의 형태 자체는 비슷하지만 각 라벨에 맞게 분류가 은근히 잘 된 결과를 얻음
   * 하이퍼 파리미터를 좀 더 다양하게 조절한다거나 적절한 에포크 수치를 찾았다면 모델의 성능 개선을 기대할 수 있었음

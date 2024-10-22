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
* torch `2.4.1+cu121`
* ultralytics `8.3.13`
* wandb `0.18.3`
* opencv-python-headless `4.10.0.84`
### | Tool
* Git Hub
* Git Bash
* WandB
### | System
* Linux | Elice Cloud

## 디렉터리 정보
### | data
* 이미지 학습을 위한 데이터셋
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
### | code.ipynb
* 라이브러리 설치 및 모델 학습을 위한 notebook 코드
### | train.py
* 딥러닝 모델 학습을 위한 python 코드

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

### | 증강 기법
전처리된 이미지를 바탕으로 각각의 거리에 맞는 증강 기법을 적용하여 다양한 상황에서 보행자를 탐지할 수 있도록 데이터를 증강

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

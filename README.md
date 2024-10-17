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

## 결과

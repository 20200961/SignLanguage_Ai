# 수어 인식 AI 프로젝트

손동작을 인식하여 수어를 실시간으로 번역하는 딥러닝 기반 프로젝트입니다.

## 프로젝트 소개
```
이 프로젝트는 MediaPipe와 LSTM 모델을 활용하여 한국 수어를 인식하고 번역합니다. 웹캠을 통해 실시간으로 손동작을 감지하고, 학습된 모델을 통해 해당하는 수어 단어를 인식합니다.
```
## 주요 기능
```
실시간 수어 인식: 웹캠을 통한 실시간 손동작 인식
게임 모드: 제시된 수어를 따라하며 학습할 수 있는 인터랙티브 게임
25개 단어 지원: 기본적인 일상 단어들을 수어로 인식
높은 정확도: 100%에 가까운 검증 정확도 달성
```


## 기술 스택
```
Python 3.10
TensorFlow/Keras: 딥러닝 모델 구축
MediaPipe: 손과 포즈 랜드마크 감지
OpenCV: 실시간 비디오 처리
Flask: 웹 서버 구축
NumPy: 데이터 처리
```

## 설치 방법
```
(1) 저장소 클론
bashgit clone [repository-url]
cd [project-directory]

(2) 필요한 패키지 설치
bashpip install tensorflow opencv-python mediapipe flask flask-cors pillow numpy

(3) 한글 폰트 설치
Windows: malgun.ttf (맑은 고딕)
프로젝트 루트 디렉토리에 위치시키기
```

## 사용 방법
```
(1) 데이터 수집
bashpython MkDataset.py

각 단어당 30초간 동작을 수집합니다
5초 카운트다운 후 자동으로 수집이 시작됩니다

(2) 모델 학습
bashjupyter notebook MkLstm.ipynb

노트북의 모든 셀을 순서대로 실행합니다
학습된 모델은 models/final.h5로 저장됩니다

(3) 모델 변환 (옵션)
bashpython ConvertTflite.py

H5 모델을 TFLite 형식으로 변환합니다

(4) 테스트 실행
python TestSL.py

웹캠을 통해 실시간으로 수어를 인식합니다
'q' 키를 눌러 종료할 수 있습니다

(5) 게임 모드 실행
cd SLGame
python SLGame.py
```

- 브라우저에서 `http://localhost:5001` 접속
- 제시된 단어를 수어로 표현하면 정답 여부를 확인할 수 있습니다

## 프로젝트 구조
```
├── MkDataset.py          # 데이터 수집 스크립트
├── MkLstm.ipynb          # 모델 학습 노트북
├── TestSL.py             # 실시간 테스트 프로그램
├── ConvertTflite.py      # 모델 변환 스크립트
├── SLGame/
│   ├── SLGame.py         # Flask 서버
│   ├── templates/
│   │   └── game1.html    # 게임 UI
│   └── final.tflite      # 변환된 모델
└── dataset/              # 수집된 데이터셋

- 지원 단어 목록
안녕하세요, 감사합니다, 사랑합니다, 어머니, 아버지, 동생, 잘, 못, 간다, 나, 이름, 만나다, 반갑다, 부탁, 학교, 생일, 월, 일, 나이, 복습, 학습, 눈치, 오다, 말, 곱다

- 모델 상세 아키텍처
입력: 30 프레임 × 225 특징점 (양손 + 상체 포즈)
LSTM Layer: 64 units
Dense Layer: 32 units (ReLU)
출력: 25 classes (Softmax)

- 학습 파라미터
Optimizer: Adam
Loss: Categorical Crossentropy
Batch Size: 32
Epochs: 200 (조기 종료 적용)
학습/검증 분할: 90/10
```

## 주의사항
```
웹캠이 제대로 연결되어 있는지 확인하세요
충분한 조명이 있는 환경에서 사용하세요
손동작이 카메라에 잘 보이도록 위치를 조정하세요
게임 모드 실행 시 5001 포트가 사용 가능한지 확인하세요
```
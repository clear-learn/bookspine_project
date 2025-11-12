# 알라딘 책등인식 YOLObook
## 1. 목적
이 프로젝트는 Ultralytics YOLOv11 기반으로 이미지 내 책등에 대한 Segmentation을 학습하고, 추론 단계에서 생성된 마스크(mask)로부터 **회전된 최소 면적 사각형(Rotated Bounding Box)**을 추출 및 시각화하는 것을 목표로 합니다.

## 2. 실험 배경
표준 Bounding Box는 회전된 객체를 정확하게 표현하지 못하는 한계가 있습니다. 이 실험은 yolov11의 Segmentation 출력을 후처리하여 객체의 방향까지 고려한 Rotated Bounding Box를 생성하는 방법론을 검증합니다.

## 3. 실행 흐름
환경 구성: Dockerfile을 이용해 실험에 필요한 PyTorch, OpenCV, Ultralytics 등의 라이브러리가 설치된 Docker 이미지를 빌드합니다.

모델 학습: src/train.py 스크립트를 실행하여 coco_2.yaml에 정의된 데이터셋으로 yolov11 Segmentation 모델을 학습시키고, 결과 가중치(.pt 파일)를 저장합니다.

추론 및 시각화: 학습된 가중치를 src/inference.py 스크립트에 로드합니다. 이 스크립트는 지정된 폴더의 이미지들을 읽어 각 객체의 마스크를 예측하고, cv2.minAreaRect를 이용해 Rotated Bounding Box를 계산하여 결과 이미지에 시각화한 후 저장합니다.

## 4. 주요 코드 설명
### 4.1. 모델 학습 (src/train.py)
yolov11의 train 함수를 사용하여 모델을 학습시키는 스크립트입니다. 데이터셋 경로, 에포크, 이미지 크기, 배치 사이즈 등의 하이퍼파라미터를 설정합니다.

```bash
from ultralytics import YOLO

# 사전 학습된 모델 또는 이전 실험에서 학습된 모델 경로를 지정
model = YOLO("yolov11n-seg.pt")
model.resume = True # 이전 학습 상태에서 재개 (옵션)

# 모델 학습 실행
results = model.train(
    data="yaml path",
    epochs=200,
    imgsz=864,
    device=[1, 0], # 사용할 GPU 장치 ID
    batch=20,
    save_period=1, # 1 에포크마다 모델 저장
    # ... 기타 augmentation 파라미터 ...
)
```
### 4.2. 추론 및 후처리 (src/inference.py)
학습된 모델을 사용하여 추론을 수행하고, 그 결과로부터 Rotated Bounding Box를 추출하는 핵심 로직을 포함합니다.

```bash

if __name__ == "__main__":
    # 1. 학습된 모델 가중치 로드
    model = YOLO("weight path ")
    
    # 2. 입출력 폴더 설정
    input_folder = "input image path"
    output_folder = ""output image path"
    os.makedirs(output_folder, exist_ok=True)

    # 3. 입력 폴더의 각 이미지에 대해 추론 실행
    for img_name in os.listdir(input_folder):
        img_path = os.path.join(input_folder, img_name)
        
        # YOLO 모델로 예측하여 마스크 결과 획득
        results = model.predict(img_path, retina_masks=True, ...)

        # 4. 후처리 함수를 호출하여 Rotated BBox 계산 및 결과 저장
        visualize_rotated_rect(
            image_path=img_path, 
            output=results,
            save_path=os.path.join(output_folder, f"rotated_{img_name}")
        )
    
    # 5. 전체 성능 지표 출력
    print(f"Average Inference Time: {avg_inference:.3f} ms")
    # ...
```

### 4.3. 실행 환경 (Dockerfile)
실험의 재현성을 보장하기 위해 PyTorch, CUDA, 필수 라이브러리 등을 정의한 Dockerfile입니다. 이를 통해 누구나 동일한 환경에서 코드를 실행할 수 있습니다.
```bash
# 베이스 이미지 설정
FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-devel

# 필수 패키지 및 라이브러리 설치
RUN apt-get update && apt-get install -y libgl1-mesa-glx python3-pip
COPY requirements.txt .
RUN python -m pip install -r requirements.txt

# 코드 복사
WORKDIR /app
COPY ./src /app/src

# 컨테이너 실행 시 bash 터미널 접속
ENTRYPOINT ["/bin/bash"]
```


- 참고: https://github.com/ultralytics/ultralytics

- weight: https://nas.aladin.co.kr/drive/d/f/12ZWNxvvDt4Uq5S2Afs3DEnrTCUg3EIt

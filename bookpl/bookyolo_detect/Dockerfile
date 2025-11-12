FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-devel

WORKDIR /opt/ml/code

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx tzdata && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 앱 복사
COPY app ./app

# SageMaker가 /opt/ml/model/ 에 YOLO 가중치를 주입
ENV MODEL_DIR=/opt/ml/model
ENV MODEL_FILE=bookYOLO_seg_model.pt
ENV HOST=0.0.0.0
ENV PORT=8080
ENV YOLO_DEVICE=0

EXPOSE 8080
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "1"]

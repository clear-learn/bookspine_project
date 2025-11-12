# 베이스 이미지 설정
FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-devel

# 타임존 및 필수 패키지 설치
ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Seoul

# 필수 패키지 설치
RUN apt-get update && apt-get install -y \
    software-properties-common \
    build-essential \
    python3-pip \
    libgl1-mesa-glx \
    curl \
    git \
    wget \
    vim \
    tzdata

# pip 업그레이드
RUN python -m pip install --upgrade pip

# requirements.txt 복사 및 패키지 설치
COPY requirements.txt /opt/ml/code/requirements.txt
RUN python -m pip install -r /opt/ml/code/requirements.txt

RUN pip install --upgrade numpy
RUN pip install --upgrade ultralytics

# 외부 코드 파일
COPY sagemaker_inference.py /opt/ml/code/sagemaker_inference.py
COPY bookYOLO.py /opt/ml/code/bookYOLO.py
COPY image.py /opt/ml/code/image.py

# 내부 코드 파일
COPY bookYOLO.py /opt/ml/code/
COPY models /opt/ml/code/models
COPY utils /opt/ml/code/utils

#ENTRYPOINT ["python", "/opt/ml/code/sagemaker_inference.py"]
ENTRYPOINT ["/bin/bash"]
CMD ["-i"]

# Python 3.10.16 기반 이미지 사용
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# Python 3.10.16 설치 (deadsnakes PPA 사용)
RUN apt-get update && apt-get install -y software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -y python3.10=3.10.16* python3.10-distutils python3.10-dev python3-pip && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1 && \
    update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1

# pip 업그레이드
RUN python -m pip install --upgrade pip

# 작업 디렉토리 설정
WORKDIR /app

# 환경변수 설정
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app 

# 시스템 패키지 설치 (한글 폰트 및 CUDA 라이브러리 포함)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    git \
    fonts-nanum \
    fonts-noto-cjk \
    locales \
    fontconfig \
    fontconfig-config \
    cuda-libraries-11-8 \
    cuda-nvtx-11-8 \
    cuda-cudart-11-8 \
    libcufft-11-8 \
    libcurand-11-8 \
    libcusolver-11-8 \
    libcusparse-11-8 \
    libcublas-11-8 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# 한글 로케일 설정
RUN sed -i -e 's/# ko_KR.UTF-8 UTF-8/ko_KR.UTF-8 UTF-8/' /etc/locale.gen && \
    locale-gen
ENV LANG=ko_KR.UTF-8
ENV LC_ALL=ko_KR.UTF-8

# 폰트 캐시 갱신
RUN fc-cache -fv

# Poetry 설치
RUN pip install poetry==2.1.1

# Poetry 가상환경 생성하지 않도록 설정
RUN poetry config virtualenvs.create false

# 프로젝트 파일 복사
COPY pyproject.toml ./
COPY poetry.lock* ./

# 종속성 설치
RUN poetry install --no-interaction --no-ansi --no-root

# data 디렉토리 생성
RUN mkdir -p /app/data

# app.py 수정 (GPU 활성화)
COPY . .
RUN sed -i 's/os.environ\["CUDA_VISIBLE_DEVICES"\] = "-1"/os.environ\["CUDA_VISIBLE_DEVICES"\] = "0"/g' app.py

# 포트 설정
EXPOSE 8777

# 실행 명령어
CMD ["streamlit", "run", "app.py", "--server.port", "8777", "--server.address", "0.0.0.0", "--server.fileWatcherType=none"]
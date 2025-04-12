# CUDA 베이스 이미지
FROM nvidia/cuda:12.2.2-cudnn8-runtime-ubuntu22.04

# 비대화형 모드 설정
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Seoul

# TensorFlow GPU 관련 환경변수 추가
ENV CUDA_VISIBLE_DEVICES=0
ENV TF_FORCE_GPU_ALLOW_GROWTH=true

# 필요한 기본 패키지 설치
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libssl-dev \
    zlib1g-dev \
    libbz2-dev \
    libreadline-dev \
    libsqlite3-dev \
    wget \
    curl \
    llvm \
    libncurses5-dev \
    xz-utils \
    tk-dev \
    libxml2-dev \
    libxmlsec1-dev \
    libffi-dev \
    liblzma-dev \
    git \
    ca-certificates \
    libpq-dev \
    fonts-nanum \
    fonts-noto-cjk \
    locales \
    fontconfig \
    fontconfig-config \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# 한글 로케일 설정
RUN sed -i -e 's/# ko_KR.UTF-8 UTF-8/ko_KR.UTF-8 UTF-8/' /etc/locale.gen && \
    locale-gen
ENV LANG=ko_KR.UTF-8
ENV LC_ALL=ko_KR.UTF-8

# 폰트 캐시 갱신
RUN fc-cache -fv

# pyenv 설치 (비루트 사용자로 설치하는 것이 권장되지만, 간단하게 루트로 진행)
ENV PYENV_ROOT=/root/.pyenv
ENV PATH=$PYENV_ROOT/bin:$PATH
RUN curl https://pyenv.run | bash

# Python 3.10.16 설치 (정확한 버전)
RUN eval "$(pyenv init -)" && \
    pyenv install 3.10.16 && \
    pyenv global 3.10.16 && \
    pip install --upgrade pip

# pyenv 설정을 글로벌 쉘에 추가
RUN echo 'export PYENV_ROOT="/root/.pyenv"' >> /root/.bashrc && \
    echo 'export PATH="$PYENV_ROOT/bin:$PATH"' >> /root/.bashrc && \
    echo 'eval "$(pyenv init -)"' >> /root/.bashrc && \
    echo 'eval "$(pyenv init --path)"' >> /root/.bashrc

# 작업 디렉토리 설정
WORKDIR /app

# 환경변수 설정
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

# Poetry 설치
RUN eval "$(pyenv init -)" && \
    pip install poetry==2.1.1

# Poetry 가상환경 생성하지 않도록 설정
RUN eval "$(pyenv init -)" && \
    poetry config virtualenvs.create false

# 프로젝트 파일 복사
COPY pyproject.toml ./
COPY poetry.lock* ./

# 종속성 설치
RUN eval "$(pyenv init -)" && \
    poetry install --no-interaction --no-ansi --no-root

# data 디렉토리 생성
RUN mkdir -p /app/data

# 소스코드 복사
COPY . .

# app.py 수정 (GPU 활성화)
RUN sed -i 's/os.environ\["CUDA_VISIBLE_DEVICES"\] = "-1"/os.environ\["CUDA_VISIBLE_DEVICES"\] = "0"/g' app.py

# 포트 설정
EXPOSE 8777

# 실행 명령어 설정 (pyenv 환경 활성화 포함)
CMD ["bash", "-c", "eval \"$(pyenv init -)\" && streamlit run app.py --server.port 8777 --server.address 0.0.0.0 --server.fileWatcherType=none"]
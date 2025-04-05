# Python 3.10.16 기반 이미지 사용
FROM python:3.10.16-slim

# 작업 디렉토리 설정
WORKDIR /app

# 환경변수 설정
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app 

# 시스템 패키지 설치
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    git \
    fonts-nanum \         # 나눔 폰트 추가
    fonts-noto-cjk \      # CJK 폰트 추가 (한중일 문자 지원)
    language-pack-ko \    # 한국어 언어팩 추가
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

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

# 소스코드 복사
COPY . .

# 포트 설정
EXPOSE 8777

# 실행 명령어
CMD ["streamlit", "run", "app.py", "--server.port", "8777", "--server.address", "0.0.0.0"]
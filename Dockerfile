# 베이스 이미지로 Python 3.11 사용
FROM python:3.11-slim

# 작업 디렉토리 설정
WORKDIR /app

# 필요한 시스템 패키지 설치
RUN apt-get update && apt-get install -y \
    build-essential \
    wget \
    && rm -rf /var/lib/apt/lists/*

# 한글 폰트 설치
RUN apt-get update && apt-get install -y fonts-nanum \
    && rm -rf /var/lib/apt/lists/*

# 파이썬 환경 설정
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# 필요한 파이썬 패키지 설치
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 애플리케이션 코드 복사
COPY . .

# 데이터 디렉토리 생성
RUN mkdir -p /app/data

# 포트 설정
EXPOSE 8777

# 환경 변수 설정
ENV STREAMLIT_SERVER_PORT=8777
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0

# Streamlit 앱 실행
CMD ["streamlit", "run", "app.py", "--server.port=8777", "--server.address=0.0.0.0"]
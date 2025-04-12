# 서울시 대기질 시계열 분석 애플리케이션

서울시 IoT 기반 대기질 데이터를 활용한 시계열 분석 및 예측 웹 애플리케이션입니다.

## 주요 기능

- **데이터 탐색**: 서울시 대기질 데이터의 기본 통계 및 시각화 제공
- **시계열 분해**: 추세(Trend), 계절성(Seasonality), 불규칙성(Irregularity) 분석
- **정상성 검정**: ADF 검정을 통한 시계열 데이터 정상성 분석
- **ACF/PACF 분석**: 자기상관함수 및 부분 자기상관함수 분석
- **모델 비교**: ARIMA/SARIMA, 지수평활법, Prophet, LSTM, Transformer 등 다양한 예측 모델 지원
- **예측 성능 평가**: RMSE, MAE, R² 등 다양한 메트릭 기반 평가
- **LLM 분석**: 인공지능을 활용한 시계열 데이터 및 예측 결과 자동 분석

## 프로젝트 구조

```
project_root/
  ├── app.py                      # 메인 애플리케이션 진입점
  ├── config/                     # 설정 파일
  │   ├── __init__.py
  │   └── settings.py             # 환경설정 관리
  ├── models/                     # 시계열 예측 모델
  │   ├── __init__.py
  │   ├── base_model.py           # 기본 모델 클래스
  │   ├── arima_model.py          # ARIMA/SARIMA 모델
  │   ├── exp_smoothing_model.py  # 지수평활법 모델
  │   ├── prophet_model.py        # Prophet 모델
  │   ├── lstm_model.py           # LSTM 모델
  │   └── model_factory.py        # 모델 팩토리
  ├── backend/                    # 백엔드 로직
  │   ├── __init__.py
  │   ├── data_service.py         # 데이터 로딩/처리 관련
  │   ├── model_service.py        # 모델 학습 관련
  │   ├── visualization_service.py # 시각화 관련
  │   └── llm_service.py          # LLM 분석 관련
  ├── frontend/                   # 프론트엔드 로직
  │   ├── __init__.py
  │   ├── sidebar.py              # 사이드바 구성
  │   ├── components.py           # 공통 UI 컴포넌트
  │   └── session_state.py        # 세션 상태 관리
  ├── streamlit_pages/            # 페이지 파일들
  │   ├── __init__.py
  │   ├── home.py            # 홈 페이지
  │   ├── visualization.py   # 시계열 시각화
  │   ├── decomposition.py   # 시계열 분해
  │   ├── stationarity.py    # 정상성 & ACF/PACF
  │   ├── modeling.py        # 모델 학습/예측
  │   └── llm_analysis.py    # LLM 분석
  ├── utils/                      # 유틸리티 함수
  │   ├── __init__.py
  │   ├── singleton.py            # 싱글톤 패턴 구현
  │   ├── data_reader.py          # 데이터 수집 모듈
  │   ├── data_processor.py       # 데이터 전처리 모듈
  │   ├── visualizer.py           # 시각화 모듈
  │   └── llm_connector.py        # LLM 연동 모듈
  ├── prompts/                    # LLM 프롬프트
  │   ├── __init__.py
  │   └── time_series_analysis_prompt.py # 시계열 분석 프롬프트
  ├── deploy/                     # 배포 관련 파일
  ├── data/                       # 데이터 저장 디렉토리
  ├── .dockerignore
  ├── .gitignore
  ├── Dockerfile                  # Docker 컨테이너 빌드 파일
  ├── pyproject.toml              # 프로젝트 의존성 관리
  └── README.md
```

## 설치 및 실행 방법

### 로컬 환경에서 실행

1. 파이썬 3.10 이상 설치
2. Poetry 설치 (의존성 관리 도구)
3. 프로젝트 클론 및 의존성 설치

```bash
git clone <repository-url>
cd time-series-analysis-app
poetry install
```

4. 환경 변수 설정 (.env 파일 생성)

```
SEOUL_API_KEY=<서울시 OpenAPI 키>
SEOUL_API_BASE_URL=http://openAPI.seoul.go.kr:8088
SEOUL_AIR_QUALITY_SERVICE=TimeAverageAirQuality
OLLAMA_SERVER=<Ollama 서버 URL>
OLLAMA_MODEL=<사용할 모델명>
```

5. 애플리케이션 실행

```bash
poetry run streamlit run app.py
```

### Docker로 실행

```bash
# 이미지 빌드
docker build -t time-series-app .

# 컨테이너 실행
docker run -p 8777:8777 --env-file .env time-series-app
```

### Kubernetes로 배포

```bash
cd deploy/time-series-app
helm install time-series-app .
```

## 기술 스택

- **백엔드**: Python 3.10+, Streamlit
- **시계열 모델링**: 
  - ARIMA/SARIMA (statsmodels, pmdarima)
  - 지수평활법 (statsmodels)
  - Prophet (Facebook)
  - LSTM (TensorFlow)
  - Transformer (TensorFlow)
- **데이터 처리**: pandas, numpy
- **시각화**: plotly, matplotlib, seaborn
- **LLM 분석**: OpenAI API (Ollama 연동)

## 환경 요구사항

- Python 3.10+
- 최소 4GB RAM 권장 (LSTM 모델 사용 시 8GB 이상 권장)

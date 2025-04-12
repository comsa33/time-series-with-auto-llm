# 서울시 대기질 시계열 분석 및 LLM 기반 지능형 예측 시스템

[![Python 3.10](https://img.shields.io/badge/Python-3.10-blue.svg)](https://www.python.org/downloads/release/python-3100/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.44.1-red.svg)](https://streamlit.io/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15.0-orange.svg)](https://www.tensorflow.org/)

서울시 IoT 기반 대기질 데이터를 활용한 고급 시계열 분석 및 LLM 기반 지능형 예측 시스템입니다. 본 프로젝트는 석사학위 논문 및 학술지 게재를 위한 연구의 일환으로 개발되었습니다.

[🔗 시계열 연구 앱 바로가기](https://time-series-analysis.po24lio.com)

## 📝 연구 개요

본 연구는 대기질 예측의 정확성 향상과 시계열 분석의 자동화를 위해 기존 통계적 모델과 딥러닝 모델을 결합하고, LLM을 활용하여 모델 최적화 및 분석 자동화를 구현한 프레임워크를 제안합니다.

### 연구 목적
- 다양한 시계열 예측 모델의 성능 비교 분석 및 평가
- LLM을 활용한 하이퍼파라미터 최적화 자동화 프레임워크 구축
- 시계열 데이터 분석 인사이트 자동 추출 시스템 개발
- 대기질 예측 정확도 향상을 위한 최적 모델링 방법론 제시

### 학술적 의의
- 전통적 통계 모델과 딥러닝 모델의 비교 분석을 통한 대기질 예측 모델링 체계 확립
- LLM을 활용한 하이퍼파라미터 최적화 자동화로 전문가 의존도 감소
- 다변량 시계열 분석과 구조적 변화점 탐지를 통한 대기질 데이터 심층 이해
- 대기질 모니터링 및 예측을 위한 통합 분석 프레임워크 제안

## 🌟 주요 기능

### 데이터 수집 및 전처리
- **실시간 API 연동**: 서울시 OpenAPI를 통한 대기질 데이터 실시간 수집
- **자동 전처리**: 결측치 처리, 이상치 탐지, 시계열 정규화

### 고급 시계열 분석
- **시계열 분해**: 추세(Trend), 계절성(Seasonality), 불규칙성(Residual) 분리 분석
- **정상성 검정**: ADF 및 KPSS 검정을 통한 다중 정상성 분석
- **ACF/PACF 분석**: 자기상관함수 및 부분 자기상관함수 분석
- **차분 분석**: 최적 차분 수준 자동 추천 및 적용
- **구조적 변화점 탐지**: 시계열 데이터의 구조적 변화 시점 자동 탐지
- **다변량 분석**: 대기질 변수 간 상관관계 및 Granger 인과성 분석

### 다중 모델 시계열 예측
- **전통적 통계 모델**: ARIMA/SARIMA, 지수평활법(Exponential Smoothing)
- **머신러닝 모델**: Prophet(Facebook)
- **딥러닝 모델**: LSTM(Long Short-Term Memory), Transformer

### LLM 기반 자동화 및 지능화
- **하이퍼파라미터 최적화**: LLM 기반 모델별 최적 하이퍼파라미터 자동 추천
- **모델 성능 분석**: 다양한 메트릭(RMSE, MAE, R², MAPE)을 통한 모델 자동 비교 분석
- **시계열 인사이트 추출**: 데이터 특성 및 예측 결과에 대한 자동 분석 리포트 생성
- **전문가 수준 해석**: 모델 선택 및 결과 해석에 대한 AI 기반 가이드 제공

## 🏗️ 프로젝트 아키텍처

```
project_root/
  ├── app.py                      # 애플리케이션 진입점
  ├── config/                     # 설정 파일
  │   ├── __init__.py
  │   └── settings.py             # 환경설정 관리
  ├── models/                     # 시계열 예측 모델
  │   ├── __init__.py
  │   ├── base_model.py           # 기본 모델 클래스 (추상화)
  │   ├── arima_model.py          # ARIMA/SARIMA 모델
  │   ├── exp_smoothing_model.py  # 지수평활법 모델
  │   ├── prophet_model.py        # Prophet 모델
  │   ├── lstm_model.py           # LSTM 모델
  │   ├── transformer_model.py    # Transformer 모델
  │   └── model_factory.py        # 모델 팩토리 패턴 구현
  ├── backend/                    # 백엔드 로직
  │   ├── __init__.py
  │   ├── data_service.py         # 데이터 로딩/처리 서비스
  │   ├── model_service.py        # 모델 학습/예측 서비스
  │   ├── visualization_service.py # 시각화 서비스
  │   └── llm_service.py          # LLM 연동 서비스
  ├── frontend/                   # 프론트엔드 로직
  │   ├── __init__.py
  │   ├── sidebar.py              # 사이드바 구성
  │   ├── components.py           # 공통 UI 컴포넌트
  │   └── session_state.py        # 세션 상태 관리
  ├── streamlit_pages/            # 페이지 컴포넌트
  │   ├── introduction.py         # 메인 페이지
  │   ├── time-series-graph.py    # 시계열 시각화
  │   ├── decomposition.py        # 시계열 분해
  │   ├── stationarity.py         # 정상성 & ACF/PACF
  │   ├── differencing.py         # 차분 분석
  │   ├── change_point_analysis.py # 구조적 변화점 분석
  │   ├── multivariate_analysis.py # 다변량 분석
  │   ├── modeling.py             # 모델 학습/예측
  │   ├── hyperparameter_optimization.py # 하이퍼파라미터 최적화
  │   └── llm_analysis.py         # LLM 분석 리포트
  ├── utils/                      # 유틸리티 모듈
  │   ├── __init__.py
  │   ├── singleton.py            # 싱글톤 패턴 구현
  │   ├── data_reader.py          # 데이터 수집 모듈
  │   ├── data_processor.py       # 데이터 전처리 모듈
  │   ├── visualizer.py           # 시각화 모듈
  │   ├── parameter_utils.py      # 파라미터 유효성 검증
  │   └── llm_connector.py        # LLM 연동 모듈
  ├── prompts/                    # LLM 프롬프트 템플릿
  │   ├── __init__.py
  │   ├── time_series_analysis_prompt.py     # 시계열 분석 프롬프트
  │   ├── hyperparameter_recommendation_prompt.py # 하이퍼파라미터 추천 프롬프트
  │   └── model_parameter_tuning_prompt.py   # 모델 파라미터 조정 프롬프트
  ├── deploy/                     # 배포 관련 파일
  │   └── time-series-app/        # Helm 차트
  ├── data/                       # 데이터 저장 디렉토리
  ├── .dockerignore
  ├── .gitignore
  ├── Dockerfile                  # Docker 컨테이너 빌드 파일
  ├── pyproject.toml              # 프로젝트 의존성 관리
  └── README.md
```

## 🔄 워크플로우

1. **데이터 수집**: 서울시 OpenAPI에서 실시간 대기질 데이터 수집
2. **데이터 전처리**: 결측치 처리, 이상치 탐지, 시계열 정규화
3. **시계열 분석**: 추세-계절성 분해, 정상성 검정, ACF/PACF 분석, 차분 분석
4. **모델 학습**: 다양한 시계열 모델(ARIMA, 지수평활법, Prophet, LSTM, Transformer) 학습
5. **LLM 기반 최적화**: 모델 성능 분석 및 하이퍼파라미터 최적화 추천
6. **성능 평가**: RMSE, MAE, R², MAPE 등 다양한 메트릭을 통한 모델 평가
7. **LLM 분석 리포트**: 데이터 특성 및 예측 결과에 대한 자동 분석 리포트 생성

## 📊 연구 결과

본 연구를 통해 다음과 같은 주요 결과를 도출하였습니다:

1. 대기질 데이터에서 Transformer 모델이 가장 우수한 예측 성능을 보임
2. LLM 기반 하이퍼파라미터 최적화를 통해 모델 성능 10-15% 향상
3. 차분 분석 적용 시 전통 통계 모델(ARIMA, 지수평활법)의 성능 개선
4. 다변량 분석을 통해 대기질 변수 간 상관관계 및 인과성 패턴 발견
5. 구조적 변화점 탐지를 통한 대기질 패턴 변화 시점 자동 식별

## 🛠️ 설치 및 실행 방법

### 로컬 환경에서 실행

1. Python 3.10 이상 설치 (특정 버전 필요)
```bash
# pyenv 사용 시
pyenv install 3.10.16
pyenv local 3.10.16
```

2. Poetry 설치 (의존성 관리 도구)
```bash
curl -sSL https://install.python-poetry.org | python3 -
```

3. 프로젝트 클론 및 의존성 설치
```bash
git clone <repository-url>
cd time-series-analysis-app
poetry install
```

4. 환경 변수 설정 (.env 파일 생성)
```env
SEOUL_API_KEY=<서울시 OpenAPI 키>
SEOUL_API_BASE_URL=http://openAPI.seoul.go.kr:8088
SEOUL_AIR_QUALITY_SERVICE=TimeAverageAirQuality
OLLAMA_SERVER=<Ollama 서버 URL>  # 예: http://localhost:11434
OLLAMA_MODEL=<사용할 모델명>      # 예: gemma3:27b
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
# 필요시 values.yaml 수정
helm install time-series-app .
```

## 🔧 기술 스택

### 백엔드
- **Python 3.10+**: 코어 로직 구현
- **Streamlit 1.44+**: 웹 인터페이스 프레임워크
- **pandas & numpy**: 데이터 처리 및 수치 연산
- **statsmodels**: 통계 모델링 및 검정

### 시계열 모델링
- **ARIMA/SARIMA**: 자기회귀 통합 이동평균 모델 (statsmodels, pmdarima)
- **지수평활법**: Holt-Winters 등 지수평활법 (statsmodels)
- **Prophet**: Facebook의 시계열 예측 라이브러리
- **LSTM**: Long Short-Term Memory 네트워크 (TensorFlow)
- **Transformer**: 자기 주의 메커니즘 기반 모델 (TensorFlow)

### 데이터 시각화
- **Plotly**: 인터랙티브 시각화
- **matplotlib & seaborn**: 통계적 시각화

### LLM 연동
- **OpenAI API 호환 인터페이스**: Ollama 연동
- **Custom Prompt Engineering**: 시계열 분석 및 하이퍼파라미터 튜닝 최적화

### 배포
- **Docker**: 컨테이너화
- **Kubernetes**: 오케스트레이션
- **Helm**: 패키지 관리

## 🌐 활용 가능 분야

- **환경 모니터링**: 대기질 예측 및 환경 오염 분석
- **정책 결정 지원**: 데이터 기반 환경 정책 수립
- **공중 보건**: 대기질과 건강 영향 관계 분석
- **스마트 시티**: 도시 환경 모니터링 및 제어
- **교통 관리**: 교통량과 대기질 관계 분석 및 예측

## 🔜 향후 연구 방향

- **멀티모달 데이터 통합**: 기상 데이터, 교통량 등 추가 데이터 소스 통합
- **공간적 분석 확장**: 지리적 위치 정보를 활용한 공간-시간 분석
- **LLM 기반 자동 모델 선택**: 데이터 특성에 따른 최적 모델 자동 선택 시스템
- **인과관계 분석 강화**: 구조적 인과 모델링 및 반사실적 분석 도입
- **실시간 알림 시스템**: 예측 결과 기반 이상치 감지 및 알림 시스템 구현

## 📝 인용 정보

본 연구를 학술적 목적으로 인용하실 경우 다음 형식을 사용해 주세요:

```
이루오, "LLM 기반 시계열 분석 자동화 프레임워크: 서울시 대기질 데이터 예측 사례 연구", 
서울과학종합대학원대학교, AI Big Data 전공 석사학위논문, 2025.
```

## 📄 라이센스

이 프로젝트는 MIT 라이센스 하에 제공됩니다. 자세한 내용은 [LICENSE](LICENSE) 파일을 참조하세요.

## 📞 연락처

- 이름: 이루오(Ruo Lee)
- 이메일: comsa333@gmail.com
- GitHub: [comsa33](https://github.com/comsa33)

## 🙏 감사의 글

이 연구는 aSSIST(서울과학종합대학원대학교) AI Big Data 전공 석사학위 논문 연구의 일환으로 진행되었습니다. 자문과 지도를 제공해 주신 교수님들과 서울시 공공데이터를 제공해 주신 서울시에 감사드립니다.
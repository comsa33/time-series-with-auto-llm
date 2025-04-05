"""
서울시 대기질 시계열 분석 메인 Streamlit 앱
"""
import os
import warnings
from datetime import datetime, timedelta

import streamlit as st
import pandas as pd
import numpy as np

from config.settings import app_config
from utils.data_reader import get_seoul_air_quality
from utils.data_processor import DataProcessor
from utils.visualizer import TimeSeriesVisualizer
from utils.llm_connector import LLMConnector
from prompts.time_series_analysis_prompt import TIME_SERIES_ANALYSIS_PROMPT

import os
# GPU 사용 비활성화 (CPU 모드만 사용)
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
# TensorFlow 로그 레벨 조정
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# 경고 메시지 무시
warnings.filterwarnings('ignore')


# 페이지 설정
st.set_page_config(
    page_title=app_config.APP_TITLE,
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)


# 객체 초기화
data_processor = DataProcessor()
visualizer = TimeSeriesVisualizer()

# 세션 상태 초기화
def initialize_session_state():
    if 'df' not in st.session_state:
        st.session_state.df = None
    if 'data_source' not in st.session_state:
        st.session_state.data_source = "API에서 가져오기"
    if 'selected_station' not in st.session_state:
        st.session_state.selected_station = None
    if 'selected_target' not in st.session_state:
        st.session_state.selected_target = None
    if 'series' not in st.session_state:
        st.session_state.series = None
    if 'train' not in st.session_state:
        st.session_state.train = None
    if 'test' not in st.session_state:
        st.session_state.test = None
    if 'period' not in st.session_state:
        st.session_state.period = 24
    if 'decomposition' not in st.session_state:
        st.session_state.decomposition = None
    if 'stationarity_result' not in st.session_state:
        st.session_state.stationarity_result = None
    if 'acf_values' not in st.session_state:
        st.session_state.acf_values = None
    if 'pacf_values' not in st.session_state:
        st.session_state.pacf_values = None
    if 'forecasts' not in st.session_state:
        st.session_state.forecasts = {}
    if 'metrics' not in st.session_state:
        st.session_state.metrics = {}
    if 'models_trained' not in st.session_state:
        st.session_state.models_trained = False
    if 'selected_models' not in st.session_state:
        st.session_state.selected_models = []
    if 'test_size' not in st.session_state:
        st.session_state.test_size = app_config.DEFAULT_TEST_SIZE
    if 'best_model' not in st.session_state:
        st.session_state.best_model = None

# 모델 팩토리 동적 로드
@st.cache_resource
def get_model_factory():
    """
    모델 팩토리를 동적으로 로드합니다.
    필요할 때만 import하여 시작 시 pmdarima 오류를 방지합니다.
    """
    try:
        from models.model_factory import ModelFactory
        return ModelFactory()
    except Exception as e:
        st.error(f"모델 로드 중 오류 발생: {e}")
        st.error("필요한 라이브러리를 재설치하세요: pip uninstall -y pmdarima numpy && pip install numpy==1.24.3 && pip install pmdarima==2.0.4")
        return None

# 데이터 불러오기 함수
@st.cache_data(ttl=3600)
def load_data(file_path=None, start_date=None, end_date=None):
    """
    CSV 파일에서 데이터를 불러오거나, 파일이 없는 경우 API를 통해 데이터를 가져옵니다.
    """
    try:
        if file_path and os.path.exists(file_path):
            df = pd.read_csv(file_path)
            if 'MSRDT' in df.columns:
                df['MSRDT'] = pd.to_datetime(df['MSRDT'])
            return df
        else:
            st.info("데이터 파일을 찾을 수 없습니다. API를 통해 데이터를 가져옵니다.")
            
            if not start_date or not end_date:
                # 기본값: 최근 한 달
                end_date = datetime.now().strftime("%Y-%m-%d")
                start_date = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
            
            df = get_seoul_air_quality(app_config.SEOUL_API_KEY, start_date, end_date)
            
            if df is not None and not df.empty:
                # 파일로 저장
                os.makedirs(app_config.DATA_DIR, exist_ok=True)
                df.to_csv(app_config.DEFAULT_DATA_FILE, index=False, encoding='utf-8-sig')
                st.success(f"데이터가 성공적으로 저장되었습니다: {app_config.DEFAULT_DATA_FILE}")
            
            return df
    except Exception as e:
        st.error(f"데이터 로딩 중 오류 발생: {e}")
        return None

# 앱 헤더 함수
def render_header():
    """
    앱 헤더 렌더링
    """
    st.title("서울시 대기질 시계열 분석")
    st.markdown("서울시 IoT 데이터를 활용한 시계열 분석 앱")
    
    # 확장 가능한 앱 소개
    with st.expander("📌 App Introduction and Usage"):
        st.markdown("""
        ### 📌 앱 소개
        이 앱은 서울시 대기질 데이터를 활용하여 시계열 데이터를 분석하고 시각화하는 도구입니다.

        ### 🌟 주요 기능
        1. **데이터 탐색**: 서울시 대기질 데이터의 기본 통계 및 시각화 제공
        2. **시계열 분해**: 추세(Trend), 계절성(Seasonality), 불규칙성(Irregularity) 분석
        3. **모델 비교**: ARIMA/SARIMA, 지수평활법, Prophet, LSTM 등 다양한 예측 모델 지원
        4. **예측 성능 평가**: RMSE, MAE, R² 등 다양한 메트릭 기반 평가

        ### 🛠️ 사용 방법
        1. 사이드바에서 데이터 업로드 또는 API 수집 옵션 선택
        2. 분석할 측정소와 변수(PM10, PM25 등) 선택
        3. 시계열 분석 옵션 설정 후 모델 학습 실행
        4. 결과 탭에서 다양한 모델의 예측 결과 비교 및 분석
        """)

# 데이터 소스 변경 콜백
def on_data_source_change():
    st.session_state.df = None
    st.session_state.series = None
    st.session_state.train = None
    st.session_state.test = None
    st.session_state.models_trained = False

# 측정소/타겟 변경 콜백
def update_series():
    if st.session_state.df is not None:
        # 선택된 측정소와 타겟 변수에 따라 시계열 데이터 전처리
        st.session_state.series = data_processor.preprocess_data(
            st.session_state.df, 
            st.session_state.selected_target, 
            st.session_state.selected_station
        )
        # 모델 학습 상태 초기화
        st.session_state.train = None
        st.session_state.test = None
        st.session_state.models_trained = False

# 모델 학습 함수
def train_models():
    # 훈련/테스트 분할
    st.session_state.train, st.session_state.test = data_processor.train_test_split(
        st.session_state.series, 
        st.session_state.test_size
    )
    
    # 모델 팩토리 가져오기
    model_factory = get_model_factory()
    
    if model_factory is None:
        st.error("모델 팩토리를 로드할 수 없습니다. pmdarima 호환성 문제일 수 있습니다.")
        st.error("아래 명령어로 문제를 해결할 수 있습니다:")
        st.code("pip uninstall -y pmdarima numpy && pip install numpy==1.24.3 && pip install pmdarima==2.0.4")
        return
    
    # 진행 상황 표시
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # 예측 결과 및 메트릭 저장
    forecasts = {}
    metrics = {}
    
    # 모델 개수
    total_models = len(st.session_state.selected_models)
    completed_models = 0
    
    # 각 모델 학습 및 예측
    for model_type in st.session_state.selected_models:
        status_text.text(f"{model_type} 모델 학습 중...")
        
        try:
            # 모델 인스턴스 생성
            model = model_factory.get_model(model_type)
            
            # 모델별 학습 매개변수 설정
            if model_type == 'arima':
                # ARIMA 모델 파라미터
                forecast, model_metrics = model.fit_predict_evaluate(
                    st.session_state.train, 
                    st.session_state.test,
                    seasonal=True,
                    m=st.session_state.period
                )
            elif model_type == 'exp_smoothing':
                # 지수평활법 모델 파라미터
                forecast, model_metrics = model.fit_predict_evaluate(
                    st.session_state.train, 
                    st.session_state.test,
                    seasonal_periods=st.session_state.period
                )
            elif model_type == 'prophet':
                # Prophet 모델 파라미터
                forecast, model_metrics = model.fit_predict_evaluate(
                    st.session_state.train, 
                    st.session_state.test,
                    daily_seasonality=True,
                    weekly_seasonality=True
                )
            elif model_type == 'lstm':
                # LSTM 모델 파라미터
                forecast, model_metrics = model.fit_predict_evaluate(
                    st.session_state.train, 
                    st.session_state.test,
                    n_steps=min(48, len(st.session_state.train) // 10),  # 시퀀스 길이
                    lstm_units=[50, 50],
                    dropout_rate=0.2,
                    epochs=100,
                    batch_size=32,
                    validation_split=0.1
                )
            else:
                # 기본 파라미터
                forecast, model_metrics = model.fit_predict_evaluate(
                    st.session_state.train, 
                    st.session_state.test
                )
            
            # 예측 결과 및 메트릭 저장
            forecasts[model.name] = forecast
            metrics[model.name] = model_metrics
            
            # 진행 상황 업데이트
            completed_models += 1
            progress_bar.progress(completed_models / total_models)
            
        except Exception as e:
            st.error(f"{model_type} 모델 학습 중 오류 발생: {e}")
    
    # 모든 모델 학습 완료 후 결과 저장
    if forecasts:
        st.session_state.forecasts = forecasts
        st.session_state.metrics = metrics
        st.session_state.models_trained = True
        
        # 최적 모델 선택
        rmse_values = {model: metrics[model]['RMSE'] for model in metrics}
        st.session_state.best_model = min(rmse_values.items(), key=lambda x: x[1])[0]
        
        status_text.text("모든 모델 학습 완료!")
    else:
        st.error("모델 학습 중 오류가 발생했습니다.")

# 메인 함수
def main():
    # 세션 상태 초기화
    initialize_session_state()
    
    # 앱 헤더 렌더링
    render_header()
    
    # 사이드바 설정
    st.sidebar.header("📊 분석 설정")

    st.sidebar.markdown("---")
    
    # 데이터 로드
    st.sidebar.subheader("서울시 대기질 데이터 로드", help="서울시 IoT 대기질 데이터 API를 통해 데이터를 실시간으로 가져옵니다.")


    # 날짜 범위 선택
    today = datetime.now().date()  # datetime.date 객체로 변환
    default_end_date = today
    default_start_date = today - timedelta(days=30)

    st.sidebar.markdown("##### 📅 분석 기간 선택", help="시계열 분석을 위한 데이터 기간을 선택하세요. (최대 30일)")

    date_col1, date_col2 = st.sidebar.columns(2)

    with date_col1:
        start_date = st.date_input(
            "시작 날짜",
            default_start_date
        )
        
    with date_col2:
        # 시작일 기준으로 최대 종료일 계산 (30일 이내)
        max_end_date = start_date + timedelta(days=30)
        if today < max_end_date:
            max_end_date = today
            
        end_date = st.date_input(
            "종료 날짜",
            min(default_end_date, max_end_date),
            min_value=start_date,
            max_value=max_end_date
        )

    # 선택된 날짜 범위 일수 계산
    date_range_days = (end_date - start_date).days

    # 기간 표시 정보 및 시각화
    progress_value = min(date_range_days / 30, 1.0)
    st.sidebar.progress(progress_value)
    st.sidebar.text(f"선택된 기간: {date_range_days + 1}일 / 최대 30일")

    if date_range_days > 25:
        st.sidebar.warning("데이터 양이 많을수록 분석 시간이 길어질 수 있습니다.")

    # 데이터 가져오기 버튼
    if st.sidebar.button("데이터 가져오기"):
        with st.spinner("서울시 API에서 데이터를 가져오는 중..."):
            df = load_data(
                start_date=start_date.strftime("%Y-%m-%d"),
                end_date=end_date.strftime("%Y-%m-%d")
            )
            if df is not None and not df.empty:
                st.session_state.df = df

    st.sidebar.markdown("---")
    
    # 데이터가 로드되면 분석 시작
    if st.session_state.df is not None and not st.session_state.df.empty:
        # 데이터 기본 정보 표시
        with st.expander("📋 데이터 미리보기", expanded=True):
            st.write(st.session_state.df.head())
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write(f"**데이터 크기:** {st.session_state.df.shape[0]} 행 × {st.session_state.df.shape[1]} 열")
                st.write(f"**기간:** {st.session_state.df['MSRDT'].min()} ~ {st.session_state.df['MSRDT'].max()}")
            
            with col2:
                if 'MSRSTE_NM' in st.session_state.df.columns:
                    st.write(f"**측정소 수:** {st.session_state.df['MSRSTE_NM'].nunique()}개")
                    st.write(f"**측정소 목록:** {', '.join(sorted(st.session_state.df['MSRSTE_NM'].unique()))}")
        
        # 분석 옵션 설정
        st.sidebar.subheader("🔍 시계열 분석 옵션")
        
        # 측정소 선택
        if 'MSRSTE_NM' in st.session_state.df.columns:
            stations = ['전체 평균'] + sorted(st.session_state.df['MSRSTE_NM'].unique().tolist())
            selected_station = st.sidebar.selectbox(
                "Select Station", 
                stations,
                index=0 if st.session_state.selected_station is None else stations.index(st.session_state.selected_station if st.session_state.selected_station else "전체 평균")
            )
            
            if selected_station == '전체 평균':
                st.session_state.selected_station = None
            else:
                st.session_state.selected_station = selected_station
        else:
            st.session_state.selected_station = None
            st.sidebar.info("측정소 정보가 없습니다.")
        
        # 타겟 변수 선택
        numeric_columns = st.session_state.df.select_dtypes(include=np.number).columns.tolist()
        target_options = [col for col in numeric_columns if col in ['PM10', 'PM25', 'O3', 'NO2', 'CO', 'SO2']]
        
        if not target_options:
            target_options = numeric_columns
        
        if target_options:
            selected_target = st.sidebar.selectbox(
                "분석할 변수 선택", 
                target_options,
                index=0 if st.session_state.selected_target is None else target_options.index(st.session_state.selected_target)
            )
            st.session_state.selected_target = selected_target
        else:
            st.error("분석 가능한 숫자형 변수가 없습니다.")
            return
        
        # 시리즈 데이터 업데이트
        update_series()
        
        # 시계열 분석 탭
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "시계열 시각화", 
            "시계열 분해", 
            "정상성 & ACF/PACF", 
            "모델 학습 및 예측",
            "LLM 분석"
        ])
        
        with tab1:
            # 시계열 데이터 시각화
            st.subheader("📈 시계열 시각화")
            
            # 선택한 측정소와 변수에 대한 시계열 그래프
            station_text = f"{st.session_state.selected_station} " if st.session_state.selected_station else "Seoul City Overall "
            fig = visualizer.plot_timeseries(
                st.session_state.series,
                title=f"{station_text}{st.session_state.selected_target} Time Series Data",
                ylabel=st.session_state.selected_target
            )
            st.pyplot(fig)
        
        with tab2:
            # 시계열 분해
            st.subheader("🔄 시계열 분해", help="시계열 분해(Time Series Decomposition)란, **시계열 데이터를 구성하는 여러 요소(성분)**를 분리해내는 기법입니다. 이를 통해 데이터의 구조를 더 잘 이해하고, 예측력 높은 모델을 만들 수 있습니다.")
            
            # 계절성 주기 선택
            min_period = 2
            max_period = min(len(st.session_state.series) // 2, 168)  # 최대 일주일(168시간) 또는 데이터 길이의 절반
            
            period = st.slider(
                "계절성 주기 (시간 단위)",
                min_value=min_period,
                max_value=max_period,
                value=st.session_state.period,
                help="계절성 주기를 선택하세요. 예: 24시간은 하루 주기를 의미합니다."
            )
            st.session_state.period = period
            
            try:
                # 시계열 분해 수행
                st.session_state.decomposition = data_processor.decompose_timeseries(st.session_state.series, period)
                
                # 분해 결과 시각화
                decomp_fig = visualizer.plot_decomposition(st.session_state.decomposition)
                st.pyplot(decomp_fig)
            except Exception as e:
                st.error(f"시계열 분해 중 오류 발생: {str(e)}")
        
        with tab3:
            # 정상성 검정
            st.subheader("🔍 정상성 검정", help="정상성 검정(Stationarity Test)이란 시계열 데이터가 시간이 지나도 통계적 특성이 일정한지(=정상인지) 확인하는 검정입니다. 즉, 평균, 분산, 자기공분산 등의 값이 시간에 따라 변하지 않는지를 확인하는 것입니다")
            
            try:
                # 정상성 검정 수행
                st.session_state.stationarity_result = data_processor.check_stationarity(st.session_state.series)
                
                # 정상성 검정 결과 표시
                col1, col2 = st.columns(2)
                with col1:
                    st.write(f"**ADF 통계량:** {st.session_state.stationarity_result['test_statistic']:.4f}")
                    st.write(f"**p-값:** {st.session_state.stationarity_result['p_value']:.4f}")
                    
                    # 정상성 여부
                    if st.session_state.stationarity_result['is_stationary']:
                        st.success("시계열 데이터가 정상성을 만족합니다.")
                    else:
                        st.warning("시계열 데이터가 정상성을 만족하지 않습니다.")
                
                with col2:
                    st.write("**임계값:**")
                    for key, value in st.session_state.stationarity_result['critical_values'].items():
                        st.write(f"{key}: {value:.4f}")
                
                # ACF, PACF 분석
                st.subheader("📊 ACF/PACF 분석")
                
                # ACF, PACF 계산
                st.session_state.acf_values, st.session_state.pacf_values = data_processor.get_acf_pacf(st.session_state.series)
                
                acf_pacf_fig = visualizer.plot_acf_pacf(st.session_state.acf_values, st.session_state.pacf_values)
                st.pyplot(acf_pacf_fig)
            except Exception as e:
                st.error(f"정상성 검정 중 오류 발생: {str(e)}")
        
        with tab4:
            # 모델 학습 및 예측
            st.subheader("🤖 모델 학습 및 예측")
            
            # 사이드바에 훈련/테스트 분할 옵션 추가
            test_size = st.sidebar.slider(
                "테스트 데이터 비율",
                min_value=0.1,
                max_value=0.5,
                value=st.session_state.test_size,
                step=0.05
            )
            st.session_state.test_size = test_size
            
            # 모델 선택
            model_factory = get_model_factory()
            
            if model_factory is None:
                st.error("모델 팩토리 로드에 실패했습니다. pmdarima 호환성 문제일 수 있습니다.")
            else:
                available_models = model_factory.get_all_available_models()
                
                selected_models = st.sidebar.multiselect(
                    "모델 선택",
                    available_models,
                    default=available_models[:] if not st.session_state.selected_models else st.session_state.selected_models
                )
                st.session_state.selected_models = selected_models
                
                # 모델 학습 및 예측 버튼
                if st.button("모델 학습 및 예측 시작"):
                    if not selected_models:
                        st.warning("최소한 하나의 모델을 선택해주세요.")
                    else:
                        with st.spinner("모델을 학습 중입니다..."):
                            train_models()
            
            # 모델 학습 결과 표시
            if st.session_state.models_trained and st.session_state.forecasts:
                # 예측 결과 비교 시각화
                st.subheader("📊 예측 비교")
                comparison_fig = visualizer.plot_forecast_comparison(
                    st.session_state.train, 
                    st.session_state.test, 
                    st.session_state.forecasts
                )
                st.pyplot(comparison_fig)
                
                # 메트릭 비교 시각화
                st.subheader("📈 모델 성능 비교")
                metrics_fig = visualizer.plot_metrics_comparison(st.session_state.metrics)
                st.pyplot(metrics_fig)
                
                # 메트릭 표 표시
                st.subheader("📋 모델 성능 메트릭")
                metrics_df = pd.DataFrame({model: st.session_state.metrics[model] for model in st.session_state.metrics})
                st.write(metrics_df)
                
                # 최적 모델 선택
                if st.session_state.best_model:
                    st.success(f"최적 모델 (RMSE 기준): {st.session_state.best_model}")
                
                # 선택한 최적 모델 상세 분석
                if st.session_state.best_model in st.session_state.forecasts:
                    st.subheader(f"📈 최적 모델 ({st.session_state.best_model}) 상세 분석")
                    
                    # 실제값과 예측값 비교
                    best_forecast = st.session_state.forecasts[st.session_state.best_model]
                    
                    # 잔차 분석
                    residuals_fig = visualizer.plot_residuals(st.session_state.test, best_forecast)
                    st.pyplot(residuals_fig)
        with tab5:
            # LLM 분석 탭
            st.subheader("🤖 LLM 시계열 데이터 분석")
            
            with st.expander("📊 LLM 분석 설정", expanded=True):
                st.info(f"Ollama 서버를 통해 {app_config.OLLAMA_MODEL} 모델로 시계열 분석 결과를 자동으로 분석합니다.")
            
            def check_analysis_ready():
                """
                LLM 분석을 위한 데이터와 모델이 준비되었는지 확인합니다.
                """
                # 모델 예측 결과와 평가 지표를 우선 확인
                if not hasattr(st.session_state, 'forecasts') or not st.session_state.forecasts:
                    return False, "모델 예측 결과가 없습니다. 먼저 '모델 학습 및 예측' 탭에서 모델 학습을 완료해주세요."
                    
                if not hasattr(st.session_state, 'metrics') or not st.session_state.metrics:
                    return False, "모델 평가 지표가 없습니다. 먼저 '모델 학습 및 예측' 탭에서 모델 학습을 완료해주세요."
                    
                if not hasattr(st.session_state, 'best_model') or st.session_state.best_model is None:
                    return False, "최적 모델 정보가 없습니다. 먼저 '모델 학습 및 예측' 탭에서 모델 학습을 완료해주세요."
                
                # 시계열 데이터는 필수 확인
                if not hasattr(st.session_state, 'series') or st.session_state.series is None:
                    return False, "시계열 데이터가 준비되지 않았습니다."
                
                # train/test 데이터는 경고만 출력하고 진행
                if not hasattr(st.session_state, 'train') or st.session_state.train is None or not hasattr(st.session_state, 'test') or st.session_state.test is None:
                    st.warning("학습/테스트 데이터가 접근 불가능하지만, 모델 결과가 있으므로 분석을 진행합니다.")
                
                return True, "분석 준비 완료"
            
            # LLM 분석 실행 버튼
            if st.button("LLM 분석 시작", type="primary"):
                
                # 데이터 및 모델 학습 상태 확인
                is_ready, message = check_analysis_ready()
                
                if not is_ready:
                    st.warning(message)
                else:
                    with st.spinner("LLM을 통해 시계열 분석 결과를 분석 중입니다..."):
                        try:
                            llm_connector = LLMConnector(
                                base_url=app_config.OLLAMA_SERVER,
                                model=app_config.OLLAMA_MODEL
                            )
                            
                            # 데이터 정보 수집 - 안전하게 값 추출
                            data_info = {
                                "target_variable": st.session_state.selected_target,
                                "station": st.session_state.selected_station if hasattr(st.session_state, 'selected_station') else None,
                                "seasonality_period": st.session_state.period if hasattr(st.session_state, 'period') else None,
                                "data_range": {
                                    "total_points": len(st.session_state.series),
                                },
                                "date_range": {
                                    "start": str(st.session_state.series.index.min()),
                                    "end": str(st.session_state.series.index.max())
                                },
                                "value_stats": {
                                    "min": float(st.session_state.series.min()),
                                    "max": float(st.session_state.series.max()),
                                    "mean": float(st.session_state.series.mean()),
                                    "std": float(st.session_state.series.std())
                                }
                            }

                            # 훈련/테스트 데이터가 있는 경우만 추가
                            if hasattr(st.session_state, 'train') and st.session_state.train is not None:
                                data_info["data_range"]["train_points"] = len(st.session_state.train)

                            if hasattr(st.session_state, 'test') and st.session_state.test is not None:
                                data_info["data_range"]["test_points"] = len(st.session_state.test)
                            
                            # 정상성 정보 추가 (있는 경우만)
                            if (hasattr(st.session_state, 'stationarity_result') 
                                and st.session_state.stationarity_result is not None):
                                try:
                                    data_info["stationarity"] = {
                                        "is_stationary": bool(st.session_state.stationarity_result["is_stationary"]),
                                        "p_value": float(st.session_state.stationarity_result["p_value"]),
                                        "test_statistic": float(st.session_state.stationarity_result["test_statistic"])
                                    }
                                except (KeyError, TypeError):
                                    # 정상성 정보에 필요한 키가 없는 경우 무시
                                    pass
                            
                            # 분해 정보 추가 (있는 경우만, 요약 정보만)
                            if (hasattr(st.session_state, 'decomposition') 
                                and st.session_state.decomposition is not None):
                                try:
                                    decomp_info = {}
                                    for comp_name, comp_data in st.session_state.decomposition.items():
                                        if comp_name != 'observed' and comp_data is not None:  # 원본 데이터는 제외
                                            clean_data = comp_data.dropna()
                                            if not clean_data.empty:
                                                decomp_info[comp_name] = {
                                                    "min": float(clean_data.min()),
                                                    "max": float(clean_data.max()),
                                                    "mean": float(clean_data.mean())
                                                }
                                    
                                    if decomp_info:  # 비어있지 않은 경우만 추가
                                        data_info["decomposition"] = decomp_info
                                except Exception as e:
                                    st.warning(f"분해 정보 처리 중 오류 발생 (무시됨): {e}")
                            
                            # 모델 결과 정보 수집
                            model_results = {
                                "best_model": st.session_state.best_model,
                                "models": {}
                            }
                            
                            # 각 모델의 메트릭 정보 추가
                            for model_name, metrics in st.session_state.metrics.items():
                                model_results["models"][model_name] = {
                                    "metrics": {}
                                }
                                for metric_name, metric_value in metrics.items():
                                    # NaN 값 처리
                                    if pd.isna(metric_value):
                                        model_results["models"][model_name]["metrics"][metric_name] = None
                                    else:
                                        model_results["models"][model_name]["metrics"][metric_name] = float(metric_value)
                                
                                # 예측값이 있는 경우만 통계 추가
                                if model_name in st.session_state.forecasts:
                                    forecast = st.session_state.forecasts[model_name]
                                    if forecast is not None and len(forecast) > 0:
                                        model_results["models"][model_name]["forecast_stats"] = {
                                            "min": float(np.min(forecast)),
                                            "max": float(np.max(forecast)),
                                            "mean": float(np.mean(forecast)),
                                            "std": float(np.std(forecast))
                                        }
                            
                            # LLM 분석 요청
                            analysis_result = llm_connector.analyze_time_series(
                                data_info,
                                model_results,
                                TIME_SERIES_ANALYSIS_PROMPT
                            )
                            
                            # 분석 결과 표시
                            st.markdown(analysis_result)
                            
                            # 세션 상태에 분석 결과 저장
                            st.session_state.llm_analysis = analysis_result
                            
                            # 분석 결과 다운로드 버튼 제공
                            st.download_button(
                                label="분석 결과 다운로드 (Markdown)",
                                data=analysis_result,
                                file_name="time_series_analysis_report.md",
                                mime="text/markdown"
                            )
                            
                        except Exception as e:
                            st.error(f"LLM 분석 중 오류 발생: {str(e)}")

            # 이전에 분석한 결과가 있으면 표시
            elif hasattr(st.session_state, 'llm_analysis') and st.session_state.llm_analysis:
                st.markdown(st.session_state.llm_analysis)
                
                # 다운로드 버튼
                st.download_button(
                    label="분석 결과 다운로드 (Markdown)",
                    data=st.session_state.llm_analysis,
                    file_name="time_series_analysis_report.md",
                    mime="text/markdown"
                )
    else:
        st.info("분석을 시작하려면 데이터를 업로드하거나 API에서 데이터를 가져오세요.")

# 앱 실행
if __name__ == "__main__":
    main()

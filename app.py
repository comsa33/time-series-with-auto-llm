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
@st.cache_data
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
    st.title("Air Quality Time Series Analysis")
    st.markdown("Seoul City IoT Data Time Series Analysis App")
    
    # 확장 가능한 앱 소개
    with st.expander("📌 App Introduction and Usage"):
        st.markdown("""
        ### App Introduction
        This app analyzes and visualizes time series data from Seoul City's air quality data.
        
        ### Main Features
        1. **Data Exploration**: Basic statistics and visualization of Seoul's air quality data
        2. **Time Series Decomposition**: Trend, seasonality, and irregularity analysis
        3. **Model Comparison**: Various prediction models including ARIMA/SARIMA, Exponential Smoothing, Prophet, LSTM, etc.
        4. **Prediction Performance Evaluation**: Various metrics-based evaluation (RMSE, MAE, R^2, etc.)
        
        ### How to Use
        1. Select data upload or API collection options in the sidebar
        2. Choose the measurement station and variable (PM10, PM25, etc.) to analyze
        3. Set time series analysis options and run model training
        4. Compare and analyze prediction results of various models in the results tab
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
    st.sidebar.header("📊 Analysis Settings")
    
    # 데이터 로드 방식 선택
    data_source = st.sidebar.radio(
        "Select Data Source",
        ["API에서 가져오기", "파일 업로드"],
        key="data_source",
        on_change=on_data_source_change
    )
    
    # 데이터 로드
    if data_source == "API에서 가져오기":
        st.sidebar.subheader("API Settings")
        
        # 날짜 범위 선택
        today = datetime.now()
        default_end_date = today.strftime("%Y-%m-%d")
        default_start_date = (today - timedelta(days=30)).strftime("%Y-%m-%d")
        
        start_date = st.sidebar.date_input(
            "Start Date",
            datetime.strptime(default_start_date, "%Y-%m-%d")
        )
        
        end_date = st.sidebar.date_input(
            "End Date",
            datetime.strptime(default_end_date, "%Y-%m-%d")
        )
        
        if st.sidebar.button("Get Data"):
            with st.spinner("Getting data from Seoul City API..."):
                df = load_data(
                    start_date=start_date.strftime("%Y-%m-%d"),
                    end_date=end_date.strftime("%Y-%m-%d")
                )
                if df is not None and not df.empty:
                    st.session_state.df = df
    else:
        st.sidebar.subheader("File Upload")
        
        # 파일 업로드
        uploaded_file = st.sidebar.file_uploader("Upload CSV File", type=["csv"])
        
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            if 'MSRDT' in df.columns:
                df['MSRDT'] = pd.to_datetime(df['MSRDT'])
            st.session_state.df = df
        else:
            # 기존 파일 사용
            if os.path.exists(app_config.DEFAULT_DATA_FILE):
                use_existing = st.sidebar.checkbox("Use Existing Data", value=True)
                if use_existing:
                    df = load_data(file_path=app_config.DEFAULT_DATA_FILE)
                    if df is not None and not df.empty:
                        st.session_state.df = df
            else:
                st.sidebar.warning("No saved data file found. Please upload a file or get data from API.")
    
    # 데이터가 로드되면 분석 시작
    if st.session_state.df is not None and not st.session_state.df.empty:
        # 데이터 기본 정보 표시
        with st.expander("📋 Data Preview", expanded=True):
            st.write(st.session_state.df.head())
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write(f"**Data Size:** {st.session_state.df.shape[0]} rows × {st.session_state.df.shape[1]} columns")
                st.write(f"**Period:** {st.session_state.df['MSRDT'].min()} ~ {st.session_state.df['MSRDT'].max()}")
            
            with col2:
                if 'MSRSTE_NM' in st.session_state.df.columns:
                    st.write(f"**Number of Stations:** {st.session_state.df['MSRSTE_NM'].nunique()}")
                    st.write(f"**Station List:** {', '.join(sorted(st.session_state.df['MSRSTE_NM'].unique()))}")
        
        # 분석 옵션 설정
        st.sidebar.subheader("🔍 Analysis Options")
        
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
            st.sidebar.info("No station information available.")
        
        # 타겟 변수 선택
        numeric_columns = st.session_state.df.select_dtypes(include=np.number).columns.tolist()
        target_options = [col for col in numeric_columns if col in ['PM10', 'PM25', 'O3', 'NO2', 'CO', 'SO2']]
        
        if not target_options:
            target_options = numeric_columns
        
        if target_options:
            selected_target = st.sidebar.selectbox(
                "Select Variable", 
                target_options,
                index=0 if st.session_state.selected_target is None else target_options.index(st.session_state.selected_target)
            )
            st.session_state.selected_target = selected_target
        else:
            st.error("No numeric variables available for analysis.")
            return
        
        # 시리즈 데이터 업데이트
        update_series()
        
        # 시계열 분석 탭
        tab1, tab2, tab3, tab4 = st.tabs(["Time Series Visualization", "Time Series Decomposition", "Stationarity & ACF/PACF", "Model Training & Prediction"])
        
        with tab1:
            # 시계열 데이터 시각화
            st.subheader("📈 Time Series Visualization")
            
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
            st.subheader("🔄 Time Series Decomposition")
            
            # 계절성 주기 선택
            min_period = 2
            max_period = min(len(st.session_state.series) // 2, 168)  # 최대 일주일(168시간) 또는 데이터 길이의 절반
            
            period = st.slider(
                "Seasonality Period (hours)",
                min_value=min_period,
                max_value=max_period,
                value=st.session_state.period
            )
            st.session_state.period = period
            
            try:
                # 시계열 분해 수행
                st.session_state.decomposition = data_processor.decompose_timeseries(st.session_state.series, period)
                
                # 분해 결과 시각화
                decomp_fig = visualizer.plot_decomposition(st.session_state.decomposition)
                st.pyplot(decomp_fig)
            except Exception as e:
                st.error(f"Error in time series decomposition: {str(e)}")
        
        with tab3:
            # 정상성 검정
            st.subheader("🔍 Stationarity Test")
            
            try:
                # 정상성 검정 수행
                st.session_state.stationarity_result = data_processor.check_stationarity(st.session_state.series)
                
                # 정상성 검정 결과 표시
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write(f"**ADF Statistic:** {st.session_state.stationarity_result['test_statistic']:.4f}")
                    st.write(f"**p-value:** {st.session_state.stationarity_result['p_value']:.4f}")
                    
                    # 정상성 여부
                    if st.session_state.stationarity_result['is_stationary']:
                        st.success("The time series data is stationary.")
                    else:
                        st.warning("The time series data is not stationary.")
                
                with col2:
                    st.write("**Critical Values:**")
                    for key, value in st.session_state.stationarity_result['critical_values'].items():
                        st.write(f"{key}: {value:.4f}")
                
                # ACF, PACF 분석
                st.subheader("📊 ACF/PACF Analysis")
                
                # ACF, PACF 계산
                st.session_state.acf_values, st.session_state.pacf_values = data_processor.get_acf_pacf(st.session_state.series)
                
                acf_pacf_fig = visualizer.plot_acf_pacf(st.session_state.acf_values, st.session_state.pacf_values)
                st.pyplot(acf_pacf_fig)
            except Exception as e:
                st.error(f"Error in stationarity test: {str(e)}")
        
        with tab4:
            # 모델 학습 및 예측
            st.subheader("🤖 Model Training & Prediction")
            
            # 사이드바에 훈련/테스트 분할 옵션 추가
            test_size = st.sidebar.slider(
                "Test Data Ratio",
                min_value=0.1,
                max_value=0.5,
                value=st.session_state.test_size,
                step=0.05
            )
            st.session_state.test_size = test_size
            
            # 모델 선택
            model_factory = get_model_factory()
            
            if model_factory is None:
                st.error("Model factory loading failed. May be pmdarima compatibility issue.")
                st.error("Try running the following command:")
                st.code("pip uninstall -y pmdarima numpy && pip install numpy==1.24.3 && pip install pmdarima==2.0.4")
            else:
                available_models = model_factory.get_all_available_models()
                
                selected_models = st.sidebar.multiselect(
                    "Select Models",
                    available_models,
                    default=available_models[:2] if not st.session_state.selected_models else st.session_state.selected_models
                )
                st.session_state.selected_models = selected_models
                
                # 모델 학습 및 예측 버튼
                if st.button("Start Model Training & Prediction"):
                    if not selected_models:
                        st.warning("Please select at least one model.")
                    else:
                        with st.spinner("Training models..."):
                            train_models()
            
            # 모델 학습 결과 표시
            if st.session_state.models_trained and st.session_state.forecasts:
                # 예측 결과 비교 시각화
                st.subheader("📊 Forecast Comparison")
                comparison_fig = visualizer.plot_forecast_comparison(
                    st.session_state.train, 
                    st.session_state.test, 
                    st.session_state.forecasts
                )
                st.pyplot(comparison_fig)
                
                # 메트릭 비교 시각화
                st.subheader("📈 Model Performance Comparison")
                metrics_fig = visualizer.plot_metrics_comparison(st.session_state.metrics)
                st.pyplot(metrics_fig)
                
                # 메트릭 표 표시
                st.subheader("📋 Model Performance Metrics")
                metrics_df = pd.DataFrame({model: st.session_state.metrics[model] for model in st.session_state.metrics})
                st.write(metrics_df)
                
                # 최적 모델 선택
                if st.session_state.best_model:
                    st.success(f"Best Model (based on RMSE): {st.session_state.best_model}")
                
                # 모델 해석 및 인사이트
                st.subheader("🔍 Model Interpretation & Insights")
                
                st.markdown(f"""
                ### Time Series Analysis Results
                
                1. **Data Characteristics**:
                   - The selected variable ({st.session_state.selected_target}) shows distinct daily and weekly patterns.
                   - As seen in the decomposition results, there is a seasonality with a {st.session_state.period}-hour period.
                
                2. **Model Performance Comparison**:
                   - The {st.session_state.best_model} model showed the best performance based on RMSE.
                   - Model characteristics:
                     - ARIMA: Statistical model utilizing autocorrelation in time series data
                     - Exponential Smoothing: Method giving higher weights to recent observations
                     - Prophet: Facebook's time series model considering trend, seasonality, and holiday effects
                     - LSTM: Deep learning-based time series prediction model using recurrent neural networks
                
                3. **Applicability**:
                   - This prediction model can be used to develop a Seoul air quality forecasting system.
                   - By predicting times when fine dust concentration is expected to be high, it can help protect citizens' health.
                """)
                
                # 선택한 최적 모델 상세 분석
                if st.session_state.best_model in st.session_state.forecasts:
                    st.subheader(f"📈 Best Model ({st.session_state.best_model}) Detailed Analysis")
                    
                    # 실제값과 예측값 비교
                    best_forecast = st.session_state.forecasts[st.session_state.best_model]
                    
                    # 잔차 분석
                    residuals_fig = visualizer.plot_residuals(st.session_state.test, best_forecast)
                    st.pyplot(residuals_fig)
    else:
        st.info("Please upload data or get data from API to start analysis.")

# 앱 실행
if __name__ == "__main__":
    main()

"""
서울시 대기질 시계열 분석 메인 Streamlit 앱
"""
import os
import warnings
import traceback
from datetime import datetime, timedelta

import streamlit as st
import pandas as pd
import numpy as np

# 설정 및 유틸리티 모듈
from config.settings import app_config
from utils.data_reader import get_seoul_air_quality
from utils.data_processor import DataProcessor
from utils.visualizer import TimeSeriesVisualizer

# 모델 모듈 - 직접 import 하지 않고 필요할 때 동적으로 가져오기
# from models.model_factory import ModelFactory

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
    st.title(app_config.APP_TITLE)
    st.markdown(app_config.APP_DESCRIPTION)
    
    # 확장 가능한 앱 소개
    with st.expander("📌 앱 소개 및 사용 방법"):
        st.markdown("""
        ### 앱 소개
        이 앱은 서울시 IoT 도시데이터 중 대기질 데이터를 활용하여 다양한 시계열 분석 기법을 비교하고 시각화합니다.
        
        ### 주요 기능
        1. **데이터 탐색**: 서울시 대기질 데이터 기본 통계 및 시각화
        2. **시계열 분해**: 추세, 계절성, 불규칙성 분해 및 분석
        3. **모델 비교**: ARIMA/SARIMA, 지수평활법, Prophet, LSTM 등 다양한 예측 모델 비교
        4. **예측 성능 평가**: RMSE, MAE, R^2 등 다양한 성능 지표 기반 평가
        
        ### 사용 방법
        1. 사이드바에서 데이터 업로드 또는 API를 통한 데이터 수집 옵션 선택
        2. 분석할 측정소와 변수(PM10, PM25 등) 선택
        3. 시계열 분석 옵션 설정 및 모델 학습 실행
        4. 결과 탭에서 다양한 모델의 예측 결과 비교 및 분석
        """)


# 메인 함수
def main():
    # 앱 헤더 렌더링
    render_header()
    
    # 사이드바 설정
    st.sidebar.header("📊 분석 설정")
    
    # 데이터 로드 방식 선택
    data_source = st.sidebar.radio(
        "데이터 소스 선택",
        ["API에서 가져오기", "파일 업로드"]
    )
    
    df = None
    
    if data_source == "API에서 가져오기":
        st.sidebar.subheader("API 설정")
        
        # 날짜 범위 선택
        today = datetime.now()
        default_end_date = today.strftime("%Y-%m-%d")
        default_start_date = (today - timedelta(days=30)).strftime("%Y-%m-%d")
        
        start_date = st.sidebar.date_input(
            "시작 날짜",
            datetime.strptime(default_start_date, "%Y-%m-%d")
        )
        
        end_date = st.sidebar.date_input(
            "종료 날짜",
            datetime.strptime(default_end_date, "%Y-%m-%d")
        )
        
        if st.sidebar.button("데이터 가져오기"):
            with st.spinner("서울시 API에서 데이터를 가져오는 중..."):
                df = load_data(
                    start_date=start_date.strftime("%Y-%m-%d"),
                    end_date=end_date.strftime("%Y-%m-%d")
                )
    else:
        st.sidebar.subheader("파일 업로드")
        
        # 파일 업로드
        uploaded_file = st.sidebar.file_uploader("CSV 파일 업로드", type=["csv"])
        
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            if 'MSRDT' in df.columns:
                df['MSRDT'] = pd.to_datetime(df['MSRDT'])
        else:
            # 기존 파일 사용
            if os.path.exists(app_config.DEFAULT_DATA_FILE):
                use_existing = st.sidebar.checkbox("기존 저장된 데이터 사용", value=True)
                if use_existing:
                    df = load_data(file_path=app_config.DEFAULT_DATA_FILE)
            else:
                st.sidebar.warning("저장된 데이터 파일이 없습니다. 파일을 업로드하거나 API에서 데이터를 가져오세요.")
    
    # 데이터가 로드되면 분석 시작
    if df is not None and not df.empty:
        analyze_data(df)
    else:
        st.info("데이터를 업로드하거나 API에서 가져와서 분석을 시작하세요.")


def analyze_data(df):
    """
    데이터 분석 수행
    """
    # 데이터 미리보기
    st.subheader("📋 데이터 미리보기")
    st.write(df.head())
    
    # 기본 정보
    st.subheader("📊 데이터 기본 정보")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write(f"**데이터 크기:** {df.shape[0]} 행 × {df.shape[1]} 열")
        st.write(f"**기간:** {df['MSRDT'].min()} ~ {df['MSRDT'].max()}")
    
    with col2:
        if 'MSRSTE_NM' in df.columns:
            st.write(f"**측정소 수:** {df['MSRSTE_NM'].nunique()}개")
            st.write(f"**측정소 목록:** {', '.join(sorted(df['MSRSTE_NM'].unique()))}")
    
    # 측정소 선택
    st.sidebar.subheader("🔍 분석 옵션")
    
    if 'MSRSTE_NM' in df.columns:
        stations = ['전체 평균'] + sorted(df['MSRSTE_NM'].unique().tolist())
        selected_station = st.sidebar.selectbox("측정소 선택", stations)
        
        if selected_station == '전체 평균':
            selected_station = None
    else:
        selected_station = None
        st.sidebar.info("측정소 정보가 없습니다.")
    
    # 타겟 변수 선택
    numeric_columns = df.select_dtypes(include=np.number).columns.tolist()
    target_options = [col for col in numeric_columns if col in ['PM10', 'PM25', 'O3', 'NO2', 'CO', 'SO2']]
    
    if not target_options:
        target_options = numeric_columns
    
    if target_options:
        selected_target = st.sidebar.selectbox(
            "분석할 변수 선택", 
            target_options,
            index=0 if 'PM10' in target_options else 0
        )
    else:
        st.error("분석할 수치형 변수가 없습니다.")
        return
    
    # 데이터 전처리
    series = data_processor.preprocess_data(df, selected_target, selected_station)
    
    # 시계열 데이터 시각화
    st.subheader("📈 시계열 데이터 시각화")
    
    # 선택한 측정소와 변수에 대한 시계열 그래프
    station_text = f"{selected_station}의 " if selected_station else "서울시 전체 "
    fig = visualizer.plot_timeseries(
        series,
        title=f"{station_text}{selected_target} 시계열 데이터",
        ylabel=selected_target
    )
    st.pyplot(fig)
    
    # 시계열 분해
    st.subheader("🔄 시계열 분해")
    
    # 계절성 주기 선택
    default_period = 24  # 기본값: 24시간(일별) 주기
    min_period = 2
    max_period = min(len(series) // 2, 168)  # 최대 일주일(168시간) 또는 데이터 길이의 절반
    
    period = st.sidebar.slider(
        "계절성 주기 (시간)",
        min_value=min_period,
        max_value=max_period,
        value=default_period
    )
    
    try:
        # 시계열 분해 수행
        decomposition = data_processor.decompose_timeseries(series, period)
        
        # 분해 결과 시각화
        decomp_fig = visualizer.plot_decomposition(decomposition)
        st.pyplot(decomp_fig)
        
        # 정상성 검정
        st.subheader("🔍 정상성 검정")
        
        stationarity_result = data_processor.check_stationarity(series)
        
        # 정상성 검정 결과 표시
        col1, col2 = st.columns(2)
        
        with col1:
            st.write(f"**ADF 통계량:** {stationarity_result['test_statistic']:.4f}")
            st.write(f"**p-값:** {stationarity_result['p_value']:.4f}")
            
            # 정상성 여부
            if stationarity_result['is_stationary']:
                st.success("시계열 데이터가 정상성(Stationary)을 만족합니다.")
            else:
                st.warning("시계열 데이터가 정상성(Stationary)을 만족하지 않습니다.")
        
        with col2:
            st.write("**임계값:**")
            for key, value in stationarity_result['critical_values'].items():
                st.write(f"{key}: {value:.4f}")
        
        # ACF, PACF 분석
        st.subheader("📊 ACF/PACF 분석")
        
        acf_values, pacf_values = data_processor.get_acf_pacf(series)
        acf_pacf_fig = visualizer.plot_acf_pacf(acf_values, pacf_values)
        st.pyplot(acf_pacf_fig)
        
    except Exception as e:
        st.error(f"시계열 분해 중 오류 발생: {traceback.format_exc()}")
    
    # 모델 학습 및 예측
    st.subheader("🤖 모델 학습 및 예측")
    
    # 훈련/테스트 분할
    test_size = st.sidebar.slider(
        "테스트 데이터 비율",
        min_value=0.1,
        max_value=0.5,
        value=app_config.DEFAULT_TEST_SIZE,
        step=0.05
    )
    
    # 분할 수행
    train, test = data_processor.train_test_split(series, test_size)
    
    # 모델 선택 - 모델 팩토리를 필요할 때만 로드
    model_factory = get_model_factory()
    
    if model_factory is None:
        st.error("모델 팩토리를 로드할 수 없습니다. pmdarima 호환성 문제일 수 있습니다.")
        st.error("아래 명령어로 문제를 해결할 수 있습니다:")
        st.code("pip uninstall -y pmdarima numpy && pip install numpy==1.24.3 && pip install pmdarima==2.0.4")
        return
    
    available_models = model_factory.get_all_available_models()
    
    selected_models = st.sidebar.multiselect(
        "분석 방법 선택",
        available_models,
        default=available_models[:2]  # 기본적으로 처음 두 모델 선택
    )
    
    # 모델 학습 및 예측 버튼
    if st.button("모델 학습 및 예측 시작"):
        if not selected_models:
            st.warning("적어도 하나의 모델을 선택하세요.")
        else:
            # 진행 상황 표시
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # 예측 결과 및 메트릭 저장
            forecasts = {}
            metrics = {}
            
            # 모델 개수
            total_models = len(selected_models)
            completed_models = 0
            
            # 각 모델 학습 및 예측
            for model_type in selected_models:
                status_text.text(f"{model_type} 모델 학습 중...")
                
                try:
                    # 모델 인스턴스 생성
                    model = model_factory.get_model(model_type)
                    
                    # 모델별 학습 매개변수 설정
                    if model_type == 'arima':
                        # ARIMA 모델 파라미터
                        forecast, model_metrics = model.fit_predict_evaluate(
                            train, test,
                            seasonal=True,
                            m=period
                        )
                    elif model_type == 'exp_smoothing':
                        # 지수평활법 모델 파라미터
                        forecast, model_metrics = model.fit_predict_evaluate(
                            train, test,
                            seasonal_periods=period
                        )
                    elif model_type == 'prophet':
                        # Prophet 모델 파라미터
                        forecast, model_metrics = model.fit_predict_evaluate(
                            train, test,
                            daily_seasonality=True,
                            weekly_seasonality=True
                        )
                    elif model_type == 'lstm':
                        # LSTM 모델 파라미터
                        forecast, model_metrics = model.fit_predict_evaluate(
                            train, test,
                            n_steps=min(48, len(train) // 10),  # 시퀀스 길이
                            lstm_units=[50, 50],
                            dropout_rate=0.2,
                            epochs=100,
                            batch_size=32,
                            validation_split=0.1
                        )
                    else:
                        # 기본 파라미터
                        forecast, model_metrics = model.fit_predict_evaluate(train, test)
                    
                    # 예측 결과 및 메트릭 저장
                    forecasts[model.name] = forecast
                    metrics[model.name] = model_metrics
                    
                    # 진행 상황 업데이트
                    completed_models += 1
                    progress_bar.progress(completed_models / total_models)
                    
                except Exception as e:
                    st.error(f"{model_type} 모델 학습 중 오류 발생: {e}")
            
            # 모든 모델 학습 완료
            if forecasts:
                status_text.text("모든 모델 학습 완료!")
                
                # 예측 결과 비교 시각화
                st.subheader("📊 예측 결과 비교")
                comparison_fig = visualizer.plot_forecast_comparison(train, test, forecasts)
                st.pyplot(comparison_fig)
                
                # 메트릭 비교 시각화
                st.subheader("📈 모델 성능 비교")
                metrics_fig = visualizer.plot_metrics_comparison(metrics)
                st.pyplot(metrics_fig)
                
                # 메트릭 표 표시
                st.subheader("📋 모델 성능 지표")
                metrics_df = pd.DataFrame({model: metrics[model] for model in metrics})
                st.write(metrics_df)
                
                # 최적 모델 선택
                rmse_values = {model: metrics[model]['RMSE'] for model in metrics}
                best_model = min(rmse_values.items(), key=lambda x: x[1])[0]
                st.success(f"RMSE 기준 최적 모델: {best_model}")
                
                # 모델 해석 및 인사이트
                st.subheader("🔍 모델 해석 및 인사이트")
                
                st.markdown(f"""
                ### 시계열 분석 결과 해석
                
                1. **데이터 특성**:
                   - 선택한 변수({selected_target})는 뚜렷한 일일 패턴과 주간 패턴을 보입니다.
                   - 시계열 분해 결과에서 볼 수 있듯이, {period}시간 주기의 계절성이 존재합니다.
                
                2. **모델 성능 비교**:
                   - RMSE 기준으로 {best_model} 모델이 가장 우수한 성능을 보였습니다.
                   - 각 모델별 특성:
                     - ARIMA: 시계열 데이터의 자기상관성을 활용한 통계적 모델
                     - 지수평활법: 최근 관측치에 더 높은 가중치를 부여하는 방법
                     - Prophet: 추세, 계절성, 휴일 효과를 고려하는 페이스북의 시계열 모델
                     - LSTM: 순환신경망을 활용한 딥러닝 기반 시계열 예측 모델
                
                3. **적용 가능성**:
                   - 이 예측 모델을 활용하여 서울시 대기질 예보 시스템을 개발할 수 있습니다.
                   - 미세먼지 농도가 높아질 것으로 예상되는 시간대를 사전에 알림으로써, 시민들의 건강을 보호하는 데 기여할 수 있습니다.
                """)
                
                # 선택한 최적 모델 상세 분석
                if best_model in forecasts:
                    st.subheader(f"📈 최적 모델 ({best_model}) 상세 분석")
                    
                    # 실제값과 예측값 비교
                    best_forecast = forecasts[best_model]
                    
                    # 잔차 분석
                    residuals_fig = visualizer.plot_residuals(test, best_forecast)
                    st.pyplot(residuals_fig)


# 앱 실행
if __name__ == "__main__":
    main()

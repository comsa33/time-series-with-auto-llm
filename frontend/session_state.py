"""
세션 상태 관리 모듈
"""
import streamlit as st

from config.settings import app_config


def initialize_session_state():
    """
    필요한 세션 상태 변수 초기화
    """
    # 기본 데이터 변수들
    if 'df' not in st.session_state:
        st.session_state.df = None
    if 'data_source' not in st.session_state:
        st.session_state.data_source = "API에서 가져오기"
    if 'selected_station' not in st.session_state:
        st.session_state.selected_station = None
    if 'selected_target' not in st.session_state:
        st.session_state.selected_target = None
    
    # 시계열 데이터 변수들
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
    
    # 정상성 및 ACF/PACF 관련 변수들
    if 'stationarity_result' not in st.session_state:
        st.session_state.stationarity_result = None
    if 'acf_values' not in st.session_state:
        st.session_state.acf_values = None
    if 'pacf_values' not in st.session_state:
        st.session_state.pacf_values = None
    
    # 모델 관련 변수들
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
    if 'complexity' not in st.session_state:
        st.session_state.complexity = '간단 (빠름, 저메모리)'
    
    # 변경 추적 변수들
    if 'active_tab' not in st.session_state:
        st.session_state.active_tab = 0
    if 'prev_target' not in st.session_state:
        st.session_state.prev_target = None
    if 'prev_station' not in st.session_state:
        st.session_state.prev_station = None
        
    # LLM 분석 결과
    if 'llm_analysis' not in st.session_state:
        st.session_state.llm_analysis = None

    # 하이퍼파라미터 최적화 관련 변수
    if 'hyperparameter_recommendations' not in st.session_state:
        st.session_state.hyperparameter_recommendations = {}
    if 'optimization_history' not in st.session_state:
        st.session_state.optimization_history = {}

    # 차분 관련 변수들
    if 'diff_order' not in st.session_state:
        st.session_state.diff_order = 0
    if 'seasonal_diff_order' not in st.session_state:
        st.session_state.seasonal_diff_order = 0
    if 'use_differencing' not in st.session_state:
        st.session_state.use_differencing = False
    if 'differenced_series' not in st.session_state:
        st.session_state.differenced_series = None
    if 'differencing_recommendation' not in st.session_state:
        st.session_state.differencing_recommendation = None
    if 'diff_train' not in st.session_state:
        st.session_state.diff_train = None
    if 'diff_test' not in st.session_state:
        st.session_state.diff_test = None


def reset_model_results():
    """
    모델 결과 관련 세션 상태 초기화
    """
    st.session_state.models_trained = False
    st.session_state.forecasts = {}
    st.session_state.metrics = {}
    st.session_state.best_model = None
    st.session_state.use_differencing = False

    
def reset_data_results():
    """
    데이터 분석 결과 관련 세션 상태 초기화
    """
    st.session_state.series = None
    st.session_state.train = None
    st.session_state.test = None
    st.session_state.decomposition = None
    st.session_state.stationarity_result = None
    st.session_state.acf_values = None
    st.session_state.pacf_values = None
    st.session_state.diff_order = 0
    st.session_state.seasonal_diff_order = 0
    st.session_state.use_differencing = False
    st.session_state.differenced_series = None
    st.session_state.differencing_recommendation = None
    st.session_state.diff_train = None
    st.session_state.diff_test = None
    reset_model_results()

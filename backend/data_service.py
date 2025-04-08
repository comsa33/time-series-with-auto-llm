"""
데이터 관련 서비스 모듈
"""
import os
from datetime import datetime, timedelta

import streamlit as st
import pandas as pd

from config.settings import app_config
from utils.data_reader import get_seoul_air_quality
from utils.data_processor import (
    cached_preprocess_data,
    cached_train_test_split,
    cached_decompose_timeseries,
    cached_check_stationarity,
    cached_get_acf_pacf
)

@st.cache_data(ttl=3600)
def load_data(file_path=None, start_date=None, end_date=None):
    """
    CSV 파일에서 데이터를 불러오거나, 파일이 없는 경우 API를 통해 데이터를 가져옵니다.
    
    Args:
        file_path: 데이터 파일 경로 (기본값: None)
        start_date: 시작 날짜 (기본값: None)
        end_date: 종료 날짜 (기본값: None)
        
    Returns:
        pandas.DataFrame: 로드된 데이터프레임
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

def update_series():
    """
    시계열 데이터 업데이트 함수 - 캐싱 활용
    """
    if st.session_state.df is not None:
        # 캐싱된 데이터 전처리 함수 호출
        st.session_state.series = cached_preprocess_data(
            st.session_state.df, 
            st.session_state.selected_target, 
            st.session_state.selected_station
        )
        
        # 이전 결과와 현재 설정의 호환성 확인
        if st.session_state.models_trained:
            # 타겟 변수나 측정소가 변경되면 결과 초기화
            if ('prev_target' in st.session_state and 
                st.session_state.prev_target != st.session_state.selected_target):
                st.session_state.models_trained = False
                st.session_state.forecasts = {}
                st.session_state.metrics = {}
            
            if ('prev_station' in st.session_state and 
                st.session_state.prev_station != st.session_state.selected_station):
                st.session_state.models_trained = False
                st.session_state.forecasts = {}
                st.session_state.metrics = {}
        
        # 현재 선택 저장
        st.session_state.prev_target = st.session_state.selected_target
        st.session_state.prev_station = st.session_state.selected_station

def prepare_train_test_data(test_size=None):
    """
    훈련/테스트 데이터 분할 준비
    
    Args:
        test_size: 테스트 데이터 비율 (기본값: None)
    """
    if test_size is None:
        test_size = st.session_state.test_size
        
    if st.session_state.series is not None:
        st.session_state.train, st.session_state.test = cached_train_test_split(
            st.session_state.series, 
            test_size
        )
        return True
    return False

def analyze_decomposition(period=None):
    """
    시계열 분해 분석 수행
    
    Args:
        period: 계절성 주기 (기본값: None)
    
    Returns:
        dict: 분해 결과 딕셔너리
    """
    if period is None:
        period = st.session_state.period
        
    if st.session_state.series is not None:
        try:
            decomposition = cached_decompose_timeseries(st.session_state.series, period)
            st.session_state.decomposition = decomposition
            return decomposition
        except Exception as e:
            st.error(f"시계열 분해 중 오류 발생: {str(e)}")
            return None
    return None

def analyze_stationarity():
    """
    정상성 검정 수행
    
    Returns:
        dict: 정상성 검정 결과 딕셔너리
    """
    if st.session_state.series is not None:
        try:
            stationarity_result = cached_check_stationarity(st.session_state.series)
            st.session_state.stationarity_result = stationarity_result
            return stationarity_result
        except Exception as e:
            st.error(f"정상성 검정 중 오류 발생: {str(e)}")
            return None
    return None

def analyze_acf_pacf(nlags=40):
    """
    ACF/PACF 분석 수행
    
    Args:
        nlags: 최대 시차 (기본값: 40)
    
    Returns:
        tuple: (ACF 값, PACF 값) 튜플
    """
    if st.session_state.series is not None:
        try:
            acf_values, pacf_values = cached_get_acf_pacf(st.session_state.series, nlags)
            st.session_state.acf_values = acf_values
            st.session_state.pacf_values = pacf_values
            return acf_values, pacf_values
        except Exception as e:
            st.error(f"ACF/PACF 분석 중 오류 발생: {str(e)}")
            return None, None
    return None, None

def safe_len(obj, default=10):
    """
    None이 아닌 객체의 길이를 안전하게 반환
    
    Args:
        obj: 길이를 확인할 객체
        default: 기본값 (기본값: 10)
    
    Returns:
        int: 객체의 길이 또는 기본값
    """
    if obj is not None:
        return len(obj)
    return default

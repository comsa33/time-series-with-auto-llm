"""
데이터 관련 서비스 모듈
"""
import os
from datetime import datetime, timedelta

import streamlit as st
import pandas as pd
import numpy as np

from config.settings import app_config
from utils.data_reader import get_seoul_air_quality
from utils.data_processor import (
    cached_preprocess_data,
    cached_train_test_split,
    cached_decompose_timeseries,
    cached_check_stationarity,
    cached_get_acf_pacf,
    cached_recommend_differencing,
    cached_perform_differencing,
    cached_apply_inverse_differencing
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
        test_size = st.session_state.test_size if 'test_size' in st.session_state else 0.2  # Default to 20% if undefined
        
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

def analyze_differencing_need():
    """
    차분 필요성 분석을 수행합니다.
    
    Returns:
        dict: 차분 추천 정보를 담은 딕셔너리
    """
    if st.session_state.series is not None:
        try:
            # 먼저 ACF, PACF 분석이 있는지 확인
            if st.session_state.acf_values is None or st.session_state.pacf_values is None:
                acf_values, pacf_values = analyze_acf_pacf()
            else:
                acf_values, pacf_values = st.session_state.acf_values, st.session_state.pacf_values
                
            # 차분 추천 실행
            recommendation = cached_recommend_differencing(st.session_state.series, acf_values, pacf_values)
            st.session_state.differencing_recommendation = recommendation
            return recommendation
        except Exception as e:
            st.error(f"차분 필요성 분석 중 오류 발생: {str(e)}")
            return None
    return None

def perform_differencing(diff_order=None, seasonal_diff_order=None, seasonal_period=None):
    """
    시계열 데이터에 차분을 적용합니다.
    
    Args:
        diff_order: 일반 차분 차수 (기본값: None, 추천값 사용)
        seasonal_diff_order: 계절 차분 차수 (기본값: None, 추천값 사용)
        seasonal_period: 계절성 주기 (기본값: None, 추천값 사용)
        
    Returns:
        차분된 시계열 데이터
    """
    if st.session_state.series is None:
        return None
        
    try:
        # 파라미터 설정
        if diff_order is None:
            if st.session_state.differencing_recommendation:
                diff_order = st.session_state.differencing_recommendation['diff_order']
            else:
                diff_order = st.session_state.diff_order or 0
                
        if seasonal_diff_order is None:
            if st.session_state.differencing_recommendation:
                seasonal_diff_order = st.session_state.differencing_recommendation['seasonal_diff_order']
            else:
                seasonal_diff_order = st.session_state.seasonal_diff_order or 0
                
        if seasonal_period is None:
            if st.session_state.differencing_recommendation and st.session_state.differencing_recommendation['seasonal_period']:
                seasonal_period = st.session_state.differencing_recommendation['seasonal_period']
            else:
                seasonal_period = st.session_state.period
        
        # 세션 상태 업데이트
        st.session_state.diff_order = diff_order
        st.session_state.seasonal_diff_order = seasonal_diff_order
        
        # 차분 실행
        differenced_series = cached_perform_differencing(
            st.session_state.series, 
            diff_order, 
            seasonal_diff_order, 
            seasonal_period
        )
        
        st.session_state.differenced_series = differenced_series
        return differenced_series
        
    except Exception as e:
        st.error(f"차분 적용 중 오류 발생: {str(e)}")
        return None

def prepare_differenced_train_test_data(test_size=None):
    """
    차분된 시계열 데이터를 훈련/테스트 세트로 분할합니다.
    
    Args:
        test_size: 테스트 데이터 비율 (기본값: None, 세션 상태 사용)
        
    Returns:
        bool: 성공 여부
    """
    if test_size is None:
        test_size = st.session_state.test_size
        
    if st.session_state.differenced_series is not None:
        st.session_state.diff_train, st.session_state.diff_test = cached_train_test_split(
            st.session_state.differenced_series, 
            test_size
        )
        
        # 원본 데이터도 함께 분할 (시각화용)
        if st.session_state.series is not None:
            st.session_state.train, st.session_state.test = cached_train_test_split(
                st.session_state.series,
                test_size
            )
            
        return True
    return False

def inverse_transform_forecast(forecast, original_series=None, diff_order=None, seasonal_diff_order=None, seasonal_period=None):
    """
    차분된 데이터로 예측한 결과를 원래 스케일로 변환합니다.
    
    Args:
        forecast: 차분된 데이터에 대한 예측값
        original_series: 원본 시계열 데이터 (기본값: None, 세션 상태 사용)
        diff_order: 적용된 일반 차분 차수 (기본값: None, 세션 상태 사용)
        seasonal_diff_order: 적용된 계절 차분 차수 (기본값: None, 세션 상태 사용)
        seasonal_period: 적용된 계절성 주기 (기본값: None, 세션 상태 사용)
        
    Returns:
        원래 스케일로 변환된 예측값
    """
    try:
        # 기본값 설정 (train/diff_train 모두 확인)
        if original_series is None:
            if hasattr(st.session_state, 'train') and st.session_state.train is not None:
                original_series = st.session_state.train
            elif hasattr(st.session_state, 'diff_train') and st.session_state.diff_train is not None:
                # 차분된 데이터를 사용할 경우, 원본 시계열로 되돌리기 위해 원래 시계열 필요
                original_series = st.session_state.series
            else:
                raise ValueError("원본 시계열 데이터를 찾을 수 없습니다.")
            
        if diff_order is None:
            diff_order = st.session_state.diff_order if hasattr(st.session_state, 'diff_order') else 0
            
        if seasonal_diff_order is None:
            seasonal_diff_order = st.session_state.seasonal_diff_order if hasattr(st.session_state, 'seasonal_diff_order') else 0
            
        if seasonal_period is None:
            if hasattr(st.session_state, 'differencing_recommendation') and st.session_state.differencing_recommendation and st.session_state.differencing_recommendation.get('seasonal_period'):
                seasonal_period = st.session_state.differencing_recommendation['seasonal_period']
            else:
                seasonal_period = st.session_state.period if hasattr(st.session_state, 'period') else 24
        
        # 예측값을 시리즈로 변환 (인덱스 설정)
        if not isinstance(forecast, pd.Series):
            if isinstance(forecast, np.ndarray):
                # 테스트 데이터 인덱스 사용
                if hasattr(st.session_state, 'test') and st.session_state.test is not None:
                    test_index = st.session_state.test.index
                    
                    # 길이 조정 - 이 부분이 핵심 수정사항
                    min_len = min(len(forecast), len(test_index))
                    forecast = forecast[:min_len]
                    test_index = test_index[:min_len]
                    
                    forecast_series = pd.Series(forecast, index=test_index)
                else:
                    # 인덱스를 만들 수 없는 경우, 예측 길이만큼의 인덱스 생성
                    last_date = original_series.index[-1]
                    freq = pd.infer_freq(original_series.index)
                    test_index = pd.date_range(start=last_date, periods=len(forecast)+1, freq=freq)[1:]
                
                forecast_series = pd.Series(forecast, index=test_index)
        else:
            forecast_series = forecast
        
        inverted_forecast = cached_apply_inverse_differencing(
            forecast_series,
            original_series,
            diff_order,
            seasonal_diff_order,
            seasonal_period
        )
        
        # 유효성 검사 추가
        if inverted_forecast is None or len(inverted_forecast) == 0:
            st.warning(f"역변환 결과가 비어 있습니다. 원본 예측 결과를 반환합니다.")
            return forecast_series
            
        return inverted_forecast
        
    except Exception as e:
        st.error(f"예측 결과 역변환 중 오류 발생: {str(e)}")
        import traceback
        st.error(f"상세 오류: {traceback.format_exc()}")
        
        # 오류 발생 시 원본 예측값 반환 - 여기도 길이 조정 추가
        st.warning("역변환 실패로 원본 예측값을 반환합니다.")
        if not isinstance(forecast, pd.Series):
            if isinstance(forecast, np.ndarray) and len(forecast) > 0:
                # test 데이터의 인덱스 사용
                if hasattr(st.session_state, 'test') and st.session_state.test is not None:
                    test_index = st.session_state.test.index
                    min_len = min(len(forecast), len(test_index))
                    
                    # 길이 맞추기
                    return pd.Series(forecast[:min_len], index=test_index[:min_len])
        return forecast

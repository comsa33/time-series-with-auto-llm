"""
모델 학습 및 예측 관련 서비스 모듈
"""
import traceback
from typing import List, Dict, Any

import streamlit as st
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from backend.data_service import safe_len
from utils.parameter_utils import validate_model_parameters

# 모델 팩토리 동적 로드
@st.cache_resource
def get_model_factory():
    """
    모델 팩토리를 동적으로 로드합니다.
    필요할 때만 import하여 시작 시 pmdarima 오류를 방지합니다.
    
    Returns:
        ModelFactory: 모델 팩토리 인스턴스 또는 None
    """
    try:
        from models.model_factory import ModelFactory
        return ModelFactory()
    except Exception as e:
        st.error(f"모델 로드 중 오류 발생: {traceback.format_exc()}")
        st.error("필요한 라이브러리를 재설치하세요: pip uninstall -y pmdarima numpy && pip install numpy==1.24.3 && pip install pmdarima==2.0.4")
        return None

# ARIMA 모델용 캐싱 함수
@st.cache_data(ttl=3600)
def cached_train_arima(train_data_key, test_data_key, seasonal, m, **kwargs):
    """
    ARIMA 모델 학습 결과를 캐싱합니다.
    
    Args:
        train_data_key: 훈련 데이터 해시
        test_data_key: 테스트 데이터 해시
        seasonal: 계절성 모델 여부
        m: 계절성 주기
        **kwargs: 추가 파라미터
        
    Returns:
        tuple: (예측값, 성능 지표)
    """
    try:
        model_factory = get_model_factory()
        if model_factory is None:
            return None, None
        
        model = model_factory.get_model('arima')
        # 모든 추가 파라미터를 model.fit_predict_evaluate로 전달
        forecast, metrics = model.fit_predict_evaluate(
            st.session_state.train, 
            st.session_state.test,
            seasonal=seasonal,
            m=m,
            **kwargs  # 여기서 arima_params의 내용이 전달됨
        )
        return forecast, metrics
    except Exception as e:
        st.error(f"ARIMA 모델 학습 오류: {traceback.format_exc()}")
        return None, None

# Prophet 모델용 캐싱 함수
@st.cache_data(ttl=3600)
def cached_train_prophet(train_data_key, test_data_key, **kwargs):
    """
    Prophet 모델 학습 결과를 캐싱합니다.
    
    Args:
        train_data_key: 훈련 데이터 해시
        test_data_key: 테스트 데이터 해시
        **kwargs: 추가 파라미터
        
    Returns:
        tuple: (예측값, 성능 지표)
    """
    try:
        model_factory = get_model_factory()
        if model_factory is None:
            return None, None
        
        # 데이터 확인 (st.session_state.diff_train, diff_test 또는 train, test)
        train_data = None
        test_data = None
        
        if hasattr(st.session_state, 'train') and st.session_state.train is not None:
            train_data = st.session_state.train
        elif hasattr(st.session_state, 'diff_train') and st.session_state.diff_train is not None:
            train_data = st.session_state.diff_train
            
        if hasattr(st.session_state, 'test') and st.session_state.test is not None:
            test_data = st.session_state.test
        elif hasattr(st.session_state, 'diff_test') and st.session_state.diff_test is not None:
            test_data = st.session_state.diff_test
            
        if train_data is None or test_data is None:
            st.error("Prophet 모델을 위한 데이터가 없습니다.")
            return None, None
        
        model = model_factory.get_model('prophet')
        # 모든 추가 파라미터를 model.fit_predict_evaluate로 전달
        forecast, metrics = model.fit_predict_evaluate(
            train_data, 
            test_data,
            **kwargs  # prophet_params의 내용이 여기로 전달됨
        )
        return forecast, metrics
    except Exception as e:
        st.error(f"Prophet 모델 학습 오류: {e}")
        return None, None

# LSTM 모델용 캐싱 함수
@st.cache_data(ttl=3600)
def cached_train_lstm(train_data_key, test_data_key, **kwargs):
    """
    LSTM 모델 학습 결과를 캐싱합니다.
    
    Args:
        train_data_key: 훈련 데이터 해시
        test_data_key: 테스트 데이터 해시
        **kwargs: 추가 파라미터
        
    Returns:
        tuple: (예측값, 성능 지표)
    """
    try:
        model_factory = get_model_factory()
        if model_factory is None:
            return None, None
        
        # 데이터 확인
        train_data = None
        test_data = None
        
        if hasattr(st.session_state, 'train') and st.session_state.train is not None:
            train_data = st.session_state.train
        elif hasattr(st.session_state, 'diff_train') and st.session_state.diff_train is not None:
            train_data = st.session_state.diff_train
            
        if hasattr(st.session_state, 'test') and st.session_state.test is not None:
            test_data = st.session_state.test
        elif hasattr(st.session_state, 'diff_test') and st.session_state.diff_test is not None:
            test_data = st.session_state.diff_test
            
        if train_data is None or test_data is None:
            st.error("LSTM 모델을 위한 데이터가 없습니다.")
            return None, None
        
        model = model_factory.get_model('lstm')
        forecast, metrics = model.fit_predict_evaluate(
            train_data, 
            test_data,
            **kwargs  # 모든 lstm_params가 여기로 전달됨
        )
        return forecast, metrics
    except Exception as e:
        st.error(f"LSTM 모델 학습 오류: {e}")
        return None, None

@st.cache_data(ttl=3600)
def cached_train_exp_smoothing(train_data_key, test_data_key, seasonal_periods):
    """
    지수평활법 모델 학습 결과를 캐싱합니다.
    
    Args:
        train_data_key: 훈련 데이터 해시
        test_data_key: 테스트 데이터 해시
        seasonal_periods: 계절성 주기
        
    Returns:
        tuple: (예측값, 성능 지표)
    """
    try:
        model_factory = get_model_factory()
        if model_factory is None:
            return None, None
        
        # 데이터 확인
        train_data = None
        test_data = None
        
        if hasattr(st.session_state, 'train') and st.session_state.train is not None:
            train_data = st.session_state.train
        elif hasattr(st.session_state, 'diff_train') and st.session_state.diff_train is not None:
            train_data = st.session_state.diff_train
            
        if hasattr(st.session_state, 'test') and st.session_state.test is not None:
            test_data = st.session_state.test
        elif hasattr(st.session_state, 'diff_test') and st.session_state.diff_test is not None:
            test_data = st.session_state.diff_test
            
        if train_data is None or test_data is None:
            st.error("지수평활법 모델을 위한 데이터가 없습니다.")
            return None, None
        
        model = model_factory.get_model('exp_smoothing')
        forecast, metrics = model.fit_predict_evaluate(
            train_data, 
            test_data,
            seasonal_periods=seasonal_periods
        )
        return forecast, metrics
    except Exception as e:
        st.error(f"지수평활법 모델 학습 오류: {e}")
        return None, None

@st.cache_data(ttl=3600)
def cached_train_arima(train_data_key, test_data_key, seasonal, m, **kwargs):
    """
    ARIMA 모델 학습 결과를 캐싱합니다.
    
    Args:
        train_data_key: 훈련 데이터 해시
        test_data_key: 테스트 데이터 해시
        seasonal: 계절성 모델 여부
        m: 계절성 주기
        **kwargs: 추가 파라미터
        
    Returns:
        tuple: (예측값, 성능 지표)
    """
    try:
        model_factory = get_model_factory()
        if model_factory is None:
            return None, None
        
        # 데이터 확인
        train_data = None
        test_data = None
        
        if hasattr(st.session_state, 'train') and st.session_state.train is not None:
            train_data = st.session_state.train
        elif hasattr(st.session_state, 'diff_train') and st.session_state.diff_train is not None:
            train_data = st.session_state.diff_train
            
        if hasattr(st.session_state, 'test') and st.session_state.test is not None:
            test_data = st.session_state.test
        elif hasattr(st.session_state, 'diff_test') and st.session_state.diff_test is not None:
            test_data = st.session_state.diff_test
            
        if train_data is None or test_data is None:
            st.error("ARIMA 모델을 위한 데이터가 없습니다.")
            return None, None
        
        model = model_factory.get_model('arima')
        # 모든 추가 파라미터를 model.fit_predict_evaluate로 전달
        forecast, metrics = model.fit_predict_evaluate(
            train_data, 
            test_data,
            seasonal=seasonal,
            m=m,
            **kwargs  # 여기서 arima_params의 내용이 전달됨
        )
        return forecast, metrics
    except Exception as e:
        st.error(f"ARIMA 모델 학습 오류: {e}")
        return None, None

@st.cache_data(ttl=3600)
def cached_train_arima_differenced(train_data_key, test_data_key, seasonal, m, **kwargs):
    """차분된 데이터로 ARIMA 모델 학습 결과를 캐싱합니다."""
    try:
        model_factory = get_model_factory()
        if model_factory is None:
            return None, None
        
        # 데이터 확인
        if hasattr(st.session_state, 'diff_train') and st.session_state.diff_train is not None and hasattr(st.session_state, 'diff_test') and st.session_state.diff_test is not None:
            train_data = st.session_state.diff_train
            test_data = st.session_state.diff_test
        else:
            st.error("차분된 ARIMA 모델을 위한 데이터가 없습니다.")
            return None, None
        
        model = model_factory.get_model('arima')
        forecast, metrics = model.fit_predict_evaluate(
            train_data, 
            test_data,
            seasonal=seasonal,
            m=m,
            **kwargs
        )
        return forecast, metrics
    except Exception as e:
        st.error(f"차분 데이터로 ARIMA 모델 학습 오류: {e}")
        return None, None

@st.cache_data(ttl=3600)
def cached_train_exp_smoothing_differenced(train_data_key, test_data_key, seasonal_periods):
    """차분된 데이터로 지수평활법 모델 학습 결과를 캐싱합니다."""
    try:
        model_factory = get_model_factory()
        if model_factory is None:
            return None, None
        
        # 데이터 확인
        if hasattr(st.session_state, 'diff_train') and st.session_state.diff_train is not None and hasattr(st.session_state, 'diff_test') and st.session_state.diff_test is not None:
            train_data = st.session_state.diff_train
            test_data = st.session_state.diff_test
        else:
            st.error("차분된 지수평활법 모델을 위한 데이터가 없습니다.")
            return None, None
        
        model = model_factory.get_model('exp_smoothing')
        forecast, metrics = model.fit_predict_evaluate(
            train_data, 
            test_data,
            seasonal_periods=seasonal_periods
        )
        return forecast, metrics
    except Exception as e:
        st.error(f"차분 데이터로 지수평활법 모델 학습 오류: {e}")
        return None, None

@st.cache_data(ttl=3600)
def cached_train_lstm_differenced(train_data_key, test_data_key, **kwargs):
    """차분된 데이터로 LSTM 모델 학습 결과를 캐싱합니다."""
    try:
        model_factory = get_model_factory()
        if model_factory is None:
            return None, None
        
        # 데이터 확인
        if hasattr(st.session_state, 'diff_train') and st.session_state.diff_train is not None and hasattr(st.session_state, 'diff_test') and st.session_state.diff_test is not None:
            train_data = st.session_state.diff_train
            test_data = st.session_state.diff_test
        else:
            st.error("차분된 LSTM 모델을 위한 데이터가 없습니다.")
            return None, None
        
        model = model_factory.get_model('lstm')
        forecast, metrics = model.fit_predict_evaluate(
            train_data, 
            test_data,
            **kwargs
        )
        return forecast, metrics
    except Exception as e:
        st.error(f"차분 데이터로 LSTM 모델 학습 오류: {e}")
        return None, None

def evaluate_prediction(actual: pd.Series, predicted: np.ndarray) -> Dict[str, float]:
    """
    예측 결과를 평가합니다.
    
    Args:
        actual: 실제 값
        predicted: 예측 값
        
    Returns:
        성능 지표 딕셔너리
    """
    
    # None 값 검사
    if actual is None or predicted is None:
        return {
            'MSE': float('nan'),
            'RMSE': float('nan'),
            'MAE': float('nan'),
            'R^2': float('nan'),
            'MAPE': float('nan')
        }
    
    # 길이 맞춤
    min_len = min(len(actual), len(predicted))
    actual = actual.iloc[:min_len]
    predicted = predicted[:min_len]
    
    # 성능 지표 계산
    mse = mean_squared_error(actual, predicted)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(actual, predicted)
    r2 = r2_score(actual, predicted)
    
    # MAPE 계산 (실제값이 0이 아닌 경우만)
    mask = actual != 0
    if mask.any():
        mape = np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100
    else:
        mape = np.nan
    
    return {
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'R^2': r2,
        'MAPE': mape
    }

def train_models(selected_models, complexity):
    """
    선택된 모델 학습 및 예측 수행
    
    Args:
        selected_models: 선택된 모델 목록
        complexity: 모델 복잡도 설정
    
    Returns:
        bool: 학습 성공 여부
    """
    # 복잡도별 파라미터 설정
    if complexity == '간단 (빠름, 저메모리)':
        arima_params = {
            'max_p': 1, 'max_q': 1, 'max_P': 0, 'max_Q': 0,
            'stepwise': True, 'n_jobs': 1
        }
        lstm_params = {
            'n_steps': min(24, safe_len(st.session_state.train, 100) // 20),
            'lstm_units': [32],
            'epochs': 30
        }
        prophet_params = {
            'daily_seasonality': False,
            'weekly_seasonality': True,
            'changepoint_prior_scale': 0.01
        }
    elif complexity == '중간':
        arima_params = {
            'max_p': 2, 'max_q': 2, 'max_P': 1, 'max_Q': 1,
            'stepwise': True, 'n_jobs': 1
        }
        lstm_params = {
            'n_steps': min(48, safe_len(st.session_state.train, 100) // 10),
            'lstm_units': [50],
            'epochs': 50
        }
        prophet_params = {
            'daily_seasonality': True,
            'weekly_seasonality': True,
            'changepoint_prior_scale': 0.05
        }
    else:  # 복잡 (정확도 높음, 고메모리)
        arima_params = {
            'max_p': 5, 'max_q': 5, 'max_P': 2, 'max_Q': 2,
            'stepwise': True, 'n_jobs': 1
        }
        lstm_params = {
            'n_steps': min(72, safe_len(st.session_state.train, 100) // 8),
            'lstm_units': [50, 50],
            'epochs': 100
        }
        prophet_params = {
            'daily_seasonality': True,
            'weekly_seasonality': True,
            'yearly_seasonality': True,
            'changepoint_prior_scale': 0.05
        }
    
    # 데이터 준비 및 키 생성 (캐싱용)
    if st.session_state.use_differencing:
        # 차분 데이터가 없으면 생성
        if st.session_state.differenced_series is None:
            from backend.data_service import perform_differencing
            perform_differencing()
            
        if st.session_state.diff_train is None or st.session_state.diff_test is None:
            from backend.data_service import prepare_differenced_train_test_data
            prepare_differenced_train_test_data()
        
        # 모든 데이터 키 생성 (차분 및 원본 모두)
        if st.session_state.diff_train is not None and st.session_state.diff_test is not None:
            train_data_key = hash(tuple(st.session_state.diff_train.values.tolist()))
            test_data_key = hash(tuple(st.session_state.diff_test.values.tolist()))
            
            # 모델마다 다른 데이터를 사용할 수 있으므로 원본 데이터 키도 생성
            if st.session_state.train is not None and st.session_state.test is not None:
                original_train_key = hash(tuple(st.session_state.train.values.tolist()))
                original_test_key = hash(tuple(st.session_state.test.values.tolist()))
            else:
                # 원본 데이터 없으면 차분 데이터 키로 대체
                original_train_key = train_data_key
                original_test_key = test_data_key
        else:
            st.error("차분 데이터를 준비할 수 없습니다. 원본 데이터를 사용합니다.")
            st.session_state.use_differencing = False
            if st.session_state.train is not None and st.session_state.test is not None:
                train_data_key = hash(tuple(st.session_state.train.values.tolist()))
                test_data_key = hash(tuple(st.session_state.test.values.tolist()))
            else:
                st.error("모델 학습에 필요한 데이터가 없습니다.")
                return False
    else:
        # 원본 데이터 학습 시
        if st.session_state.train is not None and st.session_state.test is not None:
            train_data_key = hash(tuple(st.session_state.train.values.tolist()))
            test_data_key = hash(tuple(st.session_state.test.values.tolist()))
        else:
            st.error("모델 학습에 필요한 데이터가 없습니다.")
            return False
    
    # 진행 상황 표시
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # 예측 결과 및 메트릭 저장
    forecasts = {}
    metrics = {}
    
    # 모델 개수
    total_models = len(selected_models)
    completed_models = 0

    # 모델 파라미터 저장을 위한 초기화
    if 'model_params' not in st.session_state:
        st.session_state.model_params = {}
    
    # 각 모델 학습 및 예측
    for model_type in selected_models:
        status_text.text(f"{model_type} 모델 학습 중...")
        
        try:
            # 모델별 캐싱된 학습 함수 호출
            if model_type == 'arima':
                # 파라미터 저장
                st.session_state.model_params[model_type] = {
                    'order': arima_params.get('order', (1, 1, 1)),
                    'seasonal_order': arima_params.get('seasonal_order', (1, 1, 1, st.session_state.period)),
                    'seasonal': True,
                    'm': st.session_state.period,
                    **arima_params
                }
                
                # 차분 데이터 사용 여부에 따라 다른 함수 호출
                if st.session_state.use_differencing and st.session_state.diff_train is not None and st.session_state.diff_test is not None:
                    # ARIMA 모델은 차분 기능을 내장하고 있으므로,
                    # 차분을 이미 했다면 차수를 줄일 수 있음
                    modified_arima_params = arima_params.copy()
                    if 'order' in modified_arima_params:
                        p, d, q = modified_arima_params['order']
                        # 이미 차분을 했으므로 d를 줄임
                        modified_arima_params['order'] = (p, max(0, d - st.session_state.diff_order), q)
                    
                    try:
                        forecast, model_metrics = cached_train_arima_differenced(
                            train_data_key, 
                            test_data_key,
                            seasonal=True,
                            m=st.session_state.period,
                            **modified_arima_params
                        )
                        
                        # 역변환 적용
                        if forecast is not None:
                            from backend.data_service import inverse_transform_forecast
                            try:
                                forecast = inverse_transform_forecast(forecast)
                            except Exception as e:
                                st.error(f"ARIMA 예측 결과 역변환 중 오류: {e}")
                                forecast = None
                                model_metrics = None
                    except Exception as e:
                        st.error(f"차분 데이터로 ARIMA 모델 학습 중 오류: {e}")
                        # 원본 데이터로 학습 시도
                        if st.session_state.train is not None and st.session_state.test is not None:
                            st.warning("원본 데이터로 ARIMA 모델 학습을 시도합니다.")
                            train_key = hash(tuple(st.session_state.train.values.tolist()))
                            test_key = hash(tuple(st.session_state.test.values.tolist()))
                            forecast, model_metrics = cached_train_arima(
                                train_key, 
                                test_key,
                                seasonal=True,
                                m=st.session_state.period,
                                **arima_params
                            )
                        else:
                            forecast = None
                            model_metrics = None
                else:
                    # 원본 데이터로 학습
                    if st.session_state.train is not None and st.session_state.test is not None:
                        forecast, model_metrics = cached_train_arima(
                            train_data_key, 
                            test_data_key,
                            seasonal=True,
                            m=st.session_state.period,
                            **arima_params
                        )
                    else:
                        st.error(f"ARIMA 모델 학습을 위한 데이터가 없습니다.")
                        forecast = None
                        model_metrics = None
                    
            elif model_type == 'exp_smoothing':
                # 파라미터 저장
                st.session_state.model_params[model_type] = {
                    'model_type': 'hw',
                    'trend': 'add',
                    'seasonal': 'add',
                    'seasonal_periods': st.session_state.period,
                    'damped_trend': False
                }
                
                # 차분 데이터 사용 여부에 따라 다른 함수 호출
                if st.session_state.use_differencing and st.session_state.diff_train is not None and st.session_state.diff_test is not None:
                    try:
                        forecast, model_metrics = cached_train_exp_smoothing_differenced(
                            train_data_key, 
                            test_data_key,
                            seasonal_periods=st.session_state.period
                        )
                        
                        # 역변환 적용
                        if forecast is not None:
                            from backend.data_service import inverse_transform_forecast
                            try:
                                forecast = inverse_transform_forecast(forecast)
                            except Exception as e:
                                st.error(f"지수평활법 예측 결과 역변환 중 오류: {e}")
                                forecast = None
                                model_metrics = None
                    except Exception as e:
                        st.error(f"차분 데이터로 지수평활법 모델 학습 중 오류: {e}")
                        # 원본 데이터로 학습 시도
                        if st.session_state.train is not None and st.session_state.test is not None:
                            st.warning("원본 데이터로 지수평활법 모델 학습을 시도합니다.")
                            train_key = hash(tuple(st.session_state.train.values.tolist()))
                            test_key = hash(tuple(st.session_state.test.values.tolist()))
                            forecast, model_metrics = cached_train_exp_smoothing(
                                train_key, 
                                test_key,
                                seasonal_periods=st.session_state.period
                            )
                        else:
                            forecast = None
                            model_metrics = None
                else:
                    # 원본 데이터로 학습
                    if st.session_state.train is not None and st.session_state.test is not None:
                        forecast, model_metrics = cached_train_exp_smoothing(
                            train_data_key, 
                            test_data_key,
                            seasonal_periods=st.session_state.period
                        )
                    else:
                        st.error(f"지수평활법 모델 학습을 위한 데이터가 없습니다.")
                        forecast = None
                        model_metrics = None
                    
            elif model_type == 'exp_smoothing':
                # 파라미터 저장
                st.session_state.model_params[model_type] = {
                    'model_type': 'hw',
                    'trend': 'add',
                    'seasonal': 'add',
                    'seasonal_periods': st.session_state.period,
                    'damped_trend': False
                }
                
                # 차분 데이터 사용 여부에 따라 다른 함수 호출
                if st.session_state.use_differencing:
                    forecast, model_metrics = cached_train_exp_smoothing_differenced(
                        train_data_key, 
                        test_data_key,
                        seasonal_periods=st.session_state.period
                    )
                    
                    # 역변환 적용
                    if forecast is not None:
                        from backend.data_service import inverse_transform_forecast
                        forecast = inverse_transform_forecast(forecast)
                else:
                    forecast, model_metrics = cached_train_exp_smoothing(
                        train_data_key, 
                        test_data_key,
                        seasonal_periods=st.session_state.period
                    )
                    
            elif model_type == 'prophet':
                # 파라미터 저장
                st.session_state.model_params[model_type] = {
                    'daily_seasonality': prophet_params.get('daily_seasonality', False),
                    'weekly_seasonality': prophet_params.get('weekly_seasonality', True),
                    'yearly_seasonality': prophet_params.get('yearly_seasonality', False),
                    'changepoint_prior_scale': prophet_params.get('changepoint_prior_scale', 0.05)
                }
                
                # Prophet은 차분을 내장하고 있으므로 원본 데이터를 우선적으로 사용
                # train과 test가 None이 아닌지 확인
                if st.session_state.train is not None and st.session_state.test is not None:
                    train_hash = hash(tuple(st.session_state.train.values.tolist()))
                    test_hash = hash(tuple(st.session_state.test.values.tolist()))
                    forecast, model_metrics = cached_train_prophet(
                        train_hash, 
                        test_hash,
                        **prophet_params
                    )
                else:
                    # 원본 데이터가 없는 경우, 차분 데이터로 대체
                    st.warning("Prophet 모델은 원본 데이터 사용이 권장됩니다. 차분 데이터로 학습합니다.")
                    # 차분 데이터의 해시 생성
                    train_hash = hash(tuple(st.session_state.diff_train.values.tolist()))
                    test_hash = hash(tuple(st.session_state.diff_test.values.tolist()))
                    
                    # Prophet 모델 학습
                    forecast, model_metrics = cached_train_prophet(
                        train_hash, 
                        test_hash,
                        **prophet_params
                    )
                    
                    # 역변환 적용
                    if forecast is not None:
                        from backend.data_service import inverse_transform_forecast
                        forecast = inverse_transform_forecast(forecast)
                
            elif model_type == 'lstm':
                # 파라미터 저장
                st.session_state.model_params[model_type] = {
                    'n_steps': lstm_params.get('n_steps', 24),
                    'lstm_units': lstm_params.get('lstm_units', [50]),
                    'epochs': lstm_params.get('epochs', 50)
                }
                
                # LSTM은 스케일링이 내장되어 있으므로 차분 데이터도 사용 가능
                if st.session_state.use_differencing and st.session_state.diff_train is not None and st.session_state.diff_test is not None:
                    try:
                        forecast, model_metrics = cached_train_lstm_differenced(
                            train_data_key, 
                            test_data_key,
                            **lstm_params
                        )
                        
                        # 역변환 적용
                        if forecast is not None:
                            from backend.data_service import inverse_transform_forecast
                            try:
                                forecast = inverse_transform_forecast(forecast)
                            except Exception as e:
                                st.error(f"LSTM 예측 결과 역변환 중 오류: {e}")
                                forecast = None
                                model_metrics = None
                    except Exception as e:
                        st.error(f"차분 데이터로 LSTM 모델 학습 중 오류: {e}")
                        # 원본 데이터로 학습 시도
                        if st.session_state.train is not None and st.session_state.test is not None:
                            st.warning("원본 데이터로 LSTM 모델 학습을 시도합니다.")
                            train_key = hash(tuple(st.session_state.train.values.tolist()))
                            test_key = hash(tuple(st.session_state.test.values.tolist()))
                            forecast, model_metrics = cached_train_lstm(
                                train_key, 
                                test_key,
                                **lstm_params
                            )
                        else:
                            forecast = None
                            model_metrics = None
                else:
                    # 원본 데이터로 학습
                    if st.session_state.train is not None and st.session_state.test is not None:
                        forecast, model_metrics = cached_train_lstm(
                            hash(tuple(st.session_state.train.values.tolist())), 
                            hash(tuple(st.session_state.test.values.tolist())),
                            **lstm_params
                        )
                    else:
                        st.error(f"LSTM 모델 학습을 위한 데이터가 없습니다.")
                        forecast = None
                        model_metrics = None
            
            # 유효한 결과만 저장
            if forecast is not None and model_metrics is not None:
                # 차분 데이터로 학습한 경우 메트릭 재계산 (원본 스케일로)
                if st.session_state.use_differencing:
                    model_name = model_metrics.get('name', model_type)
                    
                    # 차분 역변환 후 원본 test와 비교 (test가 있는 경우만)
                    if hasattr(st.session_state, 'test') and st.session_state.test is not None:
                        try:
                            metrics_from_test = evaluate_prediction(st.session_state.test, forecast)
                            model_metrics.update(metrics_from_test)
                        except Exception as e:
                            st.error(f"원본 스케일 메트릭 계산 중 오류: {e}")
                            # 오류 발생 시 기존 메트릭 유지
                            pass
                    else:
                        # 원본 test가 없으면 NaN 값으로 대체
                        model_metrics.update({
                            'MSE': float('nan'),
                            'RMSE': float('nan'),
                            'MAE': float('nan'),
                            'R^2': float('nan'),
                            'MAPE': float('nan')
                        })
                    
                    model_metrics['name'] = model_name  # 이름 복원
                
                forecasts[model_metrics.get('name', model_type)] = forecast
                metrics[model_metrics.get('name', model_type)] = model_metrics
            
            # 진행 상황 업데이트
            completed_models += 1
            progress_bar.progress(completed_models / total_models)
            
        except Exception as e:
            st.error(f"{model_type} 모델 학습 중 오류 발생: {traceback.format_exc()}")
    
    # 모든 모델 학습 완료 후 결과 저장
    if forecasts:
        st.session_state.forecasts = forecasts
        st.session_state.metrics = metrics
        st.session_state.models_trained = True
        
        # 최적 모델 선택
        rmse_values = {model: metrics[model]['RMSE'] for model in metrics}
        st.session_state.best_model = min(rmse_values.items(), key=lambda x: x[1])[0]
        
        status_text.text("모든 모델 학습 완료!")
        return True
    else:
        st.error("모델 학습 중 오류가 발생했습니다.")
        return False


def train_models_with_params(selected_models: List[str], tuned_params: Dict[str, Any], prefix: str = "") -> bool:
    """
    조정된 파라미터로 모델 학습 및 예측 수행
    """
    # 데이터 준비 및 캐싱 키 생성
    train_data_key = hash(tuple(st.session_state.train.values.tolist()))
    test_data_key = hash(tuple(st.session_state.test.values.tolist()))
    
    # 진행 상황 표시
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # 결과 저장용 변수
    tuned_forecasts = {}
    tuned_metrics = {}
    
    # 모델 파라미터 저장 공간 초기화
    if 'model_params' not in st.session_state:
        st.session_state.model_params = {}
    
    # 각 모델 학습 및 예측
    for i, model_type in enumerate(selected_models):
        tuned_model_name = f"{prefix}{model_type}"
        status_text.text(f"{tuned_model_name} 모델 학습 중...")
        
        try:
            # 자세한 디버깅 정보 출력
            st.write(f"모델 유형: {model_type}")
            st.write(f"조정된 파라미터: {tuned_params}")
            
            # 모델 유형에 따른 처리
            if model_type == 'exp_smoothing':
                # 지수평활법 모델에 필요한 파라미터 분리
                # 모델 생성 시 필요한 파라미터
                model_params = {
                    'model_type': tuned_params.get('model_type', 'hw'),
                    'trend': tuned_params.get('trend', 'add'),
                    'seasonal': tuned_params.get('seasonal', 'add'),
                    'seasonal_periods': tuned_params.get('seasonal_periods', st.session_state.period),
                    'damped_trend': tuned_params.get('damped_trend', False),
                    'use_boxcox': tuned_params.get('use_boxcox', False)
                }
                
                # 모델 유형 확인
                model_type_value = model_params.get('model_type')
                
                # fit 메서드에 전달할 파라미터
                fit_params = {}
                
                # 모델 유형에 따라 다른 파라미터 설정
                if model_type_value == 'simple':
                    # 단순 지수평활법은 alpha 파라미터를 받음
                    if 'alpha' in tuned_params:
                        fit_params['smoothing_level'] = tuned_params['alpha']
                elif model_type_value == 'holt':
                    # Holt 모델은 alpha, beta 파라미터를 받음
                    if 'alpha' in tuned_params:
                        fit_params['smoothing_level'] = tuned_params['alpha']
                    if 'beta' in tuned_params:
                        fit_params['smoothing_trend'] = tuned_params['beta']
                else:  # 'hw' (default)
                    # ExponentialSmoothing은 alpha, beta, gamma를 직접 받지 않음
                    # 대신 최적화 관련 파라미터 사용
                    fit_params['optimized'] = tuned_params.get('optimized', True)
                    if 'use_brute' in tuned_params:
                        fit_params['use_brute'] = tuned_params['use_brute']
                
                # 파라미터 저장
                st.session_state.model_params[tuned_model_name] = {**model_params, **fit_params}
                
                # 디버깅 정보 출력
                st.write(f"모델 생성 파라미터: {model_params}")
                st.write(f"fit 메서드 파라미터: {fit_params}")
                
                try:
                    # 직접 모델 생성 및 학습
                    model_factory = get_model_factory()
                    model = model_factory.get_model(model_type, name=tuned_model_name)
                    
                    # 모델 생성 파라미터로 fit 호출
                    model.fit(st.session_state.train, **model_params)
                    
                    # 예측 및 평가
                    forecast = model.predict(len(st.session_state.test), st.session_state.test)
                    metrics = model.evaluate(st.session_state.test, forecast)
                    metrics['name'] = tuned_model_name
                except Exception as e:
                    st.error(f"{tuned_model_name} 모델 학습 중 오류 발생: {traceback.format_exc()}")
                    st.exception(e)  # 상세 오류 정보 출력
                    continue  # 다음 모델로 진행
            
            elif model_type == 'arima':
                # ARIMA 모델에 필요한 파라미터 보장
                arima_params = {
                    'order': tuned_params.get('order', (1, 1, 1)),
                    'seasonal_order': tuned_params.get('seasonal_order', (1, 1, 1, st.session_state.period)),
                    'seasonal': tuned_params.get('seasonal', True),
                    'm': tuned_params.get('m', st.session_state.period)
                }
                
                # 파라미터 저장
                st.session_state.model_params[tuned_model_name] = arima_params
                
                # 직접 모델 생성 및 학습
                model_factory = get_model_factory()
                model = model_factory.get_model(model_type, name=tuned_model_name)
                forecast, metrics = model.fit_predict_evaluate(
                    st.session_state.train, 
                    st.session_state.test,
                    **arima_params
                )
                metrics['name'] = tuned_model_name
            
            elif model_type == 'prophet':
                # Prophet 모델에 필요한 파라미터 보장
                prophet_params = {
                    'daily_seasonality': tuned_params.get('daily_seasonality', False),
                    'weekly_seasonality': tuned_params.get('weekly_seasonality', True),
                    'yearly_seasonality': tuned_params.get('yearly_seasonality', False),
                    'seasonality_mode': tuned_params.get('seasonality_mode', 'additive'),
                    'changepoint_prior_scale': tuned_params.get('changepoint_prior_scale', 0.05)
                }
                
                # 파라미터 저장
                st.session_state.model_params[tuned_model_name] = prophet_params
                
                # 직접 모델 생성 및 학습
                model_factory = get_model_factory()
                model = model_factory.get_model(model_type, name=tuned_model_name)
                forecast, metrics = model.fit_predict_evaluate(
                    st.session_state.train, 
                    st.session_state.test,
                    **prophet_params
                )
                metrics['name'] = tuned_model_name
                
            elif model_type == 'lstm':
                # LSTM 모델에 필요한 파라미터 보장
                lstm_params = {
                    'n_steps': tuned_params.get('n_steps', 24),
                    'lstm_units': tuned_params.get('lstm_units', [50]),
                    'dropout_rate': tuned_params.get('dropout_rate', 0.2),
                    'epochs': tuned_params.get('epochs', 50)
                }
                
                # 파라미터 저장
                st.session_state.model_params[tuned_model_name] = lstm_params
                
                # 직접 모델 생성 및 학습
                model_factory = get_model_factory()
                model = model_factory.get_model(model_type, name=tuned_model_name)
                forecast, metrics = model.fit_predict_evaluate(
                    st.session_state.train, 
                    st.session_state.test,
                    **lstm_params
                )
                metrics['name'] = tuned_model_name
                
            else:
                # 기타 모델 처리
                model_factory = get_model_factory()
                model = model_factory.get_model(model_type, name=tuned_model_name)
                forecast, metrics = model.fit_predict_evaluate(
                    st.session_state.train, 
                    st.session_state.test,
                    **tuned_params
                )
                metrics['name'] = tuned_model_name
            
            # 결과 저장
            tuned_forecasts[tuned_model_name] = forecast
            tuned_metrics[tuned_model_name] = metrics
            
            # 진행 상황 업데이트
            progress_bar.progress((i + 1) / len(selected_models))
            
        except Exception as e:
            st.error(f"{tuned_model_name} 모델 학습 중 오류 발생: {traceback.format_exc()}")
            st.exception(e)  # 상세 오류 정보 출력
    
    # 모든 모델 학습 완료 후 결과 저장
    if tuned_forecasts:
        st.session_state.forecasts.update(tuned_forecasts)
        st.session_state.metrics.update(tuned_metrics)
        status_text.text("모든 모델 학습 완료!")
        return True
    else:
        st.error("조정된 파라미터로 모델 학습 중 오류가 발생했습니다.")
        return False

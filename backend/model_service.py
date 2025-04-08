"""
모델 학습 및 예측 관련 서비스 모듈
"""
import streamlit as st

from backend.data_service import safe_len

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
        st.error(f"모델 로드 중 오류 발생: {e}")
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
        st.error(f"ARIMA 모델 학습 오류: {e}")
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
        
        model = model_factory.get_model('prophet')
        forecast, metrics = model.fit_predict_evaluate(
            st.session_state.train, 
            st.session_state.test,
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
        
        model = model_factory.get_model('lstm')
        forecast, metrics = model.fit_predict_evaluate(
            st.session_state.train, 
            st.session_state.test,
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
        
        model = model_factory.get_model('exp_smoothing')
        forecast, metrics = model.fit_predict_evaluate(
            st.session_state.train, 
            st.session_state.test,
            seasonal_periods=seasonal_periods
        )
        return forecast, metrics
    except Exception as e:
        st.error(f"지수평활법 모델 학습 오류: {e}")
        return None, None

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
    
    # 데이터 키 생성 (캐싱용)
    train_data_key = hash(tuple(st.session_state.train.values.tolist()))
    test_data_key = hash(tuple(st.session_state.test.values.tolist()))
    
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
            # 모델별 캐싱된 학습 함수 호출
            if model_type == 'arima':
                forecast, model_metrics = cached_train_arima(
                    train_data_key, 
                    test_data_key,
                    seasonal=True,
                    m=st.session_state.period,
                    **arima_params
                )
            elif model_type == 'exp_smoothing':
                forecast, model_metrics = cached_train_exp_smoothing(
                    train_data_key, 
                    test_data_key,
                    seasonal_periods=st.session_state.period
                )
            elif model_type == 'prophet':
                forecast, model_metrics = cached_train_prophet(
                    train_data_key, 
                    test_data_key,
                    **prophet_params
                )
            elif model_type == 'lstm':
                forecast, model_metrics = cached_train_lstm(
                    train_data_key, 
                    test_data_key,
                    **lstm_params
                )
            else:
                # 일반적인 모델 처리
                model_factory = get_model_factory()
                model = model_factory.get_model(model_type)
                forecast, model_metrics = model.fit_predict_evaluate(
                    st.session_state.train, 
                    st.session_state.test
                )
            
            # 유효한 결과만 저장
            if forecast is not None and model_metrics is not None:
                forecasts[model_metrics.get('name', model_type)] = forecast
                metrics[model_metrics.get('name', model_type)] = model_metrics
            
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
        return True
    else:
        st.error("모델 학습 중 오류가 발생했습니다.")
        return False

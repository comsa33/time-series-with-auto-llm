# utils/parameter_utils.py
from typing import Dict, Any


def validate_model_parameters(model_type: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
    """
    모델 타입에 따른 파라미터 유효성 검사 및 형식 변환
    """
    if model_type == 'arima':
        return validate_arima_parameters(parameters)
    elif model_type == 'prophet':
        return validate_prophet_parameters(parameters)
    elif model_type == 'exp_smoothing':
        return validate_exp_smoothing_parameters(parameters)
    elif model_type == 'lstm':
        return validate_lstm_parameters(parameters)
    else:
        return parameters


def validate_arima_parameters(parameters: Dict[str, Any]) -> Dict[str, Any]:
    """
    ARIMA 모델 파라미터 검증
    """
    valid_params = {}
    
    # 기본 파라미터 검증
    if 'order' in parameters:
        try:
            # order는 (p, d, q) 형태의 튜플
            if isinstance(parameters['order'], list):
                valid_params['order'] = tuple(parameters['order'])
            else:
                valid_params['order'] = parameters['order']
        except Exception:
            pass
    
    # 계절성 파라미터 검증
    if 'seasonal_order' in parameters:
        try:
            # seasonal_order는 (P, D, Q, m) 형태의 튜플
            if isinstance(parameters['seasonal_order'], list):
                valid_params['seasonal_order'] = tuple(parameters['seasonal_order'])
            else:
                valid_params['seasonal_order'] = parameters['seasonal_order']
        except Exception:
            pass
    
    # 기타 파라미터
    for param in ['auto', 'seasonal', 'm']:
        if param in parameters:
            valid_params[param] = parameters[param]
    
    return valid_params


def validate_prophet_parameters(parameters: Dict[str, Any]) -> Dict[str, Any]:
    """
    Prophet 모델 파라미터 검증
    """
    valid_params = {}
    
    # 유효한 Prophet 파라미터만 추출
    valid_keys = [
        'daily_seasonality', 'weekly_seasonality', 'yearly_seasonality',
        'seasonality_mode', 'changepoint_prior_scale', 'seasonality_prior_scale',
        'holidays_prior_scale'
    ]
    
    for key in valid_keys:
        if key in parameters:
            valid_params[key] = parameters[key]
    
    return valid_params


def validate_exp_smoothing_parameters(parameters: Dict[str, Any]) -> Dict[str, Any]:
    """
    지수평활법 모델 파라미터 검증
    """
    valid_params = {}
    
    # 유효한 파라미터만 추출
    valid_keys = [
        'model_type', 'trend', 'seasonal', 'seasonal_periods',
        'damped_trend', 'use_boxcox'
    ]
    
    for key in valid_keys:
        if key in parameters:
            valid_params[key] = parameters[key]
    
    return valid_params


def validate_lstm_parameters(parameters: Dict[str, Any]) -> Dict[str, Any]:
    """
    LSTM 모델 파라미터 검증
    """
    valid_params = {}
    
    # 유효한 파라미터만 추출
    valid_keys = [
        'n_steps', 'lstm_units', 'dropout_rate', 'epochs', 
        'batch_size', 'validation_split', 'early_stopping', 'patience'
    ]
    
    for key in valid_keys:
        if key in parameters:
            valid_params[key] = parameters[key]
    
    return valid_params

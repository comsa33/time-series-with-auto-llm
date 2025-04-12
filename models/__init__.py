"""
시계열 모델 패키지
"""
# 기본 모델 클래스만 항상 가져오기
from models.base_model import TimeSeriesModel

# 다른 모델 클래스들은 필요할 때 동적으로 가져오기 (import 시 오류 방지)
# 외부에서 접근 가능한 클래스 및 함수 정의
__all__ = [
    'TimeSeriesModel',
    'get_available_models'
]

def get_available_models():
    """
    사용 가능한 모델 목록을 반환합니다.
    각 모델의 가용성을 확인하고 결과를 딕셔너리로 반환합니다.
    
    Returns:
        dict: 모델 이름과 가용성 여부를 담은 딕셔너리
    """
    available = {
        'TimeSeriesModel': True  # 기본 클래스는 항상 사용 가능
    }
    
    # ARIMA 모델 확인
    try:
        from models.arima_model import ArimaModel
        available['ArimaModel'] = True
    except ImportError:
        available['ArimaModel'] = False
    
    # 지수평활법 모델 확인
    try:
        from models.exp_smoothing_model import ExpSmoothingModel
        available['ExpSmoothingModel'] = True
    except ImportError:
        available['ExpSmoothingModel'] = False
    
    # Prophet 모델 확인
    try:
        from models.prophet_model import ProphetModel, PROPHET_AVAILABLE
        available['ProphetModel'] = PROPHET_AVAILABLE
    except ImportError:
        available['ProphetModel'] = False
    
    # LSTM 모델 확인
    try:
        from models.lstm_model import LSTMModel, TF_AVAILABLE
        available['LSTMModel'] = TF_AVAILABLE
    except ImportError:
        available['LSTMModel'] = False
    
    # 트랜스포머 모델 확인
    try:
        from models.transformer_model import TransformerModel, TF_AVAILABLE
        available['TransformerModel'] = TF_AVAILABLE
    except ImportError:
        available['TransformerModel'] = False
    
    # 모델 팩토리 확인
    try:
        from models.model_factory import ModelFactory
        available['ModelFactory'] = True
    except ImportError:
        available['ModelFactory'] = False
    
    return available

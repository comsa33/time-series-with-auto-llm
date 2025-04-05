"""
시계열 모델 생성을 위한 팩토리 패턴 구현 모듈
"""
import warnings
from typing import Dict, Any, List, Type

from models.base_model import TimeSeriesModel
from utils.singleton import Singleton


# 모델 클래스를 동적으로 가져오기
def get_arima_model():
    try:
        from models.arima_model import ArimaModel
        return ArimaModel
    except ImportError as e:
        warnings.warn(f"ARIMA 모델을 가져올 수 없습니다: {e}")
        return None

def get_exp_smoothing_model():
    try:
        from models.exp_smoothing_model import ExpSmoothingModel
        return ExpSmoothingModel
    except ImportError as e:
        warnings.warn(f"지수평활법 모델을 가져올 수 없습니다: {e}")
        return None

def get_prophet_model():
    try:
        from models.prophet_model import ProphetModel, PROPHET_AVAILABLE
        if PROPHET_AVAILABLE:
            return ProphetModel
        else:
            warnings.warn("Prophet 라이브러리가 설치되지 않았습니다.")
            return None
    except ImportError as e:
        warnings.warn(f"Prophet 모델을 가져올 수 없습니다: {e}")
        return None

def get_lstm_model():
    try:
        from models.lstm_model import LSTMModel, TF_AVAILABLE
        if TF_AVAILABLE:
            return LSTMModel
        else:
            warnings.warn("TensorFlow 라이브러리가 설치되지 않았습니다.")
            return None
    except ImportError as e:
        warnings.warn(f"LSTM 모델을 가져올 수 없습니다: {e}")
        return None


class ModelFactory(metaclass=Singleton):
    """
    시계열 모델 생성을 위한 팩토리 클래스
    싱글턴 패턴을 적용하여 메모리 효율성 확보
    """
    
    def __init__(self):
        """
        모델 팩토리 초기화
        """
        self.available_models = {}
        
        # 지수평활법 모델 추가 (에러 가능성이 적은 모델부터 시작)
        exp_smoothing_model = get_exp_smoothing_model()
        if exp_smoothing_model:
            self.available_models['exp_smoothing'] = exp_smoothing_model
        
        # ARIMA 모델 추가
        arima_model = get_arima_model()
        if arima_model:
            self.available_models['arima'] = arima_model
        
        # Prophet 모델 추가
        prophet_model = get_prophet_model()
        if prophet_model:
            self.available_models['prophet'] = prophet_model
        
        # LSTM 모델 추가
        lstm_model = get_lstm_model()
        if lstm_model:
            self.available_models['lstm'] = lstm_model
    
    def get_model(self, model_type: str, **kwargs) -> TimeSeriesModel:
        """
        지정된 유형의 모델 인스턴스를 생성합니다.
        
        Args:
            model_type: 모델 유형
            **kwargs: 모델 생성자에 전달할 추가 인자
            
        Returns:
            생성된 모델 인스턴스
            
        Raises:
            ValueError: 존재하지 않는 모델 유형인 경우
        """
        model_type = model_type.lower()
        
        if model_type not in self.available_models:
            available_types = ", ".join(self.available_models.keys())
            raise ValueError(f"존재하지 않는 모델 유형입니다: {model_type}. "
                           f"사용 가능한 모델 유형: {available_types}")
        
        # 모델 클래스 가져오기
        model_class = self.available_models[model_type]
        
        # 모델 인스턴스 생성
        model = model_class(**kwargs)
        
        return model
    
    def get_all_available_models(self) -> List[str]:
        """
        사용 가능한 모든 모델 유형을 반환합니다.
        
        Returns:
            사용 가능한 모델 유형 목록
        """
        return list(self.available_models.keys())
    
    def create_all_models(self, **kwargs) -> Dict[str, TimeSeriesModel]:
        """
        사용 가능한 모든 모델 인스턴스를 생성합니다.
        
        Args:
            **kwargs: 모델 생성자에 전달할 추가 인자
            
        Returns:
            모델 유형을 키로, 모델 인스턴스를 값으로 하는 딕셔너리
        """
        models = {}
        
        for model_type in self.available_models:
            try:
                models[model_type] = self.get_model(model_type, **kwargs)
            except Exception as e:
                warnings.warn(f"{model_type} 모델 생성 중 오류 발생: {e}")
        
        return models
    
    def register_model(self, model_type: str, model_class: Type[TimeSeriesModel]) -> None:
        """
        새로운 모델 유형을 등록합니다.
        
        Args:
            model_type: 등록할 모델 유형
            model_class: 등록할 모델 클래스
            
        Raises:
            ValueError: 이미 존재하는 모델 유형인 경우
        """
        model_type = model_type.lower()
        
        if model_type in self.available_models:
            raise ValueError(f"이미 존재하는 모델 유형입니다: {model_type}")
        
        self.available_models[model_type] = model_class

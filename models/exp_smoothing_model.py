"""
지수평활법 모델 구현 모듈
"""
import warnings
from typing import Dict, Any, Tuple, Optional, Union

import pandas as pd
import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing, SimpleExpSmoothing, Holt

from models.base_model import TimeSeriesModel


class ExpSmoothingModel(TimeSeriesModel):
    """
    지수평활법 모델 구현 클래스
    """
    
    def __init__(self, name: str = "지수평활법"):
        """
        지수평활법 모델 생성자
        
        Args:
            name: 모델 이름
        """
        super().__init__(name)
        self.model_type = None
        self.model_params = {}
    
    def fit(self, 
           train_data: pd.Series, 
           model_type: str = 'hw',
           trend: Optional[str] = 'add',
           seasonal: Optional[str] = 'add',
           seasonal_periods: int = 24,
           damped_trend: bool = False,
           use_boxcox: Union[bool, float] = False,
           **kwargs) -> Any:
        """
        지수평활법 모델을 학습합니다.
        
        Args:
            train_data: 학습 데이터
            model_type: 모델 유형 ('simple', 'holt', 'hw')
            trend: 추세 유형 ('add', 'mul', None)
            seasonal: 계절성 유형 ('add', 'mul', None)
            seasonal_periods: 계절성 주기
            damped_trend: 감쇠 추세 사용 여부
            use_boxcox: Box-Cox 변환 사용 여부
            **kwargs: 추가 매개변수
            
        Returns:
            학습된 모델
        """
        self.train_data = train_data
        self.model_type = model_type
        
        try:
            # 모델 유형에 따라 적절한 지수평활법 선택
            if model_type == 'simple':
                # 단순 지수평활법
                self.model = SimpleExpSmoothing(train_data, **kwargs)
                self.model_fit = self.model.fit()
                
            elif model_type == 'holt':
                # Holt 이중 지수평활법 (추세 고려)
                self.model = Holt(
                    train_data, 
                    damped_trend=damped_trend,
                    **kwargs
                )
                self.model_fit = self.model.fit()
                
            else:  # 'hw' (default)
                # Holt-Winters 삼중 지수평활법 (추세 및 계절성 고려)
                self.model = ExponentialSmoothing(
                    train_data,
                    trend=trend,
                    seasonal=seasonal,
                    seasonal_periods=seasonal_periods,
                    damped_trend=damped_trend,
                    use_boxcox=use_boxcox,
                    **kwargs
                )
                self.model_fit = self.model.fit()
            
            # 모델 파라미터 저장
            self.model_params = {
                'model_type': model_type,
                'trend': trend,
                'seasonal': seasonal,
                'seasonal_periods': seasonal_periods,
                'damped_trend': damped_trend,
                'use_boxcox': use_boxcox
            }
            
            # 학습된 파라미터 저장
            if hasattr(self.model_fit, 'params'):
                for param_name, param_value in self.model_fit.params.items():
                    self.model_params[param_name] = param_value
            
            self.is_fitted = True
            return self.model_fit
            
        except Exception as e:
            warnings.warn(f"지수평활법 모델 학습 중 오류 발생: {e}")
            self.is_fitted = False
            return None
    
    def predict(self, steps: int = 1, test_data: Optional[pd.Series] = None, **kwargs) -> np.ndarray:
        """
        미래 값을 예측합니다.
        
        Args:
            steps: 예측할 기간 수
            test_data: 테스트 데이터 (있는 경우)
            **kwargs: 추가 매개변수
            
        Returns:
            예측값 배열
        """
        if not self.is_fitted:
            raise ValueError("모델이 학습되지 않았습니다. 먼저 fit() 메서드를 호출하세요.")
        
        forecast = self.model_fit.forecast(steps)
        
        return forecast.values
    
    def get_params(self) -> Dict[str, Any]:
        """
        모델 파라미터를 반환합니다.
        
        Returns:
            모델 파라미터 딕셔너리
        """
        return self.model_params

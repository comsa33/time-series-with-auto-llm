"""
ARIMA/SARIMA 모델 구현 모듈
"""
import warnings
from typing import Dict, Any, Tuple, Optional, Union

import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX

from models.base_model import TimeSeriesModel

# pmdarima는 선택적으로 사용
try:
    import pmdarima as pm
    PMDARIMA_AVAILABLE = True
except ImportError:
    PMDARIMA_AVAILABLE = False
    warnings.warn("pmdarima 라이브러리를 사용할 수 없습니다. 수동으로 ARIMA/SARIMA 파라미터를 설정해야 합니다.")


class ArimaModel(TimeSeriesModel):
    """
    ARIMA/SARIMA 모델 구현 클래스
    """
    
    def __init__(self, name: str = "ARIMA/SARIMA"):
        """
        ARIMA/SARIMA 모델 생성자
        
        Args:
            name: 모델 이름
        """
        super().__init__(name)
        self.order = None
        self.seasonal_order = None
        self.summary = None
        self.auto_arima_result = None
    
    def fit(self, 
           train_data: pd.Series, 
           order: Optional[Tuple[int, int, int]] = None,
           seasonal_order: Optional[Tuple[int, int, int, int]] = None,
           auto: bool = True,
           seasonal: bool = True,
           m: int = 24,
           **kwargs) -> Any:
        """
        ARIMA/SARIMA 모델을 학습합니다.
        
        Args:
            train_data: 학습 데이터
            order: ARIMA 차수 (p, d, q)
            seasonal_order: 계절성 ARIMA 차수 (P, D, Q, m)
            auto: 자동으로 최적 파라미터 찾기 여부
            seasonal: 계절성 고려 여부
            m: 계절성 주기
            **kwargs: 추가 매개변수
            
        Returns:
            학습된 모델
        """
        self.train_data = train_data
        
        # 자동으로 최적 파라미터 찾기
        if auto and PMDARIMA_AVAILABLE:
            try:
                # 자동 ARIMA 모델 피팅
                auto_model = pm.auto_arima(
                    train_data,
                    seasonal=seasonal,
                    m=m,
                    start_p=0, start_q=0,
                    max_p=5, max_q=5,
                    d=None,
                    trace=True,
                    error_action='ignore',
                    suppress_warnings=True,
                    stepwise=True,
                    **kwargs
                )
                
                # 모델 파라미터 저장
                self.order = auto_model.order
                self.seasonal_order = auto_model.seasonal_order
                self.auto_arima_result = auto_model
                self.summary = auto_model.summary()
                
            except Exception as e:
                warnings.warn(f"auto_arima 실행 중 오류 발생: {e}")
                # 기본값 사용
                self.order = (1, 1, 1)
                self.seasonal_order = (1, 1, 1, m) if seasonal else None
        else:
            # 수동 파라미터 사용
            self.order = order if order is not None else (1, 1, 1)
            self.seasonal_order = seasonal_order if seasonal_order is not None else (
                (1, 1, 1, m) if seasonal else None)
        
        # 최종 모델 학습
        try:
            if self.seasonal_order is not None:
                self.model = SARIMAX(
                    train_data,
                    order=self.order,
                    seasonal_order=self.seasonal_order,
                    **kwargs
                )
            else:
                self.model = ARIMA(
                    train_data,
                    order=self.order,
                    **kwargs
                )
                
            self.model_fit = self.model.fit()
            self.summary = self.model_fit.summary()
            self.is_fitted = True
            
            return self.model_fit
            
        except Exception as e:
            warnings.warn(f"ARIMA/SARIMA 모델 학습 중 오류 발생: {e}")
            self.is_fitted = False
            return None
    
    def predict(self, 
               steps: int = 1, 
               test_data: Optional[pd.Series] = None,
               return_conf_int: bool = False,
               alpha: float = 0.05,
               **kwargs) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """
        미래 값을 예측합니다.
        
        Args:
            steps: 예측할 기간 수
            test_data: 테스트 데이터 (있는 경우)
            return_conf_int: 신뢰 구간 반환 여부
            alpha: 신뢰 구간의 유의 수준
            **kwargs: 추가 매개변수
            
        Returns:
            예측값 배열 또는 (예측값, 하한, 상한) 튜플
        """
        if not self.is_fitted:
            raise ValueError("모델이 학습되지 않았습니다. 먼저 fit() 메서드를 호출하세요.")
        
        # pmdarima를 사용한 경우
        if self.auto_arima_result is not None and PMDARIMA_AVAILABLE:
            if return_conf_int:
                forecast, conf_int = self.auto_arima_result.predict(
                    n_periods=steps, 
                    return_conf_int=True,
                    alpha=alpha
                )
                lower = conf_int[:, 0]
                upper = conf_int[:, 1]
                return forecast, lower, upper
            else:
                forecast = self.auto_arima_result.predict(n_periods=steps)
                return forecast
        
        # statsmodels 사용
        if return_conf_int:
            forecast = self.model_fit.get_forecast(steps=steps)
            mean_forecast = forecast.predicted_mean.values
            conf_int = forecast.conf_int(alpha=alpha)
            lower = conf_int.iloc[:, 0].values
            upper = conf_int.iloc[:, 1].values
            return mean_forecast, lower, upper
        else:
            forecast = self.model_fit.get_forecast(steps=steps)
            return forecast.predicted_mean.values
    
    def get_params(self) -> Dict[str, Any]:
        """
        모델 파라미터를 반환합니다.
        
        Returns:
            모델 파라미터 딕셔너리
        """
        return {
            'order': self.order,
            'seasonal_order': self.seasonal_order
        }

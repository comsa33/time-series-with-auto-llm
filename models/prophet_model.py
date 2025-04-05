"""
Facebook Prophet 모델 구현 모듈
"""
import warnings
from typing import Dict, Any, Tuple, Optional, Union, List

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from models.base_model import TimeSeriesModel

# Prophet은 선택적으로 사용
try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False
    warnings.warn("Prophet 라이브러리를 사용할 수 없습니다. Prophet 모델을 사용하려면 설치가 필요합니다.")


class ProphetModel(TimeSeriesModel):
    """
    Facebook Prophet 모델 구현 클래스
    """
    
    def __init__(self, name: str = "Prophet"):
        """
        Prophet 모델 생성자
        
        Args:
            name: 모델 이름
        """
        super().__init__(name)
        self.model_params = {}
        self.forecast_df = None
        self.component_fig = None
        
        if not PROPHET_AVAILABLE:
            warnings.warn("Prophet 라이브러리가 설치되지 않았습니다. 모델을 사용하기 전에 설치하세요.")
    
    def fit(self, 
           train_data: pd.Series, 
           daily_seasonality: bool = True,
           weekly_seasonality: bool = True,
           yearly_seasonality: bool = True,
           holidays: Optional[pd.DataFrame] = None,
           seasonality_mode: str = 'additive',
           changepoint_prior_scale: float = 0.05,
           seasonality_prior_scale: float = 10.0,
           holidays_prior_scale: float = 10.0,
           **kwargs) -> Any:
        """
        Prophet 모델을 학습합니다.
        
        Args:
            train_data: 학습 데이터
            daily_seasonality: 일별 계절성 사용 여부
            weekly_seasonality: 주별 계절성 사용 여부
            yearly_seasonality: 연별 계절성 사용 여부
            holidays: 휴일 데이터프레임
            seasonality_mode: 계절성 모드 ('additive' 또는 'multiplicative')
            changepoint_prior_scale: 변화점 사전 스케일
            seasonality_prior_scale: 계절성 사전 스케일
            holidays_prior_scale: 휴일 사전 스케일
            **kwargs: 추가 매개변수
            
        Returns:
            학습된 모델
        """
        if not PROPHET_AVAILABLE:
            raise ImportError("Prophet 라이브러리가 설치되지 않았습니다. 먼저 설치하세요.")
        
        self.train_data = train_data
        
        try:
            # Prophet 데이터 포맷으로 변환
            df_prophet = pd.DataFrame({
                'ds': train_data.index,
                'y': train_data.values
            })
            
            # 모델 학습
            self.model = Prophet(
                daily_seasonality=daily_seasonality,
                weekly_seasonality=weekly_seasonality,
                yearly_seasonality=yearly_seasonality,
                holidays=holidays,
                seasonality_mode=seasonality_mode,
                changepoint_prior_scale=changepoint_prior_scale,
                seasonality_prior_scale=seasonality_prior_scale,
                holidays_prior_scale=holidays_prior_scale,
                **kwargs
            )
            
            self.model.fit(df_prophet)
            
            # 모델 파라미터 저장
            self.model_params = {
                'daily_seasonality': daily_seasonality,
                'weekly_seasonality': weekly_seasonality,
                'yearly_seasonality': yearly_seasonality,
                'seasonality_mode': seasonality_mode,
                'changepoint_prior_scale': changepoint_prior_scale,
                'seasonality_prior_scale': seasonality_prior_scale,
                'holidays_prior_scale': holidays_prior_scale
            }
            
            self.is_fitted = True
            return self.model
            
        except Exception as e:
            warnings.warn(f"Prophet 모델 학습 중 오류 발생: {e}")
            self.is_fitted = False
            return None
    
    def predict(self, 
               steps: int = 1, 
               test_data: Optional[pd.Series] = None,
               return_components: bool = False,
               **kwargs) -> Union[np.ndarray, Tuple[np.ndarray, pd.DataFrame]]:
        """
        미래 값을 예측합니다.
        
        Args:
            steps: 예측할 기간 수
            test_data: 테스트 데이터 (있는 경우)
            return_components: 구성 요소 반환 여부
            **kwargs: 추가 매개변수
            
        Returns:
            예측값 배열 또는 (예측값, 구성 요소) 튜플
        """
        if not PROPHET_AVAILABLE:
            raise ImportError("Prophet 라이브러리가 설치되지 않았습니다. 먼저 설치하세요.")
        
        if not self.is_fitted:
            raise ValueError("모델이 학습되지 않았습니다. 먼저 fit() 메서드를 호출하세요.")
        
        # 예측 기간 설정
        if test_data is not None:
            future = pd.DataFrame({'ds': test_data.index})
        else:
            # 마지막 날짜 이후로 steps 기간 예측
            last_date = self.train_data.index[-1]
            future_dates = pd.date_range(
                start=last_date,
                periods=steps + 1,
                freq=pd.infer_freq(self.train_data.index)
            )[1:]  # 첫 번째 날짜는 마지막 훈련 데이터와 중복되므로 제외
            
            future = pd.DataFrame({'ds': future_dates})
        
        # 예측
        self.forecast_df = self.model.predict(future)
        
        # 구성 요소 그래프 생성
        if return_components:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.component_fig = self.model.plot_components(self.forecast_df)
            
            return self.forecast_df['yhat'].values, self.forecast_df
        else:
            return self.forecast_df['yhat'].values
    
    def get_components_fig(self) -> Optional[plt.Figure]:
        """
        구성 요소 시각화 그래프를 반환합니다.
        
        Returns:
            구성 요소 시각화 Matplotlib 그래프
        """
        return self.component_fig
    
    def get_params(self) -> Dict[str, Any]:
        """
        모델 파라미터를 반환합니다.
        
        Returns:
            모델 파라미터 딕셔너리
        """
        return self.model_params

"""
시계열 데이터 전처리를 위한 모듈
"""
import pandas as pd
import numpy as np
from typing import Tuple, Optional, List, Union
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller, acf, pacf
import matplotlib.pyplot as plt

from utils.singleton import Singleton
from config.settings import app_config


class DataProcessor(metaclass=Singleton):
    """
    시계열 데이터 전처리를 위한 클래스
    싱글턴 패턴을 적용하여 메모리 효율성 확보
    """
    
    def preprocess_data(self, 
                        df: pd.DataFrame, 
                        target_col: str, 
                        station: Optional[str] = None) -> pd.Series:
        """
        시계열 분석을 위한 데이터 전처리를 수행합니다.
        
        Args:
            df: 전처리할 데이터프레임
            target_col: 대상 변수 컬럼명
            station: 특정 측정소 이름 (None인 경우 전체 측정소 평균)
            
        Returns:
            전처리된 시계열 데이터
        """
        # 특정 측정소 데이터 필터링
        if station and 'MSRSTE_NM' in df.columns:
            df = df[df['MSRSTE_NM'] == station].copy()
        elif not station and 'MSRSTE_NM' in df.columns:
            # 전체 측정소 평균
            df = df.groupby('MSRDT').agg({target_col: 'mean'}).reset_index()
        
        # 시간 인덱스 설정
        if 'MSRDT' in df.columns:
            df = df.set_index('MSRDT')
        
        # 타겟 컬럼만 선택
        if isinstance(df.index, pd.DatetimeIndex):
            series = df[target_col] if target_col in df.columns else df.iloc[:, 0]
        else:
            df.index = pd.to_datetime(df.index)
            series = df[target_col] if target_col in df.columns else df.iloc[:, 0]
        
        # 시계열 데이터 정렬
        series = series.sort_index()
        
        # 결측치 처리
        series = series.interpolate(method='time')
        
        return series
    
    def train_test_split(self, 
                         series: pd.Series, 
                         test_size: float = app_config.DEFAULT_TEST_SIZE) -> Tuple[pd.Series, pd.Series]:
        """
        시계열 데이터를 훈련 세트와 테스트 세트로 분할합니다.
        
        Args:
            series: 분할할 시계열 데이터
            test_size: 테스트 세트의 비율
            
        Returns:
            (훈련 데이터, 테스트 데이터) 튜플
        """
        # 분할 지점 계산
        split_idx = int(len(series) * (1 - test_size))
        
        # 시간 순서대로 분할
        train = series[:split_idx]
        test = series[split_idx:]
        
        return train, test
    
    def decompose_timeseries(self, 
                             series: pd.Series, 
                             period: int = 24) -> dict:
        """
        시계열 데이터를 추세, 계절성, 잔차로 분해합니다.
        
        Args:
            series: 분해할 시계열 데이터
            period: 계절성 주기
            
        Returns:
            분해 결과를 담은 딕셔너리
        """
        # 결측치 보간
        series_clean = series.interpolate()
        
        # 시계열 분해
        decomposition = seasonal_decompose(series_clean, period=period)
        
        return {
            'observed': decomposition.observed,
            'trend': decomposition.trend,
            'seasonal': decomposition.seasonal,
            'resid': decomposition.resid
        }
    
    def check_stationarity(self, series: pd.Series) -> dict:
        """
        시계열 데이터의 정상성을 검정합니다.
        
        Args:
            series: 검정할 시계열 데이터
            
        Returns:
            검정 결과를 담은 딕셔너리
        """
        # ADF 검정 수행
        result = adfuller(series.dropna())
        
        # 결과 정리
        return {
            'test_statistic': result[0],
            'p_value': result[1],
            'lags_used': result[2],
            'num_observations': result[3],
            'critical_values': result[4],
            'is_stationary': result[1] < 0.05
        }
    
    def get_acf_pacf(self, 
                    series: pd.Series, 
                    nlags: int = 40) -> Tuple[np.ndarray, np.ndarray]:
        """
        시계열 데이터의 ACF와 PACF를 계산합니다.
        
        Args:
            series: 분석할 시계열 데이터
            nlags: 계산할 최대 지연값
            
        Returns:
            (ACF, PACF) 튜플
        """
        from statsmodels.tsa.stattools import acf, pacf
        
        # 결측치 제거
        series_clean = series.dropna()
        
        # statsmodels는 기본적으로 시계열 길이의 절반 이하의 lag만 허용
        max_lags_allowed = len(series_clean) // 2
        
        # 요청한 nlags와 허용 가능 maximum lag 중 더 작은 값 선택
        safe_nlags = min(nlags, max_lags_allowed)
        
        # ACF, PACF 계산
        acf_values = acf(series_clean, nlags=safe_nlags)
        pacf_values = pacf(series_clean, nlags=safe_nlags)
        
        return acf_values, pacf_values
    
    def create_features(self, 
                         df: pd.DataFrame, 
                         date_col: str = 'MSRDT') -> pd.DataFrame:
        """
        시계열 데이터에서 날짜/시간 기반 특성을 추출합니다.
        
        Args:
            df: 특성을 추출할 데이터프레임
            date_col: 날짜/시간 컬럼명
            
        Returns:
            특성이 추가된 데이터프레임
        """
        # 데이터프레임 복사
        result = df.copy()
        
        # 날짜/시간 컬럼이 인덱스인 경우
        if date_col == df.index.name or df.index.name is None and isinstance(df.index, pd.DatetimeIndex):
            dt = df.index
        else:
            # 날짜/시간 컬럼이 데이터프레임 컬럼인 경우
            dt = pd.to_datetime(df[date_col])
        
        # 시간 기반 특성 추출
        result['hour'] = dt.hour
        result['dayofweek'] = dt.dayofweek
        result['quarter'] = dt.quarter
        result['month'] = dt.month
        result['year'] = dt.year
        result['dayofyear'] = dt.dayofyear
        result['weekofyear'] = dt.isocalendar().week
        
        # 주말 여부
        result['is_weekend'] = (dt.dayofweek >= 5).astype(int)
        
        # 시간대 구분 (아침, 낮, 저녁, 밤)
        result['time_of_day'] = pd.cut(
            dt.hour, 
            bins=[0, 6, 12, 18, 24], 
            labels=['night', 'morning', 'afternoon', 'evening'],
            include_lowest=True
        )
        
        # 출퇴근 시간대 여부
        result['is_rush_hour'] = (
            ((dt.hour >= 7) & (dt.hour <= 9)) | 
            ((dt.hour >= 17) & (dt.hour <= 19))
        ).astype(int)
        
        return result

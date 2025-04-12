"""
시계열 데이터 전처리를 위한 모듈
"""
from typing import Tuple, Optional

import streamlit as st
import pandas as pd
import numpy as np
import ruptures as rpt
from arch import arch_model
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller, kpss, grangercausalitytests
from statsmodels.stats.diagnostic import acorr_ljungbox

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

    def perform_differencing(self, series: pd.Series, diff_order: int = 1, seasonal_diff_order: int = 0, seasonal_period: int = None) -> pd.Series:
        """
        시계열 데이터에 차분을 적용합니다.
        
        Args:
            series: 차분할 시계열 데이터
            diff_order: 일반 차분 차수 (기본값: 1)
            seasonal_diff_order: 계절 차분 차수 (기본값: 0)
            seasonal_period: 계절성 주기 (None인 경우 계절 차분 미적용)
            
        Returns:
            차분된 시계열 데이터
        """
        differenced_series = series.copy()
        
        # 계절 차분 적용 (seasonal_period가 있는 경우)
        if seasonal_diff_order > 0 and seasonal_period is not None:
            for _ in range(seasonal_diff_order):
                differenced_series = differenced_series.diff(seasonal_period).dropna()
        
        # 일반 차분 적용
        for _ in range(diff_order):
            differenced_series = differenced_series.diff().dropna()
        
        return differenced_series

    def recommend_differencing(self, series: pd.Series, acf_values: np.ndarray = None, pacf_values: np.ndarray = None) -> dict:
        """
        시계열 데이터의 ACF, PACF 및 정상성 검정 결과를 기반으로 차분 추천을 제공합니다.
        
        Args:
            series: 시계열 데이터
            acf_values: ACF 값 배열 (None인 경우 계산)
            pacf_values: PACF 값 배열 (None인 경우 계산)
            
        Returns:
            추천 정보를 담은 딕셔너리
        """
        # 정상성 검정
        stationarity_result = self.check_stationarity(series)
        is_stationary = stationarity_result['is_stationary']
        
        # ACF/PACF가 제공되지 않은 경우 계산
        if acf_values is None or pacf_values is None:
            acf_values, pacf_values = self.get_acf_pacf(series)
        
        # 초기 추천 사항
        recommendation = {
            'needs_differencing': not is_stationary,
            'diff_order': 0,
            'seasonal_diff_order': 0,
            'seasonal_period': None,
            'reason': []
        }
        
        # 비정상이면 차분 추천
        if not is_stationary:
            recommendation['diff_order'] = 1
            recommendation['reason'].append("시계열이 정상성을 만족하지 않습니다 (ADF 검정 p-value > 0.05).")
        
        # ACF 감소 속도가 느리면 차분 추천
        slow_decay = all(acf_values[i] > 0.5 for i in range(1, min(5, len(acf_values))))
        if slow_decay:
            recommendation['diff_order'] = max(recommendation['diff_order'], 1)
            recommendation['reason'].append("ACF가 천천히 감소하는 패턴을 보입니다 (추세 존재 가능성).")
        
        # 계절성 확인 (ACF에서 특정 lag에서 높은 값 발견)
        for period in [24, 168, 720]:  # 일별(24시간), 주별(168시간), 월별(30일) 주기
            if len(acf_values) > period and acf_values[period] > 0.3:
                recommendation['seasonal_period'] = period
                recommendation['seasonal_diff_order'] = 1
                recommendation['reason'].append(f"{period}시간 주기의 계절성이 감지되었습니다.")
                break
        
        return recommendation

    def apply_inverse_differencing(self, 
                                  differenced_series: pd.Series, 
                                  original_series: pd.Series, 
                                  diff_order: int = 1, 
                                  seasonal_diff_order: int = 0, 
                                  seasonal_period: int = None) -> pd.Series:
        """
        차분된 시계열 데이터를 원래 스케일로 되돌립니다.
        
        Args:
            differenced_series: 차분된 시계열 데이터 (예측값)
            original_series: 원본 시계열 데이터
            diff_order: 적용된 일반 차분 차수
            seasonal_diff_order: 적용된 계절 차분 차수
            seasonal_period: 적용된 계절성 주기
            
        Returns:
            원래 스케일로 변환된 시계열 데이터
        """
        # 예측값을 시리즈로 변환
        if not isinstance(differenced_series, pd.Series):
            if isinstance(differenced_series, np.ndarray):
                # 테스트 데이터 인덱스 사용
                differenced_series = pd.Series(differenced_series, index=original_series.index[-len(differenced_series):])
        
        # 마지막 값들 가져오기
        last_values = original_series.iloc[-max(diff_order + seasonal_diff_order * (seasonal_period or 0), 1):]
        
        # 역변환할 시계열 초기화
        inverted_series = differenced_series.copy()
        
        # 일반 차분 역변환
        for d in range(diff_order):
            # 초기값 설정 (원본 시리즈의 마지막 값)
            cumsum_series = inverted_series.copy()
            last_value = last_values.iloc[-1]
            
            # 누적합 계산
            cumsum_series = pd.Series([last_value] + list(inverted_series.values)).cumsum()[1:]
            cumsum_series.index = inverted_series.index
            
            inverted_series = cumsum_series
        
        # 계절 차분 역변환 (있는 경우)
        if seasonal_diff_order > 0 and seasonal_period is not None:
            for _ in range(seasonal_diff_order):
                # 원본 값에 계절 요소 추가
                for i, date in enumerate(inverted_series.index):
                    seasonal_idx = date - pd.Timedelta(hours=seasonal_period)
                    if seasonal_idx in original_series.index:
                        inverted_series.iloc[i] += original_series.loc[seasonal_idx]
        
        return inverted_series

    def perform_ljung_box_test(self, residuals, lags=None):
        """
        Ljung-Box 검정을 수행합니다.
        
        Args:
            residuals: 모델 잔차
            lags: 지연값 수 (기본값: None, min(10, n/5)으로 설정)
        
        Returns:
            검정 결과 딕셔너리
        """
        
        # lags가 None인 경우 자동 설정
        if lags is None:
            lags = min(10, len(residuals) // 5)
        
        # Ljung-Box 검정 수행
        result = acorr_ljungbox(residuals, lags=[lags])
        
        # 결과 반환
        return {
            'statistic': float(result['lb_stat'].iloc[0]),
            'p_value': float(result['lb_pvalue'].iloc[0]),
            'lags': lags,
            'is_white_noise': float(result['lb_pvalue'].iloc[0]) > 0.05
        }

    def check_stationarity_kpss(self, series: pd.Series) -> dict:
        """
        시계열 데이터의 정상성을 KPSS 테스트로 검정합니다.
        
        Args:
            series: 검정할 시계열 데이터
            
        Returns:
            검정 결과를 담은 딕셔너리
        """
        
        # KPSS 검정 수행
        result = kpss(series.dropna(), regression='c')
        
        # 결과 정리 (p-값 해석에 주의: ADF와 반대)
        return {
            'test_statistic': result[0],
            'p_value': result[1],
            'lags_used': result[2],
            'critical_values': result[3],
            'is_stationary': result[1] > 0.05  # KPSS는 p값이 크면 정상성 만족
        }

    def perform_granger_causality_test(self, x: pd.Series, y: pd.Series, max_lag: int = 12) -> dict:
        """
        두 시계열 간의 Granger 인과성 검정을 수행합니다.
        
        Args:
            x: 원인 시계열
            y: 결과 시계열
            max_lag: 최대 지연값
            
        Returns:
            검정 결과 딕셔너리
        """
        
        # 두 시계열을 데이터프레임으로 결합
        data = pd.concat([y, x], axis=1)
        data.columns = ['y', 'x']
        
        # 결측치 제거
        data = data.dropna()
        
        # Granger 인과성 검정 수행
        results = {}
        
        for lag in range(1, max_lag + 1):
            try:
                test_result = grangercausalitytests(data, maxlag=lag, verbose=False)
                results[lag] = {
                    'ssr_ftest': {
                        'statistic': float(test_result[lag][0]['ssr_ftest'][0]),
                        'p_value': float(test_result[lag][0]['ssr_ftest'][1]),
                        'is_causal': float(test_result[lag][0]['ssr_ftest'][1]) < 0.05
                    },
                    'ssr_chi2test': {
                        'statistic': float(test_result[lag][0]['ssr_chi2test'][0]),
                        'p_value': float(test_result[lag][0]['ssr_chi2test'][1]),
                        'is_causal': float(test_result[lag][0]['ssr_chi2test'][1]) < 0.05
                    }
                }
            except Exception as e:
                results[lag] = {'error': str(e)}
        
        return results

    def analyze_volatility(self, series: pd.Series) -> dict:
        """
        시계열 데이터의 변동성을 분석합니다.
        
        Args:
            series: 분석할 시계열 데이터
            
        Returns:
            변동성 분석 결과 딕셔너리
        """
        
        # 결측치 제거
        series_clean = series.dropna()
        
        # 수익률 계산 (금융 데이터 방식)
        returns = 100 * series_clean.pct_change().dropna()
        
        # GARCH(1,1) 모델 적합
        model = arch_model(returns, vol='GARCH', p=1, q=1)
        model_fit = model.fit(disp='off')
        
        # 결과 정리
        return {
            'model_summary': model_fit.summary().as_text(),
            'params': model_fit.params.to_dict(),
            'volatility': model_fit.conditional_volatility.tolist(),
            'aic': model_fit.aic,
            'bic': model_fit.bic
        }

    def detect_change_points(self, series: pd.Series, method: str = 'l2', min_size: int = 30) -> dict:
        """
        시계열 데이터의 구조적 변화점을 탐지합니다.
        
        Args:
            series: 분석할 시계열 데이터
            method: 변화점 탐지 방법 ('l1', 'l2', 'rbf', 'linear', 'normal', 'ar')
            min_size: 최소 세그먼트 크기
            
        Returns:
            변화점 탐지 결과 딕셔너리
        """
        # 결측치 제거
        series_clean = series.dropna().values
        
        # 변화점 탐지 알고리즘
        algo = rpt.Pelt(model=method, min_size=min_size)
        
        # 변화점 탐지 수행
        algo.fit(series_clean)
        change_points = algo.predict(pen=10)
        
        # 결과 정리
        result = {
            'change_points': change_points,
            'change_dates': [series.index[cp] for cp in change_points if cp < len(series)],
            'num_changes': len(change_points) - 1,  # 마지막은 시리즈 길이
            'segments': []
        }
        
        # 각 세그먼트 정보 추가
        prev_cp = 0
        for cp in change_points:
            if cp >= len(series):
                break
                
            segment_data = series.iloc[prev_cp:cp]
            result['segments'].append({
                'start_date': str(segment_data.index[0]),
                'end_date': str(segment_data.index[-1]),
                'length': len(segment_data),
                'mean': float(segment_data.mean()),
                'std': float(segment_data.std())
            })
            prev_cp = cp
        
        return result

# utils/data_processor.py 내의 함수를 직접 호출하는 대신 캐싱된 래퍼 함수 사용
@st.cache_data(ttl=3600)
def cached_preprocess_data(df, target_col, station):
    """시계열 그래프 캐싱"""
    processor = DataProcessor()
    return processor.preprocess_data(df, target_col, station)

@st.cache_data(ttl=3600)
def cached_train_test_split(series, test_size):
    """훈련/테스트 분할 캐싱"""
    processor = DataProcessor()
    return processor.train_test_split(series, test_size)

@st.cache_data(ttl=3600)
def cached_decompose_timeseries(series, period):
    """분해 결과 캐싱"""
    processor = DataProcessor()
    return processor.decompose_timeseries(series, period)

@st.cache_data(ttl=3600)
def cached_check_stationarity(series):
    """정상성 검정 결과 캐싱"""
    processor = DataProcessor()
    return processor.check_stationarity(series)

@st.cache_data(ttl=3600)
def cached_get_acf_pacf(series, nlags=40):
    """ACF/PACF 결과 캐싱"""
    processor = DataProcessor()
    return processor.get_acf_pacf(series, nlags)

@st.cache_data(ttl=3600)
def cached_perform_differencing(series, diff_order=1, seasonal_diff_order=0, seasonal_period=None):
    """차분 적용 결과 캐싱"""
    processor = DataProcessor()
    return processor.perform_differencing(series, diff_order, seasonal_diff_order, seasonal_period)

@st.cache_data(ttl=3600)
def cached_recommend_differencing(series, acf_values=None, pacf_values=None):
    """차분 추천 결과 캐싱"""
    processor = DataProcessor()
    return processor.recommend_differencing(series, acf_values, pacf_values)

@st.cache_data(ttl=3600)
def cached_apply_inverse_differencing(differenced_series, original_series, diff_order=1, seasonal_diff_order=0, seasonal_period=None):
    """차분 역변환 결과 캐싱"""
    processor = DataProcessor()
    return processor.apply_inverse_differencing(differenced_series, original_series, diff_order, seasonal_diff_order, seasonal_period)

@st.cache_data(ttl=3600)
def cached_perform_ljung_box_test(residuals, lags=None):
    """Ljung-Box 검정 결과 캐싱"""
    processor = DataProcessor()
    return processor.perform_ljung_box_test(residuals, lags)

@st.cache_data(ttl=3600)
def cached_check_stationarity_kpss(series):
    """KPSS 정상성 검정 결과 캐싱"""
    processor = DataProcessor()
    return processor.check_stationarity_kpss(series)

@st.cache_data(ttl=3600)
def cached_perform_granger_causality_test(x, y, max_lag=12):
    """Granger 인과성 검정 결과 캐싱"""
    processor = DataProcessor()
    return processor.perform_granger_causality_test(x, y, max_lag)

@st.cache_data(ttl=3600)
def cached_analyze_volatility(series):
    """변동성 분석 결과 캐싱"""
    processor = DataProcessor()
    return processor.analyze_volatility(series)

@st.cache_data(ttl=3600)
def cached_detect_change_points(series, method='binary', min_size=30):
    """변화점 탐지 결과 캐싱"""
    processor = DataProcessor()
    return processor.detect_change_points(series, method, min_size)

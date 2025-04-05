"""
시계열 데이터 시각화를 위한 모듈
"""
from typing import Dict, Tuple

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.dates as mdates
from matplotlib.figure import Figure
from matplotlib import font_manager, rc

from utils.singleton import Singleton


class TimeSeriesVisualizer(metaclass=Singleton):
    """
    시계열 데이터 시각화를 위한 클래스
    싱글턴 패턴을 적용하여 메모리 효율성 확보
    """
    
    def __init__(self):
        """
        시각화 클래스 초기화
        """
        # 기본 시각화 스타일 설정
        sns.set_style('whitegrid')
        plt.rcParams['font.size'] = 12
        plt.rcParams['figure.figsize'] = (12, 6)
        
        # 한글 폰트 설정
        font_path = '/usr/share/fonts/truetype/nanum/NanumGothic.ttf'  # 나눔고딕 폰트 경로
        font_name = font_manager.FontProperties(fname=font_path).get_name()
        rc('font', family=font_name)
        plt.rcParams['axes.unicode_minus'] = False
    
    def plot_timeseries(self, 
                       data: pd.Series, 
                       title: str = "Time Series Plot",
                       xlabel: str = "Date",
                       ylabel: str = "Value",
                       color: str = '#1f77b4',
                       figsize: Tuple[int, int] = (12, 6),
                       **kwargs) -> Figure:
        """
        기본 시계열 플롯을 생성합니다.
        
        Args:
            data: 시각화할 시계열 데이터
            title: 그래프 제목
            xlabel: x축 레이블
            ylabel: y축 레이블
            color: 선 색상
            figsize: 그래프 크기
            **kwargs: 추가 매개변수
            
        Returns:
            Matplotlib Figure 객체
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        # 시계열 데이터 플롯
        ax.plot(data.index, data.values, color=color, **kwargs)
        
        # 그래프 스타일 설정
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        
        # x축 날짜 포맷 설정
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        
        return fig
    
    def plot_decomposition(self, 
                          decomposition: Dict[str, pd.Series],
                          figsize: Tuple[int, int] = (12, 10)) -> Figure:
        """
        시계열 분해 결과를 시각화합니다.
        
        Args:
            decomposition: 시계열 분해 결과 딕셔너리
            figsize: 그래프 크기
            
        Returns:
            Matplotlib Figure 객체
        """
        fig, axes = plt.subplots(4, 1, figsize=figsize, sharex=True)
        
        # 관측값
        decomposition['observed'].plot(ax=axes[0])
        axes[0].set_title('Observed')
        axes[0].set_xlabel('')
        
        # 추세
        decomposition['trend'].plot(ax=axes[1])
        axes[1].set_title('Trend')
        axes[1].set_xlabel('')
        
        # 계절성
        decomposition['seasonal'].plot(ax=axes[2])
        axes[2].set_title('Seasonality')
        axes[2].set_xlabel('')
        
        # 잔차
        decomposition['resid'].plot(ax=axes[3])
        axes[3].set_title('Residuals')
        
        plt.tight_layout()
        
        return fig
    
    def plot_acf_pacf(self, 
                     acf_values: np.ndarray, 
                     pacf_values: np.ndarray,
                     lags: int = 40,
                     figsize: Tuple[int, int] = (12, 6)) -> Figure:
        """
        ACF 및 PACF 플롯을 생성합니다.
        
        Args:
            acf_values: ACF 값
            pacf_values: PACF 값
            lags: 지연값 수
            figsize: 그래프 크기
            
        Returns:
            Matplotlib Figure 객체
        """
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
        # ACF 플롯
        axes[0].stem(range(len(acf_values)), acf_values)
        axes[0].set_title('Autocorrelation Function')
        axes[0].set_xlabel('Lag')
        axes[0].set_ylabel('Correlation')
        
        # 신뢰 구간 (95%)
        confidence = 1.96 / np.sqrt(len(acf_values))
        axes[0].axhline(y=confidence, linestyle='--', color='gray')
        axes[0].axhline(y=-confidence, linestyle='--', color='gray')
        
        # PACF 플롯
        axes[1].stem(range(len(pacf_values)), pacf_values)
        axes[1].set_title('Partial Autocorrelation Function')
        axes[1].set_xlabel('Lag')
        axes[1].set_ylabel('Correlation')
        
        # 신뢰 구간 (95%)
        confidence = 1.96 / np.sqrt(len(pacf_values))
        axes[1].axhline(y=confidence, linestyle='--', color='gray')
        axes[1].axhline(y=-confidence, linestyle='--', color='gray')
        
        plt.tight_layout()
        
        return fig
    
    def plot_forecast_comparison(self, 
                               train: pd.Series,
                               test: pd.Series,
                               forecasts: Dict[str, np.ndarray],
                               figsize: Tuple[int, int] = (12, 6)) -> Figure:
        """
        여러 모델의 예측 결과를 비교하여 시각화합니다.
        
        Args:
            train: 훈련 데이터
            test: 테스트 데이터
            forecasts: 모델별 예측값 딕셔너리
            figsize: 그래프 크기
            
        Returns:
            Matplotlib Figure 객체
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        # 훈련 데이터
        ax.plot(train.index, train.values, label='Training Data', color='blue')
        
        # 테스트 데이터
        ax.plot(test.index, test.values, label='Actual Test Data', color='green')
        
        # 각 모델의 예측
        colors = ['red', 'purple', 'orange', 'brown', 'pink', 'gray', 'olive']
        for i, (model_name, forecast) in enumerate(forecasts.items()):
            ax.plot(test.index, forecast, 
                  label=f'{model_name} Forecast', 
                  color=colors[i % len(colors)], 
                  linestyle='--')
        
        # 그래프 스타일 설정
        ax.set_title('Forecast Comparison')
        ax.set_xlabel('Date')
        ax.set_ylabel('Value')
        ax.legend()
        
# x축 날짜 포맷 설정
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        
        return fig
    
    def plot_metrics_comparison(self, metrics: Dict[str, Dict[str, float]]) -> Figure:
        """
        여러 모델의 성능 지표를 비교하여 시각화합니다.
        
        Args:
            metrics: 모델별 성능 지표 딕셔너리
            
        Returns:
            Matplotlib Figure 객체
        """
        # 데이터 준비
        models = list(metrics.keys())
        metric_names = ['RMSE', 'MAE', 'R^2', 'MAPE']
        
        # 모든 모델에 있는 지표만 선택
        available_metrics = set.intersection(*[set(m.keys()) for m in metrics.values()])
        metric_names = [m for m in metric_names if m in available_metrics]
        
        # 그래프 설정
        fig, axes = plt.subplots(len(metric_names), 1, figsize=(12, 4 * len(metric_names)))
        
        # 단일 지표인 경우 axes를 리스트로 변환
        if len(metric_names) == 1:
            axes = [axes]
        
        for i, metric in enumerate(metric_names):
            values = [metrics[model][metric] for model in models]
            
            # 특별히 R^2는 높을수록 좋음, 나머지는 낮을수록 좋음
            if metric == 'R^2':
                colors = ['green' if v > 0 else 'red' for v in values]
                # 내림차순 정렬 (높을수록 좋음)
                sorted_idx = np.argsort(values)[::-1]
            else:
                # 값을 정규화하여 색상 결정 (낮을수록 좋음)
                max_val = max(values) if values else 1
                colors = ['lightcoral' if v/max_val > 0.7 else 'lightgreen' for v in values]
                # 오름차순 정렬 (낮을수록 좋음)
                sorted_idx = np.argsort(values)
            
            # 정렬된 모델 및 값
            sorted_models = [models[i] for i in sorted_idx]
            sorted_values = [values[i] for i in sorted_idx]
            sorted_colors = [colors[i] for i in sorted_idx]
            
            # 바 그래프
            bars = axes[i].bar(sorted_models, sorted_values, color=sorted_colors)
            axes[i].set_title(f'{metric} Comparison')
            axes[i].set_ylabel(metric)
            
            # 값 표시
            for j, (bar, v) in enumerate(zip(bars, sorted_values)):
                text_color = 'black'
                axes[i].text(
                    bar.get_x() + bar.get_width() / 2,
                    v + (max(sorted_values) * 0.02),
                    f'{v:.4f}',
                    ha='center',
                    va='bottom',
                    color=text_color,
                    fontweight='bold'
                )
        
        plt.tight_layout()
        
        return fig
    
    def plot_residuals(self, 
                      actual: pd.Series, 
                      predicted: np.ndarray,
                      title: str = "Residual Analysis",
                      figsize: Tuple[int, int] = (12, 10)) -> Figure:
        """
        잔차 분석 플롯을 생성합니다.
        
        Args:
            actual: 실제 값
            predicted: 예측 값
            title: 그래프 제목
            figsize: 그래프 크기
            
        Returns:
            Matplotlib Figure 객체
        """
        # 길이 맞춤
        min_len = min(len(actual), len(predicted))
        actual_values = actual.iloc[:min_len].values
        predicted_values = predicted[:min_len]
        
        # 잔차 계산
        residuals = actual_values - predicted_values
        
        # 그래프 설정
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        fig.suptitle(title, fontsize=16)
        
        # 잔차 시계열 플롯
        axes[0, 0].plot(actual.index[:min_len], residuals)
        axes[0, 0].set_title('Residuals Over Time')
        axes[0, 0].set_xlabel('Date')
        axes[0, 0].set_ylabel('Residual')
        axes[0, 0].axhline(y=0, linestyle='--', color='red')
        
        # 잔차 히스토그램
        axes[0, 1].hist(residuals, bins=20, edgecolor='black')
        axes[0, 1].set_title('Residual Distribution')
        axes[0, 1].set_xlabel('Residual')
        axes[0, 1].set_ylabel('Frequency')
        
        # 잔차 QQ 플롯
        from scipy import stats
        stats.probplot(residuals, dist="norm", plot=axes[1, 0])
        axes[1, 0].set_title('Normal Q-Q Plot')
        
        # 잔차 vs 예측값
        axes[1, 1].scatter(predicted_values, residuals)
        axes[1, 1].set_title('Residuals vs Predicted')
        axes[1, 1].set_xlabel('Predicted Values')
        axes[1, 1].set_ylabel('Residuals')
        axes[1, 1].axhline(y=0, linestyle='--', color='red')
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.9)
        
        return fig
    
    def plot_feature_importance(self, 
                              features: pd.DataFrame, 
                              target: pd.Series,
                              top_n: int = 10) -> Figure:
        """
        특성 중요도를 시각화합니다.
        
        Args:
            features: 특성 데이터프레임
            target: 타겟 변수
            top_n: 표시할 상위 특성 수
            
        Returns:
            Matplotlib Figure 객체
        """
        from sklearn.ensemble import RandomForestRegressor
        
        # 모든 특성이 수치형인지 확인
        numeric_features = features.select_dtypes(include=np.number).columns.tolist()
        
        # 수치형 특성만 사용
        X = features[numeric_features]
        
        # 랜덤 포레스트로 특성 중요도 계산
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X, target)
        
        # 특성 중요도 정렬
        importance = model.feature_importances_
        indices = np.argsort(importance)[::-1]
        
        # 상위 N개 특성만 선택
        top_indices = indices[:top_n]
        top_features = [numeric_features[i] for i in top_indices]
        top_importance = importance[top_indices]
        
        # 그래프 설정
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # 수평 막대 그래프
        y_pos = np.arange(len(top_features))
        ax.barh(y_pos, top_importance, align='center')
        ax.set_yticks(y_pos)
        ax.set_yticklabels(top_features)
        ax.invert_yaxis()  # 위에서부터 중요도 높은 순으로
        ax.set_xlabel('Feature Importance')
        ax.set_title('Top Feature Importance')
        
        plt.tight_layout()
        
        return fig

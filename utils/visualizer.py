"""
시계열 데이터 시각화를 위한 모듈
"""
from typing import Dict

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns
from matplotlib import font_manager, rc
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

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
    
    def plot_timeseries(
            self, 
            data: pd.Series, 
            title: str = "시계열 플롯 (Time Series Plot)",
            xlabel: str = "날짜 (Date)",
            ylabel: str = "값 (Value)",
            color: str = '#1f77b4',
            **kwargs
        ) -> go.Figure:
        """
        기본 시계열 플롯을 생성합니다 (Plotly 버전).
        """
        # Plotly 버전으로 구현
        fig = px.line(
            x=data.index, 
            y=data.values,
            labels={"x": xlabel, "y": ylabel},
            title=title
        )
        
        # 스타일 설정
        fig.update_layout(
            title=title,
            title_font_size=14,
            height=400,
            margin=dict(l=10, r=10, t=50, b=10),
        )
        
        # 날짜 형식 지정
        fig.update_xaxes(
            tickformat="%Y-%m-%d",
        )
        
        return fig
    
    def plot_decomposition(
            self, 
            decomposition: Dict[str, pd.Series],
            **kwargs
        ) -> go.Figure:
        """
        시계열 분해 결과를 시각화합니다 (Plotly 버전).
        """
        # 서브플롯 생성
        fig = make_subplots(
            rows=4, cols=1,
            subplot_titles=('관측값 (Observed)', '추세 (Trend)', '계절성 (Seasonality)', '잔차 (Residuals)'),
            shared_xaxes=True,
            vertical_spacing=0.05
        )
        
        # 관측값
        fig.add_trace(
            go.Scatter(x=decomposition['observed'].index, y=decomposition['observed'].values,
                    mode='lines', name='관측값'),
            row=1, col=1
        )
        
        # 추세
        fig.add_trace(
            go.Scatter(x=decomposition['trend'].index, y=decomposition['trend'].values,
                    mode='lines', name='추세'),
            row=2, col=1
        )
        
        # 계절성
        fig.add_trace(
            go.Scatter(x=decomposition['seasonal'].index, y=decomposition['seasonal'].values,
                    mode='lines', name='계절성'),
            row=3, col=1
        )
        
        # 잔차
        fig.add_trace(
            go.Scatter(x=decomposition['resid'].index, y=decomposition['resid'].values,
                    mode='lines', name='잔차'),
            row=4, col=1
        )
        
        # 스타일 업데이트
        fig.update_layout(
            height=800,
            margin=dict(l=10, r=10, t=30, b=10),
            showlegend=False
        )
        
        return fig
    
    def plot_acf_pacf(self,
                    acf_values: np.ndarray,
                    pacf_values: np.ndarray,
                    lags: int = 40,
                    **kwargs) -> go.Figure:
        """
        ACF 및 PACF 플롯을 생성합니다 (Plotly 버전).
        
        Args:
            acf_values: ACF 값
            pacf_values: PACF 값
            lags: 지연값 수
            
        Returns:
            Plotly Figure 객체
        """
        # 서브플롯 생성
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=(
                '자기상관 함수 (Autocorrelation Function)',
                '부분 자기상관 함수 (Partial Autocorrelation Function)'
            )
        )
        
        # x축 값 (lags)
        x = list(range(len(acf_values)))
        
        # 신뢰 구간 계산 (95%)
        confidence = 1.96 / np.sqrt(len(acf_values))
        
        # ACF 플롯 - stem 효과 (마커와 선 조합)
        for i in range(len(acf_values)):
            fig.add_trace(
                go.Scatter(
                    x=[i, i], 
                    y=[0, acf_values[i]], 
                    mode='lines',
                    line=dict(color='blue', width=1),
                    showlegend=False
                ),
                row=1, col=1
            )
        
        # ACF 마커
        fig.add_trace(
            go.Scatter(
                x=x, 
                y=acf_values, 
                mode='markers',
                marker=dict(color='blue', size=8),
                name='ACF'
            ),
            row=1, col=1
        )
        
        # 신뢰 구간 추가
        fig.add_trace(
            go.Scatter(
                x=[0, len(acf_values)-1],
                y=[confidence, confidence],
                mode='lines',
                line=dict(color='gray', width=1, dash='dash'),
                showlegend=False
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=[0, len(acf_values)-1],
                y=[-confidence, -confidence],
                mode='lines',
                line=dict(color='gray', width=1, dash='dash'),
                showlegend=False
            ),
            row=1, col=1
        )
        
        # PACF 플롯 - stem 효과
        for i in range(len(pacf_values)):
            fig.add_trace(
                go.Scatter(
                    x=[i, i], 
                    y=[0, pacf_values[i]], 
                    mode='lines',
                    line=dict(color='blue', width=1),
                    showlegend=False
                ),
                row=1, col=2
            )
        
        # PACF 마커
        fig.add_trace(
            go.Scatter(
                x=x, 
                y=pacf_values, 
                mode='markers',
                marker=dict(color='blue', size=8),
                name='PACF'
            ),
            row=1, col=2
        )
        
        # PACF 신뢰 구간
        fig.add_trace(
            go.Scatter(
                x=[0, len(pacf_values)-1],
                y=[confidence, confidence],
                mode='lines',
                line=dict(color='gray', width=1, dash='dash'),
                showlegend=False
            ),
            row=1, col=2
        )
        
        fig.add_trace(
            go.Scatter(
                x=[0, len(pacf_values)-1],
                y=[-confidence, -confidence],
                mode='lines',
                line=dict(color='gray', width=1, dash='dash'),
                showlegend=False
            ),
            row=1, col=2
        )
        
        # 레이아웃 업데이트
        fig.update_layout(
            height=400,
            margin=dict(l=10, r=10, t=50, b=10),
            showlegend=False,
        )
        
        # x축 및 y축 레이블
        fig.update_xaxes(title_text='지연 (Lag)', row=1, col=1)
        fig.update_xaxes(title_text='지연 (Lag)', row=1, col=2)
        fig.update_yaxes(title_text='상관도 (Correlation)', row=1, col=1)
        fig.update_yaxes(title_text='상관도 (Correlation)', row=1, col=2)
        
        return fig

    def plot_forecast_comparison(self,
                            train: pd.Series,
                            test: pd.Series,
                            forecasts: Dict[str, np.ndarray],
                            **kwargs) -> go.Figure:
        """
        여러 모델의 예측 결과를 비교하여 시각화합니다 (Plotly 버전).
        
        Args:
            train: 훈련 데이터
            test: 테스트 데이터
            forecasts: 모델별 예측값 딕셔너리
            
        Returns:
            Plotly Figure 객체
        """
        # 그래프 생성
        fig = go.Figure()
        
        # 훈련 데이터
        fig.add_trace(
            go.Scatter(
                x=train.index,
                y=train.values,
                mode='lines',
                name='Training Data',
                line=dict(color='blue', width=2)
            )
        )
        
        # 테스트 데이터
        fig.add_trace(
            go.Scatter(
                x=test.index,
                y=test.values,
                mode='lines',
                name='Actual Test Data',
                line=dict(color='green', width=2)
            )
        )
        
        # 각 모델의 예측
        colors = ['red', 'purple', 'orange', 'brown', 'pink', 'gray', 'olive']
        for i, (model_name, forecast) in enumerate(forecasts.items()):
            fig.add_trace(
                go.Scatter(
                    x=test.index,
                    y=forecast,
                    mode='lines',
                    name=f'{model_name} Forecast',
                    line=dict(color=colors[i % len(colors)], width=2, dash='dash')
                )
            )
        
        # 레이아웃 업데이트
        fig.update_layout(
            title='예측 비교 (Forecast Comparison)',
            xaxis_title='날짜 (Date)',
            yaxis_title='값 (Value)',
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            margin=dict(l=10, r=10, t=50, b=10),
            height=500,
        )
        
        # 날짜 형식 지정
        fig.update_xaxes(
            tickformat="%Y-%m-%d"
        )
        
        return fig

    def plot_metrics_comparison(self, metrics: Dict[str, Dict[str, float]]) -> go.Figure:
        """
        여러 모델의 성능 지표를 비교하여 시각화합니다 (Plotly 버전).
        
        Args:
            metrics: 모델별 성능 지표 딕셔너리
            
        Returns:
            Plotly Figure 객체
        """
        # 데이터 준비
        models = list(metrics.keys())
        metric_names = ['RMSE', 'MAE', 'R^2', 'MAPE']
        
        # 모든 모델에 있는 지표만 선택
        available_metrics = set.intersection(*[set(m.keys()) for m in metrics.values()])
        metric_names = [m for m in metric_names if m in available_metrics]
        
        # 서브플롯 생성
        fig = make_subplots(
            rows=len(metric_names), 
            cols=1,
            subplot_titles=[f'{metric} Comparison' for metric in metric_names]
        )
        
        # 각 메트릭별 바 차트 생성
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
            
            # 바 차트 추가
            fig.add_trace(
                go.Bar(
                    x=sorted_models,
                    y=sorted_values,
                    text=[f'{v:.4f}' for v in sorted_values],
                    textposition='outside',
                    marker_color=sorted_colors,
                    name=metric
                ),
                row=i+1, col=1
            )
        
        # 레이아웃 업데이트
        fig.update_layout(
            height=300 * len(metric_names),
            showlegend=False,
            margin=dict(l=10, r=10, t=50, b=10)
        )
        
        # y축 타이틀 업데이트
        for i, metric in enumerate(metric_names):
            fig.update_yaxes(title_text=metric, row=i+1, col=1)
        
        return fig

    def plot_residuals(
            self,
            actual: pd.Series,
            predicted: np.ndarray,
            title: str = "Residual Analysis",
            **kwargs
        ) -> go.Figure:
        """
        잔차 분석 플롯을 생성합니다 (Plotly 버전).
        
        Args:
            actual: 실제 값
            predicted: 예측 값
            title: 그래프 제목
            
        Returns:
            Plotly Figure 객체
        """
        # 길이 맞춤
        min_len = min(len(actual), len(predicted))
        actual_values = actual.iloc[:min_len].values
        predicted_values = predicted[:min_len]
        
        # 잔차 계산
        residuals = actual_values - predicted_values
        
        # 서브플롯 생성
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                '잔차 시계열 (Residuals Over Time)',
                '잔차 분포 (Residual Distribution)',
                '정규 Q-Q 플롯 (Normal Q-Q Plot)',
                '잔차 vs 예측값 (Residuals vs Predicted)'
            )
        )
        
        # 1. 잔차 시계열 플롯
        fig.add_trace(
            go.Scatter(
                x=actual.index[:min_len],
                y=residuals,
                mode='lines',
                name='Residuals'
            ),
            row=1, col=1
        )
        
        # 0 라인 추가
        fig.add_trace(
            go.Scatter(
                x=[actual.index[0], actual.index[min_len-1]],
                y=[0, 0],
                mode='lines',
                line=dict(color='red', width=1, dash='dash'),
                showlegend=False
            ),
            row=1, col=1
        )
        
        # 2. 잔차 히스토그램
        fig.add_trace(
            go.Histogram(
                x=residuals,
                nbinsx=20,
                marker_line_color='black',
                marker_line_width=1,
                opacity=0.7,
                name='Residual Distribution'
            ),
            row=1, col=2
        )
        
        osm, osr = stats.probplot(residuals, dist="norm", fit=False)
        
        fig.add_trace(
            go.Scatter(
                x=osm,
                y=osr,
                mode='markers',
                marker=dict(color='blue', size=6),
                name='Q-Q Plot'
            ),
            row=2, col=1
        )
        
        # 이론적인 정규분포 라인
        z = np.polyfit(osm, osr, 1)
        p = np.poly1d(z)
        fig.add_trace(
            go.Scatter(
                x=osm,
                y=p(osm),
                mode='lines',
                line=dict(color='red', width=1),
                showlegend=False
            ),
            row=2, col=1
        )
        
        # 4. 잔차 vs 예측값
        fig.add_trace(
            go.Scatter(
                x=predicted_values,
                y=residuals,
                mode='markers',
                marker=dict(color='blue', size=6),
                name='Residuals vs Predicted'
            ),
            row=2, col=2
        )
        
        # 0 라인 추가
        fig.add_trace(
            go.Scatter(
                x=[min(predicted_values), max(predicted_values)],
                y=[0, 0],
                mode='lines',
                line=dict(color='red', width=1, dash='dash'),
                showlegend=False
            ),
            row=2, col=2
        )
        
        # 레이아웃 업데이트
        fig.update_layout(
            title=title,
            height=800,
            showlegend=False,
            margin=dict(l=10, r=10, t=50, b=10)
        )
        
        # 축 레이블 업데이트
        fig.update_xaxes(title_text='날짜 (Date)', row=1, col=1)
        fig.update_xaxes(title_text='잔차 (Residual)', row=1, col=2)
        fig.update_xaxes(title_text='이론적 분위수 (Theoretical Quantiles)', row=2, col=1)
        fig.update_xaxes(title_text='예측값 (Predicted Values)', row=2, col=2)
        
        fig.update_yaxes(title_text='잔차 (Residual)', row=1, col=1)
        fig.update_yaxes(title_text='빈도 (Frequency)', row=1, col=2)
        fig.update_yaxes(title_text='정렬된 값 (Ordered Values)', row=2, col=1)
        fig.update_yaxes(title_text='잔차 (Residuals)', row=2, col=2)
        
        return fig

    def plot_feature_importance(self,
                            features: pd.DataFrame,
                            target: pd.Series,
                            top_n: int = 10,
                            **kwargs) -> go.Figure:
        """
        특성 중요도를 시각화합니다 (Plotly 버전).
        
        Args:
            features: 특성 데이터프레임
            target: 타겟 변수
            top_n: 표시할 상위 특성 수
            
        Returns:
            Plotly Figure 객체
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
        
        # 시각화
        fig = go.Figure()
        
        # 수평 막대 그래프
        fig.add_trace(
            go.Bar(
                y=top_features,
                x=top_importance,
                orientation='h',
                marker=dict(
                    color='rgba(50, 171, 96, 0.7)',
                    line=dict(color='rgba(50, 171, 96, 1.0)', width=1)
                )
            )
        )
        
        # 레이아웃 업데이트
        fig.update_layout(
            title='상위 특성 중요도 (Top Feature Importance)',
            xaxis_title='특성 중요도 (Feature Importance)',
            yaxis_title='특성 (Feature)',
            height=500,
            margin=dict(l=10, r=10, t=50, b=10)
        )
        
        # y축 역순 정렬 (중요도 높은 특성이 상단에 표시)
        fig.update_layout(yaxis={'categoryorder': 'total ascending'})
        
        return fig

@st.cache_data(ttl=3600)
def cached_plot_timeseries(data, title, xlabel, ylabel, color='#1f77b4'):
    """시계열 그래프 캐싱"""
    viz = TimeSeriesVisualizer()
    return viz.plot_timeseries(data, title=title, xlabel=xlabel, ylabel=ylabel, color=color)

@st.cache_data(ttl=3600)
def cached_plot_decomposition(decomposition):
    """분해 그래프 캐싱"""
    viz = TimeSeriesVisualizer()
    return viz.plot_decomposition(decomposition)

@st.cache_data(ttl=3600)
def cached_plot_acf_pacf(acf_values, pacf_values):
    """ACF/PACF 그래프 캐싱"""
    viz = TimeSeriesVisualizer()
    return viz.plot_acf_pacf(acf_values, pacf_values)

@st.cache_data(ttl=3600)
def cached_plot_forecast_comparison(train, test, forecasts):
    """예측 비교 그래프 캐싱"""
    viz = TimeSeriesVisualizer()
    return viz.plot_forecast_comparison(train, test, forecasts)

@st.cache_data(ttl=3600)
def cached_plot_metrics_comparison(metrics):
    """메트릭 비교 그래프 캐싱"""
    viz = TimeSeriesVisualizer()
    return viz.plot_metrics_comparison(metrics)

@st.cache_data(ttl=3600)
def cached_plot_residuals(actual, predicted, title="Residual Analysis"):
    """잔차 분석 그래프 캐싱"""
    viz = TimeSeriesVisualizer()
    return viz.plot_residuals(actual, predicted, title=title)
